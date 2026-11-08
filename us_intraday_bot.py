"""
US intraday momentum + liquidity bot (scanner + ORB/VWAP strategy + backtest).

Install:   pip install alpaca-py pandas numpy
Keys:      export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
Run:       python us_intraday_bot.py --start 2025-01-01 --end 2025-12-31
Test only: python us_intraday_bot.py --demo        (synthetic data, no keys)

Logic (long only):
 1. Liquidity filter: prev close >= MIN_PRICE and 20d avg dollar volume >= MIN_DOLLAR_VOL
 2. Momentum filter at 09:45: gap-up >= MIN_GAP and opening-range volume >= MIN_RVOL x normal
 3. Entry: 1-min close breaks opening-range high AND is above VWAP (fill next bar open)
 4. Stop = VWAP at entry, target = TARGET_R x risk, forced exit at 15:55
 5. Risk: RISK_PCT of equity per trade, position cap, daily loss limit
NOTE: free Alpaca data is IEX only (thin volume) -> lower MIN_DOLLAR_VOL, or use --feed sip.
Backtest only. Paper trade for weeks before any real money.
"""
import argparse, math, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

TZ = "America/New_York"
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "NFLX", "AVGO",
            "CRM", "ORCL", "ADBE", "INTC", "MU", "SMCI", "PLTR", "COIN", "UBER", "SHOP",
            "JPM", "BAC", "XOM", "BA", "DIS", "PYPL", "SQ", "SOFI", "RIVN", "F"]


@dataclass
class Cfg:
    start_equity: float = 10_000
    min_price: float = 5.0
    min_dollar_vol: float = 20_000_000   # lower this for IEX data
    min_gap: float = 0.02
    min_rvol: float = 1.5
    or_minutes: int = 15
    last_entry: str = "15:30"
    exit_time: str = "15:55"
    target_r: float = 2.0
    min_risk_pct: float = 0.004
    max_risk_pct: float = 0.03
    risk_pct: float = 0.005              # 0.5% of equity per trade
    max_pos_pct: float = 0.25
    max_trades_day: int = 4
    daily_loss_limit: float = 0.02
    slippage_bps: float = 5
    feed: str = "iex"


# ---------------- data ----------------
def fetch_minutes(symbols, start, end, feed):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    cl = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    out = {}
    for s in symbols:
        req = StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Minute,
                               start=pd.Timestamp(start, tz=TZ), end=pd.Timestamp(end, tz=TZ),
                               feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        df = cl.get_stock_bars(req).df
        if df.empty:
            continue
        df = df.droplevel(0) if isinstance(df.index, pd.MultiIndex) else df
        df.index = df.index.tz_convert(TZ)
        out[s] = df.between_time("09:30", "15:59")[["open", "high", "low", "close", "volume"]]
        print(f"loaded {s}: {len(out[s])} bars")
    return out


def demo_data(symbols, start, end):
    rng = np.random.default_rng(7)
    days = pd.bdate_range(start, end)
    out = {}
    for s in symbols[:10]:
        p, rows = rng.uniform(20, 200), []
        base_vol = rng.uniform(2e4, 8e4)
        for d in days:
            p *= 1 + rng.normal(0.002, 0.02)  # overnight gap
            idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="min", tz=TZ)
            r = rng.normal(0, 0.0012, 390).cumsum() + rng.normal(0.00002, 0, 390)
            c = p * np.exp(r)
            v = base_vol * (1 + 2 * np.exp(-np.arange(390) / 30)) * rng.uniform(0.5, 1.6)
            o = np.r_[c[0], c[:-1]]
            rows.append(pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005,
                                      "low": np.minimum(o, c) * 0.9995, "close": c, "volume": v}, index=idx))
            p = c[-1]
        out[s] = pd.concat(rows)
    return out


# ---------------- scanner ----------------
def daily_features(m: pd.DataFrame) -> pd.DataFrame:
    d = m.groupby(m.index.date).agg(open=("open", "first"), close=("close", "last"), volume=("volume", "sum"))
    d["dollar_vol"] = (m["close"] * m["volume"]).groupby(m.index.date).sum()
    d["prev_close"] = d["close"].shift(1)
    d["avg_vol20"] = d["volume"].rolling(20).mean().shift(1)       # shift(1): no look-ahead
    d["avg_dvol20"] = d["dollar_vol"].rolling(20).mean().shift(1)
    return d


def scan(sym, day, bars, feat, cfg):
    """Return candidate dict if the symbol passes liquidity + momentum filters at the end of the OR."""
    if pd.isna(feat.prev_close) or pd.isna(feat.avg_vol20) or len(bars) < cfg.or_minutes:
        return None
    orb = bars.iloc[:cfg.or_minutes]
    gap = orb["open"].iloc[0] / feat.prev_close - 1
    rvol = orb["volume"].sum() / (feat.avg_vol20 * cfg.or_minutes / 390 * 2)  # x2: open is ~2x busier than avg minute
    if feat.prev_close < cfg.min_price or feat.avg_dvol20 < cfg.min_dollar_vol:
        return None
    if gap < cfg.min_gap or rvol < cfg.min_rvol:
        return None
    return dict(sym=sym, gap=gap, rvol=rvol, orh=orb["high"].max(), orl=orb["low"].min())


# ---------------- strategy ----------------
def simulate(cand, bars, cfg):
    """One long trade at most. Returns dict (price-level trade, sizing applied later) or None."""
    h, l, c, o, v = (bars[k].to_numpy() for k in ("high", "low", "close", "open", "volume"))
    tp = (h + l + c) / 3
    vwap = np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1)
    times = bars.index
    last_entry = pd.Timestamp(f"{times[0].date()} {cfg.last_entry}", tz=TZ)
    exit_t = pd.Timestamp(f"{times[0].date()} {cfg.exit_time}", tz=TZ)
    end = int(np.searchsorted(times.values, exit_t.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right")) - 1
    slip = cfg.slippage_bps / 1e4
    for i in range(cfg.or_minutes, len(bars) - 1):
        if times[i] > last_entry:
            break
        if c[i] > cand["orh"] and c[i] > vwap[i]:
            entry = o[i + 1] * (1 + slip)
            stop = vwap[i]
            risk = entry - stop
            if not (cfg.min_risk_pct <= risk / entry <= cfg.max_risk_pct):
                continue
            target = entry + cfg.target_r * risk
            for j in range(i + 1, end + 1):
                if l[j] <= stop:
                    px, why = min(o[j], stop) * (1 - slip), "stop"
                    break
                if h[j] >= target:
                    px, why = target * (1 - slip), "target"
                    break
            else:
                px, why = c[end] * (1 - slip), "eod"
                j = end
            return dict(sym=cand["sym"], entry_t=times[i + 1], exit_t=times[j], entry=entry,
                        exit=px, risk_ps=risk, why=why, gap=cand["gap"], rvol=cand["rvol"])
    return None


# ---------------- backtest ----------------
def backtest(data, cfg):
    feats = {s: daily_features(m) for s, m in data.items()}
    days = sorted({d for m in data.values() for d in set(m.index.date)})
    equity, curve, trades = cfg.start_equity, [], []
    for day in days:
        cands = []
        for s, m in data.items():
            if day not in feats[s].index:
                continue
            bars = m[m.index.date == day]
            cand = scan(s, day, bars, feats[s].loc[day], cfg)
            if cand:
                cands.append((cand, bars))
        cands.sort(key=lambda x: x[0]["rvol"] * x[0]["gap"], reverse=True)
        day_trades = [t for t in (simulate(c, b, cfg) for c, b in cands[:cfg.max_trades_day * 2]) if t]
        day_trades.sort(key=lambda t: t["entry_t"])
        start_eq, taken = equity, []
        for t in day_trades:
            if len(taken) >= cfg.max_trades_day:
                break
            realized = sum(x["pnl"] for x in taken if x["exit_t"] <= t["entry_t"])
            if realized <= -cfg.daily_loss_limit * start_eq:
                break
            shares = min(math.floor(start_eq * cfg.risk_pct / t["risk_ps"]),
                         math.floor(start_eq * cfg.max_pos_pct / t["entry"]))
            if shares < 1:
                continue
            t["shares"], t["pnl"] = shares, shares * (t["exit"] - t["entry"])
            t["R"] = t["pnl"] / (shares * t["risk_ps"])
            t["day"] = day
            taken.append(t)
        trades += taken
        equity += sum(x["pnl"] for x in taken)
        curve.append((day, equity))
    return pd.DataFrame(trades), pd.DataFrame(curve, columns=["day", "equity"]).set_index("day")


def report(tr, curve, cfg):
    if tr.empty:
        print("No trades. Loosen filters (min_gap / min_rvol / min_dollar_vol).")
        return
    wins, losses = tr[tr.pnl > 0].pnl.sum(), -tr[tr.pnl <= 0].pnl.sum()
    dd = (curve.equity / curve.equity.cummax() - 1).min()
    print(f"\ntrades: {len(tr)} | win rate: {(tr.pnl > 0).mean():.1%} | avg R: {tr.R.mean():.2f}")
    print(f"profit factor: {wins / losses if losses else float('inf'):.2f}")
    print(f"return: {curve.equity.iloc[-1] / cfg.start_equity - 1:.1%} | max drawdown: {dd:.1%}")
    print(tr.why.value_counts().to_string())
    tr.to_csv("trades.csv", index=False)
    print("saved trades.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    cfg = Cfg(feed=a.feed)
    if a.demo:
        cfg.min_dollar_vol, cfg.min_gap, cfg.min_rvol = 0, 0.0, 0.5
        data = demo_data(UNIVERSE, a.start, a.end)
    else:
        data = fetch_minutes(UNIVERSE, a.start, a.end, a.feed)
    tr, curve = backtest(data, cfg)
    report(tr, curve, cfg)
