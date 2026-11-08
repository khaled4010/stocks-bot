"""
Telegram signal bot for the US intraday momentum strategy (signals only, no auto-trading).

Setup:
  1. Telegram -> @BotFather -> /newbot -> copy the token
  2. Send any message to your new bot, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates  and copy "chat":{"id": ...}
  3. export TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...
     export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...  EQUITY=10000
  4. pip install alpaca-py pandas numpy requests
  5. python telegram_signals.py        (keep it running during US market hours, e.g. on a VPS)

Needs us_intraday_bot.py in the same folder (same filters / entry / stop / target logic).
Flow: 09:46 scan -> watchlist message -> signal when price breaks the opening range above VWAP
      -> alerts on stop / target / 15:55 close-all. Times are US Eastern.
"""
import os, math, time
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pandas as pd
import requests
from us_intraday_bot import Cfg, UNIVERSE, TZ, scan, fetch_minutes

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
EQUITY = float(os.environ.get("EQUITY", "10000"))
cfg = Cfg()
cfg.last_entry = os.environ.get("LAST_ENTRY", cfg.last_entry)
cfg.exit_time = os.environ.get("EXIT_TIME", cfg.exit_time)
ONE_DAY = bool(os.environ.get("ONE_DAY"))   # free GitHub Actions mode: run one session then exit


def tg(text):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT, "text": text}, timeout=15)
    except Exception as e:
        print("telegram error:", e)
    print(text, "\n")


def fetch_today(client, symbols):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    now = pd.Timestamp.now(tz=TZ)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=now.normalize() + pd.Timedelta(hours=9, minutes=30), end=now,
                           feed=DataFeed.SIP if cfg.feed == "sip" else DataFeed.IEX)
    df = client.get_stock_bars(req).df
    out = {}
    if df.empty:
        return out
    for s, g in df.groupby(level=0):
        g = g.droplevel(0)
        g.index = g.index.tz_convert(TZ)
        out[s] = g[["open", "high", "low", "close", "volume"]]
    return out


def build_feats():
    now = pd.Timestamp.now(tz=TZ)
    hist = fetch_minutes(UNIVERSE, (now - timedelta(days=45)).date().isoformat(),
                         now.normalize().isoformat(), cfg.feed)
    feats = {}
    for s, m in hist.items():
        d = m.groupby(m.index.date).agg(close=("close", "last"), volume=("volume", "sum"))
        dv = (m["close"] * m["volume"]).groupby(m.index.date).sum()
        if len(d) >= 20:
            feats[s] = SimpleNamespace(prev_close=d.close.iloc[-1], avg_vol20=d.volume.tail(20).mean(),
                                       avg_dvol20=dv.tail(20).mean())
    return feats


def main():
    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    feats = build_feats()
    tg("🟢 البوت اشتغل. السكان الساعة 09:46 بتوقيت نيويورك.")
    day, scanned, cands, state, n_taken, day_pnl, halted = None, False, {}, {}, 0, 0.0, False

    while True:
        now = pd.Timestamp.now(tz=TZ)
        if now.date() != day:
            day, scanned, cands, state, n_taken, day_pnl, halted = now.date(), False, {}, {}, 0, 0.0, False
        t = now.strftime("%H:%M")
        if ONE_DAY and t > "16:00":
            tg("🔴 انتهت الجلسة. البوت توقف.")
            return
        if now.weekday() >= 5 or t < "09:31" or t > "16:00":
            time.sleep(30)
            continue
        try:
            bars_all = fetch_today(client, list(feats))
        except Exception as e:
            print("data error:", e)
            time.sleep(20)
            continue
        if not bars_all:                     # holiday / no data yet
            if ONE_DAY and t >= "10:00":
                tg("السوق مغلق اليوم (لا توجد بيانات).")
                return
            time.sleep(30)
            continue

        # 1) scan once, right after the opening range
        if not scanned and t >= "09:46":
            scanned = True
            found = []
            for s, b in bars_all.items():
                c = scan(s, day, b, feats[s], cfg)
                if c:
                    found.append(c)
            found.sort(key=lambda c: c["rvol"] * c["gap"], reverse=True)
            cands = {c["sym"]: c for c in found[:cfg.max_trades_day * 2]}
            if cands:
                lines = [f"{c['sym']}: فجوة {c['gap']:.1%} | حجم x{c['rvol']:.1f} | "
                         f"أعلى النطاق {c['orh']:.2f}" for c in cands.values()]
                tg("📋 قائمة المراقبة:\n" + "\n".join(lines))
            else:
                tg("📋 لا توجد أسهم تحقق الشروط اليوم.")

        # 2) signals + trade management
        for s, c in cands.items():
            b = bars_all.get(s)
            if b is None or len(b) <= cfg.or_minutes:
                continue
            tp = (b.high + b.low + b.close) / 3
            vwap = (tp * b.volume).cumsum() / b.volume.cumsum().clip(lower=1)
            st = state.get(s)

            if st is None:
                if halted or t > cfg.last_entry or n_taken >= cfg.max_trades_day:
                    continue
                px, vw = b.close.iloc[-1], vwap.iloc[-1]
                if px > c["orh"] and px > vw:
                    risk = px - vw
                    if not (cfg.min_risk_pct <= risk / px <= cfg.max_risk_pct):
                        continue
                    shares = min(math.floor(EQUITY * cfg.risk_pct / risk),
                                 math.floor(EQUITY * cfg.max_pos_pct / px))
                    if shares < 1:
                        continue
                    target = px + cfg.target_r * risk
                    state[s] = dict(entry=px, stop=vw, target=target, shares=shares,
                                    last_ts=b.index[-1], open=True)
                    n_taken += 1
                    tg(f"🚀 إشارة شراء {s}\nدخول تقريبي: {px:.2f}\nوقف: {vw:.2f}\n"
                       f"هدف: {target:.2f}\nالكمية المقترحة: {shares} سهم "
                       f"(مخاطرة ~{shares * risk:.0f}$)")
            elif st["open"]:
                new = b[b.index > st["last_ts"]]
                st["last_ts"] = b.index[-1]
                for _, r in new.iterrows():
                    if r.low <= st["stop"]:
                        pnl = st["shares"] * (st["stop"] - st["entry"])
                        st.update(open=False)
                        day_pnl += pnl
                        tg(f"🛑 ضرب الوقف {s} عند {st['stop']:.2f} (≈ {pnl:+.0f}$)")
                        break
                    if r.high >= st["target"]:
                        pnl = st["shares"] * (st["target"] - st["entry"])
                        st.update(open=False)
                        day_pnl += pnl
                        tg(f"✅ تحقق الهدف {s} عند {st['target']:.2f} (≈ {pnl:+.0f}$)")
                        break

        if not halted and day_pnl <= -cfg.daily_loss_limit * EQUITY:
            halted = True
            tg(f"⛔ وصلنا حد الخسارة اليومي ({day_pnl:+.0f}$). لا إشارات جديدة اليوم.")

        # 3) end-of-day close-all
        if t >= cfg.exit_time:
            for s, st in state.items():
                if st["open"]:
                    px = bars_all[s].close.iloc[-1]
                    pnl = st["shares"] * (px - st["entry"])
                    st["open"] = False
                    tg(f"⏰ أغلق {s} الآن عند ~{px:.2f} (≈ {pnl:+.0f}$)")
            if ONE_DAY:
                tg("🔴 انتهت جلسة اليوم. البوت توقف.")
                return

        time.sleep(max(5, 63 - now.second))


if __name__ == "__main__":
    main()
