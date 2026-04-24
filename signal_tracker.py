import json
import os
from datetime import datetime, date
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

SIGNALS_FILE = "signals_db.json"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _safe_float(x, default=0.0):
    try:
        if isinstance(x, str):
            x = x.replace("Rp", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return float(default)


def load_signals() -> Dict:
    if not os.path.exists(SIGNALS_FILE):
        return {"signals": []}
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"signals": []}
    data.setdefault("signals", [])
    return data


def save_signals(db: Dict):
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def _signal_key(sig: Dict) -> str:
    return f"{sig.get('ticker')}|{sig.get('entry')}|{sig.get('stop_loss')}|{sig.get('take_profit')}|{sig.get('timestamp')}"


def register_signals_from_plan(plan: Dict):
    db = load_signals()
    existing = {_signal_key(s): s for s in db.get("signals", [])}
    created = 0

    for bucket in ["POSITION", "SWING", "SCALP", "HIGH_BETA"]:
        for t in plan.get(bucket, []):
            expiry = t.get("order_expiry") or (date.today() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            sig = {
                "ticker": t["ticker"],
                "entry": _safe_float(t.get("entry", 0)),
                "stop_loss": _safe_float(t.get("stop_loss", 0)),
                "take_profit": _safe_float(t.get("take_profit", 0)),
                "timestamp": _now_iso(),
                "expiry_date": expiry,
                "initial_confidence": float(t.get("confidence_score", 0)),
                "confidence_label": t.get("confidence_label", "C"),
                "status": "ACTIVE",
                "current_price": None,
                "current_return_pct": None,
                "days_since_signal": 0,
            }

            duplicate_active = any(
                s.get("ticker") == sig["ticker"]
                and s.get("status") == "ACTIVE"
                and abs(float(s.get("entry", 0)) - sig["entry"]) < 1e-9
                for s in db.get("signals", [])
            )
            if duplicate_active:
                continue

            key = _signal_key(sig)
            if key not in existing:
                db["signals"].append(sig)
                created += 1

    if created:
        save_signals(db)
    return created


def _latest_prices(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    prices = {}
    try:
        data = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    prices[t] = float(data[t]["Close"].dropna().iloc[-1])
                except Exception:
                    pass
        else:
            # single ticker fallback
            c = data["Close"].dropna()
            if len(c):
                prices[tickers[0]] = float(c.iloc[-1])
    except Exception:
        pass
    return prices


def update_signal_statuses() -> List[Dict]:
    db = load_signals()
    signals = db.get("signals", [])
    if not signals:
        return []

    tickers = sorted({s.get("ticker") for s in signals if s.get("ticker")})
    prices = _latest_prices(tickers)
    today = date.today()

    for s in signals:
        ts_raw = s.get("timestamp")
        try:
            ts_date = datetime.fromisoformat(ts_raw).date() if ts_raw else today
        except ValueError:
            ts_date = today
        days_old = max((today - ts_date).days, 0)
        s["days_since_signal"] = days_old

        px = prices.get(s.get("ticker"))
        if px is not None:
            s["current_price"] = round(float(px), 2)
            entry = _safe_float(s.get("entry", 0), 0)
            if entry > 0:
                s["current_return_pct"] = round((px - entry) / entry * 100, 2)

        if s.get("status") not in ("TP HIT", "SL HIT", "EXPIRED"):
            try:
                exp = datetime.strptime(s.get("expiry_date", ""), "%Y-%m-%d").date()
            except Exception:
                exp = today

            entry = _safe_float(s.get("entry", 0), 0)
            sl = _safe_float(s.get("stop_loss", 0), 0)
            tp = _safe_float(s.get("take_profit", 0), 0)
            cur = s.get("current_price")

            if today > exp:
                s["status"] = "EXPIRED"
            elif cur is not None and tp > 0 and cur >= tp:
                s["status"] = "TP HIT"
            elif cur is not None and sl > 0 and cur <= sl:
                s["status"] = "SL HIT"
            else:
                s["status"] = "ACTIVE"

    save_signals(db)
    return signals


def compute_signal_performance(signals: List[Dict]) -> Dict:
    if not signals:
        return {
            "win_rate_7d": "0.0%", "win_rate_30d": "0.0%", "win_rate_all": "0.0%",
            "avg_return": "0.00%", "expectancy": "0.00%", "max_drawdown": "0.00%"
        }

    df = pd.DataFrame(signals)
    if df.empty:
        return {
            "win_rate_7d": "0.0%", "win_rate_30d": "0.0%", "win_rate_all": "0.0%",
            "avg_return": "0.00%", "expectancy": "0.00%", "max_drawdown": "0.00%"
        }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["closed"] = df["status"].isin(["TP HIT", "SL HIT", "EXPIRED"])
    df["ret"] = pd.to_numeric(df.get("current_return_pct"), errors="coerce").fillna(0.0)
    closed = df[df["closed"]].copy().sort_values("timestamp")

    def _win_rate(days=None):
        data = closed
        if days is not None:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
            data = data[data["timestamp"] >= cutoff]
        if len(data) == 0:
            return "0.0%"
        wins = (data["status"] == "TP HIT").sum()
        return f"{wins / len(data) * 100:.1f}%"

    avg_return = float(closed["ret"].mean()) if len(closed) else 0.0
    wins = closed[closed["status"] == "TP HIT"]["ret"]
    losses = closed[closed["status"].isin(["SL HIT", "EXPIRED"])]["ret"]
    win_rate = (len(wins) / len(closed)) if len(closed) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    if len(closed):
        equity = (1 + closed["ret"].fillna(0) / 100).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min() * 100)
    else:
        max_dd = 0.0

    return {
        "win_rate_7d": _win_rate(7),
        "win_rate_30d": _win_rate(30),
        "win_rate_all": _win_rate(None),
        "avg_return": f"{avg_return:.2f}%",
        "expectancy": f"{expectancy:.2f}%",
        "max_drawdown": f"{max_dd:.2f}%",
    }
