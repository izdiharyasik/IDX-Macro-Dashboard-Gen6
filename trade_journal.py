"""
Trade Journal — logs every trade, tracks results, adjusts system weights.
Stored as JSON on disk. Persists across sessions.
"""

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

JOURNAL_FILE = "trade_journal.json"

DEFAULT_WEIGHTS = {
    "macro": 0.30,
    "technical": 0.30,
    "sentiment": 0.20,
    "fundamental": 0.20,
}

def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return {"trades": [], "learned_weights": DEFAULT_WEIGHTS.copy(), "stats": {}}

    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            journal = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"trades": [], "learned_weights": DEFAULT_WEIGHTS.copy(), "stats": {}}

    # Backward/partial compatibility for older journal schema.
    journal.setdefault("trades", [])
    journal.setdefault("learned_weights", DEFAULT_WEIGHTS.copy())
    journal.setdefault("stats", {})

    # Ensure all weight keys exist even if older file is missing one.
    for key, default_value in DEFAULT_WEIGHTS.items():
        journal["learned_weights"].setdefault(key, default_value)
    return journal

def save_journal(journal):
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, default=str)

def log_trade(ticker, trade_type, composite, macro_score, tech_score,
              sent_score, fund_score, entry_price, stop_loss, take_profit,
              lots, regime, sector, playbook_action):
    """Log a new trade setup when placed."""
    journal = load_journal()
    trade = {
        "id":            len(journal["trades"]) + 1,
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "ticker":        ticker,
        "trade_type":    trade_type,
        "regime":        regime,
        "sector":        sector,
        "composite":     composite,
        "macro_score":   macro_score,
        "tech_score":    tech_score,
        "sent_score":    sent_score,
        "fund_score":    fund_score,
        "entry_price":   entry_price,
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
        "lots":          lots,
        "playbook":      playbook_action,
        "status":        "OPEN",
        "exit_price":    None,
        "exit_date":     None,
        "pnl_pct":       None,
        "result":        None,
        "expiry_date":   None,   # set based on trade type
    }
    # Set expiry
    hold = {"SCALP": 1, "SWING": 5, "POSITION": 20}.get(trade_type, 5)
    expiry = pd.Timestamp.now() + pd.Timedelta(days=hold + 2)
    trade["expiry_date"] = expiry.strftime("%Y-%m-%d")

    journal["trades"].append(trade)
    save_journal(journal)
    return trade["id"]

def close_trade(trade_id, exit_price, exit_date=None):
    """Mark a trade as closed and record result."""
    journal = load_journal()
    changed = False
    for t in journal["trades"]:
        if t["id"] == trade_id and t["status"] == "OPEN":
            ep  = t["entry_price"]
            pnl = (exit_price - ep) / ep * 100
            t.update({
                "exit_price": exit_price,
                "exit_date":  exit_date or datetime.now().strftime("%Y-%m-%d"),
                "pnl_pct":    round(pnl, 2),
                "result":     "WIN" if pnl > 0 else "LOSS",
                "status":     "CLOSED",
            })
            changed = True
            break
    if changed:
        _update_learned_weights(journal)
        save_journal(journal)
    return changed

def expire_stale_trades():
    """Auto-expire trades that hit their expiry date without being filled."""
    journal = load_journal()
    today   = datetime.now().date()
    changed = False
    for t in journal["trades"]:
        if t["status"] == "OPEN" and t.get("expiry_date"):
            try:
                exp = datetime.strptime(t["expiry_date"], "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            if today > exp:
                t["status"]  = "EXPIRED"
                t["result"]  = "EXPIRED"
                t["exit_date"] = str(today)
                changed = True
    if changed:
        save_journal(journal)
    return [t for t in journal["trades"] if t.get("result") == "EXPIRED" and
            t.get("exit_date") == str(today)]

def _update_learned_weights(journal):
    """
    Adaptive learning: boost weights of factors that predicted wins,
    reduce weights that predicted losses.
    Only updates after 10+ closed trades.
    """
    closed = [t for t in journal["trades"] if t["status"] == "CLOSED"]
    if len(closed) < 10:
        return  # not enough data yet

    wins  = [t for t in closed if t["result"] == "WIN"]
    losses= [t for t in closed if t["result"] == "LOSS"]

    if not wins or not losses:
        return

    # Average scores for wins vs losses
    def avg(trades, key):
        vals = [t.get(key, 0) for t in trades if t.get(key) is not None]
        return np.mean(vals) if vals else 0

    factors = ["macro_score", "tech_score", "sent_score", "fund_score"]
    keys    = ["macro", "technical", "sentiment", "fundamental"]

    win_avgs  = {k: avg(wins,   f) for k, f in zip(keys, factors)}
    loss_avgs = {k: avg(losses, f) for k, f in zip(keys, factors)}

    # Factor that separates wins from losses most = higher weight
    separations = {}
    for k in keys:
        separations[k] = max(win_avgs[k] - loss_avgs[k], 0.001)

    total = sum(separations.values())
    new_weights = {k: round(v / total, 3) for k, v in separations.items()}

    # Blend 70% learned, 30% default (don't drift too far)
    blended = {}
    for k in keys:
        blended[k] = round(0.70 * new_weights[k] + 0.30 * DEFAULT_WEIGHTS[k], 3)

    # Renormalize
    total_b = sum(blended.values())
    journal["learned_weights"] = {k: round(v / total_b, 3) for k, v in blended.items()}
    journal["stats"] = {
        "total_trades":  len(closed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      f"{len(wins)/len(closed)*100:.1f}%",
        "avg_win_pct":   f"{np.mean([t['pnl_pct'] for t in wins]):.2f}%",
        "avg_loss_pct":  f"{np.mean([t['pnl_pct'] for t in losses]):.2f}%",
        "last_updated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

def get_learned_weights():
    journal = load_journal()
    return journal.get("learned_weights", DEFAULT_WEIGHTS.copy())

def get_journal_df():
    journal = load_journal()
    if not journal["trades"]:
        return pd.DataFrame()
    df = pd.DataFrame(journal["trades"])
    return df

def get_journal_stats():
    journal = load_journal()
    return journal.get("stats", {}), journal.get("learned_weights", DEFAULT_WEIGHTS.copy())

def get_open_trades():
    journal = load_journal()
    return [t for t in journal["trades"] if t["status"] == "OPEN"]
