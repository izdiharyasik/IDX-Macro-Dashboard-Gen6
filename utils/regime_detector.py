"""Operating mode and regime detection utilities for Gen 7."""

from __future__ import annotations

from typing import Dict, Tuple


def classify_ihsg_regime(price: float, ema50: float) -> str:
    """Classify IHSG regime from price relative to EMA50."""
    if ema50 <= 0:
        return "NEUTRAL"
    distance = (price - ema50) / ema50
    if abs(distance) <= 0.005:
        return "NEUTRAL"
    return "RISK-ON" if distance > 0 else "RISK-OFF"


def classify_vix_bucket(vix: float) -> str:
    if vix > 30:
        return "BUY ZONE SIGNAL"
    if vix >= 20:
        return "ELEVATED"
    return "CALM"


def detect_operating_mode(vix: float, dg_score: int, guardrails: Dict[str, bool]) -> Dict[str, str]:
    """Return operating mode metadata."""
    guardrail_triggered = any(guardrails.values())

    if vix > 30:
        if dg_score >= 4 and not guardrail_triggered:
            return {
                "mode": "CRISIS ENTRY",
                "color": "green",
                "message": "CRISIS DETECTED. DG framework active. Check all 5 boxes before sizing up.",
                "risk_guidance": "1.5%–2.0% risk/trade",
            }
        return {
            "mode": "CRISIS ENTRY",
            "color": "red",
            "message": "CRISIS DETECTED but guardrails active. Stay selective.",
            "risk_guidance": "0.5%–1.0% risk/trade",
        }

    if 20 <= vix <= 30:
        return {
            "mode": "ELEVATED RISK",
            "color": "amber",
            "message": "Market elevated. Half sizing. Wait for confirmation.",
            "risk_guidance": "0.5% risk/trade (score > 0.7, 3/5 DG minimum)",
        }

    return {
        "mode": "DAILY GRIND",
        "color": "blue",
        "message": "Normal market conditions. Clip singles. Standard sizing.",
        "risk_guidance": "1.0% risk/trade",
    }
