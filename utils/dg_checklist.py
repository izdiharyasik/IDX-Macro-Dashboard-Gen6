"""DG 5-box checklist scoring helpers."""

from __future__ import annotations

from typing import Dict, Tuple


def build_dg_metrics(vix: float, fed_stance: str, margin_direction: str, sector_leader: str, earnings_beats: int) -> Dict[str, Dict[str, object]]:
    """Build DG metric status dictionary."""
    metric1 = vix > 30
    metric2 = fed_stance in {"CUTTING", "HOLDING"}
    metric3 = margin_direction == "DELEVERAGING"
    metric4 = bool(sector_leader)
    metric5 = earnings_beats >= 2

    return {
        "Metric 1: VIX > 30": {"ok": metric1, "detail": f"Current: {vix:.2f}"},
        "Metric 2: Fed not hiking": {"ok": metric2, "detail": f"Status: {fed_stance}"},
        "Metric 3: Margin deleveraging": {"ok": metric3, "detail": margin_direction},
        "Metric 4: Clear sector leader": {"ok": metric4, "detail": sector_leader or "No clear leader"},
        "Metric 5: Earnings beats": {"ok": metric5, "detail": f"{earnings_beats} leaders beating"},
    }


def score_dg(metrics: Dict[str, Dict[str, object]]) -> Tuple[int, str]:
    score = sum(1 for payload in metrics.values() if bool(payload.get("ok")))
    if score == 5:
        verdict = "MAXIMUM CONVICTION — Full DG crisis entry. 2x sizing."
    elif score == 4:
        verdict = "STRONG SIGNAL — Size up to 1.5x."
    elif score == 3:
        verdict = "PARTIAL SIGNAL — Standard sizing only."
    else:
        verdict = "INSUFFICIENT — Do not use crisis sizing. Normal mode only."
    return score, verdict
