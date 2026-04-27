"""FRED API client helpers."""

from __future__ import annotations

from typing import Dict, List, Optional
import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred(series_id: str, api_key: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch descending-ordered FRED observations for a series."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    response = requests.get(FRED_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload.get("observations", [])


def _to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ".", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_latest_value(observations: List[Dict[str, str]]) -> Optional[float]:
    """Return latest numeric value from observations."""
    for obs in observations:
        parsed = _to_float(obs.get("value"))
        if parsed is not None:
            return parsed
    return None


def parse_change_arrow(observations: List[Dict[str, str]]) -> str:
    """Return arrow based on latest two valid observations."""
    values: List[float] = []
    for obs in observations:
        parsed = _to_float(obs.get("value"))
        if parsed is not None:
            values.append(parsed)
        if len(values) == 2:
            break

    if len(values) < 2:
        return "→"
    if values[0] > values[1]:
        return "↑"
    if values[0] < values[1]:
        return "↓"
    return "→"
