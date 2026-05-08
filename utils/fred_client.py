"""Macro data client helpers.

FRED's JSON API requires a key, but FRED also publishes public graph CSV
endpoints for the same series.  The app prefers the API when a key is present
and falls back to that public CSV feed so Streamlit Cloud deployments do not
need a FRED secret just to boot with live macro data.
"""

from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, List, Optional

import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred(series_id: str, api_key: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch descending-ordered FRED observations for a series via the keyed API."""
    if not api_key:
        return []
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


def fetch_fred_csv(series_id: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch descending observations from FRED's public CSV graph endpoint.

    The CSV endpoint does not require an API key.  It returns ascending rows
    named ``observation_date`` and ``<series_id>``; this normalizes them to the
    same ``date``/``value`` shape returned by the JSON API helper.
    """
    params = {"id": series_id}
    response = requests.get(FRED_GRAPH_CSV_URL, params=params, timeout=10)
    response.raise_for_status()

    observations: List[Dict[str, str]] = []
    reader = csv.DictReader(StringIO(response.text))
    for row in reader:
        value = row.get(series_id) or row.get("value") or "."
        observations.append({"date": row.get("observation_date", ""), "value": value})

    observations = [obs for obs in observations if obs.get("date")]
    observations.sort(key=lambda obs: obs["date"], reverse=True)
    return observations[:limit]


def fetch_macro_observations(series_id: str, api_key: Optional[str] = None, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch macro observations, using FRED API key when available, then CSV fallback."""
    if api_key:
        try:
            observations = fetch_fred(series_id, api_key, limit=limit)
            if observations:
                return observations
        except Exception:
            pass
    return fetch_fred_csv(series_id, limit=limit)


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
