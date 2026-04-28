"""Supabase helper functions with safe fallbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _read_secret(secrets: Dict[str, Any], *keys: str) -> Optional[str]:
    """Read first matching secret from flat keys or common nested sections."""
    sections = ("supabase", "SUPABASE")
    for key in keys:
        value = secrets.get(key)
        if value not in (None, ""):
            return str(value)
    for section in sections:
        nested = secrets.get(section)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            value = nested.get(key)
            if value not in (None, ""):
                return str(value)
            value = nested.get(key.lower())
            if value not in (None, ""):
                return str(value)
            value = nested.get(key.upper())
            if value not in (None, ""):
                return str(value)
    return None


def get_supabase_client(secrets: Dict[str, Any]):
    """Create a Supabase client from Streamlit secrets, or return None."""
    try:
        from supabase import create_client
    except Exception:
        return None

    url = _read_secret(secrets, "SUPABASE_URL", "supabase_url", "url")
    key = _read_secret(
        secrets,
        "SUPABASE_KEY",
        "supabase_key",
        "SUPABASE_ANON_KEY",
        "anon_key",
        "key",
    )
    if not url or not key:
        return None

    url = str(url).strip()
    key = str(key).strip()
    # Supabase python client expects project base URL (without /rest/v1)
    if "/rest/v1" in url:
        url = url.split("/rest/v1")[0]
    url = url.rstrip("/")

    try:
        return create_client(url, key)
    except Exception:
        return None


def safe_select(client, table: str, columns: str = "*", order_by: Optional[str] = None, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Select rows safely from a table."""
    if client is None:
        return [], "Supabase client is not configured."
    try:
        query = client.table(table).select(columns)
        if order_by:
            query = query.order(order_by, desc=True)
        if limit:
            query = query.limit(limit)
        result = query.execute()
        return result.data or [], None
    except Exception as exc:
        return [], str(exc)


def safe_insert(client, table: str, payload: Dict[str, Any]) -> Optional[str]:
    """Insert a single row safely."""
    if client is None:
        return "Supabase client is not configured."
    try:
        client.table(table).insert(payload).execute()
        return None
    except Exception as exc:
        return str(exc)


def compute_portfolio_heat(open_trades: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Sum risk percentage across open trades."""
    total = 0.0
    for row in open_trades:
        try:
            total += float(row.get("risk_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total, len(open_trades)
