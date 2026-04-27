"""Supabase helper functions with safe fallbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def get_supabase_client(secrets: Dict[str, Any]):
    """Create a Supabase client from Streamlit secrets, or return None."""
    try:
        from supabase import create_client
    except Exception:
        return None

    url = secrets.get("SUPABASE_URL") or secrets.get("supabase_url")
    key = secrets.get("SUPABASE_KEY") or secrets.get("supabase_key") or secrets.get("SUPABASE_ANON_KEY")
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
