"""
Supabase products table upsert.
"""
from datetime import datetime, timezone
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY is not set (use .env or environment)")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def upsert_product(row: dict) -> None:
    """Insert or update one product row. Uses (source, product_url) as conflict key."""
    payload = dict(row)
    payload["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    client = get_client()
    client.table("products").upsert(
        payload,
        on_conflict="source,product_url",
    ).execute()
