"""
Fetch all products from Divin by Divin via Shopify JSON API.
Collections are paginated with ?page=1&limit=250 until empty.
"""
import requests
from typing import Iterator
from config import BASE_URL, COLLECTION_HANDLES, PRODUCTS_PER_PAGE


def fetch_collection_page(handle: str, page: int) -> list[dict]:
    url = f"{BASE_URL}/collections/{handle}/products.json"
    params = {"limit": PRODUCTS_PER_PAGE, "page": page}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("products") or []


def fetch_all_products_from_collection(handle: str) -> Iterator[dict]:
    page = 1
    while True:
        products = fetch_collection_page(handle, page)
        if not products:
            break
        for p in products:
            yield p
        if len(products) < PRODUCTS_PER_PAGE:
            break
        page += 1


def fetch_all_products() -> Iterator[tuple[str, dict]]:
    """Yield (collection_handle, product) for every product. Same product may appear in multiple collections."""
    seen_ids: set[int] = set()
    for handle in COLLECTION_HANDLES:
        for product in fetch_all_products_from_collection(handle):
            pid = product.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                yield handle, product
