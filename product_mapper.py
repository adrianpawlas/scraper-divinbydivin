"""
Map Shopify product JSON + collection to our DB row shape.
"""
import json
import hashlib
from bs4 import BeautifulSoup
from typing import Any
from config import BASE_URL, SOURCE, BRAND, GENDER


# Collection handle -> category string (comma-separated if multiple)
COLLECTION_TO_CATEGORY = {
    "zips-and-hoodies": "Zips, Hoodies",
    "denim-pants": "Denim, Pants",
    "jackets": "Jackets",
    "t-shirts": "T-Shirts",
    "knits-and-crewnecks": "Knits, Crewnecks",
    "accessories": "Accessories",
    "jorts-shorts": "Jorts, Shorts",
    "tops-shirts": "Tops, Shirts",
}


def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def product_id_from_url(product_url: str) -> str:
    return hashlib.sha256(f"{SOURCE}:{product_url}".encode()).hexdigest()[:32]


def build_price_string(product: dict) -> str:
    """Store price in EUR (site currency). Format: 49.95EUR (and optionally other currencies)."""
    variants = product.get("variants") or []
    if not variants:
        return ""
    prices = set()
    for v in variants:
        p = v.get("price")
        if p:
            prices.add(f"{p}EUR")
    return ",".join(sorted(prices))


def build_sale(product: dict) -> str | None:
    """If any variant has compare_at_price > price, consider it on sale."""
    for v in product.get("variants") or []:
        compare = v.get("compare_at_price")
        if compare and str(compare).strip():
            return "true"
    return None


def build_size(product: dict) -> str | None:
    options = product.get("options") or []
    for opt in options:
        if (opt.get("name") or "").lower() in ("size", "taille"):
            vals = opt.get("values") or []
            return ",".join(vals) if vals else None
    return None


def build_metadata(product: dict, category: str) -> str:
    d: dict[str, Any] = {
        "shopify_id": product.get("id"),
        "handle": product.get("handle"),
        "vendor": product.get("vendor"),
        "product_type": product.get("product_type"),
        "tags": product.get("tags") or [],
        "category": category,
        "options": [o.get("name") for o in (product.get("options") or [])],
        "variant_count": len(product.get("variants") or []),
    }
    return json.dumps(d, ensure_ascii=False)


def build_info_text(product: dict, category: str, product_url: str) -> str:
    """Full text used for info_embedding: title, price, description, category, gender, metadata."""
    title = (product.get("title") or "").strip()
    body = strip_html(product.get("body_html") or "")
    price_str = build_price_string(product)
    parts = [
        f"Title: {title}",
        f"Category: {category}",
        f"Gender: {GENDER}",
        f"Price: {price_str}",
        f"URL: {product_url}",
    ]
    if body:
        parts.append(f"Description: {body}")
    meta = build_metadata(product, category)
    parts.append(f"Metadata: {meta}")
    return " ".join(parts)


def map_product_to_row(
    product: dict,
    collection_handle: str,
    image_embedding: list[float],
    info_embedding: list[float],
) -> dict[str, Any]:
    """Build one products table row from Shopify product + collection + embeddings."""
    handle = product.get("handle") or ""
    product_url = f"{BASE_URL}/products/{handle}" if handle else ""
    id_ = product_id_from_url(product_url)

    images = product.get("images") or []
    image_url = ""
    additional_urls: list[str] = []
    if images:
        image_url = (images[0].get("src") or "").strip()
        for img in images[1:]:
            src = (img.get("src") or "").strip()
            if src:
                additional_urls.append(src)
    additional_images = ", ".join(additional_urls) if additional_urls else None

    category = COLLECTION_TO_CATEGORY.get(collection_handle, collection_handle)
    title = (product.get("title") or "").strip()
    description = strip_html(product.get("body_html") or "") or None
    price = build_price_string(product) or None
    sale = build_sale(product)
    size = build_size(product)
    metadata = build_metadata(product, category)

    return {
        "id": id_,
        "source": SOURCE,
        "product_url": product_url or None,
        "affiliate_url": None,
        "image_url": image_url or "",  # required
        "brand": BRAND,
        "title": title or "Untitled",
        "description": description,
        "category": category,
        "gender": GENDER,
        "search_tsv": None,
        "metadata": metadata,
        "size": size,
        "second_hand": False,
        "image_embedding": image_embedding,
        "country": None,
        "compressed_image_url": None,
        "tags": product.get("tags") or None,
        "search_vector": None,
        "title_tsv": None,
        "brand_tsv": None,
        "description_tsv": None,
        "other": None,
        "price": price,
        "sale": sale,
        "additional_images": additional_images,
        "info_embedding": info_embedding,
    }
