"""
Divin by Divin full scraper: fetch products -> image + text embeddings (SigLIP 768) -> Supabase.
"""
import argparse
import logging
import sys
from scraper import fetch_all_products
from product_mapper import map_product_to_row, build_info_text, COLLECTION_TO_CATEGORY
from embeddings import get_image_embedding, get_text_embedding
from db import upsert_product
from config import BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run(limit: int | None = None, dry_run: bool = False):
    total = 0
    skipped = 0
    errors = 0
    for collection_handle, product in fetch_all_products():
        if limit is not None and total >= limit:
            break
        total += 1
        handle = product.get("handle") or ""
        title = (product.get("title") or "").strip()
        product_url = f"{BASE_URL}/products/{handle}" if handle else ""
        images = product.get("images") or []
        if not images:
            log.warning("Skip %s: no images", title or handle)
            skipped += 1
            continue
        image_url = (images[0].get("src") or "").strip()
        if not image_url:
            log.warning("Skip %s: empty image src", title or handle)
            skipped += 1
            continue
        category = COLLECTION_TO_CATEGORY.get(collection_handle, collection_handle)
        try:
            if dry_run:
                log.info("[DRY-RUN] Would process: %s", title or handle)
                continue
            log.info("Embedding image + text: %s", title or handle)
            image_emb = get_image_embedding(image_url)
            info_text = build_info_text(product, category, product_url)
            info_emb = get_text_embedding(info_text)
            row = map_product_to_row(product, collection_handle, image_emb, info_emb)
            upsert_product(row)
            log.info("Upserted: %s", title or handle)
        except Exception as e:
            log.exception("Error processing %s: %s", title or handle, e)
            errors += 1
            continue
    log.info("Done. Total seen=%s, skipped=%s, errors=%s", total, skipped, errors)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape Divin by Divin and upsert to Supabase")
    p.add_argument("--limit", type=int, default=None, help="Max number of products to process (for testing)")
    p.add_argument("--dry-run", action="store_true", help="Fetch and log only; no embeddings or DB")
    args = p.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)
