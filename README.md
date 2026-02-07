# Divin by Divin Scraper

Scrapes all products from [divinbydivin.com](https://divinbydivin.com), generates 768-dim image and text embeddings with **google/siglip-base-patch16-384**, and upserts into your Supabase `products` table.

## Setup

1. **Python 3.10+** and a virtualenv:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or: source .venv/bin/activate  # Linux/macOS
   ```

2. **Install dependencies** (includes PyTorch and Hugging Face Transformers):

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment**: Copy `.env.example` to `.env` and set your Supabase credentials:

   ```env
   SUPABASE_URL=https://yqawmzggcgpeyaaynrjk.supabase.co
   SUPABASE_KEY=your-service-role-or-anon-key
   ```

## Run

```bash
python main.py
```

- Fetches all products from the 8 category collections via Shopify’s JSON API (no browser/infinite scroll).
- For each product: downloads the first image, computes **image_embedding** (SigLIP vision), builds a text blob (title, price, description, category, gender, metadata) and computes **info_embedding** (SigLIP text), then upserts into `products` with `source=scraper`, `brand=Divin`, `gender=man`, `second_hand=false`.
- Products without images are skipped. Failures on single products are logged and the run continues.

## Fields written

| Field | Source |
|-------|--------|
| `id` | SHA256(source + product_url) first 32 chars |
| `source` | `"scraper"` |
| `product_url` | `https://divinbydivin.com/products/{handle}` |
| `image_url` | First product image URL (used for embedding) |
| `additional_images` | Other image URLs, comma-space separated |
| `brand` | `"Divin"` |
| `title` | Product title |
| `description` | Stripped HTML from body |
| `category` | From collection (e.g. `"Zips, Hoodies"`) |
| `gender` | `"man"` |
| `price` | e.g. `49.95EUR` (store uses EUR) |
| `sale` | Set if any variant has compare_at_price |
| `size` | Size option values, comma-separated |
| `second_hand` | `false` |
| `metadata` | JSON: shopify_id, handle, vendor, tags, etc. |
| `image_embedding` | 768-dim SigLIP image embedding |
| `info_embedding` | 768-dim SigLIP text embedding of title + price + description + category + gender + metadata |
| `created_at` | Set to current time on each upsert |

Conflict key for upsert: `(source, product_url)`.

## GitHub Actions (daily + manual)

A workflow runs the scraper **every day at midnight UTC** and can be **run manually** anytime:

1. In your repo go to **Settings → Secrets and variables → Actions**.
2. Add repository secrets:
   - `SUPABASE_URL`: your Supabase project URL
   - `SUPABASE_KEY`: your Supabase anon or service role key
3. **Manual run**: go to **Actions → Run Divin Scraper → Run workflow**.

To change the schedule, edit `.github/workflows/run-scraper.yml` (e.g. `cron: '0 2 * * *'` for 02:00 UTC).

## Collections scraped

- Zips & Hoodies, Denim & Pants, Jackets, T-Shirts, Knits & Crewnecks, Accessories, Jorts & Shorts, Tops & Shirts
