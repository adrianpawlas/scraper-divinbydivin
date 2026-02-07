import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://divinbydivin.com"
COLLECTION_HANDLES = [
    "zips-and-hoodies",
    "denim-pants",
    "jackets",
    "t-shirts",
    "knits-and-crewnecks",
    "accessories",
    "jorts-shorts",
    "tops-shirts",
]

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

SOURCE = "scraper"
BRAND = "Divin"
GENDER = "man"

SIGLIP_MODEL_ID = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

PRODUCTS_PER_PAGE = 250
