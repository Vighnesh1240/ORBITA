# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── News Fetching ─────────────────────────────────────────────────────────────
MAX_ARTICLES        = 15
MIN_ARTICLES        = 6
ARTICLES_PER_STANCE = 3

# ── Scraping ──────────────────────────────────────────────────────────────────
SCRAPE_TIMEOUT    = 12
MIN_ARTICLE_CHARS = 200   # lowered — some valid articles are short

# ── Deduplication ─────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.88

# ── Storage Paths ─────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
REPORTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "reports")

# ── Stance Labels ─────────────────────────────────────────────────────────────
STANCE_LABELS     = ["Supportive", "Critical", "Neutral"]
STANCE_HYPOTHESIS = "This article is {} toward the topic."

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400    # slightly smaller = better retrieval precision
CHUNK_OVERLAP = 60     # slightly more overlap = fewer boundary losses

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM   = 768  # dimension for gemini-embedding-001
EMBEDDING_BATCH = 5
EMBEDDING_DELAY = 0.8   # increased — more breathing room on free tier

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "orbita_articles"
TOP_K_RETRIEVAL   = 15   # reduced — better precision than 20

# ── Agent Settings ────────────────────────────────────────────────────────────
# Use a validated model from the provided list; avoid unsupported / unavailable names.
# Switched to flash model for better free tier quota availability
GEMINI_MODEL         = "models/gemini-2.5-flash-lite"
AGENT_TEMPERATURE    = 0.1   # lower = more deterministic outputs
AGENT_MAX_TOKENS     = 2048
AGENT_A_TOP_K        = 12
AGENT_B_TOP_K        = 12
AGENT_C_TOP_K        = 15
HALLUCINATION_THRESH = 0.35  # slightly lower = catches more flags

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_DIR            = os.path.join(os.path.dirname(__file__), "..", "cache")
ENABLE_TOPIC_CACHE   = True   # skip re-embedding if topic was run before