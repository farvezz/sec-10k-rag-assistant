"""Central configuration: companies, models, paths, credentials.

Everything downstream imports from here so the 8-company / 5-year scope is
defined in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "all_chunks.jsonl"
FACT_TABLE_FILE = DATA_DIR / "fact_table.json"
MANIFEST_FILE = DATA_DIR / "filings_manifest.json"
EVAL_DIR = ROOT / "eval"

for _d in (RAW_DIR, CLEAN_DIR, CHUNKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Scope (Phase 0) -----------------------------------------------------
# CIKs confirmed against SEC EDGAR.
COMPANIES = {
    "GOOGL": {
        "company": "Alphabet Inc.",
        "cik": "0001652044",
        "sector": "Communication Services",
        "fiscal_year_end": "December 31",
    },
    "MCD": {
        "company": "McDonald's Corporation",
        "cik": "0000063908",
        "sector": "Consumer Discretionary",
        "fiscal_year_end": "December 31",
    },
    "AAPL": {
        "company": "Apple Inc.",
        "cik": "0000320193",
        "sector": "Technology",
        "fiscal_year_end": "Last Saturday of September",
    },
    "KO": {
        "company": "The Coca-Cola Company",
        "cik": "0000021344",
        "sector": "Consumer Staples",
        "fiscal_year_end": "December 31",
    },
    "MSFT": {
        "company": "Microsoft Corporation",
        "cik": "0000789019",
        "sector": "Technology",
        "fiscal_year_end": "June 30",
    },
    "NKE": {
        "company": "NIKE, Inc.",
        "cik": "0000320187",
        "sector": "Consumer Discretionary",
        "fiscal_year_end": "May 31",
    },
    "NVDA": {
        "company": "NVIDIA Corporation",
        "cik": "0001045810",
        "sector": "Technology",
        "fiscal_year_end": "Last Sunday of January",
    },
    "DIS": {
        "company": "The Walt Disney Company",
        "cik": "0001744489",
        "sector": "Communication Services",
        "fiscal_year_end": "Sunday closest to September 30",
    },
}

TICKERS = list(COMPANIES)
FILINGS_PER_COMPANY = 5

# Aliases used by the entity detector (Phase 10.2). Lower-cased, matched as
# whole words / phrases against the condensed question.
COMPANY_ALIASES = {
    "GOOGL": ["googl", "goog", "google", "alphabet"],
    "MCD": ["mcd", "mcdonald", "mcdonalds", "mcdonald's", "macdonalds"],
    "AAPL": ["aapl", "apple"],
    "KO": ["ko", "coca-cola", "coca cola", "cocacola", "coke"],
    "MSFT": ["msft", "microsoft"],
    "NKE": ["nke", "nike"],
    "NVDA": ["nvda", "nvidia"],
    "DIS": ["dis", "disney", "walt disney"],
}

# --- SEC -----------------------------------------------------------------
# SEC requires a descriptive User-Agent ("Name email") on every request.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{accn}-index.htm"
)

# --- OpenAI (Phase 2) ----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o"
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS = 800
CONDENSE_MAX_TOKENS = 200

# --- Chunking (Phase 7) --------------------------------------------------
CHUNK_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 90  # ~13%
MIN_CHUNK_TOKENS = 50

# --- Qdrant (Phase 8) ----------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sec_10k_chunks")

# --- Retrieval / rerank (Phase 10-11) ------------------------------------
VECTOR_TOP_K = 20
RERANK_TOP_K = 5
RERANK_TOP_K_MULTI = 10  # multi-company / multi-year comparisons
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# --- Deployment guardrail (Phase 16) -------------------------------------
MAX_QUERIES_PER_SESSION = int(os.getenv("MAX_QUERIES_PER_SESSION", "20"))


def require(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example) or to "
            f"Streamlit secrets before running this step."
        )
    return value
