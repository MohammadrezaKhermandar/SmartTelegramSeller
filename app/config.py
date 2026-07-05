"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CSV_FILENAME = "products_500.csv"
CSV_PATH = PROJECT_ROOT / CSV_FILENAME
CSV_DOWNLOAD_URL = os.getenv(
    "CSV_DOWNLOAD_URL",
    "https://raw.githubusercontent.com/example/products_500.csv",
)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()

# Vector store: keyword (offline) | tfidf | chroma | auto
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "keyword").lower()

# LLM settings (Groq – فاز ۱)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
USE_LLM = os.getenv("USE_LLM", "true").lower() in ("1", "true", "yes")

# xAI (Grok) – OpenAI-compatible API
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.3")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
# Optional HTTP proxy for LLM calls (defaults to the Telegram proxy)
LLM_PROXY = os.getenv("LLM_PROXY", os.getenv("TELEGRAM_PROXY", "")).strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Follow-up job intervals (seconds) – override for demo/testing
FOLLOWUP_1H_SECONDS = int(os.getenv("FOLLOWUP_1H_SECONDS", str(3600)))
DISCOUNT_2D_SECONDS = int(os.getenv("DISCOUNT_2D_SECONDS", str(2 * 24 * 3600)))

# RAG settings
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MIN_RECOMMENDATIONS = 3

# Retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
