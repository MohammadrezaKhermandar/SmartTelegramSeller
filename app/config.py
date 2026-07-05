"""Central configuration loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = directory containing this package's parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str) -> Path:
    if value in {":memory:", "ephemeral"}:
        return Path(value)
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- LLM ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter"))
    use_llm: bool = field(default_factory=lambda: _bool(os.getenv("USE_LLM"), True))

    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_app_name: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "SINWAY Sales Assistant")
    )

    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    xai_model: str = field(default_factory=lambda: os.getenv("XAI_MODEL", "grok-2-latest"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # --- Telegram ---
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_proxy: str = field(default_factory=lambda: os.getenv("TELEGRAM_PROXY", ""))

    # --- Data paths ---
    products_csv_path: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("PRODUCTS_CSV_PATH", "products_500.csv"))
    )
    vector_db_path: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("VECTOR_DB_PATH", ".data/chroma"))
    )
    memory_db_path: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("MEMORY_DB_PATH", ".data/memory.sqlite"))
    )
    # keyword = in-memory hash (offline, stable on Windows); chroma = ChromaDB
    vector_store_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_BACKEND", "keyword").lower().strip()
    )

    # --- Runtime ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- Follow-up timing (seconds); configurable so demos can shrink them ---
    followup_idle_seconds: int = field(
        default_factory=lambda: int(os.getenv("FOLLOWUP_IDLE_SECONDS", str(60 * 60)))
    )
    followup_purchase_seconds: int = field(
        default_factory=lambda: int(os.getenv("FOLLOWUP_PURCHASE_SECONDS", str(2 * 24 * 60 * 60)))
    )
    followup_check_interval: int = field(
        default_factory=lambda: int(os.getenv("FOLLOWUP_CHECK_INTERVAL", "60"))
    )
    discount_code: str = field(default_factory=lambda: os.getenv("DISCOUNT_CODE", "SALE10"))

    # --- Retry policy ---
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "800")))

    @property
    def llm_api_key(self) -> str:
        return {
            "openrouter": self.openrouter_api_key,
            "xai": self.xai_api_key,
            "groq": self.groq_api_key,
        }.get(self.llm_provider, self.openrouter_api_key)

    @property
    def llm_model(self) -> str:
        return {
            "openrouter": self.openrouter_model,
            "xai": self.xai_model,
            "groq": self.groq_model,
        }.get(self.llm_provider, self.openrouter_model)

    @property
    def llm_base_url(self) -> str:
        return {
            "openrouter": self.openrouter_base_url,
            "xai": "https://api.x.ai/v1",
            "groq": "https://api.groq.com/openai/v1",
        }.get(self.llm_provider, self.openrouter_base_url)

    @property
    def uses_chroma(self) -> bool:
        return self.vector_store_backend == "chroma"


settings = Settings()
