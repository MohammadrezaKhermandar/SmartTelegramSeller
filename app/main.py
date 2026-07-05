"""Application entry point and initialization."""

from __future__ import annotations

import sys

from app.config import CSV_PATH
from app.data.product_loader import load_products
from app.data.product_repository import ProductRepository
from app.data.vector_store import get_vector_store, reset_vector_store
from app.tools.compare_tools import init_compare_tools
from app.tools.pandas_tools import init_pandas_tools
from app.tools.rag_tools import init_rag_tools
from app.utils.logging import logger, setup_logging


def initialize_app(csv_path=None) -> ProductRepository:
    """Load products, build vector store, init tools."""
    setup_logging()
    path = csv_path or CSV_PATH
    df, col_map = load_products(path)
    repo = ProductRepository(df)

    reset_vector_store()
    init_pandas_tools(repo)
    init_rag_tools(df)
    init_compare_tools(repo)
    get_vector_store(df)

    logger.info("App initialized with %d products", len(df))
    return repo


def main() -> None:
    """Run the Telegram bot."""
    initialize_app()
    from app.telegram.bot import run_bot

    run_bot()


if __name__ == "__main__":
    main()
