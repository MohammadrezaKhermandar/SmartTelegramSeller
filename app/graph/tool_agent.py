"""LLM tool-calling agent for hybrid product search (LangGraph Tool Calling)."""

from __future__ import annotations

from typing import Any

from app.config import USE_LLM
from app.graph.nlp import normalize_category
from app.tools.pandas_tools import hybrid_recommend
from app.tools.rag_tools import semantic_search
from app.utils.logging import logger


def run_search_tool_agent(
    requirements: dict[str, Any],
    query: str,
    *,
    max_iterations: int = 6,
) -> tuple[list[dict[str, Any]], str] | None:
    """
    Run product search through the strict hybrid pipeline.
    Category is mandatory; RAG only re-ranks hard-filtered candidates.
    Returns None when category is missing (caller should ask clarifying questions).
    """
    del max_iterations  # LLM tool loop reserved for future use; pipeline is filter-first.

    category = normalize_category(requirements.get("category"))
    if not category:
        logger.info("Tool agent skipped: category is required before search")
        return None

    if not USE_LLM:
        return None

    try:
        rag_results = semantic_search(query)
        products, note = hybrid_recommend(
            query=query,
            category=category,
            brand=requirements.get("brand"),
            min_price=requirements.get("min_price"),
            max_price=requirements.get("max_price"),
            rag_results=rag_results,
        )
        return products, note
    except Exception as exc:
        logger.error("Tool agent search failed: %s", exc)
        return None
