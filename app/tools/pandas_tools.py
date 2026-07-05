"""Pandas-based LangChain tools for structured product queries."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.config import MIN_RECOMMENDATIONS
from app.data.product_loader import product_to_dict
from app.data.product_repository import ProductRepository
from app.graph.nlp import normalize_category, product_matches_category
from app.graph.prompts import SEARCH_LIMITED_MATCH_MESSAGES, SEARCH_NO_MATCH_MESSAGE
from app.utils.json_safe import products_to_json_safe

_repo: ProductRepository | None = None


def init_pandas_tools(repo: ProductRepository) -> None:
    global _repo
    _repo = repo


def _get_repo() -> ProductRepository:
    if _repo is None:
        raise RuntimeError("ProductRepository not initialized")
    return _repo


@tool
def filter_by_category_tool(category: str) -> list[dict[str, Any]]:
    """Filter products by category name."""
    category = normalize_category(category) or category
    df = _get_repo().filter_by_category(category)
    return _get_repo().to_product_list(df, limit=20)


@tool
def filter_by_price_range_tool(
    min_price: float | None = None, max_price: float | None = None
) -> list[dict[str, Any]]:
    """Filter products by price range in Tomans."""
    df = _get_repo().filter_by_price_range(min_price, max_price)
    return _get_repo().to_product_list(df, limit=20)


@tool
def filter_by_brand_tool(brand: str) -> list[dict[str, Any]]:
    """Filter products by brand name."""
    df = _get_repo().filter_by_brand(brand)
    return _get_repo().to_product_list(df, limit=20)


@tool
def filter_by_availability_tool(in_stock_only: bool = True) -> list[dict[str, Any]]:
    """Filter products by stock availability."""
    df = _get_repo().filter_by_availability(in_stock_only)
    return _get_repo().to_product_list(df, limit=20)


@tool
def sort_products_tool(
    field: str = "price", ascending: bool = True, limit: int = 10
) -> list[dict[str, Any]]:
    """Sort products by price, rating, or discount."""
    repo = _get_repo()
    df = repo.sort_by(repo.df, field=field, ascending=ascending)
    return repo.to_product_list(df, limit=limit)


@tool
def get_product_by_id_tool(product_id: str) -> dict[str, Any] | None:
    """Retrieve a single product by its ID."""
    return _get_repo().get_by_id(product_id)


def hybrid_recommend(
    query: str,
    category: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    rag_results: list[dict[str, Any]] | None = None,
    min_count: int = MIN_RECOMMENDATIONS,
) -> tuple[list[dict[str, Any]], str]:
    """
    Hard-filter via Pandas, then rank ONLY within that set using RAG scores.
    Category is a mandatory constraint when provided — never widens to other categories.
    """
    from app.tools.rag_tools import semantic_search

    repo = _get_repo()
    category = normalize_category(category)

    if not category:
        return [], "لطفاً دسته محصول را مشخص کنید (مثلاً لپ‌تاپ، موبایل، هدفون)."

    filtered = repo.apply_filters(
        category=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=True,
    )

    if filtered.empty:
        return [], SEARCH_NO_MATCH_MESSAGE

    candidates: dict[str, dict[str, Any]] = {}
    for _, row in filtered.iterrows():
        pid = str(row["product_id"])
        base = product_to_dict(row)
        base["product_id"] = pid
        base["score"] = 0.1
        base["pandas_match"] = True
        candidates[pid] = base

    if rag_results is None:
        rag_results = semantic_search(query)

    for hit in rag_results:
        pid = str(hit.get("product_id", ""))
        if pid not in candidates:
            continue
        candidates[pid]["score"] = candidates[pid].get("score", 0.1) + float(
            hit.get("score", 0.5) or 0.5
        )
        candidates[pid]["rag_match"] = True

    ranked = sorted(candidates.values(), key=lambda x: x.get("score", 0), reverse=True)
    ranked = [p for p in ranked if product_matches_category(p, category)]

    if not ranked:
        return [], SEARCH_NO_MATCH_MESSAGE

    # 1–2 results: honest "only N options" note; 3+: normal intro (empty note).
    # The no-match message is reserved for the zero-result case above.
    note = SEARCH_LIMITED_MATCH_MESSAGES.get(len(ranked), "")

    limit = max(min_count, min(len(ranked), 10))
    return products_to_json_safe(ranked[:limit]), note
