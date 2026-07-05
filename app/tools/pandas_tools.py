"""Pandas-based LangChain tools for structured product queries."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.config import MIN_RECOMMENDATIONS
from app.data.product_repository import ProductRepository

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
    Merge Pandas hard filters with RAG semantic results.
    Returns (products, note) where note explains relaxations.
    """
    from app.tools.rag_tools import semantic_search

    repo = _get_repo()
    note = ""

    # Hard filters via Pandas
    filtered = repo.apply_filters(
        category=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=True,
    )

    # RAG semantic matching
    if rag_results is None:
        rag_results = semantic_search(query)

    # Score and merge
    scored: dict[str, dict[str, Any]] = {}
    rag_ids = {r["product_id"] for r in rag_results}

    for _, row in filtered.iterrows():
        pid = str(row["product_id"])
        base = {k: v for k, v in row.items() if not hasattr(v, "item")}
        base["product_id"] = pid
        base["pandas_match"] = True
        base["score"] = 0.5
        scored[pid] = base

    for r in rag_results:
        pid = r["product_id"]
        if pid in scored:
            scored[pid]["score"] = scored[pid].get("score", 0) + r.get("score", 0.5)
            scored[pid]["rag_match"] = True
        else:
            product = repo.get_by_id(pid)
            if product:
                product["score"] = r.get("score", 0.3)
                product["rag_match"] = True
                scored[pid] = product

    ranked = sorted(scored.values(), key=lambda x: x.get("score", 0), reverse=True)

    # Relax constraints if fewer than min_count
    if len(ranked) < min_count:
        note = "تعداد محصولات دقیق کم بود؛ محدودیت‌ها را کمی شل کردم و گزینه‌های جایگزین پیشنهاد می‌دهم."
        relaxed = repo.apply_filters(
            category=category,
            brand=None,
            min_price=None if max_price else min_price,
            max_price=max_price,
            in_stock_only=True,
        )
        for _, row in relaxed.iterrows():
            pid = str(row["product_id"])
            if pid not in scored:
                base = repo.get_by_id(pid) or {}
                base["score"] = 0.2
                base["relaxed"] = True
                ranked.append(base)
        ranked = sorted(ranked, key=lambda x: x.get("score", 0), reverse=True)[: max(min_count, 10)]

    else:
        ranked = ranked[: max(min_count, 10)]

    if len(ranked) < min_count and rag_results:
        for r in rag_results:
            if len(ranked) >= min_count:
                break
            pid = r["product_id"]
            if not any(p.get("product_id") == pid for p in ranked):
                product = repo.get_by_id(pid)
                if product:
                    product["score"] = r.get("score", 0.1)
                    product["alternative"] = True
                    ranked.append(product)
        note = note or "محصول دقیق پیدا نشد؛ محصولات مشابه پیشنهاد می‌شوند."

    return ranked[:max(min_count, len(ranked))], note
