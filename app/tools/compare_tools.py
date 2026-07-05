"""Product comparison tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.data.product_repository import ProductRepository

_repo: ProductRepository | None = None


def init_compare_tools(repo: ProductRepository) -> None:
    global _repo
    _repo = repo


def _get_repo() -> ProductRepository:
    if _repo is None:
        raise RuntimeError("ProductRepository not initialized")
    return _repo


COMPARE_FIELDS = ["title", "brand", "category", "price", "rating", "discount", "features", "stock", "availability"]


def compare_products(product_ids: list[str]) -> dict[str, Any]:
    """Compare multiple products side by side."""
    repo = _get_repo()
    products = repo.get_multiple_by_ids(product_ids)
    if len(products) < 2:
        return {"error": "حداقل دو محصول برای مقایسه لازم است.", "products": products}

    comparison: dict[str, dict[str, Any]] = {}
    for p in products:
        pid = str(p.get("product_id", ""))
        comparison[pid] = {f: p.get(f, "—") for f in COMPARE_FIELDS if f in p or f == "stock"}

    # Highlight differences
    highlights: list[str] = []
    prices = [p.get("price") for p in products if p.get("price")]
    if prices:
        cheapest = min(products, key=lambda x: x.get("price", float("inf")))
        highlights.append(f"ارزان‌ترین: {cheapest.get('title', '')}")

    ratings = [p.get("rating") for p in products if p.get("rating")]
    if ratings:
        best = max(products, key=lambda x: x.get("rating", 0) or 0)
        highlights.append(f"بالاترین امتیاز: {best.get('title', '')}")

    return {
        "products": products,
        "comparison": comparison,
        "highlights": highlights,
    }


@tool
def compare_products_tool(product_ids: list[str]) -> dict[str, Any]:
    """Compare two or more products by ID."""
    return compare_products(product_ids)
