"""LangChain tool: side-by-side comparison of previously recommended products."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.services.memory_service import get_memory_service


COMPARE_FIELDS = (
    ("price", "قیمت"),
    ("effective_price", "قیمت با تخفیف"),
    ("brand", "برند"),
    ("category", "دسته"),
    ("rating", "امتیاز"),
    ("review_count", "تعداد نظرات"),
    ("stock", "موجودی"),
    ("warranty", "گارانتی"),
    ("features", "ویژگی‌ها"),
)


@tool
def compare_products_tool(chat_id: str, positions: list[int]) -> dict[str, Any]:
    """مقایسه محصولات معرفی‌شدهٔ قبلی بر اساس شماره گزینه‌ها — از حافظه.

    Args:
        chat_id: شناسه چت.
        positions: شماره گزینه‌ها، مثلا [1, 3] برای «گزینه اول و سوم».
    """
    memory = get_memory_service()
    products = [
        p for pos in positions
        if (p := memory.get_recommendation_by_position(chat_id, pos)) is not None
    ]
    if len(products) < 2:
        return {"ok": False, "reason": "not_enough_products", "products": products}

    table: list[dict[str, Any]] = []
    for key, label in COMPARE_FIELDS:
        row: dict[str, Any] = {"field": label}
        for product in products:
            row[product.get("name", "?")] = product.get(key, "-")
        table.append(row)

    return {"ok": True, "products": products, "table": table}
