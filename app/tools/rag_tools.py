"""LangChain tool: semantic product search via ChromaDB."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from app.services.rag_service import get_rag_service


@tool
def rag_product_search_tool(
    query: str,
    n_results: int = 10,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
) -> list[dict[str, Any]]:
    """جستجوی معنایی محصولات فروشگاه بر اساس متن آزاد (نام، ویژگی، کاربرد).

    Args:
        query: متن جستجو، مثلا «لپ‌تاپ سبک برای برنامه‌نویسی».
        n_results: حداکثر تعداد نتیجه.
        max_price: سقف قیمت (تومان) در صورت وجود.
        in_stock_only: فقط محصولات موجود.
    """
    where: Optional[dict[str, Any]] = None
    clauses: list[dict[str, Any]] = []
    if max_price is not None:
        clauses.append({"effective_price": {"$lte": float(max_price)}})
    if in_stock_only:
        clauses.append({"availability": True})
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}
    return get_rag_service().search(query, n_results=n_results, where=where)
