"""LangChain tools: hybrid recommendation, memory lookup, image/link similarity."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from app.services.image_similarity_service import build_image_query
from app.services.link_product_service import extract_product_text
from app.services.memory_service import get_memory_service
from app.services.recommendation_service import get_recommendation_service


@tool
def recommend_products_tool(
    query_text: str,
    budget: Optional[float] = None,
    category: Optional[str] = None,
    brands: Optional[list[str]] = None,
    use_case: Optional[str] = None,
    top_k: int = 3,
    strict_budget: bool = False,
    hard_max_price: bool = False,
    allow_budget_overflow: Optional[bool] = None,
) -> dict[str, Any]:
    """پیشنهاد ترکیبی محصولات (Pandas + RAG) بر اساس نیاز کاربر.

    خروجی شامل محصولات رتبه‌بندی‌شده با score و اجزای امتیاز است.
    با strict_budget یا hard_max_price هیچ محصول بالاتر از بودجه برنمی‌گردد.
    """
    requirements: dict[str, Any] = {
        "budget": budget,
        "category": category,
        "brands": brands or [],
        "use_case": use_case,
        "hard_max_price": hard_max_price,
    }
    if allow_budget_overflow is not None:
        requirements["allow_budget_overflow"] = allow_budget_overflow
    return get_recommendation_service().recommend(
        query_text, requirements, top_k=top_k, strict_budget=strict_budget
    )


@tool
def get_product_from_memory_tool(
    chat_id: str,
    position: Optional[int] = None,
    product_name_query: Optional[str] = None,
) -> dict[str, Any]:
    """بازیابی محصولات معرفی‌شدهٔ قبلی از حافظه (بدون جستجوی مجدد CSV/RAG).

    Args:
        chat_id: شناسه چت.
        position: شماره گزینه (۱ = گزینه اول).
        product_name_query: بخشی از نام محصول معرفی‌شده برای resolve کردن.
    """
    memory = get_memory_service()
    if position is not None:
        product = memory.get_recommendation_by_position(chat_id, position)
        return {"found": product is not None, "product": product}
    if product_name_query:
        product = memory.find_product_by_name_query(chat_id, product_name_query)
        if product is not None:
            return {"found": True, "product": product}
        products = memory.get_active_recommendations(chat_id)
        return {"found": bool(products), "products": products}
    products = memory.get_active_recommendations(chat_id)
    return {"found": bool(products), "products": products}


@tool
def similar_products_by_image_tool(
    caption: Optional[str] = None, file_name: Optional[str] = None, top_k: int = 3
) -> dict[str, Any]:
    """پیدا کردن محصولات مشابه تصویر ارسالی کاربر (بر اساس کپشن/متادیتا).

    اگر متادیتای کافی نبود، `needs_user_help=True` برمی‌گردد تا از کاربر
    نوع محصول پرسیده شود.
    """
    info = build_image_query(caption, file_name)
    if info["needs_user_help"]:
        return {"needs_user_help": True, "products": []}
    result = get_recommendation_service().recommend(info["query"], {}, top_k=top_k)
    return {"needs_user_help": False, **result}


@tool
def similar_products_by_url_tool(url: str, top_k: int = 3) -> dict[str, Any]:
    """پیدا کردن محصولات مشابه یک لینک خارجی از داخل فروشگاه.

    صفحه fetch می‌شود و title/description استخراج می‌شود؛ اگر fetch شکست
    بخورد از کلمات داخل خود URL استفاده می‌شود.
    """
    info = extract_product_text(url)
    result = get_recommendation_service().recommend(info["query"], {}, top_k=top_k)
    return {"source": info["source"], "external_title": info["title"], **result}
