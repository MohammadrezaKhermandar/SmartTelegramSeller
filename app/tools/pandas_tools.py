"""LangChain tool: exact structured filtering with Pandas."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from app.services.pandas_query_service import ProductFilter, get_pandas_service


@tool
def pandas_filter_products_tool(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    brands: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    keywords: Optional[list[str]] = None,
    in_stock_only: bool = True,
    min_rating: Optional[float] = None,
    sort_by: str = "rating",
    limit: int = 20,
) -> dict[str, Any]:
    """فیلتر دقیق محصولات با پانداز: قیمت، برند، دسته‌بندی، موجودی و امتیاز.

    اگر نتیجه دقیق نبود، فیلترها به‌تدریج شل می‌شوند و فیلد `relaxed`
    نشان می‌دهد چه چیزی تغییر کرده است.
    """
    f = ProductFilter(
        max_price=max_price,
        min_price=min_price,
        brands=brands or [],
        categories=categories or [],
        keywords=keywords or [],
        in_stock_only=in_stock_only,
        min_rating=min_rating,
        sort_by=sort_by,  # type: ignore[arg-type]
        limit=limit,
    )
    result = get_pandas_service().query(f)
    # Trim the heavy search_text column from tool output
    for product in result["products"]:
        product.pop("search_text", None)
    return result
