"""Image-based product similarity tools."""

from __future__ import annotations

from typing import Any

from app.tools.rag_tools import find_similar_products
from app.utils.logging import logger


def process_image_input(
    caption: str | None = None,
    file_name: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Process image input pragmatically.
    Uses caption/metadata if available; otherwise asks user via returned query.
    """
    description_parts: list[str] = []

    if caption:
        description_parts.append(caption)
    if file_name:
        description_parts.append(file_name.replace("_", " ").replace("-", " "))

    if description_parts:
        query = " ".join(description_parts)
        logger.info("Image query from metadata: %s", query)
        similar = find_similar_products(query, top_k=5)
        return query, similar

    # Fallback: generic electronics query
    fallback_query = "محصول الکترونیکی مشابه"
    similar = find_similar_products(fallback_query, top_k=5)
    return fallback_query, similar


def build_image_acknowledgment(query: str, needs_clarification: bool) -> str:
    if needs_clarification:
        return (
            "عکس رو دریافت کردم. برای پیشنهاد دقیق‌تر، لطفاً بگو این محصول چه نوع کالاییه "
            "(مثلاً لپ‌تاپ، هدفون، تلویزیون) و برندش چیه؟"
        )
    return f"عکس رو دیدم و محصولات مشابه با «{query}» رو پیدا کردم."
