"""Similar-product lookup from a user-sent image.

Implemented as the documented "minimal acceptable" version with a clean
upgrade path:

1. If the photo has a caption, use it as the search query.
2. Otherwise, if the LLM provider supports vision, we could describe the
   image (hook provided, disabled by default to stay provider-agnostic).
3. Otherwise, ask the user one short question about the product type and
   remember (in state) that we are waiting for that answer.

A CLIP-based image embedding index can be plugged in later by implementing
`embed_image` and adding an image collection to ChromaDB; the graph node
(`process_image`) would not need to change.
"""

from __future__ import annotations

from typing import Any, Optional

from app.utils.logger import get_logger
from app.utils.text_normalizer import enrich_english_keywords

logger = get_logger(__name__)

ASK_ABOUT_IMAGE_TEXT = (
    "عکس رو گرفتم 👍 برای اینکه دقیق‌ترین مشابهش رو از فروشگاه پیدا کنم، "
    "بگو این چه محصولیه؟ (مثلاً: هدفون بی‌سیم سونی، یا کتری برقی)"
)


def build_image_query(
    caption: Optional[str],
    file_name: Optional[str] = None,
) -> dict[str, Any]:
    """Derive a text query from image metadata.

    Returns {'query': str | None, 'needs_user_help': bool}.
    """
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if file_name:
        cleaned = (
            file_name.rsplit(".", 1)[0]
            .replace("_", " ")
            .replace("-", " ")
            .strip()
        )
        # Telegram photo file names are random ids — only use meaningful names
        if cleaned and not cleaned.lower().startswith(("img", "photo", "image", "file")):
            parts.append(cleaned)

    if parts:
        query = enrich_english_keywords(" ".join(parts))
        logger.info("Image query derived from metadata: %.80s", query)
        return {"query": query, "needs_user_help": False}

    logger.info("Image has no usable metadata — asking user for product type")
    return {"query": None, "needs_user_help": True}
