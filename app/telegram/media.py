"""Telegram reply helpers – text + product photos."""

from __future__ import annotations

from typing import Any

from telegram import Update

from app.utils.logging import logger
from app.utils.product_media import product_image_url

MAX_PRODUCT_PHOTOS = 3


async def reply_with_recommendations(
    update: Update,
    response_text: str,
    products: list[dict[str, Any]] | None,
    *,
    max_photos: int = MAX_PRODUCT_PHOTOS,
) -> None:
    """Send recommendation text and optional product images."""
    message = update.message
    if message is None:
        return

    await message.reply_text(response_text)

    if not products:
        return

    sent = 0
    for product in products:
        if sent >= max_photos:
            break

        url = product_image_url(product)
        if not url:
            continue

        title = product.get("title", "محصول")
        price = int(float(product.get("price", 0) or 0))
        caption = f"{title}\nقیمت: {price:,} تومان"

        try:
            await message.reply_photo(photo=url, caption=caption[:1024])
            sent += 1
        except Exception as exc:
            logger.warning("send_photo failed for product %s: %s", product.get("product_id"), exc)
