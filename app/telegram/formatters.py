"""Telegram message formatting helpers (Persian, HTML parse mode)."""

from __future__ import annotations

import html
from typing import Any, Optional

from app.utils.text_normalizer import format_price

PURCHASE_COMING_SOON_TEXT = (
    "قابلیت خرید به‌زودی فعال می‌شه. فعلاً می‌تونم مشخصات محصول رو بیشتر توضیح بدم "
    "یا با گزینه‌های دیگه مقایسه کنم."
)

PRODUCT_IMAGE_OFFER_TEXT = "اگه بخوای می‌تونم عکس این گزینه‌ها رو هم برات بفرستم."

START_TEXT = (
    "سلام! 👋 من <b>سینا</b>م، فروشنده هوشمند فروشگاه SINWAY.\n\n"
    "می‌تونم:\n"
    "• بر اساس نیاز و بودجه‌ات محصول پیشنهاد بدم\n"
    "• محصولات رو باهم مقایسه کنم\n"
    "• از روی <b>عکس</b> یا <b>لینک</b> محصول، مشابهش رو توی فروشگاه پیدا کنم\n\n"
    "فقط کافیه بگی دنبال چی هستی. مثلاً:\n"
    "<i>«یه لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام»</i>"
)

HELP_TEXT = (
    "<b>راهنمای استفاده</b>\n\n"
    "🛍 <b>جستجوی محصول:</b> بنویس چی می‌خوای؛ اگه بودجه و کاربرد رو بگی سریع‌تر به نتیجه می‌رسیم.\n"
    "🔁 <b>تغییر نظر:</b> هر وقت بودجه یا برند عوض شد فقط همون رو بگو، از اول شروع نمی‌کنیم.\n"
    "❓ <b>سؤال درباره پیشنهادها:</b> مثلاً «گزینه دوم گارانتیش چقدره؟»\n"
    "⚖️ <b>مقایسه:</b> «گزینه اول و سوم رو مقایسه کن»\n"
    "🖼 <b>عکس:</b> عکس محصول رو بفرست (ترجیحاً با توضیح کوتاه).\n"
    "🔗 <b>لینک:</b> لینک محصول از هر سایتی رو بفرست تا مشابهش رو پیدا کنم.\n"
    "✅ <b>خرید:</b> اگه خرید کردی بگو «خریدم» تا پیگیری‌ها متوقف بشه.\n"
    "🔄 <b>شروع از صفر:</b> /reset — پاک کردن کامل حافظه و استیت این چت\n"
    "🛒 <b>خرید:</b> /mark_purchased — فعلاً فقط اطلاع‌رسانی (خرید آنلاین به‌زودی)\n"
    "📂 <b>دسته‌ها:</b> /categories — لیست دسته‌بندی‌های فروشگاه\n\n"
    "دستورات: /start /help /reset /categories /mark_purchased"
)


def format_response(text: str) -> str:
    """Escape a graph response for HTML parse mode (keeps it plain-safe)."""
    return html.escape(text)


def format_product_caption(product: dict[str, Any]) -> str:
    """Caption for a product photo message."""
    name = html.escape(str(product.get("name", "")))
    price = format_price(product.get("effective_price") or product.get("price", 0))
    lines = [f"<b>{name}</b>", f"💰 {price}"]
    if product.get("discount"):
        lines.append(f"🔖 {int(product['discount'])}٪ تخفیف")
    if product.get("rating"):
        lines.append(f"⭐ {product['rating']} از ۵")
    reason = product.get("_reason")
    if reason:
        lines.append(f"✅ {html.escape(str(reason))}")
    return "\n".join(lines)


def get_product_image_url(product: dict[str, Any]) -> Optional[str]:
    url = str(product.get("image_url") or "").strip()
    return url if url.startswith("http") else None


def build_categories_message() -> str:
    """Sorted Persian list of catalog categories for /categories."""
    import html

    from app.services.pandas_query_service import get_pandas_service
    from app.utils.text_normalizer import to_persian_digits

    categories = sorted(get_pandas_service().list_categories())
    if not categories:
        return "فعلاً دسته‌بندی‌ای در کاتالوگ پیدا نشد."

    lines = ["<b>دسته‌بندی‌های موجود در فروشگاه SINWAY</b>", ""]
    for i, category in enumerate(categories, 1):
        lines.append(f"{to_persian_digits(i)}. {html.escape(category)}")
    lines.append("")
    lines.append(
        f"جمعاً {to_persian_digits(len(categories))} دسته — می‌تونی نام دسته رو مستقیم بنویسی."
    )
    return "\n".join(lines)
