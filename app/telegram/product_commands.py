"""Telegram slash commands for product search and browsing."""

from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from app.graph.prompts import format_product_recommendation
from app.memory.store import session_store
from app.telegram.jobs import schedule_followups
from app.telegram.media import reply_with_recommendations
from app.tools.pandas_tools import _get_repo, filter_by_category_tool, get_product_by_id_tool
from app.tools.rag_tools import semantic_search
from app.utils.json_safe import products_to_json_safe
from app.utils.logging import logger

MAX_LIST = 5


def _save_recommendations(user_id: str, products: list[dict]) -> None:
    safe_products = products_to_json_safe(products)
    session_store.update_from_state(
        user_id,
        {
            "recommended_products": safe_products,
            "conversation_stage": "recommending",
            "purchase_status": "pending",
        },
    )


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/search <query> — semantic product search."""
    user_id = str(update.effective_user.id)
    query = " ".join(context.args).strip()
    logger.info("Command /search from user %s: %s", user_id, query or "(empty)")

    if not query:
        await update.message.reply_text(
            "🔍 جستجوی محصول\n\n"
            "نحوه استفاده:\n"
            "/search لپ‌تاپ گیمینگ\n"
            "/search هدفون بی‌سیم سونی"
        )
        return

    products = semantic_search(query)[:MAX_LIST]
    if not products:
        await update.message.reply_text(f"محصولی برای «{query}» پیدا نشد.")
        return

    note = f"نتایج جستجو برای: {query}"
    response = format_product_recommendation(products, note)
    _save_recommendations(user_id, products)
    await reply_with_recommendations(update, response, products)
    schedule_followups(context.application, user_id)


async def browse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/browse <category> — list products in a category."""
    user_id = str(update.effective_user.id)
    category = " ".join(context.args).strip()

    if not category:
        await update.message.reply_text(
            "📂 مرور دسته‌بندی\n\n"
            "نحوه استفاده:\n"
            "/browse لپ‌تاپ\n"
            "/browse هدفون\n\n"
            "لیست دسته‌ها: /categories"
        )
        return

    products = filter_by_category_tool.invoke({"category": category})[:MAX_LIST]
    if not products:
        await update.message.reply_text(
            f"محصولی در دسته «{category}» پیدا نشد.\n"
            "دسته‌های موجود: /categories"
        )
        return

    note = f"محصولات دسته {category}"
    response = format_product_recommendation(products, note)
    _save_recommendations(user_id, products)
    await reply_with_recommendations(update, response, products)
    schedule_followups(context.application, user_id)


async def product_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/product <id> — show one product by ID."""
    if not context.args:
        await update.message.reply_text(
            "📦 جزئیات محصول\n\n"
            "نحوه استفاده:\n"
            "/product 124"
        )
        return

    product_id = context.args[0].strip()
    product = get_product_by_id_tool.invoke({"product_id": product_id})
    if not product:
        await update.message.reply_text(f"محصول با شناسه {product_id} پیدا نشد.")
        return

    title = product.get("title", "—")
    price = int(float(product.get("price", 0) or 0))
    brand = product.get("brand", "—")
    category = product.get("category", "—")
    rating = product.get("rating", "—")
    stock = product.get("availability", product.get("stock", "—"))
    features = product.get("features", product.get("description", "—"))

    text = (
        f"📦 {title}\n"
        f"🆔 شناسه: {product_id}\n"
        f"🏷️ برند: {brand} | دسته: {category}\n"
        f"💰 قیمت: {price:,} تومان\n"
        f"⭐ امتیاز: {rating}\n"
        f"📦 موجودی: {stock}\n"
        f"📝 ویژگی‌ها: {str(features)[:200]}"
    )
    await reply_with_recommendations(update, text, [product], max_photos=1)


async def categories_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/categories — list available product categories."""
    logger.info("Command /categories from user %s", update.effective_user.id)
    repo = _get_repo()
    categories = repo.list_categories()
    if not categories:
        await update.message.reply_text("دسته‌بندی‌ای پیدا نشد.")
        return

    lines = ["📂 دسته‌بندی‌های فروشگاه:\n"]
    for name, count in categories:
        lines.append(f"• {name} ({count} محصول)")
    lines.append("\nبرای دیدن محصولات یک دسته:\n/browse نام_دسته")
    await update.message.reply_text("\n".join(lines))


async def my_products_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/my_products — show last recommended products from memory."""
    user_id = str(update.effective_user.id)
    session = session_store.get_or_create(user_id)
    products = session.recommended_products

    if not products:
        await update.message.reply_text(
            "هنوز محصولی پیشنهاد نشده.\n"
            "جستجو: /search لپ‌تاپ\n"
            "یا نیازت رو به فارسی بنویس."
        )
        return

    response = format_product_recommendation(
        products[:MAX_LIST],
        "آخرین پیشنهادهای شما:",
    )
    await reply_with_recommendations(update, response, products[:MAX_LIST])
