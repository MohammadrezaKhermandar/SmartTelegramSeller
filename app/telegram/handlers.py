"""Telegram message handlers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import ContextTypes

from app.graph.runner import invoke_graph, reset_user_state
from app.graph.prompts import GREETING_RESPONSE, HELP_RESPONSE
from app.memory.store import session_store
from app.telegram.jobs import schedule_followups
from app.telegram.media import reply_with_recommendations
from app.utils.logging import logger

if TYPE_CHECKING:
    pass

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _should_send_product_photos(result: dict) -> bool:
    products = result.get("recommended_products") or []
    if not products:
        return False
    stage = result.get("conversation_stage") or ""
    return stage in ("recommending", "image_search", "url_search")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    reset_user_state(user_id)
    await update.message.reply_text(GREETING_RESPONSE)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_RESPONSE)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    reset_user_state(user_id)
    await update.message.reply_text("مکالمه ریست شد. از نو شروع کنیم! بگو دنبال چه محصولی هستی.")


async def mark_purchased_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    session_store.mark_purchased(user_id)
    await update.message.reply_text("خریدت ثبت شد. ممنون از اعتمادت! 🎉")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text or ""

    url_match = URL_RE.search(text)
    if url_match:
        result = invoke_graph(user_id, text=text, url_input=url_match.group(0))
    else:
        result = invoke_graph(user_id, text=text)

    response = result.get("response_text", "متوجه نشدم. لطفاً دوباره بگو.")
    products = result.get("recommended_products") or []

    if _should_send_product_photos(result):
        await reply_with_recommendations(update, response, products)
    else:
        await update.message.reply_text(response)

    if products and result.get("conversation_stage") == "recommending":
        schedule_followups(context.application, user_id)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    photo = update.message.photo[-1] if update.message.photo else None
    caption = update.message.caption or ""

    image_input = {
        "file_id": photo.file_id if photo else None,
        "caption": caption,
        "file_name": None,
    }

    result = invoke_graph(user_id, text=caption or "عکس محصول", image_input=image_input)
    response = result.get("response_text", "عکس دریافت شد.")
    products = result.get("recommended_products") or []

    if _should_send_product_photos(result):
        await reply_with_recommendations(update, response, products)
    else:
        await update.message.reply_text(response)

    if products:
        schedule_followups(context.application, user_id)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle messages that are primarily URLs."""
    await handle_text(update, context)
