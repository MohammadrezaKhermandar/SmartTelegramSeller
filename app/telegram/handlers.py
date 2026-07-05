"""Telegram update handlers.

Graph execution is CPU/IO-bound and synchronous, so it runs in a worker
thread (asyncio.to_thread) to keep the bot responsive. All outgoing sends
are wrapped with retry.

Each user message triggers exactly one graph turn and one primary text reply.
Product photos (if any) are sent only after the main response.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Set

from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from app.graph.builder import reset_chat_state, run_turn
from app.services.followup_service import get_followup_service
from app.telegram.formatters import (
    HELP_TEXT,
    PURCHASE_COMING_SOON_TEXT,
    START_TEXT,
    build_categories_message,
    format_product_caption,
    format_response,
    get_product_image_url,
)
from app.utils.logger import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

USER_FACING_ERROR = (
    "الان برای بررسی دقیق محصول مشکلی پیش اومده، اما می‌تونم با اطلاعاتی که "
    "ازت دارم راهنمایی‌ات کنم. چند لحظه دیگه دوباره پیام بده. 🙏"
)

_processing_chats: Set[str] = set()


@retry(max_attempts=3, base_delay=1.0, exceptions=(TelegramError,))
async def _send_text(message: Message, text: str) -> None:
    await message.reply_text(
        format_response(text), parse_mode=ParseMode.HTML, disable_web_page_preview=True
    )


@retry(max_attempts=2, base_delay=1.0, exceptions=(TelegramError,))
async def _send_product_photo(message: Message, product: dict[str, Any]) -> bool:
    url = get_product_image_url(product)
    if not url:
        return False
    await message.reply_photo(
        photo=url, caption=format_product_caption(product), parse_mode=ParseMode.HTML
    )
    return True


async def _reply_with_result(message: Message, result: dict[str, Any]) -> None:
    """Send exactly one graph final_response, then optional product photos."""
    response = result.get("final_response") or USER_FACING_ERROR
    await _send_text(message, response)

    for product in result.get("send_product_images") or []:
        try:
            sent = await _send_product_photo(message, product)
            if not sent:
                name = product.get("name", "این محصول")
                await _send_text(message, f"عکس برای «{name}» در کاتالوگ ثبت نشده.")
        except TelegramError as exc:
            logger.warning("Product image send failed (%s)", exc)

    products = result.get("last_recommended_products") or []
    if not products or not response:
        return
    for product in products[:3]:
        try:
            await _send_product_photo(message, product)
        except TelegramError as exc:
            logger.warning("Product photo send failed (%s) — text already sent", exc)


async def _run_graph(update: Update, **kwargs: Any) -> dict[str, Any]:
    user = update.effective_user
    chat = update.effective_chat
    return await asyncio.to_thread(
        run_turn,
        user_id=str(user.id if user else "unknown"),
        chat_id=str(chat.id if chat else "unknown"),
        **kwargs,
    )


def _chat_lock_id(update: Update) -> Optional[str]:
    chat = update.effective_chat
    return str(chat.id) if chat else None


async def _handle_turn(update: Update, **graph_kwargs: Any) -> None:
    message = update.message
    if not message:
        return

    chat_id = _chat_lock_id(update)
    if chat_id and chat_id in _processing_chats:
        logger.info("Skipping duplicate concurrent turn for chat %s", chat_id)
        return

    if chat_id:
        _processing_chats.add(chat_id)
    try:
        await message.chat.send_action(ChatAction.TYPING)
        result = await _run_graph(update, **graph_kwargs)
        await _reply_with_result(message, result)
    except Exception:
        logger.exception("Graph turn failed")
        await _safe_error_reply(message)
    finally:
        if chat_id:
            _processing_chats.discard(chat_id)


# ------------------------------------------------------------------ commands

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(START_TEXT, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def categories_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    try:
        text = await asyncio.to_thread(build_categories_message)
    except Exception:
        logger.exception("categories_command failed")
        text = "الان نتونستم لیست دسته‌بندی‌ها رو بخونم. چند لحظه دیگه دوباره /categories بزن."
    await message.reply_text(text, parse_mode=ParseMode.HTML)


async def mark_purchased_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message:
        await message.reply_text(
            format_response(PURCHASE_COMING_SOON_TEXT), parse_mode=ParseMode.HTML
        )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear SQLite memory and LangGraph checkpoint for this chat."""
    message = update.message
    chat = update.effective_chat
    if not message or not chat:
        return

    chat_id = str(chat.id)
    if chat_id in _processing_chats:
        _processing_chats.discard(chat_id)

    await asyncio.to_thread(reset_chat_state, chat_id)
    await message.reply_text(
        "حافظه مکالمه و استیت این چت پاک شد. ✅\n"
        "از اینجا می‌تونی مثل اول تست کنی — /start یا مستقیم نیازت رو بنویس.",
        parse_mode=ParseMode.HTML,
    )


# ------------------------------------------------------------------ messages

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return
    await _handle_turn(update, message=message.text, message_type="text")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.photo:
        return
    file_name: Optional[str] = None
    if message.document and message.document.file_name:
        file_name = message.document.file_name
    await _handle_turn(
        update,
        message=message.caption or "",
        message_type="image",
        image_caption=message.caption,
        image_file_name=file_name,
    )


async def _safe_error_reply(message: Message) -> None:
    try:
        await _send_text(message, USER_FACING_ERROR)
    except TelegramError:
        logger.error("Could not deliver error message to chat %s", message.chat_id)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global PTB error handler — logs everything the handlers didn't catch."""
    logger.error("Telegram update error: %s", context.error, exc_info=context.error)


# ------------------------------------------------------------------ followups

async def followup_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic job: deliver due follow-ups (1h idle / 2d no purchase)."""
    service = get_followup_service()
    try:
        due = await asyncio.to_thread(service.check_and_build)
    except Exception:
        logger.exception("Follow-up check failed")
        return
    for item in due:
        try:
            await context.bot.send_message(chat_id=int(item["chat_id"]), text=item["text"])
            await asyncio.to_thread(service.mark_sent, item["followup_id"])
            logger.info("Follow-up sent to chat %s", item["chat_id"])
        except TelegramError as exc:
            logger.warning("Follow-up delivery to %s failed: %s", item["chat_id"], exc)
