"""Telegram bot setup."""

from __future__ import annotations

from telegram import BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_PROXY
from app.telegram.handlers import (
    handle_photo,
    handle_text,
    help_command,
    mark_purchased_command,
    reset_command,
    start_command,
)
from app.telegram.product_commands import (
    browse_command,
    categories_command,
    my_products_command,
    product_command,
    search_command,
)
from app.utils.logging import logger

BOT_COMMANDS = [
    BotCommand("start", "شروع مکالمه"),
    BotCommand("help", "راهنمای دستورات"),
    BotCommand("search", "جستجوی محصول"),
    BotCommand("browse", "مرور یک دسته"),
    BotCommand("categories", "لیست دسته‌بندی‌ها"),
    BotCommand("product", "جزئیات محصول با ID"),
    BotCommand("my_products", "آخرین پیشنهادها"),
    BotCommand("reset", "ریست مکالمه"),
    BotCommand("mark_purchased", "ثبت خرید"),
]


async def _set_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)
    logger.info("Telegram bot command menu configured (%d commands)", len(BOT_COMMANDS))


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler error: %s", context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "متأسفانه خطایی رخ داد. لطفاً دوباره امتحان کن یا /help بزن."
        )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback for unsupported slash commands."""
    command = (update.message.text or "").split()[0]
    await update.message.reply_text(
        f"دستور {command} شناخته نشد.\n"
        "دستورات موجود: /help — /search — /browse — /categories — /product — /my_products"
    )


def create_bot_application() -> Application:
    """Create and configure the Telegram bot application."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    if TELEGRAM_PROXY:
        builder = builder.proxy(TELEGRAM_PROXY).get_updates_proxy(TELEGRAM_PROXY)
        logger.info("Using Telegram proxy: %s", TELEGRAM_PROXY)
    application = builder.post_init(_set_bot_commands).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("browse", browse_command))
    application.add_handler(CommandHandler("categories", categories_command))
    application.add_handler(CommandHandler("product", product_command))
    application.add_handler(CommandHandler("my_products", my_products_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("mark_purchased", mark_purchased_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(_on_error)

    logger.info("Telegram bot application configured")
    return application


def run_bot() -> None:
    """Start the Telegram bot polling loop."""
    app = create_bot_application()
    logger.info("Starting Telegram bot...")
    app.run_polling(allowed_updates=["message"])
