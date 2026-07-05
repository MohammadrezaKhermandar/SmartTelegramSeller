"""Telegram bot application setup (python-telegram-bot v21, async)."""

from __future__ import annotations

from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.config import settings
from app.telegram import handlers
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    builder = ApplicationBuilder().token(settings.telegram_bot_token)

    if settings.telegram_proxy:
        logger.info("Using Telegram proxy: %s", settings.telegram_proxy)
        builder = (
            builder.proxy(settings.telegram_proxy)
            .get_updates_proxy(settings.telegram_proxy)
            .connect_timeout(30)
            .read_timeout(30)
        )

    app = builder.build()

    app.add_handler(CommandHandler("start", handlers.start_command))
    app.add_handler(CommandHandler("help", handlers.help_command))
    app.add_handler(CommandHandler("categories", handlers.categories_command))
    app.add_handler(CommandHandler("mark_purchased", handlers.mark_purchased_command))
    app.add_handler(CommandHandler("reset", handlers.reset_command))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_text))
    app.add_error_handler(handlers.error_handler)

    if app.job_queue:
        app.job_queue.run_repeating(
            handlers.followup_job,
            interval=settings.followup_check_interval,
            first=10,
            name="followups",
        )
        logger.info(
            "Follow-up job scheduled every %ds (idle=%ds, purchase=%ds)",
            settings.followup_check_interval,
            settings.followup_idle_seconds,
            settings.followup_purchase_seconds,
        )

    return app


def run_bot() -> None:
    """Warm up services eagerly (fail fast), then start polling."""
    from app.graph.builder import get_graph
    from app.services.rag_service import get_rag_service

    logger.info("Warming up catalog, vector index and graph…")
    get_rag_service()
    get_graph()

    app = build_application()
    logger.info("Bot is up — polling for updates.")
    app.run_polling(drop_pending_updates=True)
