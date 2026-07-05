"""Scheduled follow-up jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.config import DISCOUNT_2D_SECONDS, FOLLOWUP_1H_SECONDS
from app.graph.prompts import DISCOUNT_2D_MESSAGE, FOLLOWUP_1H_MESSAGE
from app.memory.store import session_store
from app.utils.logging import logger

if TYPE_CHECKING:
    from telegram.ext import Application


def _products_summary(products: list[dict]) -> str:
    lines = []
    for i, p in enumerate(products[:3], 1):
        title = p.get("title", "")
        price = int(p.get("price", 0) or 0)
        lines.append(f"{i}. {title} – {price:,} تومان")
    return "\n".join(lines)


async def send_followup_1h(context: Any) -> None:
    """Send 1-hour follow-up if user hasn't responded."""
    job = context.job
    user_id = str(job.data.get("user_id", ""))
    session = session_store.get_or_create(user_id)

    if session.purchase_status == "purchased":
        return

    last = session.last_message_at
    if last:
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        if elapsed < FOLLOWUP_1H_SECONDS * 0.9:
            return  # User responded recently

    try:
        await context.bot.send_message(chat_id=int(user_id), text=FOLLOWUP_1H_MESSAGE)
        logger.info("Sent 1h follow-up to user %s", user_id)
    except Exception as exc:
        logger.warning("Failed 1h follow-up for %s: %s", user_id, exc)


async def send_discount_2d(context: Any) -> None:
    """Send 2-day discount follow-up."""
    job = context.job
    user_id = str(job.data.get("user_id", ""))
    session = session_store.get_or_create(user_id)

    if session.purchase_status == "purchased":
        return

    products = session.recommended_products
    if not products:
        return

    message = DISCOUNT_2D_MESSAGE.format(products_summary=_products_summary(products))
    try:
        await context.bot.send_message(chat_id=int(user_id), text=message, parse_mode="Markdown")
        logger.info("Sent 2d discount follow-up to user %s", user_id)
    except Exception as exc:
        logger.warning("Failed 2d follow-up for %s: %s", user_id, exc)


def schedule_followups(application: "Application", user_id: str) -> None:
    """Schedule follow-up jobs after recommendation."""
    job_queue = application.job_queue
    if job_queue is None:
        logger.warning("Job queue not available")
        return

    job_queue.run_once(
        send_followup_1h,
        when=FOLLOWUP_1H_SECONDS,
        data={"user_id": user_id},
        name=f"followup_1h_{user_id}",
    )
    job_queue.run_once(
        send_discount_2d,
        when=DISCOUNT_2D_SECONDS,
        data={"user_id": user_id},
        name=f"discount_2d_{user_id}",
    )
    logger.info(
        "Scheduled follow-ups for user %s (1h=%ds, 2d=%ds)",
        user_id,
        FOLLOWUP_1H_SECONDS,
        DISCOUNT_2D_SECONDS,
    )
