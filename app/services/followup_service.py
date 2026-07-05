"""Follow-up engine.

Two follow-ups per conversation, both stored in SQLite and rescheduled on
every user interaction:

- idle_1h:      user has been silent for 1 hour -> gentle check-in.
- purchase_2d:  no purchase after 2 days -> resend best previous pick with a
                discount code.

`FollowupService.check_and_build` is called periodically by the Telegram
job queue; it returns messages to send so the transport layer stays separate
from the business logic (which also makes this unit-testable).
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import settings
from app.services.memory_service import MemoryService, get_memory_service
from app.utils.logger import get_logger
from app.utils.text_normalizer import format_price

logger = get_logger(__name__)

IDLE_MESSAGE = (
    "هنوز دنبال همون محصولی هستی؟ اگه بودجه یا مدل مدنظرت عوض شده بگو تا "
    "گزینه‌ها رو دقیق‌تر کنم."
)


def _purchase_message(product: Optional[dict[str, Any]]) -> str:
    lines = [
        "اگه هنوز قصد خرید داری، از بین گزینه‌هایی که بررسی کردیم، "
    ]
    if product:
        lines = [
            f"اگه هنوز قصد خرید داری، از بین گزینه‌هایی که بررسی کردیم، "
            f"«{product.get('name', '')}» با قیمت {format_price(product.get('effective_price') or product.get('price', 0))} "
            "هنوز انتخاب منطقی‌تریه."
        ]
    else:
        lines = ["اگه هنوز قصد خرید داری، خوشحال می‌شم دوباره کمکت کنم."]
    lines.append("")
    lines.append("برای نهایی کردن خرید می‌تونی از کد تخفیف زیر استفاده کنی:")
    lines.append("")
    lines.append(settings.discount_code)
    return "\n".join(lines)


class FollowupService:
    def __init__(self, memory: Optional[MemoryService] = None) -> None:
        self.memory = memory or get_memory_service()

    def check_and_build(self) -> list[dict[str, Any]]:
        """Return [{'chat_id', 'text', 'followup_id'}] for every due follow-up."""
        messages: list[dict[str, Any]] = []
        for followup in self.memory.due_followups():
            chat_id = followup["chat_id"]

            if followup["kind"] == "purchase_2d":
                # Skip if the user already marked the purchase as done
                if self.memory.get_purchase_status(chat_id) == "purchased":
                    self.memory.mark_followup_sent(followup["id"])
                    continue
                recs = self.memory.get_active_recommendations(chat_id)
                best = recs[0] if recs else None
                text = _purchase_message(best)
            else:  # idle_1h
                recs = self.memory.get_active_recommendations(chat_id)
                if not recs:
                    # Nothing was recommended yet; a nag would be noise
                    self.memory.mark_followup_sent(followup["id"])
                    continue
                text = IDLE_MESSAGE

            messages.append(
                {"chat_id": chat_id, "text": text, "followup_id": followup["id"]}
            )
        if messages:
            logger.info("Due follow-ups: %d", len(messages))
        return messages

    def mark_sent(self, followup_id: int) -> None:
        self.memory.mark_followup_sent(followup_id)


_service: Optional[FollowupService] = None


def get_followup_service() -> FollowupService:
    global _service
    if _service is None:
        _service = FollowupService()
    return _service
