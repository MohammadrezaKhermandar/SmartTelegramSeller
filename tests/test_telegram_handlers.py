"""Tests for Telegram handlers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.main import initialize_app

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_app():
    initialize_app(CSV_PATH)


@pytest.mark.asyncio
async def test_handle_text_invokes_graph_with_user_id():
    from app.telegram.handlers import handle_text

    update = MagicMock()
    update.effective_user.id = 12345
    update.message.text = "یه لپ‌تاپ می‌خوام"
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.application = MagicMock()
    context.application.job_queue = None

    with patch("app.telegram.handlers.invoke_graph") as mock_invoke:
        mock_invoke.return_value = {
            "response_text": "سلام",
            "recommended_products": [],
            "conversation_stage": "gathering_requirements",
        }
        await handle_text(update, context)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args
        assert call_kwargs[0][0] == "12345" or call_kwargs[0][0] == 12345 or str(call_kwargs[0][0]) == "12345"


@pytest.mark.asyncio
async def test_start_command():
    from app.telegram.handlers import start_command

    update = MagicMock()
    update.effective_user.id = 99
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    await start_command(update, context)
    update.message.reply_text.assert_called_once()
