"""Tests for checkpointer reset."""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph, reset_graph
from app.graph.runner import reset_user_state
from app.main import initialize_app
from app.memory.checkpointer import get_checkpointer, reset_checkpointer

CSV_PATH = Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture(autouse=True)
def setup_app():
    reset_graph()
    reset_checkpointer()
    initialize_app(CSV_PATH)
    yield
    reset_graph()
    reset_checkpointer()


def test_reset_clears_checkpointer_thread():
    user_id = "reset_test_user"
    graph = build_graph()
    config = {"configurable": {"thread_id": user_id}}

    graph.invoke(
        {
            "user_id": user_id,
            "messages": [HumanMessage(content="سلام")],
            "retry_count": 0,
            "errors": [],
        },
        config,
    )

    cp = get_checkpointer()
    assert user_id in cp.storage

    reset_user_state(user_id)
    assert user_id not in cp.storage
