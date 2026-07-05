"""Pytest configuration – force local TF-IDF, no external APIs."""

from __future__ import annotations

import os

# Must run before app imports – force offline mode for tests
os.environ["GROQ_API_KEY"] = ""
os.environ["USE_LLM"] = "false"
os.environ["OPENAI_API_KEY"] = ""
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["VECTOR_STORE_BACKEND"] = "keyword"

import pytest
from pathlib import Path

from app.data.vector_store import reset_vector_store
from app.graph.graph_builder import reset_graph
from app.memory.checkpointer import reset_checkpointer


@pytest.fixture(scope="session")
def csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "products_500.csv"


@pytest.fixture
def reset_singletons():
    reset_graph()
    reset_checkpointer()
    reset_vector_store()
    yield
    reset_graph()
    reset_checkpointer()
    reset_vector_store()
