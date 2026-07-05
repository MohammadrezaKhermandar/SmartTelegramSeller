"""Shared test fixtures: isolated data dir, no LLM, keyword RAG backend."""

from __future__ import annotations

import os

os.environ["USE_LLM"] = "false"
os.environ["MEMORY_DB_PATH"] = ":memory:"
os.environ["VECTOR_STORE_BACKEND"] = "keyword"
os.environ["EMBEDDING_MODE"] = "hash"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402

from app.services.memory_service import MemoryService  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: end-to-end graph tests (offline keyword backend)",
    )
    config.addinivalue_line(
        "markers",
        "smoke: quick offline demo flow (run_turn + demo scenarios)",
    )


@pytest.fixture(autouse=True)
def _reset_service_singletons():
    """Prevent cross-test singleton pollution."""
    import app.graph.builder as builder_mod
    import app.services.rag_service as rag_mod
    import app.services.recommendation_service as rec_mod

    rag_mod.reset_rag_service()
    rec_mod._service = None
    builder_mod._graph = None
    yield
    rag_mod.reset_rag_service()
    rec_mod._service = None
    builder_mod._graph = None


@pytest.fixture()
def memory() -> MemoryService:
    return MemoryService(db_path=":memory:")
