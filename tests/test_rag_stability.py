"""Offline RAG backend and smoke/integration stability tests."""

from __future__ import annotations

import uuid

import pytest

from app.config import settings
from app.graph.builder import run_turn
from app.services.memory_service import get_memory_service
from app.services.rag_service import KeywordRAGService, get_rag_service, reset_rag_service


def test_keyword_backend_is_default_in_tests():
    assert settings.vector_store_backend == "keyword"


def test_get_rag_service_without_chromadb():
    reset_rag_service()
    service = get_rag_service()
    assert service.backend == "keyword"
    assert service.document_count() > 0
    hits = service.search("لپ‌تاپ برنامه نویسی", n_results=5)
    assert hits
    assert all("similarity" in h for h in hits)


def test_chromadb_not_imported_for_keyword_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "keyword")
    reset_rag_service()
    get_rag_service()
    # Keyword path must not pull in chromadb (may already be loaded elsewhere)
    import app.services.rag_service as rag_mod

    assert rag_mod._rag is not None
    assert rag_mod._rag.backend == "keyword"


def test_run_turn_without_chromadb():
    chat_id = f"offline-{uuid.uuid4().hex[:8]}"
    result = run_turn("u", chat_id, "یه لپ‌تاپ می‌خوام")
    assert result.get("final_response")
    assert "؟" in result["final_response"]


@pytest.mark.integration
def test_run_turn_full_recommendation_flow():
    chat_id = f"flow-{uuid.uuid4().hex[:8]}"
    run_turn("u", chat_id, "یه لپ‌تاپ می‌خوام")
    result = run_turn(
        "u", chat_id, "تا ۵۰ میلیون برای برنامه‌نویسی، برند فرقی نداره"
    )
    recs = get_memory_service().get_active_recommendations(chat_id)
    assert len(recs) >= 3
    assert result.get("final_response")


@pytest.mark.smoke
def test_demo_scenarios_main_offline(monkeypatch):
    """Exercise demo_scenarios.main without ChromaDB or LLM."""
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "keyword")
    monkeypatch.setenv("USE_LLM", "false")
    monkeypatch.setenv("MEMORY_DB_PATH", ":memory:")

    reset_rag_service()
    import app.graph.builder as builder_mod
    import app.services.recommendation_service as rec_mod

    builder_mod._graph = None
    rec_mod._service = None

    import scripts.demo_scenarios as demo

    demo.main()
