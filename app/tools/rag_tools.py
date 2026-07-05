"""RAG tools for semantic product search."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.config import RAG_TOP_K
from app.data.vector_store import get_vector_store

_df = None


def init_rag_tools(df) -> None:
    global _df
    _df = df
    get_vector_store(df)


@tool
def semantic_search_tool(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """Semantic search over product catalog using RAG."""
    return semantic_search(query, top_k)


def semantic_search(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """Run semantic search – Chroma or TF-IDF fallback."""
    store = get_vector_store(_df)
    return store.search(query, top_k=top_k)


def find_similar_products(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Find similar products for image/URL inputs."""
    return semantic_search(query, top_k=top_k)
