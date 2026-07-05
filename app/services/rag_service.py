"""RAG layer with pluggable vector backends.

Backends (``VECTOR_STORE_BACKEND``):
- ``keyword`` — in-memory hashed n-gram similarity (offline, no ChromaDB).
- ``chroma``   — ChromaDB persistent/ephemeral index (lazy-imported).

If ChromaDB init or query fails, the service automatically falls back to the
keyword backend so demos and tests stay stable on Windows.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from app.config import settings
from app.services.product_loader import ProductCatalog, get_catalog
from app.utils.logger import get_logger
from app.utils.retry import retry
from app.utils.text_normalizer import normalize

logger = get_logger(__name__)

COLLECTION_NAME = "products"
EMBED_DIM = 512


class HashedNGramEmbedder:
    """Deterministic character n-gram hashing — language-agnostic, offline."""

    def __init__(self, dim: int = EMBED_DIM, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    @staticmethod
    def name() -> str:
        return "hashed_ngram"

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = re.sub(r"\s+", " ", text.lower()).strip()
        for word in text.split(" "):
            token = f" {word} "
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(token) - n + 1):
                    gram = token[i : i + n]
                    h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                    vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": str(record["product_id"]),
        "name": record["name"],
        "brand": record["brand"],
        "category": record["category"],
        "price": float(record["price"]),
        "effective_price": float(record["effective_price"]),
        "availability": bool(record["stock"] > 0),
        "stock": int(record["stock"]),
        "rating": float(record["rating"]),
        "image_url": record.get("image_url") or "",
        "product_url": record.get("product_url") or "",
    }


def _metadata_matches(meta: dict[str, Any], where: Optional[dict[str, Any]]) -> bool:
    """Minimal Chroma-style where filter for the keyword backend."""
    if not where:
        return True
    if "$and" in where:
        return all(_metadata_matches(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_metadata_matches(meta, clause) for clause in where["$or"])
    for key, cond in where.items():
        value = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond:
                if str(value) not in {str(v) for v in cond["$in"]}:
                    return False
            if "$lte" in cond and not (value is not None and float(value) <= float(cond["$lte"])):
                return False
            if "$gte" in cond and not (value is not None and float(value) >= float(cond["$gte"])):
                return False
            if "$eq" in cond and value != cond["$eq"]:
                return False
        elif value != cond:
            return False
    return True


@dataclass
class _KeywordEntry:
    product_id: str
    embedding: list[float]
    metadata: dict[str, Any]
    search_text: str


class KeywordRAGService:
    """In-memory lexical/semantic retrieval without ChromaDB."""

    backend = "keyword"

    def __init__(self, catalog: Optional[ProductCatalog] = None) -> None:
        self.catalog = catalog or get_catalog()
        self.embedder = HashedNGramEmbedder()
        self._entries: list[_KeywordEntry] = []
        self._build_index()

    def _build_index(self) -> None:
        records = self.catalog.to_records()
        texts = [r["search_text"] for r in records]
        vectors = self.embedder.embed_batch(texts)
        self._entries = [
            _KeywordEntry(
                product_id=str(r["product_id"]),
                embedding=vec,
                metadata=_record_metadata(r),
                search_text=r["search_text"],
            )
            for r, vec in zip(records, vectors)
        ]
        logger.info("Keyword index ready (%d products, hash embeddings)", len(self._entries))

    def document_count(self) -> int:
        return len(self._entries)

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        allowed_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if allowed_ids is not None and not allowed_ids:
            return []
        allowed = {str(i) for i in allowed_ids} if allowed_ids is not None else None
        q_vec = self.embedder.embed_one(query)
        q_norm = normalize(query).lower()
        q_tokens = set(q_norm.split())

        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._entries:
            if allowed is not None and entry.product_id not in allowed:
                continue
            if not _metadata_matches(entry.metadata, where):
                continue
            sim = _cosine(q_vec, entry.embedding)
            text_norm = normalize(entry.search_text).lower()
            token_hits = sum(1 for t in q_tokens if len(t) > 2 and t in text_norm)
            if token_hits:
                sim = min(1.0, sim + 0.05 * token_hits)
            scored.append((sim, {**entry.metadata, "similarity": round(max(0.0, sim), 4)}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored[:n_results]]


class ChromaRAGService:
    """ChromaDB-backed vector search (imported lazily)."""

    backend = "chroma"

    def __init__(self, catalog: Optional[ProductCatalog] = None) -> None:
        import chromadb  # lazy — only when backend is chroma

        self.catalog = catalog or get_catalog()
        self._chromadb = chromadb
        db_path = str(settings.vector_db_path)
        if db_path in {":memory:", "ephemeral"}:
            self.client = chromadb.EphemeralClient()
        else:
            settings.vector_db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = self._get_embedding_function()
        self.collection = self._ensure_index()

    def _get_embedding_function(self) -> Any:
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

        embedder = HashedNGramEmbedder()
        mode = os.getenv("EMBEDDING_MODE", "hash").lower()
        if mode == "onnx":
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                logger.info("Using ONNX MiniLM embeddings")
                return ONNXMiniLM_L6_V2()
            except Exception as exc:
                logger.warning("ONNX embeddings unavailable (%s); falling back to hash", exc)

        class _HashEmbedding(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return embedder.embed_batch(list(input))

            @staticmethod
            def name() -> str:
                return HashedNGramEmbedder.name()

        return _HashEmbedding()

    def _ensure_index(self) -> Any:
        expected_meta = {
            "csv_hash": self.catalog.csv_hash,
            "embedding": getattr(self.embedding_fn, "name", lambda: "default")(),
        }
        try:
            collection = self.client.get_collection(
                COLLECTION_NAME, embedding_function=self.embedding_fn
            )
            meta = collection.metadata or {}
            if (
                meta.get("csv_hash") == expected_meta["csv_hash"]
                and meta.get("embedding") == expected_meta["embedding"]
                and collection.count() == len(self.catalog.df)
            ):
                logger.info(
                    "Vector index up to date (%d docs), skipping rebuild", collection.count()
                )
                return collection
            logger.info("CSV or embedding changed — rebuilding vector index")
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            logger.info("No existing vector index found — building a new one")

        collection = self.client.create_collection(
            COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={**expected_meta, "hnsw:space": "cosine"},
        )
        self._index_products(collection)
        return collection

    @retry(max_attempts=3)
    def _index_products(self, collection: Any) -> None:
        records = self.catalog.to_records()
        batch = 200
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            collection.add(
                ids=[str(r["product_id"]) for r in chunk],
                documents=[r["search_text"] for r in chunk],
                metadatas=[_record_metadata(r) for r in chunk],
            )
        logger.info("Indexed %d products into ChromaDB", len(records))

    @retry(max_attempts=3)
    def _query(self, text: str, n_results: int, where: Optional[dict]) -> dict[str, Any]:
        return self.collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

    def document_count(self) -> int:
        return int(self.collection.count())

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        allowed_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            id_filter = {"product_id": {"$in": allowed_ids}}
            where = {"$and": [where, id_filter]} if where else id_filter
        try:
            res = self._query(query, n_results, where)
        except Exception as exc:
            logger.error("Chroma query failed after retries: %s", exc)
            raise
        hits: list[dict[str, Any]] = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            similarity = max(0.0, 1.0 - dist / 2.0)
            hits.append({**meta, "similarity": round(similarity, 4)})
        return hits


class RAGBackend(Protocol):
    backend: str

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        allowed_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]: ...

    def document_count(self) -> int: ...


class ResilientRAGService:
    """Wraps Chroma with automatic keyword fallback on init/query failure."""

    def __init__(self, inner: RAGBackend, fallback: KeywordRAGService) -> None:
        self._inner = inner
        self._fallback = fallback
        self.backend = inner.backend

    def document_count(self) -> int:
        try:
            return self._inner.document_count()
        except Exception:
            return self._fallback.document_count()

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        allowed_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        try:
            return self._inner.search(query, n_results, where, allowed_ids)
        except Exception as exc:
            logger.warning("Chroma search failed (%s) — using keyword fallback", exc)
            return self._fallback.search(query, n_results, where, allowed_ids)


# Backward-compatible alias used in type hints across the codebase.
RAGService = KeywordRAGService

_rag: Optional[RAGBackend] = None


def _create_backend() -> RAGBackend:
    backend = settings.vector_store_backend
    keyword = KeywordRAGService()
    if backend == "keyword":
        logger.info("RAG backend: keyword (in-memory hash)")
        return keyword
    try:
        chroma = ChromaRAGService()
        logger.info("RAG backend: chroma")
        return ResilientRAGService(chroma, keyword)
    except Exception as exc:
        logger.warning("ChromaDB unavailable (%s) — using keyword backend", exc)
        return keyword


def get_rag_service() -> RAGBackend:
    """Singleton accessor — backend chosen from ``VECTOR_STORE_BACKEND``."""
    global _rag
    if _rag is None:
        _rag = _create_backend()
    return _rag


def reset_rag_service() -> None:
    """Clear singleton (tests only)."""
    global _rag
    _rag = None
