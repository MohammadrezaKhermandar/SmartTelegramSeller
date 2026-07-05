"""Vector store with Chroma (online) or local keyword/TF-IDF fallback."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd

from app.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, RAG_TOP_K, VECTOR_STORE_BACKEND
from app.utils.errors import VectorStoreError
from app.utils.logging import logger
from app.utils.retry import with_retry

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


class KeywordVectorStore:
    """Pure-Python local search – no network, no sklearn. Best for offline/tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._docs: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        df_docs = df["combined_text"].fillna("").astype(str).tolist()
        doc_count = len(df_docs)
        df_counts: Counter[str] = Counter()

        for text in df_docs:
            tokens = _tokenize(text)
            self._docs.append(tokens)
            tf = Counter(tokens)
            self._tf.append(tf)
            df_counts.update(set(tokens))

        for term, dfreq in df_counts.items():
            self._idf[term] = math.log((1 + doc_count) / (1 + dfreq)) + 1

        logger.info("Keyword vector store initialized with %d products", len(df))

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        tf = self._tf[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term in tf:
                score += (tf[term] / max(len(self._docs[doc_idx]), 1)) * self._idf.get(term, 1.0)
        return score

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        scored = [
            (idx, self._score(query_tokens, idx))
            for idx in range(len(self.df))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:top_k]:
            if score <= 0:
                continue
            results.append(_row_to_search_result(self.df.iloc[idx], score))
        return results


class TfidfVectorStore:
    """scikit-learn TF-IDF – local but heavier import."""

    def __init__(self, df: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.df = df
        self._cosine_similarity = cosine_similarity
        texts = df["combined_text"].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF vector store initialized with %d products", len(df))

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
        query_vec = self.vectorizer.transform([query])
        scores = self._cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = self.df.iloc[idx]
            results.append(_row_to_search_result(row, float(scores[idx])))
        return results


class ChromaVectorStore:
    """Chroma + OpenAI embeddings – requires network and chromadb package."""

    def __init__(self, df: pd.DataFrame) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.df = df
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
        self.collection = self.client.get_or_create_collection(
            name="products",
            embedding_function=embed_fn,
        )
        self._index_products()

    @with_retry()
    def _index_products(self) -> None:
        if self.collection.count() >= len(self.df):
            logger.info("Chroma collection already indexed")
            return
        ids = self.df["product_id"].astype(str).tolist()
        documents = self.df["combined_text"].fillna("").astype(str).tolist()
        metadatas = [
            {
                "title": str(row.get("title", "")),
                "price": float(row.get("price", 0) or 0),
                "brand": str(row.get("brand", "")),
                "category": str(row.get("category", "")),
            }
            for _, row in self.df.iterrows()
        ]
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        logger.info("Indexed %d products in Chroma", len(ids))

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        output = []
        if not results or not results.get("ids"):
            return output
        for i, pid in enumerate(results["ids"][0]):
            score = 1.0 - (results["distances"][0][i] if results.get("distances") else 0)
            row_match = self.df[self.df["product_id"].astype(str) == str(pid)]
            if row_match.empty:
                continue
            output.append(_row_to_search_result(row_match.iloc[0], score))
        return output


def _row_to_search_result(row: pd.Series, score: float) -> dict[str, Any]:
    desc = str(row.get("description", ""))[:200]
    return {
        "product_id": str(row.get("product_id", "")),
        "title": str(row.get("title", "")),
        "price": float(row.get("price", 0) or 0),
        "brand": str(row.get("brand", "")),
        "category": str(row.get("category", "")),
        "description": desc,
        "image_url": str(row.get("image_url", "")) if pd.notna(row.get("image_url", None)) else "",
        "score": round(score, 4),
        "rating": float(row.get("rating", 0) or 0) if "rating" in row.index else None,
        "discount": float(row.get("discount", 0) or 0) if "discount" in row.index else None,
        "features": str(row.get("features", "")),
        "stock": int(float(row.get("availability", row.get("stock", 0)) or 0)),
    }


_vector_store: Any | None = None


def create_vector_store(df: pd.DataFrame) -> KeywordVectorStore | TfidfVectorStore | ChromaVectorStore:
    """Create vector store based on VECTOR_STORE_BACKEND (keyword default offline)."""
    backend = VECTOR_STORE_BACKEND

    if backend == "chroma" or (backend == "auto" and OPENAI_API_KEY):
        try:
            return ChromaVectorStore(df)
        except Exception as exc:
            logger.warning("Chroma unavailable (%s), falling back to keyword search", exc)

    if backend == "tfidf":
        try:
            return TfidfVectorStore(df)
        except ImportError:
            logger.warning("scikit-learn not installed, using keyword search")
        except Exception as exc:
            logger.warning("TF-IDF init failed (%s), using keyword search", exc)

    return KeywordVectorStore(df)


def get_vector_store(df: pd.DataFrame | None = None) -> Any:
    """Singleton accessor for vector store."""
    global _vector_store
    if _vector_store is None:
        if df is None:
            raise VectorStoreError("DataFrame required for first vector store init")
        _vector_store = create_vector_store(df)
    return _vector_store


def reset_vector_store() -> None:
    """Reset singleton (for tests)."""
    global _vector_store
    _vector_store = None
