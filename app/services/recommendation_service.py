"""Hybrid recommendation: Pandas exact filtering + RAG semantic ranking.

final_score =
    0.40 * semantic_similarity_score
  + 0.25 * budget_match_score
  + 0.15 * use_case_match_score
  + 0.10 * brand_match_score
  + 0.10 * availability_score

Every component degrades gracefully when its inputs are missing (e.g. no
budget given -> neutral budget score).
"""

from __future__ import annotations

from typing import Any, Optional

from app.services.pandas_query_service import ProductFilter, get_pandas_service
from app.services.rag_service import get_rag_service
from app.utils.logger import get_logger
from app.utils.text_normalizer import brands_match, normalize

logger = get_logger(__name__)

WEIGHTS = {
    "semantic": 0.40,
    "budget": 0.25,
    "use_case": 0.15,
    "brand": 0.10,
    "availability": 0.10,
}

USE_CASE_KEYWORDS: dict[str, list[str]] = {
    "برنامه‌نویسی": ["رم", "پردازنده", "ssd", "core", "ryzen", "لپ تاپ"],
    "گیمینگ": ["گرافیک", "rtx", "گیمینگ", "بازی", "rgb", "هرتز"],
    "بازی": ["گرافیک", "rtx", "گیمینگ", "بازی", "rgb"],
    "دانشجویی": ["سبک", "باتری", "اقتصادی", "وزن"],
    "کار اداری": ["باتری", "سبک", "وزن", "صفحه"],
    "عکاسی": ["دوربین", "لنز", "کیفیت تصویر", "مگاپیکسل"],
    "ورزش": ["ضد آب", "باتری", "سنسور", "ورزشی"],
    "آشپزی": ["ظرفیت", "وات", "آشپزخانه"],
    "سفر": ["سبک", "قابل حمل", "باتری", "وزن"],
}

_PRODUCT_FIELDS = (
    "product_id", "name", "brand", "category", "model", "price",
    "effective_price", "discount", "stock", "rating", "review_count",
    "features", "description", "warranty", "color",
    "image_url", "product_url", "search_text",
)


def _budget_score(price: float, budget: Optional[float]) -> float:
    if not budget or budget <= 0:
        return 0.6
    ratio = price / budget
    if ratio <= 0.5:
        return 0.85
    if ratio <= 1.0:
        return 1.0
    if ratio <= 1.15:
        return 0.5
    return 0.0


def _use_case_score(search_text: str, use_case: Optional[str]) -> float:
    if not use_case:
        return 0.6
    text = normalize(search_text).lower()
    keywords = USE_CASE_KEYWORDS.get(use_case, [use_case])
    if not keywords:
        return 0.6
    hits = sum(1 for k in keywords if normalize(k).lower() in text)
    return min(1.0, 0.3 + 0.7 * hits / max(1, min(len(keywords), 4)))


def _brand_score(brand: str, preferred: list[str]) -> float:
    if not preferred:
        return 0.6
    return 1.0 if any(brands_match(p, brand) for p in preferred) else 0.2


def _availability_score(stock: int) -> float:
    if stock <= 0:
        return 0.0
    if stock < 10:
        return 0.7
    return 1.0


def _in_category(product: dict[str, Any], category: str) -> bool:
    cat_norm = normalize(category)
    prod_cat = normalize(str(product.get("category", "")))
    return cat_norm in prod_cat or prod_cat in cat_norm


def _product_snapshot(product: dict[str, Any], **extra: Any) -> dict[str, Any]:
    snap = {k: product.get(k) for k in _PRODUCT_FIELDS if k in product or k != "search_text"}
    snap.update(extra)
    return snap


def _effective_price(product: dict[str, Any]) -> float:
    return float(product.get("effective_price") or product.get("price") or 0)


def filter_products_by_max_price(
    products: list[dict[str, Any]], max_price: Optional[float]
) -> list[dict[str, Any]]:
    """Drop products above max_price (hard budget guard)."""
    if max_price is None or max_price <= 0:
        return products
    return [p for p in products if _effective_price(p) <= max_price]


class RecommendationService:
    """Combines Pandas candidates with RAG similarity into a ranked list."""

    def __init__(self) -> None:
        self.pandas = get_pandas_service()
        self.rag = get_rag_service()

    def recommend(
        self,
        query_text: str,
        requirements: dict[str, Any],
        top_k: int = 3,
        strict_budget: bool = False,
    ) -> dict[str, Any]:
        budget = requirements.get("budget")
        category = requirements.get("category")
        brands = requirements.get("brands") or []
        strict = strict_budget or bool(requirements.get("hard_max_price"))

        f = ProductFilter(
            max_price=budget,
            brands=brands,
            categories=[category] if category else [],
            in_stock_only=True,
            limit=50,
            sort_by="rating",
            strict_budget=strict,
        )
        pandas_result = self.pandas.query(f)
        candidates = pandas_result["products"]
        relaxed = list(pandas_result["relaxed"])

        if strict and budget:
            candidates = filter_products_by_max_price(candidates, budget)

        allowed_ids = [str(p["product_id"]) for p in candidates] or None
        rag_hits = self.rag.search(query_text, n_results=30, allowed_ids=allowed_ids)
        similarity_by_id = {h["product_id"]: h["similarity"] for h in rag_hits}

        if not candidates and category and not strict:
            # Hard category filter: never jump to whole catalog while category has stock
            if self.pandas.category_has_products(category):
                logger.info("Category %s has products but filters returned none", category)
            else:
                logger.info("Category %s empty in catalog — cross-category fallback", category)
                rag_hits = self.rag.search(query_text, n_results=30)
                similarity_by_id = {h["product_id"]: h["similarity"] for h in rag_hits}
                catalog = self.pandas.catalog
                candidates = [
                    p for h in rag_hits
                    if (p := catalog.get_by_id(h["product_id"])) is not None
                ]
                relaxed.append("category_empty->catalog")
        elif not candidates and not strict:
            rag_hits = self.rag.search(query_text, n_results=30)
            similarity_by_id = {h["product_id"]: h["similarity"] for h in rag_hits}
            catalog = self.pandas.catalog
            candidates = [
                p for h in rag_hits
                if (p := catalog.get_by_id(h["product_id"])) is not None
            ]

        if strict and budget:
            candidates = filter_products_by_max_price(candidates, budget)

        scored = self._score_candidates(
            candidates, similarity_by_id, budget, requirements.get("use_case"), brands
        )
        scored.sort(key=lambda p: p["score"], reverse=True)

        if strict and budget:
            scored = filter_products_by_max_price(scored, budget)

        slice_size = top_k if strict else max(top_k, 3)
        if category:
            in_cat = [p for p in scored if _in_category(p, category)]
            out_cat = [p for p in scored if not _in_category(p, category)]
            if in_cat:
                top = in_cat[:slice_size]
            else:
                for p in out_cat:
                    p["is_alternative"] = True
                top = out_cat[:slice_size]
        else:
            top = scored[:slice_size]

        if not strict and len(top) < 3:
            top = self._pad_with_alternatives(top, query_text, requirements, category)

        if strict and budget:
            top = filter_products_by_max_price(top, budget)[:top_k]
        else:
            top = top[:max(top_k, 3)]

        empty_reason = ""
        if strict and budget and not top:
            empty_reason = "داخل این بودجه گزینه مناسبی پیدا نکردم."

        logger.info(
            "Hybrid recommend: %d candidates -> top %d (best=%s, strict=%s)",
            len(scored), len(top), top[0]["name"] if top else "-", strict,
        )
        return {
            "products": top,
            "exact_match": pandas_result["exact_match"] and not any(
                p.get("is_alternative") for p in top
            ),
            "relaxed": relaxed,
            "empty_reason": empty_reason,
        }

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        similarity_by_id: dict[str, float],
        budget: Optional[float],
        use_case: Optional[str],
        brands: list[str],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for product in candidates:
            pid = str(product["product_id"])
            components = {
                "semantic": similarity_by_id.get(pid, 0.3),
                "budget": _budget_score(float(product["effective_price"]), budget),
                "use_case": _use_case_score(product.get("search_text", ""), use_case),
                "brand": _brand_score(product.get("brand", ""), brands),
                "availability": _availability_score(int(product.get("stock", 0))),
            }
            final = sum(WEIGHTS[k] * v for k, v in components.items())
            scored.append(
                _product_snapshot(
                    product,
                    score=round(final, 4),
                    score_components={k: round(v, 3) for k, v in components.items()},
                )
            )
        return scored

    def _pad_with_alternatives(
        self,
        top: list[dict[str, Any]],
        query_text: str,
        requirements: dict[str, Any],
        category: Optional[str],
    ) -> list[dict[str, Any]]:
        existing_ids = {str(p["product_id"]) for p in top}
        catalog = self.pandas.catalog
        budget = requirements.get("budget")
        category_has_stock = bool(category and self.pandas.category_has_products(category))

        if category and category_has_stock:
            f = ProductFilter(
                categories=[category],
                max_price=budget * 1.5 if budget else None,
                in_stock_only=True,
                limit=20,
            )
            for product in self.pandas.query(f)["products"]:
                if len(top) >= 3:
                    break
                pid = str(product["product_id"])
                if pid in existing_ids:
                    continue
                top.append(
                    _product_snapshot(
                        product,
                        score=0.3,
                        score_components={},
                        is_alternative=bool(budget and float(product["effective_price"]) > budget),
                        over_budget=bool(
                            budget and float(product["effective_price"]) > budget
                        ),
                    )
                )
                existing_ids.add(pid)

        if len(top) >= 3:
            return top

        if category and category_has_stock:
            return top

        for hit in self.rag.search(query_text, n_results=20):
            if len(top) >= 3:
                break
            pid = str(hit["product_id"])
            if pid in existing_ids:
                continue
            product = catalog.get_by_id(pid)
            if product is None or int(product.get("stock", 0)) <= 0:
                continue
            top.append(
                _product_snapshot(
                    product,
                    score=round(hit["similarity"] * WEIGHTS["semantic"], 4),
                    score_components={"semantic": hit["similarity"]},
                    is_alternative=True,
                    over_budget=bool(
                        budget and float(product["effective_price"]) > budget
                    ),
                )
            )
            existing_ids.add(pid)
        return top


_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service
