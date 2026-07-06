# Project Memory — SINWAY Sales Assistant

Living notes for contributors and agents. Read this before changing routing, NLU, or recommendations.

## Architecture map

| Area | Files | Role |
|------|-------|------|
| NLU / phrases | `app/utils/text_normalizer.py` | Budget extraction, hard ceiling, preference rejection |
| Intent + slots | `app/graph/nodes.py` → `extract_user_intent_and_requirements` | Merges rule-based + optional LLM slots |
| Memory vs search | `app/graph/nodes.py` → `check_memory_relevance` | Decides `answer_from_memory` vs `product_search` |
| Router | `app/graph/router.py` | `conditional_router` priority order |
| Recommendations | `app/services/recommendation_service.py` | Pandas filter + RAG rank + optional padding |
| Persistence | `app/services/memory_service.py` | SQLite sessions + active recommendations |

Legacy names from other repos (`nlp.py`, `routers.py`, `update_requirements`) map to **`text_normalizer.py`**, **`router.py`**, and **`extract_user_intent_and_requirements`** here.

## Requirements dict (session-persisted)

| Key | Meaning |
|-----|---------|
| `budget` | Max budget in Toman |
| `category` | Product category (hard slot) |
| `hard_max_price` | Strict ceiling — no pandas relaxation, no padding |
| `allow_budget_overflow` | `True` (default): may pad with slightly-over-budget alternatives. `False`: strict in-budget only |

Setting `allow_budget_overflow=false` implies the same enforcement as `hard_max_price=true` in the recommendation engine.

## Intents

| Intent | When |
|--------|------|
| `product_request` | New search or slot update (budget/category/brand) |
| `change_preferences` | User rejects over-budget alternatives (see phrases below) |
| `memory_question` | Ordinal/name/attribute question about prior recs |

**Priority:** `change_preferences` is detected **before** memory heuristics. Messages containing «قیمت» used to be mis-routed to `answer_from_memory` via `is_attribute_question`.

## change_preferences phrases

Detected by `is_reject_budget_overflow()` / `is_change_preferences()`:

- نه قیمت بالاتر نمیتونم
- بیشتر از این نمی‌تونم هزینه کنم
- بودجه‌ام همینه
- نمی‌خوام بیشتر هزینه کنم
- سقف بودجه همینه
- گرون‌تر نمی‌خوام
- فقط داخل همین بودجه

**Flow:**

1. `extract_user_intent_and_requirements` → `intent=change_preferences`, `allow_budget_overflow=false`, `hard_max_price=true`
2. `check_memory_relevance` → `should_search_products=true`, purge over-budget active recs from SQLite
3. `conditional_router` → `product_search`
4. `hybrid_product_search` → strict recommend inside current `category`
5. `generate_sales_response` → only in-budget products, or `داخل این بودجه گزینه مناسبی پیدا نکردم.`

## Budget enforcement layers

1. **Pandas** — `ProductFilter.strict_budget` skips +25% relaxation
2. **Recommendation** — no `_pad_with_alternatives` when strict; no backfill to top-3
3. **Graph nodes** — `_apply_hard_budget_guard` after search/rank
4. **Response** — strip forbidden product lines; `validate_polish(forbidden_names=...)`

## Tests

- `tests/test_hard_budget.py` — numeric ceiling phrases («بیشتر از X ندارم»)
- `tests/test_preference_update.py` — `change_preferences` / reject overflow
- Run: `pytest tests/ -q`

## Known pitfalls

- Do not match bare «قیمت» as memory reference when the user is rejecting overflow.
- First-time budget («تا ۵۰ میلیون») keeps `allow_budget_overflow=true` so padding can reach 3 items.
- Preference rejection re-searches the catalog; it does not merely filter the previous text in memory.
