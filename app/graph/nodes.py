"""LangGraph nodes.

Each node is a pure-ish function: SalesState -> partial state update.
Nodes call the LangChain tools (app.tools.*) so tool-calling is real, and
every fragile operation is wrapped so failures land in state['error'] and
route to the fallback node instead of crashing the graph.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.graph.state import ALL_SLOTS, HARD_SLOTS, SalesState
from app.services import llm_service
from app.services.image_similarity_service import ASK_ABOUT_IMAGE_TEXT, build_image_query
from app.services.memory_service import get_memory_service
from app.services.recommendation_service import filter_products_by_max_price
from app.tools.comparison_tools import compare_products_tool
from app.tools.product_search_tools import (
    get_product_from_memory_tool,
    recommend_products_tool,
    similar_products_by_url_tool,
)
from app.telegram.formatters import PRODUCT_IMAGE_OFFER_TEXT, PURCHASE_COMING_SOON_TEXT
from app.utils.logger import get_logger
from app.utils.text_normalizer import (
    brand_in_text,
    detect_category,
    extract_budget,
    extract_ordinal,
    extract_ordinals,
    extract_urls,
    format_price,
    is_attribute_question,
    is_hard_max_budget,
    is_memory_reference,
    is_product_image_request,
    is_purchase_request,
    match_product_name_query,
    normalize,
    to_persian_digits,
)

logger = get_logger(__name__)

HARD_BUDGET_EMPTY_MESSAGE = "داخل این بودجه گزینه مناسبی پیدا نکردم."

FALLBACK_TEXT = (
    "الان برای بررسی دقیق محصول مشکلی پیش اومده، اما می‌تونم با اطلاعاتی که "
    "ازت دارم راهنمایی‌ات کنم. اگه بودجه یا نوع محصول مدنظرت رو دوباره بگی، "
    "سریع چند گزینه معرفی می‌کنم."
)

PERSONA = (
    "تو «سینا»، فروشنده‌ی حرفه‌ای فروشگاه اینترنتی SINWAY هستی. "
    "فارسی، صمیمی ولی حرفه‌ای صحبت می‌کنی. خلاصه و کاربردی جواب می‌دی، "
    "پرحرفی نمی‌کنی، و همیشه آخر پیام یک اقدام بعدی پیشنهاد می‌دی. "
    "فقط از اطلاعات محصولاتی که بهت داده می‌شه استفاده کن و هیچ محصولی از "
    "خودت نساز. اگر محصولی موجود نیست، صادقانه بگو."
)

# --------------------------------------------------------------------------
# Keyword tables for rule-based intent detection (LLM refines when available)
# --------------------------------------------------------------------------

_COMPARE_WORDS = (
    "مقایسه", "مقایس", "مقایسش", "کدوم بهتره", "کدام بهتر",
    "فرقشون", "فرق این", "باهم مقایس", "مقایسشون",
)
_GREETING_WORDS = ("سلام", "درود", "خسته نباشی", "هی ", "hello", "hi")
_PRODUCT_WORDS = (
    "میخوام", "می خوام", "لازم دارم", "دنبال", "معرفی کن", "پیشنهاد",
    "قیمت", "ارزون", "ارزان",
)
_REQUIREMENT_UPDATE_KEYS = (
    "budget", "category", "brands", "use_case", "priorities", "constraints",
)


def _persist(chat_id: str, **fields: Any) -> None:
    get_memory_service().update_session(chat_id, **fields)


def _strict_budget_active(state: SalesState, requirements: Optional[dict[str, Any]] = None) -> bool:
    """True when max_price must be enforced as a hard ceiling."""
    requirements = requirements if requirements is not None else state.get("requirements", {})
    return bool(requirements.get("hard_max_price"))


def _strip_lines_with_product_names(text: str, names: list[str]) -> str:
    """Remove lines that mention any of the given product names."""
    blocked = [n for n in names if n]
    if not blocked:
        return text.strip()
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if any(name in line for name in blocked):
            if i + 1 < len(lines) and lines[i + 1].startswith("   "):
                i += 2
            else:
                i += 1
            continue
        out.append(line)
        i += 1
    alt_phrases = ("نزدیک‌ترین جایگزین", "کمی بالاتر از بودجه")
    out = [ln for ln in out if not any(p in ln for p in alt_phrases)]
    return "\n".join(out).strip()


def _over_budget_product_names(
    products: list[dict[str, Any]], max_price: float
) -> list[str]:
    return [
        str(p.get("name", ""))
        for p in products
        if p.get("name") and float(p.get("effective_price") or p.get("price") or 0) > max_price
    ]


def _apply_hard_budget_guard(
    products: list[dict[str, Any]], requirements: dict[str, Any], strict: bool
) -> list[dict[str, Any]]:
    if not strict:
        return products
    budget = requirements.get("budget")
    if not budget:
        return products
    return filter_products_by_max_price(products, budget)


def _hard_budget_empty_message(requirements: dict[str, Any]) -> str:
    return (
        f"{HARD_BUDGET_EMPTY_MESSAGE} "
        f"(سقف بودجه: {format_price(requirements['budget'])}). "
        "اگه محدوده بودجه یا دسته رو کمی عوض کنی، دوباره جستجو می‌کنم."
    )


# ==========================================================================
# 1. load_or_create_session
# ==========================================================================

def load_or_create_session(state: SalesState) -> dict[str, Any]:
    """Hydrate state from SQLite so every message continues the conversation
    instead of starting from scratch."""
    try:
        memory = get_memory_service()
        session = memory.get_or_create_session(state["chat_id"], state["user_id"])
        recs = memory.get_active_recommendations(state["chat_id"])
        logger.info(
            "Session loaded chat=%s stage=%s reqs=%s recs=%d",
            state["chat_id"], session["conversation_stage"],
            session["requirements"], len(recs),
        )
        return {
            "requirements": session["requirements"],
            "user_profile": session["user_profile"],
            "conversation_stage": session["conversation_stage"],
            "memory_summary": session["memory_summary"],
            "last_recommended_products": recs,
            "error": None,
        }
    except Exception as exc:
        logger.exception("load_or_create_session failed")
        return {"error": f"session:{exc}"}


# ==========================================================================
# 2. classify_message
# ==========================================================================

def classify_message(state: SalesState) -> dict[str, Any]:
    """Detect message type (text/image/link) — cheap, no LLM needed."""
    if state.get("error"):
        return {}
    if state.get("message_type") == "image":
        return {"message_type": "image"}
    urls = extract_urls(state.get("current_message", ""))
    if urls:
        logger.info("Message classified as link: %s", urls[0])
        return {"message_type": "link"}
    return {"message_type": "text"}


# ==========================================================================
# 3. extract_user_intent_and_requirements
# ==========================================================================

def _rule_based_intent(state: SalesState) -> str:
    text = normalize(state.get("current_message", "")).lower()
    recs = state.get("last_recommended_products") or []

    if is_purchase_request(text):
        return "purchase_requested"
    if recs and is_product_image_request(text):
        return "request_product_images"
    if recs and any(w in text for w in _COMPARE_WORDS):
        return "compare_products"
    if recs and extract_budget(state.get("current_message", "")) is not None:
        return "product_request"
    if recs and is_memory_reference(state.get("current_message", ""), recs):
        return "memory_question"
    if extract_budget(text) is not None or any(w in text for w in _PRODUCT_WORDS):
        return "product_request"
    if state.get("conversation_stage") == "clarifying":
        return "product_request"
    if any(text.startswith(w) or f" {w}" in f" {text}" for w in _GREETING_WORDS) and len(text) < 30:
        return "greeting"
    return "chitchat"


def _extract_slots_rule_based(state: SalesState) -> dict[str, Any]:
    """Deterministic slot extraction that also works without any LLM."""
    from app.services.pandas_query_service import get_pandas_service

    text = normalize(state.get("current_message", ""))
    text_lower = text.lower()
    updates: dict[str, Any] = {}

    budget = extract_budget(text)
    if budget:
        updates["budget"] = budget
        if is_hard_max_budget(text):
            updates["hard_max_price"] = True

    service = get_pandas_service()
    category = detect_category(text, service.list_categories())
    if category:
        updates["category"] = category

    matched_brands = [
        b for b in service.list_brands()
        if brand_in_text(b, text_lower)
    ]
    if matched_brands:
        updates["brands"] = matched_brands
    elif any(w in text_lower for w in ("فرقی نداره", "فرق نداره", "مهم نیست", "هرچی", "نه فرقی")):
        updates["brands"] = []
        updates["brands_no_preference"] = True

    use_cases = ("برنامه نویسی", "گیمینگ", "بازی", "دانشجویی", "کار اداری",
                 "عکاسی", "ورزش", "سفر", "آشپزی")
    for uc in use_cases:
        if normalize(uc).lower() in text_lower:
            updates["use_case"] = uc.replace("برنامه نویسی", "برنامه‌نویسی")
            break
    return updates


_EXTRACT_SYSTEM = (
    "تو یک استخراج‌کننده نیازمندی خرید هستی. از پیام کاربر این فیلدها را "
    "استخراج کن (اگر در پیام نبود null بگذار):\n"
    '{"intent": "product_request|memory_question|compare_products|greeting|chitchat|purchase_requested|request_product_images",'
    ' "budget": عدد تومان یا null, "category": "دسته محصول یا null",'
    ' "brands": [برندها] یا [], "use_case": "کاربرد یا null",'
    ' "priorities": [اولویت‌ها] یا [], "constraints": [محدودیت‌ها] یا []}\n'
    "نکته: budget را به تومان کامل تبدیل کن (مثلاً ۵۰ میلیون = 50000000). "
    "اگر کاربر درباره محصولاتی که قبلاً معرفی شده سؤال می‌پرسد intent را "
    "memory_question بگذار."
)


def extract_user_intent_and_requirements(state: SalesState) -> dict[str, Any]:
    """Merge new information from this message into requirements.

    Rule-based extraction always runs (deterministic backbone); the LLM
    refines intent and fills subtler fields when available. Only *changed*
    slots are updated — the rest of the requirement dict survives, which is
    what makes scenario 4 («بودجه‌ام شد ۴۰ میلیون») a partial update instead
    of a restart.
    """
    if state.get("error"):
        return {}
    try:
        requirements = dict(state.get("requirements", {}))
        old_requirements = dict(requirements)

        slot_updates = _extract_slots_rule_based(state)
        intent = _rule_based_intent(state)

        # LLM refinement (skipped for links/images and when disabled)
        if state.get("message_type") == "text":
            context = (
                f"نیازمندی‌های فعلی: {requirements}\n"
                f"محصولات معرفی‌شده قبلی: "
                f"{[p.get('name') for p in state.get('last_recommended_products', [])]}\n"
                f"پیام کاربر: {state['current_message']}"
            )
            llm_out = llm_service.chat_json(_EXTRACT_SYSTEM, context)
            if llm_out:
                llm_intent = llm_out.get("intent")
                stage = state.get("conversation_stage", "")
                if llm_intent in (
                    "product_request", "memory_question", "compare_products",
                    "greeting", "chitchat", "purchase_requested", "request_product_images",
                    "purchase_done",
                ):
                    if intent in ("memory_question", "compare_products", "request_product_images"):
                        pass
                    elif llm_intent == "purchase_done":
                        intent = "purchase_requested"
                    elif stage == "clarifying" and intent == "product_request":
                        pass
                    elif intent in ("chitchat", "greeting", "product_request"):
                        intent = llm_intent
                if llm_out.get("budget") and "budget" not in slot_updates:
                    if extract_budget(state.get("current_message", "")) is not None:
                        slot_updates["budget"] = llm_out["budget"]
                for key in ("category", "use_case"):
                    if llm_out.get(key) and key not in slot_updates:
                        slot_updates[key] = llm_out[key]
                if llm_out.get("brands") and "brands" not in slot_updates:
                    slot_updates["brands"] = llm_out["brands"]
                for key in ("priorities", "constraints"):
                    if llm_out.get(key):
                        slot_updates[key] = llm_out[key]

        no_pref = slot_updates.pop("brands_no_preference", False)
        explicit_requirement_update = any(k in slot_updates for k in _REQUIREMENT_UPDATE_KEYS)
        if explicit_requirement_update:
            intent = "product_request"

        requirements.update({k: v for k, v in slot_updates.items() if v not in (None, "")})
        if "budget" in slot_updates:
            message = state.get("current_message", "")
            if is_hard_max_budget(message):
                requirements["hard_max_price"] = True
            elif (
                old_requirements.get("budget") is not None
                and slot_updates["budget"] != old_requirements.get("budget")
            ):
                requirements["hard_max_price"] = True
        if no_pref:
            requirements["brands"] = []
            requirements["brands_asked"] = True

        requirements_changed = any(
            old_requirements.get(k) != requirements.get(k) for k in ("budget", "category", "brands", "use_case")
        )

        missing = [
            s for s in ALL_SLOTS
            if s not in requirements and not requirements.get(f"{s}_asked")
        ]

        logger.info(
            "Intent=%s changed=%s explicit=%s updates=%s missing=%s",
            intent, requirements_changed, explicit_requirement_update, slot_updates, missing,
        )
        return {
            "intent": intent,
            "requirements": requirements,
            "missing_slots": missing,
            "requirements_changed": requirements_changed,
            "explicit_requirement_update": explicit_requirement_update,
        }
    except Exception as exc:
        logger.exception("extract_user_intent_and_requirements failed")
        return {"error": f"extract:{exc}"}


# ==========================================================================
# 4. check_memory_relevance
# ==========================================================================

def check_memory_relevance(state: SalesState) -> dict[str, Any]:
    """Decide whether this message can be answered purely from memory."""
    if state.get("error"):
        return {}

    recs = state.get("last_recommended_products") or []
    message = state.get("current_message", "")
    intent = state.get("intent", "")
    requirements = state.get("requirements", {})
    stage = state.get("conversation_stage", "")
    explicit_update = state.get("explicit_requirement_update", False)
    req_changed = state.get("requirements_changed", False)

    if explicit_update or req_changed:
        updates: dict[str, Any] = {
            "should_use_memory": False,
            "should_search_products": bool(recs) or intent == "product_request",
            "intent": "product_request",
        }
        budget = requirements.get("budget")
        if explicit_update and budget and requirements.get("hard_max_price"):
            memory = get_memory_service()
            memory.deactivate_over_budget_recommendations(state["chat_id"], budget)
            recs = filter_products_by_max_price(recs, budget)
            updates["last_recommended_products"] = recs
        return updates

    memory_ref = (
        intent not in ("purchase_requested", "compare_products", "request_product_images")
        and bool(recs)
        and (intent == "memory_question" or is_memory_reference(message, recs))
    )
    if memory_ref:
        return {
            "should_use_memory": True,
            "should_search_products": False,
            "intent": "memory_question",
        }

    missing_hard = [s for s in HARD_SLOTS if s not in requirements]
    missing_soft = state.get("missing_slots") or []
    should_search = False

    if intent == "product_request":
        if stage == "clarifying" and not missing_hard and not missing_soft:
            should_search = True
        elif state.get("requirements_changed"):
            should_search = True
        elif not recs and not missing_hard:
            should_search = True
        elif recs and state.get("requirements_changed"):
            should_search = True

    return {
        "should_use_memory": False,
        "should_search_products": should_search,
    }


# ==========================================================================
# 5a. ask_clarifying_questions
# ==========================================================================

_SLOT_QUESTIONS = {
    "budget": "حدود بودجه‌ات چقدره؟",
    "use_case": "بیشتر برای چه کاری می‌خوایش؟",
    "brands": "برند خاصی مدنظرت هست یا فرقی نداره؟",
    "category": "دنبال چه نوع محصولی هستی؟ (مثلاً لپ‌تاپ، گوشی، هدفون...)",
}


def ask_clarifying_questions(state: SalesState) -> dict[str, Any]:
    """Ask only for the missing slots — never re-ask something we know."""
    try:
        missing = state.get("missing_slots", [])[:3]
        requirements = dict(state.get("requirements", {}))
        category = requirements.get("category")

        intro = "حتماً. برای اینکه گزینه اشتباه پیشنهاد ندم، فقط این"
        if category:
            intro = (
                f"خوبه، پس دنبال {category} هستی. برای اینکه گزینه اشتباه "
                "پیشنهاد ندم، فقط این"
            )
        count_word = {1: "۱ مورد رو", 2: "۲ مورد رو", 3: "۳ مورد رو"}.get(len(missing), "چند مورد رو")
        lines = [f"{intro} {count_word} بگو:"]
        for i, slot in enumerate(missing, 1):
            lines.append(f"{i}. {_SLOT_QUESTIONS.get(slot, slot)}")

        # mark soft slots as asked so we never repeat the question
        for slot in missing:
            requirements[f"{slot}_asked"] = True

        _persist(
            state["chat_id"],
            conversation_stage="clarifying",
            requirements=requirements,
        )
        return {
            "final_response": "\n".join(lines),
            "conversation_stage": "clarifying",
            "requirements": requirements,
        }
    except Exception as exc:
        logger.exception("ask_clarifying_questions failed")
        return {"error": f"clarify:{exc}"}


# ==========================================================================
# 5b. generate_memory_based_answer
# ==========================================================================

def generate_memory_based_answer(state: SalesState) -> dict[str, Any]:
    """Answer questions about previously recommended products from memory
    (via get_product_from_memory_tool) — no CSV/RAG/Pandas."""
    if state.get("intent") == "request_product_images":
        return _respond_product_images(state)
    try:
        chat_id = state["chat_id"]
        message = state.get("current_message", "")
        position = extract_ordinal(message)
        focus: Optional[dict[str, Any]] = None
        products: list[dict[str, Any]] = []

        if position is not None:
            result = get_product_from_memory_tool.invoke(
                {"chat_id": chat_id, "position": position}
            )
            if result.get("found") and result.get("product"):
                focus = result["product"]
                products = [focus]
        else:
            name_match = match_product_name_query(
                message, state.get("last_recommended_products") or []
            )
            if name_match is not None:
                result = get_product_from_memory_tool.invoke(
                    {
                        "chat_id": chat_id,
                        "product_name_query": message,
                    }
                )
                if result.get("found") and result.get("product"):
                    focus = result["product"]
                    products = [focus]
            else:
                session = get_memory_service().get_or_create_session(
                    chat_id, state["user_id"]
                )
                focus_id = session.get("focus_product_id")
                recs = state.get("last_recommended_products") or []
                if focus_id and is_attribute_question(message):
                    focus = next(
                        (p for p in recs if str(p.get("product_id")) == str(focus_id)),
                        None,
                    )
                    if focus:
                        products = [focus]
                if not products:
                    result = get_product_from_memory_tool.invoke({"chat_id": chat_id})
                    products = result.get("products", [])

        if not products:
            return {
                "final_response": (
                    "هنوز محصولی معرفی نکردم که بخوام درباره‌اش توضیح بدم. "
                    "بگو دنبال چی هستی تا چند گزینه خوب پیشنهاد بدم."
                ),
                "conversation_stage": state.get("conversation_stage", "recommended"),
            }

        focus = focus or products[0]
        answer = _answer_about_product(state, focus)

        _persist(
            chat_id,
            conversation_stage="recommended",
            focus_product_id=str(focus.get("product_id", "")),
        )
        return {
            "final_response": answer,
            "selected_product_ids": [str(p.get("product_id")) for p in products],
            "conversation_stage": "recommended",
        }
    except Exception as exc:
        logger.exception("generate_memory_based_answer failed")
        return {"error": f"memory_answer:{exc}"}


def _respond_product_images(state: SalesState) -> dict[str, Any]:
    """Send photos of active recommendations from memory — no new search."""
    chat_id = state["chat_id"]
    memory = get_memory_service()
    all_recs = memory.get_active_recommendations(chat_id)
    if not all_recs:
        return {
            "final_response": (
                "هنوز محصولی معرفی نکردم که عکسش رو بفرستم. "
                "اول بگو دنبال چی هستی تا چند گزینه پیشنهاد بدم."
            ),
        }

    positions = extract_ordinals(state.get("current_message", ""))
    if positions:
        products = [
            p
            for pos in positions
            if (p := memory.get_recommendation_by_position(chat_id, pos)) is not None
        ]
    else:
        products = list(all_recs)

    if not products:
        return {
            "final_response": (
                "گزینه‌ای که گفتی بین پیشنهادهای فعلی پیدا نشد. "
                "مثلاً بگو «عکس گزینه اول» یا «عکس گزینه‌ها رو بفرست»."
            ),
        }

    names = "، ".join(str(p.get("name", "")) for p in products)
    return {
        "final_response": f"باشه، عکس {names} رو برات می‌فرستم.",
        "send_product_images": products,
        "conversation_stage": state.get("conversation_stage", "recommended"),
    }


def _product_card(product: dict[str, Any]) -> str:
    stock = int(product.get("stock", 0) or 0)
    return (
        f"«{product.get('name', '')}»\n"
        f"قیمت: {format_price(product.get('effective_price') or product.get('price', 0))}"
        + (f" (با {to_persian_digits(int(product['discount']))}٪ تخفیف)" if product.get("discount") else "")
        + f"\nامتیاز: {to_persian_digits(product.get('rating', '-'))} از ۵"
        + f"\nموجودی: {'موجود' if stock > 0 else 'ناموجود'}"
        + (f"\nگارانتی: {product['warranty']}" if product.get("warranty") else "")
        + (f"\nویژگی‌ها: {product['features']}" if product.get("features") else "")
    )


def _answer_about_product(state: SalesState, focus: dict[str, Any]) -> str:
    """Deterministic memory answer — attribute questions never re-recommend."""
    question = normalize(state.get("current_message", "")).lower()
    attr_answer = _answer_attribute_question(focus, question)
    if attr_answer is not None:
        pos = to_persian_digits(focus.get("_position", 1))
        name = focus.get("name", "")
        return f"گزینه {pos} ({name}):\n{attr_answer}"

    pos = to_persian_digits(focus.get("_position", 1))
    return (
        f"گزینه {pos} که معرفی کردم این مشخصات رو داره:\n\n{_product_card(focus)}\n\n"
        "اگه بخوای می‌تونم با گزینه‌های دیگه مقایسه‌اش کنم."
    )


def _answer_attribute_question(product: dict[str, Any], question: str) -> Optional[str]:
    """Return a single-attribute answer or None for a general product card."""
    features = normalize(str(product.get("features", ""))).lower()
    desc = normalize(str(product.get("description", ""))).lower()
    combined = f"{features} {desc}"

    if any(k in question for k in ("رم", "رمش", "حافظه")):
        for token in combined.split():
            if "gb" in token or "گیگ" in token or "رم" in token:
                return f"رم/حافظه: {product.get('features', token)}"
        if "رم" in combined or "حافظه" in combined:
            return f"رم/حافظه: {product.get('features', '')}"
        return "این مشخصه در اطلاعات محصول ثبت نشده."

    if any(k in question for k in ("قیمت", "قیمتش")):
        return f"قیمت: {format_price(product.get('effective_price') or product.get('price', 0))}"

    if any(k in question for k in ("گارانتی", "گارانتیش")):
        if product.get("warranty"):
            return f"گارانتی: {product['warranty']}"
        return "این مشخصه در اطلاعات محصول ثبت نشده."

    if any(k in question for k in ("موجود", "موجوده", "موجودی")):
        stock = int(product.get("stock", 0) or 0)
        return "موجود" if stock > 0 else "ناموجود"

    if any(k in question for k in ("امتیاز", "امتیازش")):
        return f"امتیاز: {to_persian_digits(product.get('rating', '-'))} از ۵"

    if any(k in question for k in ("باتری", "پردازنده", "مشخصات", "رنگ")):
        if product.get("features"):
            return f"ویژگی‌ها: {product['features']}"
        return "این مشخصه در اطلاعات محصول ثبت نشده."

    if is_attribute_question(question):
        return "این مشخصه در اطلاعات محصول ثبت نشده."

    return None


# ==========================================================================
# 5c. hybrid_product_search  ->  rank_recommendations
# ==========================================================================

def hybrid_product_search(state: SalesState) -> dict[str, Any]:
    """Run the hybrid recommender (Pandas filter + RAG rank) via tool call."""
    try:
        requirements = state.get("requirements", {})
        strict_budget = _strict_budget_active(state, requirements)
        query_parts = [
            str(requirements.get("category") or ""),
            str(requirements.get("use_case") or ""),
            " ".join(requirements.get("brands") or []),
            " ".join(requirements.get("priorities") or []),
            state.get("current_message", ""),
        ]
        query_text = " ".join(p for p in query_parts if p).strip()

        result = recommend_products_tool.invoke(
            {
                "query_text": query_text,
                "budget": requirements.get("budget"),
                "category": requirements.get("category"),
                "brands": requirements.get("brands") or [],
                "use_case": requirements.get("use_case"),
                "top_k": 3,
                "strict_budget": strict_budget,
                "hard_max_price": bool(requirements.get("hard_max_price")),
            }
        )
        products = _apply_hard_budget_guard(
            result["products"], requirements, strict_budget
        )
        return {
            "last_query_result": products,
            "comparison_context": {
                "exact_match": result["exact_match"],
                "relaxed": result["relaxed"],
                "empty_reason": result.get("empty_reason", ""),
                "strict_budget": strict_budget,
            },
        }
    except Exception as exc:
        logger.exception("hybrid_product_search failed")
        return {"error": f"search:{exc}"}


def rank_recommendations(state: SalesState) -> dict[str, Any]:
    """Final ranked top-3 with human-readable reasons per product."""
    if state.get("error"):
        return {}
    try:
        requirements = state.get("requirements", {})
        strict = _strict_budget_active(state, requirements)
        products = _apply_hard_budget_guard(
            state.get("last_query_result", []), requirements, strict
        )
        if strict:
            products = products[:3]
        else:
            products = products[:3]
        for product in products:
            product["_reason"] = _selection_reason(product, requirements, strict=strict)
        return {"last_recommended_products": products}
    except Exception as exc:
        logger.exception("rank_recommendations failed")
        return {"error": f"rank:{exc}"}


def _selection_reason(
    product: dict[str, Any],
    requirements: dict[str, Any],
    *,
    strict: bool = False,
) -> str:
    reasons: list[str] = []
    components = product.get("score_components", {})
    budget = requirements.get("budget")
    if product.get("is_alternative") and not strict:
        note = "گزینه دقیق بیشتری موجود نبود؛ این نزدیک‌ترین جایگزینه"
        if product.get("over_budget"):
            note += " (کمی بالاتر از بودجه‌ات)"
        reasons.append(note)
    if budget and components.get("budget", 0) >= 0.9:
        reasons.append("توی بودجه‌ات جا می‌شه")
    if components.get("use_case", 0) >= 0.6 and requirements.get("use_case"):
        reasons.append(f"برای {requirements['use_case']} مناسبه")
    if requirements.get("brands") and components.get("brand", 0) >= 0.9:
        reasons.append("از برند مورد علاقه‌ته")
    if float(product.get("rating", 0) or 0) >= 4.3:
        reasons.append(f"امتیاز {to_persian_digits(product['rating'])} از ۵ بین خریدارها")
    if float(product.get("discount", 0) or 0) >= 15:
        reasons.append(f"{to_persian_digits(int(product['discount']))}٪ تخفیف داره")
    if not reasons:
        reasons.append("بین محصولات فروشگاه بهترین تطابق رو با درخواستت داره")
    return "، ".join(reasons[:3])


# ==========================================================================
# 5d. generate_sales_response
# ==========================================================================

def generate_sales_response(state: SalesState) -> dict[str, Any]:
    """Build the salesperson-style reply for recommendations."""
    if state.get("final_response"):
        # An upstream node (e.g. captionless image) already produced the reply
        return {}
    try:
        requirements = state.get("requirements", {})
        strict = _strict_budget_active(state, requirements)
        products = _apply_hard_budget_guard(
            state.get("last_recommended_products", []), requirements, strict
        )
        chat_id = state["chat_id"]

        if not products:
            if strict and requirements.get("budget"):
                msg = _hard_budget_empty_message(requirements)
            else:
                msg = (
                    "با این مشخصات محصولی توی فروشگاه پیدا نکردم. "
                    "اگه بودجه یا برند رو کمی تغییر بدی احتمالاً گزینه‌های خوبی داریم. "
                    "می‌خوای محدوده رو عوض کنیم؟"
                )
            memory = get_memory_service()
            memory.save_recommendations(chat_id, [])
            _persist(chat_id, conversation_stage="recommended")
            return {
                "final_response": msg,
                "last_recommended_products": [],
                "conversation_stage": "recommended",
            }

        response = _build_recommendation_text(state, products, requirements, strict=strict)

        memory = get_memory_service()
        memory.save_recommendations(
            chat_id, products, [p.get("_reason", "") for p in products]
        )
        _persist(chat_id, conversation_stage="recommended")
        return {
            "final_response": response,
            "last_recommended_products": products,
            "conversation_stage": "recommended",
        }
    except Exception as exc:
        logger.exception("generate_sales_response failed")
        return {"error": f"respond:{exc}"}


def _build_recommendation_text(
    state: SalesState,
    products: list[dict[str, Any]],
    requirements: dict[str, Any],
    *,
    strict: bool = False,
) -> str:
    ctx = state.get("comparison_context", {})
    strict = strict or bool(ctx.get("strict_budget")) or bool(requirements.get("hard_max_price"))
    budget = requirements.get("budget")
    forbidden_names: list[str] = []
    if strict and budget:
        forbidden_names = _over_budget_product_names(products, budget)
        products = filter_products_by_max_price(products, budget)

    # Similarity flows (image / link) get their own header — the user's
    # requirements context would be misleading there.
    if ctx.get("from_link") or ctx.get("similarity_query"):
        external = ctx.get("external_title")
        header = (
            f"نزدیک‌ترین محصولات فروشگاه ما به «{external}» این‌ها هستن:"
            if external
            else "نزدیک‌ترین محصولات فروشگاه ما به چیزی که فرستادی این‌ها هستن:"
        )
        lines = [header, ""]
        for i, product in enumerate(products, 1):
            price = format_price(product.get("effective_price") or product.get("price", 0))
            lines.append(f"{i}. {product.get('name', '')} — {price}")
            lines.append(f"   {product.get('_reason', '')}")
            lines.append("")
        lines.append("اگه یکی‌شون نظرت رو گرفت بگو تا مشخصات کاملش رو بفرستم.")
        return "\n".join(lines)

    context_bits = []
    if requirements.get("use_case"):
        context_bits.append(f"برای {requirements['use_case']} می‌خوایش")
    if requirements.get("budget"):
        context_bits.append(f"بودجه‌ات حدود {format_price(requirements['budget'])}ه")
    context = " و ".join(context_bits)

    relaxed = state.get("comparison_context", {}).get("relaxed") or []
    notes = ""
    if strict and budget:
        notes = f"فقط گزینه‌های زیر {format_price(budget)} رو آوردم.\n"
    elif "brand" in relaxed:
        notes = "از برند موردنظرت گزینه مناسبی موجود نبود؛ نزدیک‌ترین‌ها از برندهای دیگه رو آوردم.\n"
    elif any(r.startswith("budget") for r in relaxed):
        notes = "دقیقاً توی این بودجه چیزی نبود؛ نزدیک‌ترین گزینه‌ها رو آوردم.\n"
    elif "stock" in relaxed:
        notes = "گزینه دقیق موجود نبود؛ این‌ها نزدیک‌ترین محصولات مشابه فروشگاه‌ان.\n"

    count = to_persian_digits(len(products))
    header = (
        f"با توجه به اینکه {context}، این {count} گزینه منطقی‌ترن:"
        if context
        else f"این {count} گزینه بهترین تطابق رو با درخواستت دارن:"
    )

    lines = [notes + header, ""]
    for i, product in enumerate(products, 1):
        price = format_price(product.get("effective_price") or product.get("price", 0))
        lines.append(f"{i}. {product.get('name', '')} — {price}")
        lines.append(f"   {product.get('_reason', '')}")
        lines.append("")
    lines.append(
        "اگه بخوای، می‌تونم همین‌ها رو از نظر قیمت و مشخصات باهم مقایسه کنم "
        "یا درباره هرکدوم بیشتر توضیح بدم."
    )
    lines.append(PRODUCT_IMAGE_OFFER_TEXT)
    draft = "\n".join(lines)
    if forbidden_names:
        draft = _strip_lines_with_product_names(draft, forbidden_names)

    product_names = [str(p.get("name", "")) for p in products if p.get("name")]
    polished = llm_service.chat(
        PERSONA
        + "\nمتن زیر پیشنهادهای نهایی فروشگاه است. فقط لحن جمله‌های آغاز و پایان را "
        "طبیعی‌تر کن. خط‌های شماره‌دار محصولات و قیمت‌ها را دقیقاً همان‌طور نگه دار. "
        "هیچ محصول یا عددی اضافه یا کم نکن.",
        draft,
    )
    polished = llm_service.validate_polish(
        draft,
        polished,
        product_names,
        max_product_count=len(products),
        forbidden_names=forbidden_names,
    )
    if forbidden_names:
        polished = _strip_lines_with_product_names(polished, forbidden_names)
    return polished


# ==========================================================================
# 5e. compare_selected_products -> generate_comparison_response
# ==========================================================================

def compare_selected_products(state: SalesState) -> dict[str, Any]:
    """Resolve which options the user wants compared and pull them from memory."""
    try:
        positions = extract_ordinals(state.get("current_message", ""))
        if len(positions) >= 2:
            selected = positions
        elif not positions:
            selected = [
                p.get("_position", i + 1)
                for i, p in enumerate(state.get("last_recommended_products", []))
            ]
        else:
            selected = positions
        selected = selected[:4]

        result = compare_products_tool.invoke(
            {"chat_id": state["chat_id"], "positions": selected}
        )
        if not result["ok"]:
            return {
                "final_response": (
                    "برای مقایسه حداقل دو تا از گزینه‌هایی که معرفی کردم رو انتخاب کن، "
                    "مثلاً بگو «گزینه اول و سوم رو مقایسه کن»."
                ),
                "comparison_context": {},
            }
        return {
            "comparison_context": result,
            "selected_product_ids": [
                str(p.get("product_id")) for p in result["products"]
            ],
        }
    except Exception as exc:
        logger.exception("compare_selected_products failed")
        return {"error": f"compare:{exc}"}


def _comparison_verdict(products: list[dict[str, Any]]) -> str:
    """Pick a single best option with a short grounded reason."""
    if len(products) < 2:
        return ""
    cheapest = min(
        products,
        key=lambda p: float(p.get("effective_price") or p.get("price", 0)),
    )
    best_rated = max(products, key=lambda p: float(p.get("rating", 0) or 0))
    if len(products) == 2:
        a, b = products[0], products[1]
        price_a = float(a.get("effective_price") or a.get("price", 0))
        price_b = float(b.get("effective_price") or b.get("price", 0))
        rating_a = float(a.get("rating", 0) or 0)
        rating_b = float(b.get("rating", 0) or 0)
        if price_a <= price_b and rating_a >= rating_b:
            winner = a
            reason = "هم از نظر قیمت به‌صرفه‌تره هم امتیاز خریدارها بالاتره"
        elif rating_a >= rating_b and price_a <= price_b * 1.1:
            winner = a
            reason = "امتیاز بهتری داره و قیمتش هم منطقیه"
        elif price_b < price_a and rating_b >= rating_a:
            winner = b
            reason = "ارزان‌تره و امتیاز خریدارها هم قوی‌تره"
        elif rating_b > rating_a:
            winner = best_rated
            reason = "رضایت خریدارها بالاتره"
        else:
            winner = cheapest
            reason = "از نظر قیمت به‌صرفه‌تره"
        pos = to_persian_digits(winner.get("_position", "?"))
        return f"جمع‌بندی: گزینه {pos} («{winner.get('name', '')}») {reason}."
    verdict = (
        f"جمع‌بندی: «{cheapest.get('name')}» از نظر قیمت به‌صرفه‌تره"
        + (
            f" و «{best_rated.get('name')}» رضایت خریدارها بالاتره."
            if best_rated.get("product_id") != cheapest.get("product_id")
            else " و امتیاز خریدارهاش هم بهتره؛ انتخاب منطقی همینه."
        )
    )
    if best_rated.get("product_id") != cheapest.get("product_id"):
        verdict += (
            f" اگر کیفیت و امتیاز برات مهم‌تره، «{best_rated.get('name')}» گزینه قوی‌تریه."
        )
    return verdict


def generate_comparison_response(state: SalesState) -> dict[str, Any]:
    if state.get("error"):
        return {}
    if state.get("final_response"):  # compare node already produced guidance
        return {}
    try:
        ctx = state.get("comparison_context", {})
        products = ctx.get("products", [])

        lines = ["مقایسه گزینه‌هایی که گفتی:", ""]
        for product in products:
            pos = product.get("_position", "?")
            stock = int(product.get("stock", 0) or 0)
            lines.append(f"گزینه {to_persian_digits(pos)}: {product.get('name', '')}")
            lines.append(f"  قیمت: {format_price(product.get('effective_price') or product.get('price', 0))}")
            lines.append(
                f"  امتیاز: {to_persian_digits(product.get('rating', '-'))} از ۵ "
                f"({to_persian_digits(product.get('review_count', 0))} نظر)"
            )
            lines.append(f"  موجودی: {'موجود' if stock > 0 else 'ناموجود'}")
            if product.get("warranty"):
                lines.append(f"  گارانتی: {product['warranty']}")
            if product.get("features"):
                lines.append(f"  ویژگی‌ها: {product['features']}")
            lines.append("")

        lines.append(_comparison_verdict(products))
        lines.append("اگه بگی کدوم برات مهم‌تره (قیمت یا کیفیت)، دقیق‌تر راهنمایی‌ات می‌کنم.")
        lines.append(PRODUCT_IMAGE_OFFER_TEXT)
        draft = "\n".join(lines)

        product_names = [str(p.get("name", "")) for p in products if p.get("name")]
        polished = llm_service.chat(
            PERSONA
            + "\nمتن زیر مقایسه محصولات است. فقط روان‌ترش کن؛ همه اعداد، نام‌ها و "
            "واقعیت‌ها را دقیقاً حفظ کن. جمع‌بندی نهایی و پیشنهاد گزینه بهتر را حذف یا "
            "تضعیف نکن. خلاصه و ساخت‌یافته بماند.",
            draft,
        )

        _persist(state["chat_id"], conversation_stage="comparing")
        return {
            "final_response": llm_service.validate_polish(draft, polished, product_names),
            "conversation_stage": "comparing",
        }
    except Exception as exc:
        logger.exception("generate_comparison_response failed")
        return {"error": f"compare_respond:{exc}"}


# ==========================================================================
# 5f. process_image / process_external_link -> find_similar_products
# ==========================================================================

def process_image(state: SalesState) -> dict[str, Any]:
    """Minimal-viable image path: caption/metadata -> text query.

    If there is nothing to work with, ask the user one short question
    (documented fallback when no vision model is available).
    """
    try:
        info = build_image_query(
            state.get("image_caption"), state.get("image_file_name")
        )
        if info["needs_user_help"]:
            _persist(state["chat_id"], conversation_stage="awaiting_image_info")
            return {
                "final_response": ASK_ABOUT_IMAGE_TEXT,
                "conversation_stage": "awaiting_image_info",
            }
        return {"comparison_context": {"similarity_query": info["query"]}}
    except Exception as exc:
        logger.exception("process_image failed")
        return {"error": f"image:{exc}"}


def process_external_link(state: SalesState) -> dict[str, Any]:
    """Fetch the external product page and derive a similarity query."""
    try:
        urls = extract_urls(state.get("current_message", ""))
        if not urls:
            return {"error": "link:no_url_found"}
        result = similar_products_by_url_tool.invoke({"url": urls[0], "top_k": 3})
        return {
            "last_query_result": result["products"],
            "comparison_context": {
                "external_title": result.get("external_title"),
                "source": result.get("source"),
                "from_link": True,
            },
        }
    except Exception as exc:
        logger.exception("process_external_link failed")
        return {"error": f"link:{exc}"}


def find_similar_products(state: SalesState) -> dict[str, Any]:
    """Shared similarity node for the image path (link path already searched
    inside its tool). Produces last_recommended_products with reasons."""
    if state.get("error") or state.get("final_response"):
        return {}
    try:
        ctx = state.get("comparison_context", {})
        if not ctx.get("from_link"):
            query = ctx.get("similarity_query", "")
            result = recommend_products_tool.invoke(
                {"query_text": query, "top_k": 3,
                 "budget": state.get("requirements", {}).get("budget")}
            )
            products = result["products"]
        else:
            products = state.get("last_query_result", [])

        for product in products:
            product["_reason"] = "شبیه‌ترین گزینه موجود فروشگاه به چیزی که فرستادی"
        return {"last_recommended_products": products[:3]}
    except Exception as exc:
        logger.exception("find_similar_products failed")
        return {"error": f"similar:{exc}"}


# ==========================================================================
# 5g. small-talk / purchase / retry-fallback
# ==========================================================================

def generate_smalltalk_response(state: SalesState) -> dict[str, Any]:
    """Greeting / chit-chat / purchase confirmation — no search needed."""
    try:
        intent = state.get("intent", "chitchat")
        chat_id = state["chat_id"]

        if intent in ("purchase_requested", "purchase_done"):
            _persist(chat_id, conversation_stage="purchase_requested")
            return {
                "final_response": PURCHASE_COMING_SOON_TEXT,
                "conversation_stage": "purchase_requested",
            }

        if intent == "greeting":
            text = (
                "سلام! من سینا هستم، فروشنده SINWAY 🙌\n"
                "بگو دنبال چه محصولی هستی (مثلاً لپ‌تاپ، گوشی، هدفون...) تا "
                "بهترین گزینه‌های فروشگاه رو با دلیل برات پیدا کنم."
            )
        else:
            llm_text = llm_service.chat(
                PERSONA + "\nکوتاه و دوستانه جواب بده و مکالمه را به سمت کمک برای "
                "خرید محصول هدایت کن.",
                state.get("current_message", ""),
            )
            text = llm_text or (
                "در خدمتم! اگه دنبال محصول خاصی هستی بگو تا سریع چند گزینه "
                "مناسب از فروشگاه معرفی کنم."
            )
        return {"final_response": text.strip()}
    except Exception as exc:
        logger.exception("generate_smalltalk_response failed")
        return {"error": f"smalltalk:{exc}"}


def retry_or_fallback_response(state: SalesState) -> dict[str, Any]:
    """Terminal error handler: log, count, and reply naturally.

    Per-operation retries happen inside services (utils.retry); this node is
    the last line of defense so the user always gets a helpful answer.
    """
    error = state.get("error")
    retry_count = int(state.get("retry_count", 0)) + 1
    logger.error("Fallback response (retry_count=%d) due to: %s", retry_count, error)

    response = FALLBACK_TEXT
    requirements = state.get("requirements", {})
    if requirements.get("category"):
        response += f"\nتا جایی که یادمه دنبال {requirements['category']}"
        if requirements.get("budget"):
            response += f" با بودجه حدود {format_price(requirements['budget'])}"
        response += " بودی."

    return {
        "final_response": response,
        "retry_count": retry_count,
        "error": None,
    }


# ==========================================================================
# 6. save_memory (terminal node for every path)
# ==========================================================================

def save_memory(state: SalesState) -> dict[str, Any]:
    """Persist the merged session and (re)schedule follow-ups."""
    try:
        chat_id = state["chat_id"]
        memory = get_memory_service()

        summary = _update_summary(state)
        memory.update_session(
            chat_id,
            requirements=state.get("requirements", {}),
            user_profile=state.get("user_profile", {}),
            conversation_stage=state.get("conversation_stage", "new"),
            memory_summary=summary,
            last_message_at=time.time(),
        )
        if memory.get_purchase_status(chat_id) != "purchased":
            memory.schedule_followups(chat_id)
        return {"memory_summary": summary}
    except Exception:
        # Never let memory persistence break the reply that was already built
        logger.exception("save_memory failed (response still delivered)")
        return {}


def _update_summary(state: SalesState) -> str:
    requirements = state.get("requirements", {})
    parts: list[str] = []
    if requirements.get("category"):
        parts.append(f"دنبال {requirements['category']}")
    if requirements.get("budget"):
        parts.append(f"بودجه {format_price(requirements['budget'])}")
    if requirements.get("use_case"):
        parts.append(f"برای {requirements['use_case']}")
    if requirements.get("brands"):
        parts.append(f"برند {', '.join(requirements['brands'])}")
    recs = state.get("last_recommended_products", [])
    if recs:
        parts.append(f"معرفی‌شده: {', '.join(str(p.get('name', '')) for p in recs[:3])}")
    parts.append(f"مرحله: {state.get('conversation_stage', 'new')}")
    return " | ".join(parts)
