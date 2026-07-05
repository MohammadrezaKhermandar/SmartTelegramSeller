"""LangGraph node implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nlp
from app.graph.prompts import (
    CLARIFYING_QUESTIONS,
    DISCOUNT_2D_MESSAGE,
    FOLLOWUP_1H_MESSAGE,
    GREETING_RESPONSE,
    HELP_RESPONSE,
    answer_company_question,
    answer_general_chat,
    format_compare_result,
    format_memory_answer,
    format_product_recommendation,
)
from app.graph.state import SalesAssistantState
from app.graph.tool_agent import run_search_tool_agent
from app.tools.compare_tools import compare_products
from app.tools.image_tools import build_image_acknowledgment, process_image_input
from app.tools.pandas_tools import get_product_by_id_tool, hybrid_recommend
from app.tools.rag_tools import semantic_search
from app.tools.url_tools import process_url_input
from app.llm.client import polish_response
from app.utils.logging import logger


def load_memory_node(state: SalesAssistantState) -> dict[str, Any]:
    """LangGraph Memory: load prior state from checkpointer (no-op merge)."""
    logger.debug("load_memory_node for user %s", state.get("user_id"))
    return {}


def input_classifier_node(state: SalesAssistantState) -> dict[str, Any]:
    """Classify input type: text, image, or URL."""
    if state.get("image_input"):
        return {"current_intent": "image_input"}
    if state.get("url_input") or (state.get("messages") and nlp.detect_url(_last_human_text(state))):
        url = state.get("url_input") or nlp.detect_url(_last_human_text(state))
        return {"current_intent": "url_input", "url_input": url}
    return {}


def intent_detector_node(state: SalesAssistantState) -> dict[str, Any]:
    """Detect user intent from message."""
    text = _last_human_text(state)
    has_recs = bool(state.get("recommended_products"))
    has_image = bool(state.get("image_input"))

    intent = nlp.detect_intent(text, has_image=has_image, has_recommendations=has_recs)

    # Continue requirement gathering when user answers clarifying questions
    if intent == "unknown" and state.get("conversation_stage") == "gathering_requirements":
        req_hint = nlp.extract_requirements(text)
        if req_hint.get("usage") or req_hint.get("max_price") or req_hint.get("min_price"):
            intent = "new_product_request"

    logger.info("Detected intent: %s for text: %s", intent, text[:50])
    return {"current_intent": intent}


def requirement_extractor_node(state: SalesAssistantState) -> dict[str, Any]:
    """Extract product requirements from user message."""
    text = _last_human_text(state)
    req = nlp.extract_requirements(text)
    missing = nlp.get_missing_slots(req)

    # Merge with existing requirements
    existing = dict(state.get("requirements") or {})
    for k, v in req.items():
        if v is not None and v != [] and v != "":
            existing[k] = v

    return {
        "requirements": existing,
        "missing_slots": missing,
        "product_category": existing.get("category"),
        "conversation_stage": "gathering_requirements",
        "last_search_query": text,
    }


def update_requirements_node(state: SalesAssistantState) -> dict[str, Any]:
    """Update requirements when user changes preferences."""
    text = _last_human_text(state)
    new_req = nlp.extract_requirements(text)
    existing = dict(state.get("requirements") or {})

    for k, v in new_req.items():
        if v is not None and v != [] and v != "":
            existing[k] = v

    missing = nlp.get_missing_slots(existing)
    logger.info("Updated requirements: %s", existing)
    return {
        "requirements": existing,
        "missing_slots": missing,
        "product_category": existing.get("category"),
        "recommended_products": [],  # clear for refresh
        "conversation_stage": "gathering_requirements",
    }


def enough_info_checker_node(state: SalesAssistantState) -> dict[str, Any]:
    """Check if we have enough info to search."""
    missing = state.get("missing_slots") or []
    req = state.get("requirements") or {}

    # Enough if category + (budget or usage)
    has_category = bool(req.get("category") or req.get("raw_query"))
    has_budget = req.get("max_price") is not None or req.get("min_price") is not None
    has_usage = bool(req.get("usage"))

    enough = has_category and (has_budget or has_usage)
    if enough:
        return {"missing_slots": [], "conversation_stage": "recommending"}
    return {"missing_slots": missing[:2]}  # at most 2 questions


def ask_clarifying_question_node(state: SalesAssistantState) -> dict[str, Any]:
    """Ask up to 2 clarifying questions in Persian."""
    missing = state.get("missing_slots") or ["budget", "usage"]
    questions = []
    slot_questions = {
        "budget": CLARIFYING_QUESTIONS["budget"],
        "usage": CLARIFYING_QUESTIONS["usage"],
        "category": CLARIFYING_QUESTIONS["category"],
        "brand": CLARIFYING_QUESTIONS["brand"],
    }
    for slot in missing[:2]:
        if slot in slot_questions:
            questions.append(slot_questions[slot])

    if not questions:
        questions = [CLARIFYING_QUESTIONS["budget"], CLARIFYING_QUESTIONS["usage"]]

    response = (
        "حتماً. برای اینکه گزینه درست پیشنهاد بدم، این موارد رو بگو:\n"
        + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    )
    return {
        "response_text": response,
        "conversation_stage": "gathering_requirements",
        "messages": [AIMessage(content=response)],
    }


def hybrid_search_node(state: SalesAssistantState) -> dict[str, Any]:
    """Hybrid RAG + Pandas search (fallback when tool agent unavailable)."""
    req = state.get("requirements") or {}
    query = req.get("raw_query") or state.get("last_search_query") or ""
    if req.get("usage"):
        query = f"{query} {req['usage']}"

    try:
        rag_results = semantic_search(query)
        products, note = hybrid_recommend(
            query=query,
            category=req.get("category"),
            brand=req.get("brand"),
            min_price=req.get("min_price"),
            max_price=req.get("max_price"),
            rag_results=rag_results,
        )
        return {
            "last_search_result": products,
            "search_note": note,
            "retry_count": 0,
        }
    except Exception as exc:
        logger.error("hybrid_search failed: %s", exc)
        return {
            "errors": (state.get("errors") or []) + [str(exc)],
            "retry_count": (state.get("retry_count") or 0) + 1,
        }


def tool_agent_node(state: SalesAssistantState) -> dict[str, Any]:
    """LangGraph Tool Calling: LLM agent with search/filter/compare tools."""
    req = state.get("requirements") or {}
    query = req.get("raw_query") or state.get("last_search_query") or ""
    if req.get("usage"):
        query = f"{query} {req['usage']}"

    try:
        agent_result = run_search_tool_agent(req, query)
        if agent_result:
            products, note = agent_result
            return {
                "last_search_result": products,
                "search_note": note,
                "retry_count": 0,
                "from_tool_agent": True,
            }

        logger.info("Tool agent unavailable or empty; falling back to hybrid_search")
        fallback = hybrid_search_node(state)
        fallback["from_tool_agent"] = False
        return fallback
    except Exception as exc:
        logger.error("tool_agent failed: %s", exc)
        return {
            "errors": (state.get("errors") or []) + [str(exc)],
            "retry_count": (state.get("retry_count") or 0) + 1,
        }


def recommend_products_node(state: SalesAssistantState) -> dict[str, Any]:
    """Format and store product recommendations."""
    products = state.get("last_search_result") or []
    note = state.get("search_note") or ""
    response = format_product_recommendation(products, note)

    product_ids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]

    return {
        "recommended_products": products,
        "selected_product_ids": product_ids,
        "response_text": response,
        "conversation_stage": "recommending",
        "purchase_status": "pending",
        "from_memory": False,
        "messages": [AIMessage(content=response)],
    }


def answer_from_memory_node(state: SalesAssistantState) -> dict[str, Any]:
    """Answer follow-up about previously recommended products – no full search."""
    text = _last_human_text(state)
    recommended = state.get("recommended_products") or []
    product = nlp.resolve_ordinal_reference(text, recommended)
    field = nlp.detect_field_question(text)

    if not product:
        return {
            "response_text": "هنوز محصولی پیشنهاد ندادم. بگو دنبال چه محصولی هستی.",
            "messages": [AIMessage(content="هنوز محصولی پیشنهاد ندادم. بگو دنبال چه محصولی هستی.")],
            "from_memory": True,
        }

    # Try to get field from memory first
    if field and field in product and product.get(field):
        value = product[field]
        if field == "price":
            value = f"{int(float(value)):,} تومان"
        response = format_memory_answer(product, field, value)
    elif field:
        # Fallback to tool only if field missing in state
        pid = str(product.get("product_id", ""))
        details = get_product_by_id_tool.invoke({"product_id": pid})
        if details and field in details:
            value = details[field]
            if field == "price":
                value = f"{int(float(value)):,} تومان"
            response = format_memory_answer(details, field, value)
        else:
            title = product.get("title", "محصول")
            features = product.get("features", product.get("description", ""))
            response = f"در مورد {title}:\n{features}"
    else:
        title = product.get("title", "")
        price = product.get("price", 0)
        features = product.get("features", "")
        response = (
            f"**{title}**\n"
            f"قیمت: {int(float(price)):,} تومان\n"
            f"ویژگی‌ها: {features}\n"
            f"امتیاز: {product.get('rating', '—')}"
        )

    return {
        "response_text": response,
        "conversation_stage": "answering_followup",
        "from_memory": True,
        "messages": [AIMessage(content=response)],
    }


def compare_products_node(state: SalesAssistantState) -> dict[str, Any]:
    """Compare products from recommendations or by ID."""
    text = _last_human_text(state)
    recommended = state.get("recommended_products") or []
    indices = nlp.extract_compare_indices(text, len(recommended))

    if recommended:
        product_ids = [
            str(recommended[i].get("product_id", ""))
            for i in indices
            if i < len(recommended)
        ]
    else:
        product_ids = []

    if len(product_ids) < 2:
        response = "برای مقایسه، حداقل دو محصول از پیشنهادها رو مشخص کن. مثلاً: «اولی و دومی رو مقایسه کن»"
        return {
            "response_text": response,
            "messages": [AIMessage(content=response)],
        }

    result = compare_products(product_ids)
    response = format_compare_result(result)
    return {
        "compare_result": result,
        "response_text": response,
        "conversation_stage": "comparing",
        "messages": [AIMessage(content=response)],
    }


def image_similarity_node(state: SalesAssistantState) -> dict[str, Any]:
    """Process image and find similar products."""
    image_data = state.get("image_input") or {}
    caption = image_data.get("caption")
    file_name = image_data.get("file_name")

    needs_clarification = not caption and not file_name
    query, similar = process_image_input(caption, file_name)

    if similar:
        response = build_image_acknowledgment(query, False)
        response += "\n\n" + format_product_recommendation(similar[:3], "محصولات مشابه عکس شما:")
        return {
            "last_search_result": similar,
            "recommended_products": similar[:5],
            "response_text": response,
            "conversation_stage": "image_search",
            "image_input": None,
            "messages": [AIMessage(content=response)],
        }

    response = build_image_acknowledgment(query, needs_clarification)
    return {
        "response_text": response,
        "conversation_stage": "image_search",
        "image_input": None,
        "messages": [AIMessage(content=response)],
    }


def url_similarity_node(state: SalesAssistantState) -> dict[str, Any]:
    """Process URL and find similar products."""
    url = state.get("url_input") or nlp.detect_url(_last_human_text(state)) or ""
    query, similar, error = process_url_input(url)

    if error:
        return {
            "response_text": error,
            "url_input": None,
            "messages": [AIMessage(content=error)],
        }

    response = f"لینک رو بررسی کردم و محصولات مشابه پیدا کردم:\n\n"
    response += format_product_recommendation(similar[:3], f"بر اساس: {query[:80]}")
    return {
        "last_search_result": similar,
        "recommended_products": similar[:5],
        "last_search_query": query,
        "response_text": response,
        "conversation_stage": "url_search",
        "url_input": None,
        "messages": [AIMessage(content=response)],
    }


def error_handler_node(state: SalesAssistantState) -> dict[str, Any]:
    """Graceful Persian error response with retry awareness."""
    errors = state.get("errors") or []
    retry = state.get("retry_count") or 0
    last_error = errors[-1] if errors else "خطای ناشناخته"

    if retry < 3:
        response = "متأسفانه مشکلی پیش اومد. دوباره تلاش می‌کنم..."
    else:
        response = (
            "متأسفانه الان نتونستم درخواستت رو پردازش کنم. "
            "لطفاً کمی بعد دوباره امتحان کن یا نیازت رو ساده‌تر بنویس."
        )
    logger.error("error_handler: %s (retry=%d)", last_error, retry)
    return {
        "response_text": response,
        "conversation_stage": "error",
        "messages": [AIMessage(content=response)],
    }


def final_response_node(state: SalesAssistantState) -> dict[str, Any]:
    """Ensure response_text is set for greeting/help/purchase/general chat."""
    intent = state.get("current_intent")
    if state.get("response_text"):
        return {}

    text = _last_human_text(state)
    stage = "completed"

    if text == "/help":
        response = HELP_RESPONSE
    elif intent == "greeting" or text in ("/start",):
        response = GREETING_RESPONSE
        stage = "greeting"
    elif intent == "company_question":
        response = answer_company_question(text)
        stage = "company_info"
    elif intent == "general_chat":
        response = answer_general_chat(text)
        stage = "general_chat"
    elif intent == "purchase":
        response = "ممنون از خریدت! 🎉 امیدوارم راضی باشی. اگه سوال دیگه‌ای داری بپرس."
    elif intent == "unknown":
        response = "متوجه نشدم. لطفاً بگو دنبال چه محصولی هستی یا از /help استفاده کن."
    else:
        response = state.get("response_text", "")

    if response:
        return {
            "response_text": response,
            "conversation_stage": stage,
            "messages": [AIMessage(content=response)],
        }
    return {}


def save_memory_node(state: SalesAssistantState) -> dict[str, Any]:
    """LangGraph Memory: persist state via checkpointer."""
    now = datetime.now(timezone.utc).isoformat()
    return {"last_user_message_at": now}


def _with_polish(state: SalesAssistantState, response: str) -> str:
    """Optionally polish response with Groq LLM."""
    try:
        return polish_response(
            response,
            user_message=_last_human_text(state),
            stage=state.get("conversation_stage") or "",
        )
    except Exception:
        return response


def llm_polish_node(state: SalesAssistantState) -> dict[str, Any]:
    """LLM node: polish final Persian response via Groq."""
    draft = state.get("response_text")
    if not draft:
        return {}
    polished = _with_polish(state, draft)
    if polished == draft:
        return {}
    return {
        "response_text": polished,
        "messages": [AIMessage(content=polished)],
    }


def _last_human_text(state: SalesAssistantState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, dict) and msg.get("type") == "human":
            return msg.get("content", "")
    return ""
