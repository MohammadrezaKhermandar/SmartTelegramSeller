"""LLM tool-calling agent for hybrid product search (LangGraph Tool Calling)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import MIN_RECOMMENDATIONS, USE_LLM
from app.llm.client import get_llm
from app.tools.compare_tools import compare_products_tool
from app.tools.pandas_tools import (
    filter_by_brand_tool,
    filter_by_category_tool,
    filter_by_price_range_tool,
    filter_by_availability_tool,
    get_product_by_id_tool,
    hybrid_recommend,
    sort_products_tool,
)
from app.tools.rag_tools import semantic_search_tool
from app.utils.logging import logger

SEARCH_AGENT_TOOLS = [
    semantic_search_tool,
    filter_by_category_tool,
    filter_by_price_range_tool,
    filter_by_brand_tool,
    filter_by_availability_tool,
    sort_products_tool,
    compare_products_tool,
    get_product_by_id_tool,
]

AGENT_SYSTEM_PROMPT = """You are a product search agent for an online store.
Use the provided tools to find products matching the customer's requirements.

Rules:
1. Always call semantic_search_tool first with a rich Persian/English query.
2. Then apply filter tools (category, price range, brand) when requirements are known.
3. Use compare_products_tool only when you already have at least two product IDs.
4. Do not invent product IDs – only use IDs returned by tools.
5. Make multiple tool calls until you have enough candidates (at least 3)."""


def _extract_products_from_tool_result(result: Any) -> list[dict[str, Any]]:
    """Normalize tool outputs into a list of product dicts."""
    if result is None:
        return []
    if isinstance(result, list):
        return [p for p in result if isinstance(p, dict) and p.get("product_id")]
    if isinstance(result, dict):
        if result.get("product_id"):
            return [result]
        if "products" in result and isinstance(result["products"], list):
            return [p for p in result["products"] if isinstance(p, dict)]
    return []


def _merge_agent_products(
    collected: list[dict[str, Any]],
    requirements: dict[str, Any],
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    """Deduplicate agent tool results and top up via hybrid_recommend if needed."""
    by_id: dict[str, dict[str, Any]] = {}
    for product in collected:
        pid = str(product.get("product_id", ""))
        if not pid:
            continue
        existing = by_id.get(pid)
        score = float(product.get("score", 0.5) or 0.5)
        if existing:
            existing["score"] = existing.get("score", 0) + score
            existing["agent_match"] = True
        else:
            product["score"] = score
            product["agent_match"] = True
            by_id[pid] = product

    ranked = sorted(by_id.values(), key=lambda p: p.get("score", 0), reverse=True)

    if len(ranked) < MIN_RECOMMENDATIONS:
        rag_results = semantic_search_tool.invoke({"query": query})
        extra, note = hybrid_recommend(
            query=query,
            category=requirements.get("category"),
            brand=requirements.get("brand"),
            min_price=requirements.get("min_price"),
            max_price=requirements.get("max_price"),
            rag_results=rag_results if isinstance(rag_results, list) else None,
        )
        for product in extra:
            pid = str(product.get("product_id", ""))
            if pid and pid not in by_id:
                ranked.append(product)
        ranked = sorted(ranked, key=lambda p: p.get("score", 0), reverse=True)
        return ranked[: max(MIN_RECOMMENDATIONS, len(ranked))], note or "ترکیب نتایج Agent و Hybrid"

    return ranked[: max(MIN_RECOMMENDATIONS, len(ranked))], "نتایج از Agent با Tool Calling"


def run_search_tool_agent(
    requirements: dict[str, Any],
    query: str,
    *,
    max_iterations: int = 6,
) -> tuple[list[dict[str, Any]], str] | None:
    """
    Run LLM + bind_tools search loop.
    Returns None when LLM is unavailable (caller should use hybrid fallback).
    """
    if not USE_LLM:
        return None

    llm = get_llm()
    if llm is None:
        return None

    llm_with_tools = llm.bind_tools(SEARCH_AGENT_TOOLS)
    tools_by_name = {tool.name: tool for tool in SEARCH_AGENT_TOOLS}

    user_prompt = (
        f"Customer requirements:\n"
        f"- category: {requirements.get('category')}\n"
        f"- brand: {requirements.get('brand')}\n"
        f"- min_price: {requirements.get('min_price')}\n"
        f"- max_price: {requirements.get('max_price')}\n"
        f"- usage: {requirements.get('usage')}\n"
        f"- query: {query}\n\n"
        "Use tools to find at least 3 suitable in-stock products."
    )

    messages: list[Any] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    collected: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.info("Tool agent finished after %d iterations (no more tool calls)", iteration + 1)
            break

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool = tools_by_name.get(tool_name)
            if tool is None:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=call.get("id", ""),
                    )
                )
                continue

            try:
                result = tool.invoke(call.get("args", {}))
                collected.extend(_extract_products_from_tool_result(result))
                payload = result if isinstance(result, (str, int, float)) else json.dumps(
                    result, ensure_ascii=False, default=str
                )
                messages.append(
                    ToolMessage(
                        content=str(payload)[:6000],
                        tool_call_id=call.get("id", ""),
                    )
                )
            except Exception as exc:
                logger.warning("Tool %s failed: %s", tool_name, exc)
                messages.append(
                    ToolMessage(
                        content=f"Tool error: {exc}",
                        tool_call_id=call.get("id", ""),
                    )
                )

    if not collected:
        return None

    products, note = _merge_agent_products(collected, requirements, query)
    if not products:
        return None
    return products, note
