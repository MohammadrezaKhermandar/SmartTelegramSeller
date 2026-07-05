"""Build the LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    answer_from_memory_node,
    ask_clarifying_question_node,
    compare_products_node,
    enough_info_checker_node,
    error_handler_node,
    final_response_node,
    hybrid_search_node,
    tool_agent_node,
    image_similarity_node,
    input_classifier_node,
    intent_detector_node,
    llm_polish_node,
    load_memory_node,
    recommend_products_node,
    requirement_extractor_node,
    save_memory_node,
    update_requirements_node,
    url_similarity_node,
)
from app.graph.routers import (
    route_after_classifier,
    route_after_error,
    route_after_hybrid_search,
    route_after_info_check,
    route_after_intent,
    route_after_update,
)
from app.graph.state import SalesAssistantState
from app.memory.checkpointer import get_checkpointer


def build_graph():
    """Construct and compile the sales assistant LangGraph."""
    workflow = StateGraph(SalesAssistantState)

    # Register nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("input_classifier", input_classifier_node)
    workflow.add_node("intent_detector", intent_detector_node)
    workflow.add_node("requirement_extractor", requirement_extractor_node)
    workflow.add_node("update_requirements", update_requirements_node)
    workflow.add_node("enough_info_checker", enough_info_checker_node)
    workflow.add_node("ask_clarifying", ask_clarifying_question_node)
    workflow.add_node("tool_agent", tool_agent_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("recommend_products", recommend_products_node)
    workflow.add_node("answer_from_memory", answer_from_memory_node)
    workflow.add_node("compare_products", compare_products_node)
    workflow.add_node("image_similarity", image_similarity_node)
    workflow.add_node("url_similarity", url_similarity_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("final_response", final_response_node)
    workflow.add_node("llm_polish", llm_polish_node)
    workflow.add_node("save_memory", save_memory_node)

    # Entry
    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "input_classifier")

    # Conditional routing after classifier
    workflow.add_conditional_edges(
        "input_classifier",
        route_after_classifier,
        {
            "image_similarity": "image_similarity",
            "url_similarity": "url_similarity",
            "intent_detector": "intent_detector",
        },
    )

    workflow.add_conditional_edges(
        "intent_detector",
        route_after_intent,
        {
            "requirement_extractor": "requirement_extractor",
            "answer_from_memory": "answer_from_memory",
            "compare_products": "compare_products",
            "update_requirements": "update_requirements",
            "final_response": "final_response",
            "url_similarity": "url_similarity",
        },
    )

    workflow.add_edge("requirement_extractor", "enough_info_checker")

    workflow.add_conditional_edges(
        "enough_info_checker",
        route_after_info_check,
        {
            "ask_clarifying": "ask_clarifying",
            "tool_agent": "tool_agent",
        },
    )

    workflow.add_conditional_edges(
        "update_requirements",
        route_after_update,
        {
            "enough_info_checker": "enough_info_checker",
            "tool_agent": "tool_agent",
        },
    )

    workflow.add_conditional_edges(
        "tool_agent",
        route_after_hybrid_search,
        {
            "recommend_products": "recommend_products",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "hybrid_search",
        route_after_hybrid_search,
        {
            "recommend_products": "recommend_products",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "hybrid_search": "hybrid_search",
            "save_memory": "llm_polish",
        },
    )

    # Terminal paths → Groq polish → save memory
    for node in [
        "ask_clarifying",
        "recommend_products",
        "answer_from_memory",
        "compare_products",
        "image_similarity",
        "url_similarity",
        "final_response",
    ]:
        workflow.add_edge(node, "llm_polish")

    workflow.add_edge("llm_polish", "save_memory")

    workflow.add_edge("save_memory", END)

    checkpointer = get_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    """Singleton graph accessor."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    """Reset graph singleton (for tests)."""
    global _graph
    _graph = None
