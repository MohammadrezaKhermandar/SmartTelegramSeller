"""Builds and compiles the LangGraph StateGraph.

Topology (see graph.png):

START -> load_or_create_session -> classify_message
      -> extract_user_intent_and_requirements -> check_memory_relevance
      -> [conditional_router]
           ├─ needs_clarification -> ask_clarifying_questions ─────────────┐
           ├─ answer_from_memory  -> generate_memory_based_answer ─────────┤
           ├─ product_search      -> hybrid_product_search                 │
           │                          -> rank_recommendations              │
           │                          -> generate_sales_response ──────────┤
           ├─ compare_products    -> compare_selected_products             │
           │                          -> generate_comparison_response ─────┤
           ├─ image_similarity    -> process_image -> find_similar_products│
           │                          -> generate_sales_response ──────────┤
           ├─ link_similarity     -> process_external_link                 │
           │                          -> find_similar_products             │
           │                          -> generate_sales_response ──────────┤
           ├─ smalltalk           -> generate_smalltalk_response ──────────┤
           └─ error               -> retry_or_fallback_response ───────────┤
                                                                           v
                                                        save_memory -> END

Every processing branch passes through an error gate: if a node stored
state['error'], the flow is diverted to retry_or_fallback_response so the
user always receives a natural reply.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.router import conditional_router
from app.graph.state import SalesState, initial_state
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _error_gate(next_node: str):
    """Route to fallback if the previous node recorded an error."""
    def gate(state: SalesState) -> str:
        return "retry_or_fallback_response" if state.get("error") else next_node
    return gate


def build_graph() -> Any:
    """Construct and compile the sales assistant graph."""
    graph = StateGraph(SalesState)

    # --- nodes ---
    graph.add_node("load_or_create_session", nodes.load_or_create_session)
    graph.add_node("classify_message", nodes.classify_message)
    graph.add_node(
        "extract_user_intent_and_requirements",
        nodes.extract_user_intent_and_requirements,
    )
    graph.add_node("check_memory_relevance", nodes.check_memory_relevance)

    graph.add_node("ask_clarifying_questions", nodes.ask_clarifying_questions)
    graph.add_node("generate_memory_based_answer", nodes.generate_memory_based_answer)

    graph.add_node("hybrid_product_search", nodes.hybrid_product_search)
    graph.add_node("rank_recommendations", nodes.rank_recommendations)
    graph.add_node("generate_sales_response", nodes.generate_sales_response)

    graph.add_node("compare_selected_products", nodes.compare_selected_products)
    graph.add_node("generate_comparison_response", nodes.generate_comparison_response)

    graph.add_node("process_image", nodes.process_image)
    graph.add_node("process_external_link", nodes.process_external_link)
    graph.add_node("find_similar_products", nodes.find_similar_products)

    graph.add_node("generate_smalltalk_response", nodes.generate_smalltalk_response)
    graph.add_node("retry_or_fallback_response", nodes.retry_or_fallback_response)
    graph.add_node("save_memory", nodes.save_memory)

    # --- linear intake pipeline ---
    graph.add_edge(START, "load_or_create_session")
    graph.add_edge("load_or_create_session", "classify_message")
    graph.add_edge("classify_message", "extract_user_intent_and_requirements")
    graph.add_edge("extract_user_intent_and_requirements", "check_memory_relevance")

    # --- main conditional router ---
    graph.add_conditional_edges(
        "check_memory_relevance",
        conditional_router,
        {
            "needs_clarification": "ask_clarifying_questions",
            "answer_from_memory": "generate_memory_based_answer",
            "product_search": "hybrid_product_search",
            "compare_products": "compare_selected_products",
            "image_similarity": "process_image",
            "link_similarity": "process_external_link",
            "smalltalk": "generate_smalltalk_response",
            "error": "retry_or_fallback_response",
        },
    )

    # --- product search branch (with error gates) ---
    graph.add_conditional_edges(
        "hybrid_product_search",
        _error_gate("rank_recommendations"),
        {"rank_recommendations": "rank_recommendations",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )
    graph.add_conditional_edges(
        "rank_recommendations",
        _error_gate("generate_sales_response"),
        {"generate_sales_response": "generate_sales_response",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )

    # --- comparison branch ---
    graph.add_conditional_edges(
        "compare_selected_products",
        _error_gate("generate_comparison_response"),
        {"generate_comparison_response": "generate_comparison_response",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )

    # --- image / link branches ---
    graph.add_conditional_edges(
        "process_image",
        _error_gate("find_similar_products"),
        {"find_similar_products": "find_similar_products",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )
    graph.add_conditional_edges(
        "process_external_link",
        _error_gate("find_similar_products"),
        {"find_similar_products": "find_similar_products",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )
    graph.add_conditional_edges(
        "find_similar_products",
        _error_gate("generate_sales_response"),
        {"generate_sales_response": "generate_sales_response",
         "retry_or_fallback_response": "retry_or_fallback_response"},
    )

    # --- response nodes converge on save_memory ---
    for terminal in (
        "ask_clarifying_questions",
        "generate_memory_based_answer",
        "generate_sales_response",
        "generate_comparison_response",
        "generate_smalltalk_response",
        "retry_or_fallback_response",
    ):
        if terminal in ("generate_sales_response", "generate_memory_based_answer",
                        "ask_clarifying_questions", "generate_comparison_response",
                        "generate_smalltalk_response"):
            graph.add_conditional_edges(
                terminal,
                _error_gate("save_memory"),
                {"save_memory": "save_memory",
                 "retry_or_fallback_response": "retry_or_fallback_response"},
            )
        else:
            graph.add_edge(terminal, "save_memory")

    graph.add_edge("save_memory", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    logger.info("Sales graph compiled with %d nodes", len(graph.nodes))
    return compiled


_graph: Optional[Any] = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_thread(chat_id: str) -> None:
    """Clear LangGraph in-memory checkpoint for this chat (thread_id)."""
    graph = get_graph()
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(chat_id)
        logger.info("Cleared LangGraph checkpoint for chat %s", chat_id)


def reset_chat_state(chat_id: str) -> None:
    """Wipe SQLite session memory and LangGraph checkpoint for a fresh start."""
    from app.services.memory_service import get_memory_service

    get_memory_service().reset_chat(chat_id)
    reset_thread(chat_id)


def run_turn(
    user_id: str,
    chat_id: str,
    message: str,
    message_type: str = "text",
    image_caption: Optional[str] = None,
    image_file_name: Optional[str] = None,
) -> SalesState:
    """Execute one conversational turn through the graph.

    The LangGraph MemorySaver checkpoint is keyed by chat_id (thread_id), and
    long-term memory lives in SQLite — so state survives process restarts.
    """
    graph = get_graph()
    state = initial_state(
        user_id=user_id,
        chat_id=chat_id,
        message=message,
        message_type=message_type,
        image_caption=image_caption,
        image_file_name=image_file_name,
    )
    config = {"configurable": {"thread_id": chat_id}}
    try:
        result: SalesState = graph.invoke(state, config=config)
    except Exception:
        logger.exception("Graph execution crashed — returning fallback")
        result = dict(state)  # type: ignore[assignment]
        result["final_response"] = nodes.FALLBACK_TEXT
    if not result.get("final_response"):
        result["final_response"] = nodes.FALLBACK_TEXT
    return result
