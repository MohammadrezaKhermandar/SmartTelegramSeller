"""Render the compiled graph to graph.png.

Strategy (in order):
1. LangGraph's built-in Mermaid renderer (draw_mermaid_png, uses mermaid.ink).
2. Pyppeteer-free local fallback: draw with matplotlib + networkx-style layout.

Run:  python -m app.graph.graph_visualizer
"""

from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT
from app.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT = PROJECT_ROOT / "graph.png"


def render_with_mermaid(compiled_graph) -> bool:
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        OUTPUT.write_bytes(png_bytes)
        logger.info("graph.png rendered via Mermaid (%d bytes)", len(png_bytes))
        return True
    except Exception as exc:
        logger.warning("Mermaid rendering failed: %s", exc)
        return False


def render_with_matplotlib(compiled_graph) -> bool:
    """Offline fallback: manual layered drawing with matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        g = compiled_graph.get_graph()
        edges = [(e.source, e.target, bool(e.conditional)) for e in g.edges]

        # Manual layered layout tuned for this topology
        layers: dict[str, tuple[float, float]] = {
            "__start__": (6.0, 11.0),
            "load_or_create_session": (6.0, 10.0),
            "classify_message": (6.0, 9.0),
            "extract_user_intent_and_requirements": (6.0, 8.0),
            "check_memory_relevance": (6.0, 7.0),
            "ask_clarifying_questions": (0.8, 5.0),
            "generate_memory_based_answer": (2.6, 5.0),
            "hybrid_product_search": (4.4, 5.5),
            "rank_recommendations": (4.4, 4.4),
            "compare_selected_products": (6.6, 5.5),
            "generate_comparison_response": (6.6, 4.4),
            "process_image": (8.4, 5.8),
            "process_external_link": (10.2, 5.8),
            "find_similar_products": (9.3, 4.7),
            "generate_sales_response": (5.5, 3.2),
            "generate_smalltalk_response": (11.4, 5.0),
            "retry_or_fallback_response": (11.4, 3.2),
            "save_memory": (6.0, 1.8),
            "__end__": (6.0, 0.8),
        }

        fig, ax = plt.subplots(figsize=(20, 13))
        ax.axis("off")

        for source, target, conditional in edges:
            if source not in layers or target not in layers:
                continue
            x1, y1 = layers[source]
            x2, y2 = layers[target]
            style = "--" if conditional else "-"
            color = "#e67e22" if conditional else "#34495e"
            ax.annotate(
                "",
                xy=(x2, y2 + 0.18), xytext=(x1, y1 - 0.18),
                arrowprops=dict(
                    arrowstyle="-|>", linestyle=style, color=color,
                    lw=1.4, alpha=0.75, shrinkA=12, shrinkB=12,
                    connectionstyle="arc3,rad=0.08",
                ),
            )

        colors = {
            "__start__": "#2ecc71", "__end__": "#e74c3c",
            "save_memory": "#9b59b6", "retry_or_fallback_response": "#e67e22",
            "check_memory_relevance": "#f1c40f",
        }
        for node, (x, y) in layers.items():
            face = colors.get(node, "#3498db")
            label = node.replace("__", "").replace("_", "\n")
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - 0.75, y - 0.28), 1.5, 0.56,
                    boxstyle="round,pad=0.06",
                    facecolor=face, edgecolor="#2c3e50", alpha=0.92,
                )
            )
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=7.5, color="white", weight="bold")

        ax.set_xlim(-0.5, 12.7)
        ax.set_ylim(0.2, 11.6)
        ax.set_title(
            "SINWAY Sales Assistant — LangGraph Workflow\n"
            "(dashed orange = conditional edges)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(OUTPUT, dpi=160)
        plt.close(fig)
        logger.info("graph.png rendered via matplotlib fallback")
        return True
    except Exception as exc:
        logger.error("Matplotlib rendering failed: %s", exc)
        return False


def generate_graph_png() -> Path:
    from app.graph.builder import get_graph

    compiled = get_graph()
    if not render_with_mermaid(compiled):
        if not render_with_matplotlib(compiled):
            raise RuntimeError("Could not render graph.png with any backend")
    return OUTPUT


if __name__ == "__main__":
    path = generate_graph_png()
    print(f"Graph image written to {path}")
