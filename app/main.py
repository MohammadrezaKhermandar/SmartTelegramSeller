"""Entry point.

Usage:
    python -m app.main            # run the Telegram bot
    python -m app.main --index    # (re)build the vector index only
    python -m app.main --graph    # regenerate graph.png only
"""

from __future__ import annotations

import argparse
import sys

from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="SINWAY Sales Assistant")
    parser.add_argument("--index", action="store_true", help="build/refresh the vector index and exit")
    parser.add_argument("--graph", action="store_true", help="render graph.png and exit")
    args = parser.parse_args()

    if args.index:
        from app.services.rag_service import get_rag_service

        service = get_rag_service()
        logger.info(
            "Vector index ready: backend=%s documents=%d",
            getattr(service, "backend", "?"),
            service.document_count(),
        )
        return 0

    if args.graph:
        from app.graph.graph_visualizer import generate_graph_png

        path = generate_graph_png()
        logger.info("Graph image written to %s", path)
        return 0

    from app.telegram.bot import run_bot

    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
