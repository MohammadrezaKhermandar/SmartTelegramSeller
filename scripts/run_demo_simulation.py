#!/usr/bin/env python
"""
Run all demo scenarios from docs/demo_script.md without Telegram.
Useful before recording the demo video to verify every flow works.

Usage:
    python scripts/run_demo_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage

from app.graph.graph_builder import build_graph, reset_graph
from app.graph.runner import reset_user_state
from app.main import initialize_app
from app.memory.checkpointer import reset_checkpointer


CSV_PATH = ROOT / "products_500.csv"


def run_scenario(name: str, user_id: str, messages: list[str]) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": user_id}}
    print(f"\n=== {name} ===")

    for text in messages:
        print(f"USER: {text}")
        result = graph.invoke(
            {
                "user_id": user_id,
                "messages": [HumanMessage(content=text)],
                "retry_count": 0,
                "errors": [],
            },
            config,
        )
        response = (result.get("response_text") or "")[:300]
        products = len(result.get("recommended_products") or [])
        stage = result.get("conversation_stage")
        from_memory = result.get("from_memory")
        from_agent = result.get("from_tool_agent")
        print(f"BOT ({stage}, products={products}, memory={from_memory}, agent={from_agent}):")
        print(response)
        if products:
            print(f"  -> {products} product(s) recommended")


def main() -> int:
    reset_graph()
    reset_checkpointer()
    initialize_app(CSV_PATH)

    uid = "demo_simulation_user"
    reset_user_state(uid)

    run_scenario("1. Start / greeting", uid, ["سلام"])
    reset_user_state(uid)

    run_scenario("2. Incomplete laptop request", uid, ["یه لپ‌تاپ می‌خوام"])
    reset_user_state(uid)

    run_scenario(
        "3. Complete requirements",
        uid,
        [
            "یه لپ‌تاپ می‌خوام",
            "برای برنامه‌نویسی، بودجه ۸۰ میلیون",
        ],
    )
    reset_user_state(uid)

    run_scenario(
        "4. Follow-up from memory",
        uid,
        [
            "لپ‌تاپ برای برنامه‌نویسی با بودجه ۱۰۰ میلیون",
            "اون دومی قیمتش چقدره؟",
        ],
    )
    reset_user_state(uid)

    run_scenario(
        "5. Compare",
        uid,
        [
            "لپ‌تاپ برای برنامه‌نویسی با بودجه ۱۰۰ میلیون",
            "اولی و دومی رو مقایسه کن",
        ],
    )
    reset_user_state(uid)

    run_scenario(
        "6. Budget change",
        uid,
        [
            "لپ‌تاپ برای اداری با بودجه ۱۰۰ میلیون",
            "بودجه‌ام رو به ۳۰ میلیون تغییر دادم",
        ],
    )

    print("\n=== Demo simulation complete ===")
    print("Next: record Telegram video using docs/demo_script.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
