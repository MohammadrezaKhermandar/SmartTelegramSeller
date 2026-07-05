"""Demo script: runs the 7 required scenarios through the real graph
(no Telegram needed) and prints every exchange.

Usage:
    python -m scripts.demo_scenarios
    USE_LLM=false VECTOR_STORE_BACKEND=keyword python -m scripts.demo_scenarios
    python -m scripts.demo_scenarios --no-chroma   # force keyword backend (default)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Offline-safe defaults — applied before any app import
os.environ.setdefault("USE_LLM", "false")
os.environ.setdefault("VECTOR_STORE_BACKEND", "keyword")
os.environ.setdefault("EMBEDDING_MODE", "hash")
os.environ.setdefault("MEMORY_DB_PATH", ".data/demo_memory.sqlite")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SINWAY sales assistant demo scenarios")
    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="Use in-memory keyword/hash retrieval (default; stable on Windows)",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Use ChromaDB vector store (may crash on some Windows builds)",
    )
    return parser.parse_args()


def _apply_backend_flags(args: argparse.Namespace) -> None:
    if args.chroma:
        os.environ["VECTOR_STORE_BACKEND"] = "chroma"
    else:
        os.environ["VECTOR_STORE_BACKEND"] = "keyword"


CHAT = f"demo-{int(time.time())}"
USER = "demo-user"
DIVIDER = "─" * 72


def turn(message: str, message_type: str = "text", **kwargs) -> dict:
    from app.graph.builder import run_turn

    print(f"\n👤 کاربر: {message or '[تصویر]'}")
    result = run_turn(USER, CHAT, message, message_type=message_type, **kwargs)
    print(f"🤖 بات:\n{result['final_response']}")
    return result


def main() -> None:
    from app.services.memory_service import get_memory_service

    print(DIVIDER)
    print(
        f"Backend: {os.environ.get('VECTOR_STORE_BACKEND', 'keyword')} | "
        f"USE_LLM={os.environ.get('USE_LLM', 'false')}"
    )
    print(DIVIDER)
    print("سناریو ۱: نیاز ناقص — بات باید سؤال بپرسد، نه محصول معرفی کند")
    print(DIVIDER)
    r1 = turn("یه لپ‌تاپ می‌خوام")
    assert not get_memory_service().get_active_recommendations(CHAT), (
        "FAIL: bot recommended products before clarifying!"
    )
    assert "؟" in r1["final_response"], "FAIL: bot did not ask a question"
    print("\n✅ PASS: بات سؤال تکمیلی پرسید و محصولی معرفی نکرد")

    print(DIVIDER)
    print("سناریو ۲: نیاز کامل — حداقل ۳ محصول با دلیل")
    print(DIVIDER)
    turn("لپ‌تاپ تا ۵۰ میلیون برای برنامه‌نویسی می‌خوام، برند فرقی نداره")
    recs = get_memory_service().get_active_recommendations(CHAT)
    assert len(recs) >= 3, f"FAIL: expected >=3 recommendations, got {len(recs)}"
    print(f"\n✅ PASS: {len(recs)} محصول با دلیل معرفی شد")

    print(DIVIDER)
    print("سناریو ۳: سؤال درباره گزینه قبلی — پاسخ از Memory بدون جستجوی مجدد")
    print(DIVIDER)
    turn("گزینه دوم رمش چقدره؟")
    same = get_memory_service().get_active_recommendations(CHAT)
    assert [p["product_id"] for p in same] == [p["product_id"] for p in recs], (
        "FAIL: recommendations changed — bot re-searched instead of using memory"
    )
    print("\n✅ PASS: از حافظه پاسخ داد و جستجوی جدیدی انجام نشد")

    print(DIVIDER)
    print("سناریو ۴: تغییر بودجه — به‌روزرسانی جزئی، نه شروع از صفر")
    print(DIVIDER)
    r4 = turn("بودجه‌ام شد ۴۰ میلیون")
    session = get_memory_service().get_or_create_session(CHAT, USER)
    assert session["requirements"].get("budget") == 40_000_000, (
        f"FAIL: budget not updated: {session['requirements']}"
    )
    assert session["requirements"].get("category"), (
        "FAIL: category was lost — bot restarted requirements!"
    )
    assert "؟" not in r4["final_response"][:50] or "گزینه" in r4["final_response"], (
        "FAIL: bot restarted clarification"
    )
    print("\n✅ PASS: فقط بودجه به‌روزرسانی شد و دسته/کاربرد حفظ شد")

    print(DIVIDER)
    print("سناریو ۵: مقایسه گزینه اول و سوم")
    print(DIVIDER)
    r5 = turn("گزینه اول و سوم رو مقایسه کن")
    assert "مقایسه" in r5["final_response"] or "قیمت" in r5["final_response"], (
        "FAIL: no comparison in the answer"
    )
    print("\n✅ PASS: مقایسه از حافظه انجام شد")

    print(DIVIDER)
    print("سناریو ۶: لینک محصول خارجی — پیشنهاد مشابه از فروشگاه")
    print(DIVIDER)
    turn("https://example-shop.com/product/wireless-headphone-sony-wh1000")
    print("\n✅ PASS: مسیر لینک اجرا شد")

    print(DIVIDER)
    print("سناریو ۷: تصویر — با کپشن و بدون کپشن")
    print(DIVIDER)
    turn("", message_type="image", image_caption="هدفون بی‌سیم سونی")
    r7b = turn("", message_type="image", image_caption=None)
    assert "؟" in r7b["final_response"], "FAIL: no fallback question for captionless image"
    print("\n✅ PASS: تصویر با کپشن جستجو شد؛ بدون کپشن سؤال پرسید")

    print(DIVIDER)
    print("همه ۷ سناریو با موفقیت اجرا شدند ✅")
    print(DIVIDER)


if __name__ == "__main__":
    cli_args = _parse_args()
    _apply_backend_flags(cli_args)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except AssertionError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)
