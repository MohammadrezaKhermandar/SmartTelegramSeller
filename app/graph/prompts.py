"""Persian system prompts and templates."""

from __future__ import annotations

import re
from typing import Any

BUSINESS_PROFILE = {
    "store_name": "فروشگاه سین وی",
    "manager": "خانم سمیعی",
    "working_hours": "شنبه تا پنج‌شنبه ۹ صبح تا ۹ شب، جمعه‌ها ۱۰ صبح تا ۶ عصر",
    "address": "برای آدرس دقیق شعب با پشتیبانی آنلاین تماس بگیرید.",
    "about": "ما فروشگاه اینترنتی سین وی هستیم و من دستیار فروش شماام.",
}

SALES_SYSTEM_PROMPT = """تو یک دستیار فروش هوشمند فروشگاه سین وی با مدیریت خانم سمیعی هستی برای یک فروشگاه اینترنتی هستی. زبان پاسخ تو فارسی است.

شخصیت تو:

* حرفه‌ای، صمیمی و قابل اعتماد
* شبیه یک فروشنده واقعی، نه یک ربات خشک
* صادق درباره مزایا و محدودیت‌های محصولات
* بدون اغراق و بدون وعده غیرواقعی
* متمرکز بر کمک به خرید درست، نه فقط فروش سریع

قوانین رفتاری:

1. اگر نیاز کاربر ناقص است، مستقیم محصول پیشنهاد نده. اول حداکثر دو سؤال کوتاه و مهم بپرس.
2. اگر اطلاعات کافی داری، حداقل سه محصول مناسب پیشنهاد بده.
3. برای هر محصول توضیح بده چرا مناسب نیاز کاربر است.
4. برای هر محصول در صورت امکان یک نقطه ضعف یا محدودیت هم بگو.
5. اگر محصول دقیق موجود نبود، محصولات مشابه یا جایگزین پیشنهاد بده و شفاف بگو که جایگزین هستند.
6. اگر کاربر درباره محصولی سؤال کرد که قبلاً پیشنهاد داده‌ای، از حافظه مکالمه و لیست محصولات پیشنهادی قبلی استفاده کن و جستجوی کامل جدید انجام نده.
7. اگر کاربر نظرش را تغییر داد، مثل تغییر بودجه، برند یا نوع استفاده، نیازمندی‌های قبلی را به‌روزرسانی کن و پیشنهادها را اصلاح کن.
8. پاسخ‌ها باید کوتاه، کاربردی و فروشنده‌محور باشند.
9. کاربر را برای قدم بعدی راهنمایی کن؛ مثلاً بپرس کدام گزینه را می‌خواهد دقیق‌تر بررسی کند.
10. اگر مطمئن نیستی، شفاف بگو و سؤال تکمیلی بپرس.

فرمت پیشنهاد محصول:

* نام محصول
* قیمت، اگر موجود است
* دلیل انتخاب
* مناسب برای چه نوع کاربری
* نقطه ضعف احتمالی
* پیشنهاد قدم بعدی

"""

CLARIFYING_QUESTIONS = {
    "budget": "حدود بودجه‌ات چقدره؟ (به تومان)",
    "usage": "بیشتر برای چه کاری می‌خوای؟ (مثلاً برنامه‌نویسی، بازی، کار اداری، طراحی، دانشجویی)",
    "category": "دنبال چه نوع محصولی هستی؟",
    "brand": "برند خاصی مدنظرت هست؟",
}

GREETING_RESPONSE = (
    "سلام! 👋 من دستیار فروش فروشگاه هستم.\n"
    "می‌تونم محصول مناسب پیشنهاد بدم، مقایسه کنم، یا درباره پیشنهادهای قبلی جواب بدم.\n\n"
    "دستورات پرکاربرد:\n"
    "/search لپ‌تاپ — جستجو\n"
    "/categories — دسته‌بندی‌ها\n"
    "/help — راهنمای کامل\n\n"
    "یا مثلاً بنویس: «یه لپ‌تاپ برای برنامه‌نویسی می‌خوام»"
)

def answer_company_question(text: str) -> str:
    """Answer store/manager/hours/address questions from business profile."""
    text_lower = text.lower()
    store = BUSINESS_PROFILE["store_name"]
    manager = BUSINESS_PROFILE["manager"]
    polite_open = ""

    if re.search(r"چطوری|چطورید|خوبی|حالت|حال\s*داری|سلام", text_lower):
        polite_open = "سلام! حالم خوبه، ممنون از احوالپرسی‌تون 😊 "

    if re.search(r"مدیر|مدیریت|صاحب|مالک", text_lower):
        return (
            f"{polite_open}مدیر مجموعه ما {manager} هستند. "
            "اگه سوالی درباره محصولات دارید خوشحال می‌شم کمک کنم."
        )
    if re.search(r"ساعت کاری|ساعات کار|کی باز|کی بسته|چه ساعتی", text_lower):
        return f"{polite_open}ساعت کاری {store}: {BUSINESS_PROFILE['working_hours']}"
    if re.search(r"آدرس|کجاست|موقعیت|لوکیشن", text_lower):
        return f"{polite_open}آدرس {store}: {BUSINESS_PROFILE['address']}"
    if re.search(r"شما کی|کی هستید|معرفی|فروشگاه.*چیه|مجموعه.*چیه", text_lower):
        return f"{polite_open}{BUSINESS_PROFILE['about']} مدیر مجموعه {manager} هستند."

    return (
        f"{polite_open}{BUSINESS_PROFILE['about']} "
        f"مدیر مجموعه {manager} هستند. چطور می‌تونم کمکتون کنم؟"
    )


def answer_general_chat(text: str) -> str:
    """Answer non-product small talk without entering product flow."""
    text_lower = text.lower()
    store = BUSINESS_PROFILE["store_name"]

    if re.search(r"چطوری|چطورید|خوبی|حالت|حال\s*داری", text_lower):
        return (
            "مرسی از احوالپرسی‌تون! من خوبم 😊 "
            f"من دستیار فروش {store} هستم. دنبال چه محصولی می‌گردید؟"
        )
    if re.search(r"ممنون|مرسی|تشکر|متشکر", text_lower):
        return "خواهش می‌کنم! اگه سوال دیگه‌ای دارید در خدمتم."

    return (
        f"در خدمتم! من دستیار فروش {store} هستم. "
        "اگه دنبال محصول خاصی هستید بگید تا پیشنهاد بدم."
    )


HELP_RESPONSE = (
    "راهنمای دستورات:\n\n"
    "🛒 خرید و پیشنهاد\n"
    "• متن آزاد: نیازت رو بگو (بودجه، برند، کاربرد)\n"
    "• /search <عبارت> — جستجوی محصول\n"
    "• /browse <دسته> — مرور یک دسته (مثلاً /browse لپ‌تاپ)\n"
    "• /categories — لیست دسته‌بندی‌ها\n"
    "• /product <id> — جزئیات یک محصول (مثلاً /product 124)\n"
    "• /my_products — آخرین پیشنهادها\n\n"
    "📎 رسانه\n"
    "• عکس: عکس محصول مشابه بفرست\n"
    "• لینک: آدرس محصول از سایت دیگر\n\n"
    "⚙️ مکالمه\n"
    "• /start — شروع\n"
    "• /reset — شروع مجدد\n"
    "• /mark_purchased — ثبت خرید\n\n"
    "💬 مقایسه: «اولی و دومی رو مقایسه کن»"
)

FOLLOWUP_1H_MESSAGE = (
    "سلام دوباره! 😊\n"
    "دیدم هنوز تصمیم نگرفتی. اگه سوالی درباره پیشنهادها داری، خوشحال می‌شم کمک کنم."
)

DISCOUNT_2D_MESSAGE = (
    "سلام! هنوز فرصت داری 😊\n"
    "پیشنهادهای قبلیت هنوز موجوده. با کد تخفیف **SALES10** می‌تونی ۱۰٪ تخفیف بگیری.\n"
    "{products_summary}"
)


def format_product_recommendation(products: list[dict], note: str = "") -> str:
    """Format product recommendations in Persian."""
    if not products:
        return "متأسفانه محصول مناسبی پیدا نکردم. لطفاً نیازت رو دقیق‌تر بگو."

    lines = ["بر اساس نیازت، این گزینه‌ها رو پیشنهاد می‌دم:\n"]
    if note:
        lines.append(f"ℹ️ {note}\n")

    for i, p in enumerate(products[:5], 1):
        title = p.get("title", "بدون عنوان")
        price = p.get("price", 0)
        brand = p.get("brand", "")
        rating = p.get("rating", "")
        features = p.get("features", "")
        weakness = _guess_weakness(p)

        price_str = f"{int(price):,}" if price else "نامشخص"
        lines.append(f"**{i}. {title}**")
        lines.append(f"   • برند: {brand} | قیمت: {price_str} تومان")
        if rating:
            lines.append(f"   • امتیاز: {rating}")
        if features:
            lines.append(f"   • ویژگی‌ها: {str(features)[:100]}")
        lines.append(f"   • چرا مناسبه: {_why_fit(p)}")
        lines.append(f"   • نقطه ضعف احتمالی: {weakness}")
        lines.append("")

    lines.append("کدوم بیشتر جذبت کرد؟ می‌تونم مقایسه کنم یا جزئیات بیشتر بدم.")
    return "\n".join(lines)


def _why_fit(product: dict) -> str:
    parts = []
    if product.get("rating") and float(product.get("rating", 0) or 0) >= 4:
        parts.append("امتیاز خوب کاربران")
    if product.get("discount") and float(product.get("discount", 0) or 0) > 0:
        parts.append(f"{int(product['discount'])}٪ تخفیف")
    if product.get("features"):
        parts.append("ویژگی‌های متنوع")
    if product.get("rag_match"):
        parts.append("تطابق خوب با نیاز شما")
    return " و ".join(parts) if parts else "تناسب با جستجوی شما"


def _guess_weakness(product: dict) -> str:
    stock = product.get("stock") or product.get("availability")
    if stock is not None and int(float(stock)) < 5:
        return "موجودی محدود"
    rating = product.get("rating")
    if rating and float(rating) < 3.5:
        return "امتیاز کاربران متوسط"
    price = product.get("price", 0)
    if price and float(price) > 50_000_000:
        return "قیمت بالاتر از میانگین"
    return "ممکنه رنگ یا مدل دقیق مورد نظرت نباشه"


def format_compare_result(result: dict) -> str:
    if result.get("error"):
        return result["error"]

    lines = ["مقایسه محصولات:\n"]
    for h in result.get("highlights", []):
        lines.append(f"• {h}")

    for p in result.get("products", []):
        lines.append(f"\n**{p.get('title', '')}**")
        lines.append(f"  قیمت: {int(p.get('price', 0)):,} تومان")
        if p.get("rating"):
            lines.append(f"  امتیاز: {p.get('rating')}")
        if p.get("features"):
            lines.append(f"  ویژگی‌ها: {str(p.get('features', ''))[:120]}")

    lines.append("\nکدوم رو ترجیح می‌دی؟")
    return "\n".join(lines)


def format_memory_answer(product: dict, field: str, value: Any) -> str:
    title = product.get("title", "محصول")
    field_labels = {
        "price": "قیمت",
        "features": "ویژگی‌ها",
        "rating": "امتیاز",
        "brand": "برند",
        "stock": "موجودی",
        "availability": "موجودی",
        "description": "توضیحات",
        "warranty": "گارانتی",
        "color": "رنگ",
    }
    label = field_labels.get(field, field)
    return f"بله، {title}:\n{label}: {value}"
