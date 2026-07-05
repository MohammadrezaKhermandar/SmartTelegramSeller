"""Rule-based NLP for intent and requirement extraction (no API key required)."""

from __future__ import annotations

import re
from typing import Any

# Persian/English category keywords mapped to CSV categories
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "لپ‌تاپ": ["لپ تاپ", "لپتاپ", "لپ‌تاپ", "notebook", "laptop"],
    "هدفون": ["هدفون", "هدفون بی‌سیم", "headphone", "earphone"],
    "اسپیکر": ["اسپیکر", "بلندگو", "speaker"],
    "تلویزیون": ["تلویزیون", "tv", "تی‌وی"],
    "موبایل": ["موبایل", "گوشی", "phone", "smartphone"],
    "تبلت": ["تبلت", "tablet", "آیپد", "ipad"],
    "دسته بازی": ["دسته بازی", "گیم پد", "gamepad", "controller"],
    "ساعت هوشمند": ["ساعت هوشمند", "smartwatch", "ساعت"],
    "ماوس": ["ماوس", "mouse"],
    "کیبورد": ["کیبورد", "keyboard"],
    "مانیتور": ["مانیتور", "monitor"],
    "پاوربانک": ["پاوربانک", "powerbank"],
    "چای‌ساز": ["چای‌ساز", "کتری"],
    "جاروبرقی": ["جاروبرقی", "جارو"],
    "یخچال": ["یخچال", "refrigerator"],
    "ماشین لباسشویی": ["ماشین لباسشویی", "لباسشویی"],
    "عطر": ["عطر", "ادکلن", "perfume"],
}

USAGE_KEYWORDS = [
    "برنامه‌نویسی", "برنامه نویسی", "coding", "گیم", "بازی", "گیمینگ",
    "اداری", "طراحی", "دانشجویی", "سفر", "ورزش", "موسیقی",
]

BRAND_KEYWORDS = [
    "سامسونگ", "اپل", "شیائومی", "ایسوس", "لنوو", "سونی", "جی‌بی‌ال",
    "بوش", "مایکروسافت", "دل", "اچ‌پی", "سونی", "ایکس‌ویژن",
]

COMPARE_PATTERNS = [
    r"مقایسه", r"فرق", r"تفاوت", r"vs", r"با هم", r"کدوم بهتر",
]

FOLLOWUP_PATTERNS = [
    r"اون\s*اول", r"اولی", r"دومی", r"سومی", r"همون", r"اون\s*دوم",
    r"قیمتش", r"رمش", r"حافظه", r"گارانتی", r"موجودی", r"رنگش",
    r"ویژگی", r"امتیاز", r"توضیح",
]

CHANGE_PATTERNS = [
    r"بودجه‌ام", r"بودجه ام", r"بودجه‌مو", r"بودجه مو",
    r"ارزون‌تر", r"ارزان‌تر", r"گرون‌تر", r"گران‌تر",
    r"برند", r"عوض", r"تغییر", r"به جاش", r"نمی‌خوام", r"نمیخوام",
]

GREETING_EXACT = {"/start", "سلام", "درود", "hello", "hi"}

BUYING_PHRASES = [
    "میخوام",
    "می‌خوام",
    "می خواهم",
    "می‌خواهم",
    "دنبال",
    "پیشنهاد بده",
    "پیشنهاد بدید",
    "معرفی کن",
    "معرفی کنید",
    "خرید",
    "بخرم",
    "لازم دارم",
    "نیاز دارم",
    "نیاز",
]

BUYING_PATTERNS = [
    r"پیشنهاد\s*بده",
    r"پیشنهاد\s*بدید",
    r"معرفی\s*کن",
    r"معرفی\s*کنید",
    r"می\s*خوام",
    r"می\s*خواهم",
    r"لازم\s*دارم",
    r"نیاز\s*دارم",
]

AVAILABILITY_PATTERNS = [
    r"دارید\??",
    r"دارین\??",
    r"موجود",
    r"هست\??",
    r"هستن\??",
    r"have\b",
]

COMPANY_QUESTION_PATTERNS = [
    r"مدیر",
    r"مدیریت",
    r"صاحب",
    r"مالک",
    r"مجموعه",
    r"فروشگاه.*کیه",
    r"شما\s*کی",
    r"کی\s*هستید",
    r"ساعت\s*کاری",
    r"ساعات\s*کار",
    r"آدرس",
    r"کجاست",
    r"موقعیت",
    r"معرفی\s*فروشگاه",
]

GENERAL_CHAT_PATTERNS = [
    r"چطوری",
    r"چطورید",
    r"خوبی",
    r"حالت",
    r"حال\s*داری",
    r"خسته\s*نباش",
    r"ممنون",
    r"مرسی",
    r"تشکر",
    r"متشکر",
]

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

ORDINAL_MAP = {
    "اول": 0, "اولی": 0, "یک": 0, "۱": 0,
    "دوم": 1, "دومی": 1, "دو": 1, "۲": 1,
    "سوم": 2, "سومی": 2, "سه": 2, "۳": 2,
    "چهارم": 3, "۴": 3,
    "پنجم": 4, "۵": 4,
}

FIELD_PATTERNS: dict[str, list[str]] = {
    "price": [r"قیمت", r"چنده", r"چقدر"],
    "features": [r"ویژگی", r"مشخصات", r"رم", r"حافظه", r"پردازنده"],
    "rating": [r"امتیاز", r"نظر", r"رضایت"],
    "stock": [r"موجود", r"انبار", r"stock"],
    "warranty": [r"گارانتی", r"ضمانت"],
    "color": [r"رنگ"],
    "brand": [r"برند"],
    "description": [r"توضیح", r"درباره"],
}


_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _normalize_persian_digits(text: str) -> str:
    return text.translate(_PERSIAN_DIGITS)


def detect_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    return match.group(0) if match else None


def _normalize_text(text: str) -> str:
    return _normalize_persian_digits(text).lower().strip()


def _has_buying_intent(text_lower: str) -> bool:
    if any(phrase in text_lower for phrase in BUYING_PHRASES):
        return True
    return any(re.search(pattern, text_lower) for pattern in BUYING_PATTERNS)


def _has_availability_intent(text_lower: str) -> bool:
    return any(re.search(pattern, text_lower) for pattern in AVAILABILITY_PATTERNS)


def _has_budget_mention(text_lower: str) -> bool:
    return bool(re.search(r"بودجه|میلیون|میلیارد|تومان|\d", text_lower))


def _has_usage_mention(text: str) -> bool:
    text_lower = _normalize_text(text)
    return any(usage in text_lower for usage in USAGE_KEYWORDS)


def _is_product_request(text: str) -> bool:
    text_lower = _normalize_text(text)
    if _has_buying_intent(text_lower):
        return True

    category = _detect_category(text)
    if category and (
        _has_availability_intent(text_lower)
        or _has_usage_mention(text)
        or _has_budget_mention(text_lower)
    ):
        return True

    return False


def _is_greeting(text_lower: str) -> bool:
    if text_lower in GREETING_EXACT:
        return True
    return bool(re.fullmatch(r"(سلام|درود|hello|hi)[\s!؟?.]*", text_lower))


def _is_company_question(text_lower: str) -> bool:
    return any(re.search(pattern, text_lower) for pattern in COMPANY_QUESTION_PATTERNS)


def _is_general_chat(text_lower: str) -> bool:
    return any(re.search(pattern, text_lower) for pattern in GENERAL_CHAT_PATTERNS)


def detect_intent(
    text: str,
    has_image: bool = False,
    has_recommendations: bool = False,
) -> str:
    text = _normalize_persian_digits(text)
    text_lower = _normalize_text(text)

    if has_image:
        return "image_input"
    if detect_url(text):
        return "url_input"
    if _is_greeting(text_lower):
        return "greeting"
    if any(re.search(p, text_lower) for p in COMPARE_PATTERNS):
        return "compare"
    if has_recommendations and any(re.search(p, text_lower) for p in FOLLOWUP_PATTERNS):
        return "followup_question"
    if has_recommendations and any(re.search(p, text_lower) for p in CHANGE_PATTERNS):
        return "change_preferences"
    if any(re.search(p, text_lower) for p in [r"تغییر", r"عوض", r"به جاش"]):
        return "change_preferences"
    if any(kw in text_lower for kw in ["خریدم", "خرید کردم", "سفارش دادم"]):
        return "purchase"
    if _is_product_request(text):
        return "new_product_request"
    if _is_company_question(text_lower):
        return "company_question"
    if _is_general_chat(text_lower):
        return "general_chat"
    if has_recommendations:
        return "followup_question"

    return "unknown"


def _detect_category(text: str) -> str | None:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return category
    return None


def extract_requirements(text: str) -> dict[str, Any]:
    """Extract structured requirements from user text."""
    text = _normalize_persian_digits(text)
    req: dict[str, Any] = {
        "category": _detect_category(text),
        "brand": None,
        "min_price": None,
        "max_price": None,
        "usage": None,
        "features": [],
        "raw_query": text,
    }

    for brand in BRAND_KEYWORDS:
        if brand in text:
            req["brand"] = brand
            break

    for usage in USAGE_KEYWORDS:
        if usage in text:
            req["usage"] = usage
            break

    # Price extraction (Tomans)
    price_patterns = [
        (r"تا\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "max"),
        (r"حداکثر\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "max"),
        (r"زیر\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "max"),
        (r"از\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "min"),
        (r"حداقل\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "min"),
        (r"بودجه\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "max"),
        (r"به\s*(\d[\d,\.]*)\s*(میلیون|میلیارد|تومان)?", "max"),
    ]
    for pattern, kind in price_patterns:
        m = re.search(pattern, text)
        if m:
            amount = _parse_price(m.group(1), m.group(2) if m.lastindex >= 2 else None)
            if kind == "max":
                req["max_price"] = amount
            else:
                req["min_price"] = amount

    return req


def _parse_price(num_str: str, unit: str | None) -> float:
    num = float(num_str.replace(",", "").replace("،", ""))
    if unit:
        if "میلیارد" in unit:
            return num * 1_000_000_000
        if "میلیون" in unit:
            return num * 1_000_000
    # Assume millions if small number
    if num < 1000:
        return num * 1_000_000
    return num


def get_missing_slots(requirements: dict[str, Any]) -> list[str]:
    """Determine which slots are missing for a good recommendation."""
    missing = []
    if not requirements.get("category") and not requirements.get("raw_query"):
        missing.append("category")
    if requirements.get("max_price") is None and requirements.get("min_price") is None:
        missing.append("budget")
    if not requirements.get("usage"):
        missing.append("usage")
    return missing


def resolve_ordinal_reference(text: str, recommended: list[dict]) -> dict | None:
    """Resolve 'دومی' etc. to a product from recommendations."""
    if not recommended:
        return None
    for word, idx in ORDINAL_MAP.items():
        if word in text and idx < len(recommended):
            return recommended[idx]
    return recommended[0] if len(recommended) == 1 else None


def detect_field_question(text: str) -> str | None:
    """Detect which product field user is asking about."""
    for field, patterns in FIELD_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                return field
    return None


def extract_compare_indices(text: str, count: int) -> list[int]:
    """Extract product indices for comparison."""
    indices: list[int] = []
    for word, idx in ORDINAL_MAP.items():
        if word in text and idx < count:
            indices.append(idx)
    if len(indices) < 2:
        # Default: first two
        indices = list(range(min(2, count)))
    return sorted(set(indices))
