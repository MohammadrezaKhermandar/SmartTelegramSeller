"""Persian text normalization and lightweight NLU helpers.

These helpers let the bot work deterministically even when the LLM is
unavailable (USE_LLM=false) and make LLM extraction more robust by
pre-normalizing the user's message.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Optional

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

_DIGIT_MAP = {ord(p): str(i) for i, p in enumerate(PERSIAN_DIGITS)}
_DIGIT_MAP.update({ord(a): str(i) for i, a in enumerate(ARABIC_DIGITS)})

# Arabic -> Persian letter unification
_CHAR_MAP = {
    ord("ي"): "ی",
    ord("ك"): "ک",
    ord("ة"): "ه",
    ord("\u200c"): " ",  # ZWNJ -> space (simplifies keyword matching)
    ord("ؤ"): "و",
    ord("أ"): "ا",
    ord("إ"): "ا",
}

_ORDINAL_WORDS = {
    "اول": 1, "اولی": 1, "یک": 1, "یکم": 1, "نخست": 1,
    "دوم": 2, "دو": 2, "دومی": 2,
    "سوم": 3, "سه": 3, "سومی": 3,
    "چهارم": 4, "چهار": 4, "چهارمی": 4,
    "پنجم": 5, "پنج": 5, "پنجمی": 5,
}

_BUDGET_MARKERS = (
    "بودجه", "تا", "زیر", "حدود", "میلیون", "ملیون",
    "تومن", "تومان", "قیمت", "هزار", "ندارم", "بیشتر از",
)

_UNIT_MULTIPLIERS = [
    (r"میلیارد", 1_000_000_000),
    (r"ملیارد", 1_000_000_000),
    (r"میلیون", 1_000_000),
    (r"ملیون", 1_000_000),
    (r"هزار\s*تومن", 1_000),
    (r"هزار\s*تومان", 1_000),
    (r"هزار", 1_000),
    (r"تومن", 1),
    (r"تومان", 1),
]

CATEGORY_SYNONYMS: dict[str, str] = {
    "لپ تاپ": "لپ‌تاپ",
    "لپتاپ": "لپ‌تاپ",
    "لپتاب": "لپ‌تاپ",
    "لبتاب": "لپ‌تاپ",
    "لپ‌تاپ": "لپ‌تاپ",
    "نوت بوک": "لپ‌تاپ",
    "notebook": "لپ‌تاپ",
    "laptop": "لپ‌تاپ",
    "گوشی": "گوشی موبایل",
    "موبایل": "گوشی موبایل",
    "هدفون": "هدفون",
    "هندزفری": "هدفون",
}

MEMORY_ATTRIBUTE_HINTS = (
    "رمش", "رم", "قیمتش", "قیمت", "گارانتیش", "گارانتی",
    "موجوده", "موجود", "حافظه", "باتری", "پردازنده", "مشخصات",
    "امتیازش", "رنگش", "وزنش",
)

URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def normalize(text: str) -> str:
    """Normalize Persian text: unify digits/letters, collapse whitespace."""
    if not text:
        return ""
    text = text.translate(_DIGIT_MAP).translate(_CHAR_MAP)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    norm = normalize(text).lower()
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)
    return [t for t in norm.split() if len(t) >= 2]


def extract_urls(text: str) -> list[str]:
    """Return every http(s) URL found in the text."""
    return URL_RE.findall(text or "")


def extract_budget(text: str) -> Optional[int]:
    """Extract a budget in Toman from Persian text.

    Only returns a value when explicit budget markers are present.
    Handles: "۵۰ میلیون", "تا 50 میلیون تومن", "بودجه‌ام ۴۰ میلیونه",
    "500 هزار تومان", "زیر ۳۰ تومن" (shop shorthand -> 30M).
    "بیشتر از X میلیون ندارم" budget ceiling phrases.
    """
    norm = normalize(text).lower()
    if not any(marker in norm for marker in _BUDGET_MARKERS):
        return None

    for unit_pattern, mult in _UNIT_MULTIPLIERS:
        m = re.search(rf"(\d+(?:[.,]\d+)?)\s*{unit_pattern}", norm)
        if m:
            value = float(m.group(1).replace(",", "."))
            amount = int(value * mult)
            if mult == 1 and unit_pattern in (r"تومن", r"تومان"):
                # Shop shorthand: «۴۰ تومن» ≈ ۴۰ میلیون تومان
                if value < 1000:
                    amount = int(value * 1_000_000)
                elif amount < 1_000_000:
                    return None
            elif amount < 1_000_000 and "هزار" not in unit_pattern:
                if unit_pattern not in (r"تومان",) or value >= 1000:
                    return None
            return amount

    # Bare large number (>= 1M) when a budget marker is present
    m = re.search(r"\b(\d{7,})\b", norm)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{6})\b", norm)
    if m and int(m.group(1)) >= 1_000_000:
        return int(m.group(1))
    return None


def is_hard_max_budget(text: str) -> bool:
    """True when the user states a ceiling budget that must not be exceeded.

    Covers phrases like «بیشتر از X ندارم», «نهایتا X», «سقف بودجه X»,
    «بالاتر از X نمی‌خوام».
    """
    norm = normalize(text).lower()
    if not norm:
        return False
    compact = norm.replace(" ", "")
    has_amount = any(m in norm for m in ("میلیون", "ملیون", "تومن", "تومان", "هزار"))
    if "بیشتر از" in norm and "ندارم" in norm:
        return True
    if "نهایتا" in norm and has_amount:
        return True
    if "سقف بودجه" in norm:
        return True
    if "بالاتر از" in norm and ("نمیخوام" in compact or "نمی‌خوام" in norm):
        return True
    return False


def is_reject_budget_overflow(text: str) -> bool:
    """True when user refuses recommendations above the stated budget.

    Routes to ``change_preferences`` (sets allow_budget_overflow=false).
    Must be checked before memory/attribute heuristics that match «قیمت».
    """
    norm = normalize(text).lower()
    if not norm:
        return False
    compact = norm.replace(" ", "").replace("\u200c", "")

    if "فقط داخل همین بودجه" in norm or "فقطداخلهمینبودجه" in compact:
        return True
    if "سقف بودجه همین" in norm or "سقفبودجههمین" in compact:
        return True
    if "بودجه ام همین" in norm or "بودجه‌ام همین" in norm or "بودجهامهمین" in compact:
        return True
    if "نمیخوام بیشتر هزینه" in compact or (
        "نمی" in compact and "بیشتر" in norm and "هزینه" in norm
    ):
        return True
    if "بیشتر از این" in norm and "هزینه" in norm and "نمی" in compact:
        return True
    if "گرون تر نمی" in norm or "گرون‌تر نمی" in norm or "گرونترنمی" in compact:
        return True
    if "قیمت" in norm and "بالا" in norm and "نمی" in compact:
        return True
    if "بالاتر" in norm and "نمی" in compact and not extract_budget(text):
        return True
    return False


def is_change_preferences(text: str) -> bool:
    """Preference update that should re-run recommendations (not memory Q&A)."""
    return is_reject_budget_overflow(text)


def extract_ordinal(text: str) -> Optional[int]:
    """Detect references like 'گزینه دوم' / 'دومی' / 'مورد 3'."""
    norm = normalize(text)
    m = re.search(r"(?:گزینه|مورد|شماره|محصول)\s*(\d+)", norm)
    if m:
        return int(m.group(1))
    for word, value in _ORDINAL_WORDS.items():
        if re.search(rf"(?:گزینه|مورد|شماره|محصول)\s*{word}\b", norm):
            return value
        if re.search(rf"\b{word}\b", norm):
            return value
    m = re.search(r"\b([123])\b", norm)
    if m:
        return int(m.group(1))
    return None


def extract_ordinals(text: str) -> list[int]:
    """Detect *all* ordinal references (for comparisons: 'اول و سوم')."""
    norm = normalize(text)
    found: list[int] = []
    for m in re.finditer(r"(?:گزینه|مورد|شماره|محصول)?\s*\b(\d)\b", norm):
        v = int(m.group(1))
        if 1 <= v <= 9 and v not in found:
            found.append(v)
    # Longer ordinal phrases first (e.g. «چهارمی» before «چهار»)
    for word, value in sorted(_ORDINAL_WORDS.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(word)}\b", norm) and value not in found:
            found.append(value)
    return sorted(found)


_IMAGE_REQUEST_HINTS = (
    "عکس", "تصویر", "عکساش", "عکسش", "عکس ها", "عکس‌ها", "عکسا",
)


def is_product_image_request(text: str) -> bool:
    """True when user asks to see photos of previously recommended products."""
    norm = normalize(text).lower()
    return any(hint in norm for hint in _IMAGE_REQUEST_HINTS)


def is_purchase_request(text: str) -> bool:
    """True when user expresses purchase/payment intent (not yet available)."""
    norm = normalize(text).lower()
    phrases = (
        "میخوام بخرم", "می خوام بخرم", "خرید میکنم", "خرید می‌کنم", "خرید می کنم",
        "سفارش میدم", "سفارش می‌دم", "سفارش می دم", "همینو میخوام", "همین رو میخوام",
        "همینو می‌خوام", "ثبت سفارش", "پرداخت", "خریدم", "سفارش دادم", "نهایی کردم",
        "خرید انجام شد",
    )
    if any(p in norm for p in phrases):
        return True
    # Standalone «خرید» / «بخرم» only when not a general product search
    if norm.strip() in ("خرید", "بخرم", "میخرم", "می‌خرم"):
        return True
    if re.search(r"\bخرید\b", norm) and not any(
        w in norm for w in ("میخوام", "می خوام", "دنبال", "پیشنهاد", "معرفی")
    ):
        if any(w in norm for w in ("میکنم", "می‌کنم", "می کنم", "کنم", "سفارش", "پرداخت")):
            return True
    return False


def detect_category(text: str, catalog_categories: Optional[list[str]] = None) -> Optional[str]:
    """Map user text to a canonical catalog category."""
    norm = normalize(text).lower()
    for phrase, cat in sorted(CATEGORY_SYNONYMS.items(), key=lambda x: -len(x[0])):
        phrase_norm = normalize(phrase).lower()
        if phrase_norm and phrase_norm in norm:
            return cat
    if catalog_categories:
        for category in catalog_categories:
            cat_norm = normalize(category).lower()
            if cat_norm and cat_norm in norm:
                return category
    return None


def brand_in_text(brand: str, text: str) -> bool:
    """Match brand as a distinct token — «دل» must not match «دلسی»."""
    brand_norm = normalize(brand).lower()
    if not brand_norm or len(brand_norm) < 2:
        return False
    text_norm = normalize(text).lower()
    if brand_norm == text_norm:
        return True
    return bool(
        re.search(rf"(?:^|\s|-){re.escape(brand_norm)}(?:\s|$|-)", text_norm)
    )


def brands_match(preferred: str, product_brand: str) -> bool:
    """True when preferred brand matches product brand without substring traps."""
    pref = normalize(preferred).lower()
    pb = normalize(product_brand).lower()
    if not pref or not pb:
        return False
    if pref == pb:
        return True
    return bool(re.search(rf"(?:^|\s|-){re.escape(pref)}(?:\s|$|-)", pb))


def is_attribute_question(text: str) -> bool:
    norm = normalize(text).lower()
    return any(hint in norm for hint in MEMORY_ATTRIBUTE_HINTS)


def product_name_in_message(text: str, products: list[dict[str, Any]]) -> bool:
    """True when part of a previously recommended product name appears in text."""
    return match_product_name_query(text, products) is not None


def match_product_name_query(
    text: str, products: list[dict[str, Any]], min_score: float = 0.35
) -> Optional[dict[str, Any]]:
    """Find the best matching recommended product by token overlap / fuzzy name."""
    query_tokens = set(_tokenize(text))
    if not query_tokens or not products:
        return None

    best: Optional[dict[str, Any]] = None
    best_score = 0.0
    norm_text = normalize(text).lower()

    for product in products:
        name = str(product.get("name", ""))
        name_tokens = set(_tokenize(name))
        if not name_tokens:
            continue
        overlap = len(query_tokens & name_tokens) / max(len(query_tokens), 1)
        ratio = difflib.SequenceMatcher(
            None, normalize(name).lower(), norm_text
        ).ratio()
        score = max(overlap, ratio * 0.85)
        if score > best_score:
            best_score = score
            best = product

    if best is not None and best_score >= min_score:
        return best
    return None


def is_memory_reference(text: str, products: list[dict[str, Any]]) -> bool:
    """True when the message refers to a prior recommendation (ordinal/name/attribute)."""
    if not products:
        return False
    if is_change_preferences(text):
        return False
    if extract_ordinal(text) is not None:
        return True
    if is_attribute_question(text):
        return True
    if product_name_in_message(text, products):
        return True
    norm = normalize(text).lower()
    pronoun_hints = ("همون", "همین", "قبلی", "گزینه")
    return any(h in norm for h in pronoun_hints)


# English product/brand words -> Persian equivalents used in the catalog.
_EN_FA_KEYWORDS = {
    "headphone": "هدفون", "headphones": "هدفون", "earbud": "هدفون", "earbuds": "هدفون",
    "laptop": "لپ‌تاپ", "notebook": "لپ‌تاپ",
    "phone": "گوشی موبایل", "smartphone": "گوشی موبایل", "mobile": "گوشی موبایل",
    "watch": "ساعت هوشمند", "smartwatch": "ساعت هوشمند",
    "speaker": "اسپیکر", "tv": "تلویزیون", "television": "تلویزیون",
    "tablet": "تبلت", "camera": "دوربین", "printer": "پرینتر",
    "monitor": "مانیتور", "mouse": "ماوس", "keyboard": "کیبورد",
    "console": "کنسول", "gamepad": "دسته بازی", "controller": "دسته بازی",
    "wireless": "بی‌سیم", "gaming": "گیمینگ", "vacuum": "جاروبرقی",
    "sony": "سونی", "samsung": "سامسونگ", "apple": "اپل", "xiaomi": "شیائومی",
    "asus": "ایسوس", "lenovo": "لنوو", "dell": "دل", "hp": "اچ‌پی",
    "huawei": "هواوی", "lg": "ال‌جی", "bosch": "بوش", "philips": "فیلیپس",
    "microsoft": "مایکروسافت", "logitech": "لاجیتک", "jbl": "جی‌بی‌ال",
}


def enrich_english_keywords(text: str) -> str:
    """Append Persian equivalents of known English words to the query text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    extras = [_EN_FA_KEYWORDS[w] for w in words if w in _EN_FA_KEYWORDS]
    return f"{text} {' '.join(dict.fromkeys(extras))}".strip() if extras else text


def to_persian_digits(text: str | int | float) -> str:
    """Convert Latin digits to Persian for display."""
    return str(text).translate(str.maketrans("0123456789", PERSIAN_DIGITS))


def format_price(price: float | int) -> str:
    """Format a Toman price for Persian display: 12455000 -> ۱۲,۴۵۵,۰۰۰ تومان."""
    return f"{to_persian_digits(f'{int(price):,}')} تومان"
