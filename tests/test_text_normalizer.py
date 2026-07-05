from app.utils.text_normalizer import (
    extract_budget,
    extract_ordinal,
    extract_ordinals,
    extract_urls,
    format_price,
    normalize,
)


def test_normalize_persian_digits():
    assert normalize("۵۰ میلیون") == "50 میلیون"


def test_extract_budget_million():
    assert extract_budget("لپ‌تاپ تا ۵۰ میلیون می‌خوام") == 50_000_000
    assert extract_budget("بودجه‌ام شد ۴۰ میلیون") == 40_000_000


def test_extract_budget_thousand_and_raw():
    assert extract_budget("تا 500 هزار تومان") == 500_000
    assert extract_budget("بودجه 12000000") == 12_000_000


def test_extract_budget_none():
    assert extract_budget("یه لپ‌تاپ می‌خوام") is None
    assert extract_budget("یه لپ‌تاپ برای برنامه‌نویسی می‌خوام") is None


def test_extract_budget_50_million_phrase():
    assert extract_budget("بودجه‌ام ۵۰ میلیونه") == 50_000_000


def test_extract_budget_40_million_ta():
    assert extract_budget("تا ۴۰ میلیون") == 40_000_000


def test_is_hard_max_budget():
    from app.utils.text_normalizer import is_hard_max_budget

    assert is_hard_max_budget("نه بیشتر از 40 میلیون ندارم")
    assert not is_hard_max_budget("تا ۴۰ میلیون")


def test_extract_budget_30_toman_shorthand():
    assert extract_budget("زیر ۳۰ تومن") == 30_000_000


def test_extract_ordinal():
    assert extract_ordinal("گزینه دوم رمش چقدره؟") == 2
    assert extract_ordinal("دومی رمش چنده") == 2
    assert extract_ordinal("مورد 3 چطوره") == 3
    assert extract_ordinal("سلام") is None


def test_extract_ordinals_for_comparison():
    result = extract_ordinals("گزینه اول و سوم رو مقایسه کن")
    assert 1 in result and 3 in result


def test_extract_urls():
    urls = extract_urls("این خوبه؟ https://example.com/p/laptop-1 ممنون")
    assert urls == ["https://example.com/p/laptop-1"]


def test_format_price_persian():
    assert format_price(12455000) == "۱۲,۴۵۵,۰۰۰ تومان"
