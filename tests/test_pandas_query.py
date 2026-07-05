from app.services.pandas_query_service import ProductFilter, get_pandas_service


def test_price_filter():
    service = get_pandas_service()
    result = service.query(ProductFilter(max_price=5_000_000, limit=100))
    assert result["products"]
    assert all(p["effective_price"] <= 5_000_000 for p in result["products"])


def test_stock_filter_default():
    service = get_pandas_service()
    result = service.query(ProductFilter(limit=100))
    assert all(p["stock"] > 0 for p in result["products"])


def test_category_filter():
    service = get_pandas_service()
    category = service.list_categories()[0]
    result = service.query(ProductFilter(categories=[category], limit=100))
    assert result["products"]
    assert all(category in p["category"] for p in result["products"])


def test_relaxation_when_no_exact_match():
    service = get_pandas_service()
    # Impossible combo: real category but absurdly low budget -> relaxation
    category = service.list_categories()[0]
    result = service.query(ProductFilter(categories=[category], max_price=1, limit=10))
    assert not result["exact_match"] or not result["products"]


def test_empty_intermediate_filter_does_not_drop_columns():
    """Regression: boolean filter on empty DataFrame must not strip columns."""
    service = get_pandas_service()
    result = service.query(
        ProductFilter(
            max_price=810,
            brands=["دل"],
            categories=["لپ‌تاپ"],
            limit=10,
        )
    )
    assert isinstance(result["products"], list)


def test_sorting_price_asc():
    service = get_pandas_service()
    result = service.query(ProductFilter(sort_by="price_asc", limit=10))
    prices = [p["effective_price"] for p in result["products"]]
    assert prices == sorted(prices)
