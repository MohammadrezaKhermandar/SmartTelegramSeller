from app.services.product_loader import get_catalog


def test_catalog_loads_and_cleans():
    catalog = get_catalog()
    assert len(catalog.df) > 0
    # required canonical columns exist
    for col in ("product_id", "name", "price", "effective_price", "search_text"):
        assert col in catalog.df.columns
    # no invalid prices survived cleaning
    assert (catalog.df["price"] > 0).all()
    # ids unique
    assert catalog.df["product_id"].is_unique


def test_search_text_contains_key_fields():
    catalog = get_catalog()
    row = catalog.df.iloc[0]
    assert row["name"] in row["search_text"]
    assert "قیمت" in row["search_text"]


def test_get_by_id():
    catalog = get_catalog()
    pid = catalog.df.iloc[0]["product_id"]
    product = catalog.get_by_id(pid)
    assert product is not None
    assert product["product_id"] == pid
    assert catalog.get_by_id("nonexistent-id-xyz") is None
