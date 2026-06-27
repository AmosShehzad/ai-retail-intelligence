"""Day 17 tests — validates data loaders work without Streamlit running."""

def test_kpi_loader():
    from analytics.engines import get_store_kpis
    kpis = get_store_kpis()
    assert isinstance(kpis, dict)
    assert "total_revenue" in kpis
    assert kpis["total_revenue"] > 0


def test_category_loader():
    from analytics.engines import get_category_margins
    df = get_category_margins()
    assert len(df) > 0
    assert "category" in df.columns
    assert "margin_pct" in df.columns


def test_inventory_health_loader():
    from analytics.inventory import get_inventory_health_summary
    health = get_inventory_health_summary()
    assert isinstance(health, dict)
    assert "total_products" in health


def test_fmt_pkr():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from frontend.dashboard import fmt_pkr
    assert fmt_pkr(1_500_000) == "Rs. 1.5M"
    assert fmt_pkr(45_000)    == "Rs. 45K"
    assert fmt_pkr(500)       == "Rs. 500"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")