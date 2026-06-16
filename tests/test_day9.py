from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_kpis():
    r = client.get("/api/v1/analytics/kpis")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert "total_revenue" in r.json()["data"]


def test_revenue_daily():
    r = client.get("/api/v1/analytics/revenue?period=D")
    assert r.status_code == 200
    assert r.json()["period"] == "D"


def test_revenue_invalid_period():
    r = client.get("/api/v1/analytics/revenue?period=X")
    assert r.status_code == 400


def test_margins():
    r = client.get("/api/v1/analytics/margins")
    assert r.status_code == 200
    assert isinstance(r.json()["data"], list)


def test_velocity():
    r = client.get("/api/v1/analytics/velocity?top_n=5")
    assert r.status_code == 200
    assert "top_sellers" in r.json()
    assert "slow_movers" in r.json()


def test_inventory_health():
    r = client.get("/api/v1/inventory/health")
    assert r.status_code == 200
    assert "total_products" in r.json()["data"]


def test_dead_stock():
    r = client.get("/api/v1/inventory/dead-stock")
    assert r.status_code == 200
    assert "dead_stock_count" in r.json()


def test_alerts():
    r = client.get("/api/v1/inventory/alerts")
    assert r.status_code == 200


def test_restock():
    r = client.get("/api/v1/inventory/restock")
    assert r.status_code == 200


def test_products_all():
    r = client.get("/api/v1/inventory/products")
    assert r.status_code == 200
    assert r.json()["count"] > 0


def test_products_filtered():
    r = client.get("/api/v1/inventory/products?low_stock_only=true")
    assert r.status_code == 200


def test_rag_placeholder():
    r = client.post(
        "/api/v1/rag/ask",
        json={"question": "What are my top products?"}
    )
    assert r.status_code == 200
    assert r.json()["success"] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")