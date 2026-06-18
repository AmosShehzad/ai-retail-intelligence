import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app import app

client = TestClient(app)


def test_kpis_response_shape():
    r = client.get("/api/v1/analytics/kpis")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    # Verify all required fields exist and are correct types
    assert isinstance(data["total_revenue"], float)
    assert isinstance(data["total_products"], int)
    assert isinstance(data["overall_margin_pct"], float)


def test_revenue_response_shape():
    r = client.get("/api/v1/analytics/revenue?period=D")
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    assert isinstance(body["data"], list)


def test_invalid_period_rejected():
    r = client.get("/api/v1/analytics/revenue?period=INVALID")
    assert r.status_code == 400


def test_rag_empty_question_rejected():
    r = client.post("/api/v1/rag/ask", json={"question": "  "})
    assert r.status_code == 422


def test_rag_question_too_short_rejected():
    r = client.post("/api/v1/rag/ask", json={"question": "Hi"})
    assert r.status_code == 422


def test_rag_valid_question_accepted():
    r = client.post(
        "/api/v1/rag/ask",
        json={"question": "What are my best selling products?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "answer" in body
    assert isinstance(body["sources"], list)


def test_top_k_out_of_range_rejected():
    r = client.post(
        "/api/v1/rag/ask",
        json={"question": "What are my top products?", "top_k": 999}
    )
    assert r.status_code == 422


def test_dead_stock_response_shape():
    r = client.get("/api/v1/inventory/dead-stock")
    assert r.status_code == 200
    body = r.json()
    assert "dead_stock_count" in body
    assert "total_capital_locked" in body
    assert isinstance(body["data"], list)


def test_alerts_urgency_values():
    r = client.get("/api/v1/inventory/alerts")
    assert r.status_code == 200
    body = r.json()
    for item in body["data"]:
        assert item["urgency"] in ("CRITICAL", "HIGH", "MEDIUM")


def test_restock_response_shape():
    r = client.get("/api/v1/inventory/restock")
    assert r.status_code == 200
    body = r.json()
    assert "total_estimated_cost_pkr" in body


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")