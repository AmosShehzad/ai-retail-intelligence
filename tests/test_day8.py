from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["database"] == "connected"


def test_analytics_router_alive():
    r = client.get("/api/v1/analytics/health-check")
    assert r.status_code == 200


def test_inventory_router_alive():
    r = client.get("/api/v1/inventory/health-check")
    assert r.status_code == 200


def test_rag_router_alive():
    r = client.get("/api/v1/rag/health-check")
    assert r.status_code == 200


def test_404_returns_json():
    r = client.get("/nonexistent-route")
    assert r.status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")