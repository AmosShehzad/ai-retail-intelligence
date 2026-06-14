import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from analytics.inventory import (
    get_dead_stock,
    get_low_stock_alerts,
    get_restock_recommendations,
    get_inventory_health_summary,
)

summary = get_inventory_health_summary()
assert isinstance(summary, dict), "Summary must be a dict"
assert "total_products" in summary, "Missing total_products key"
assert summary["total_products"] > 0, "No products found"

dead = get_dead_stock()
assert hasattr(dead, "columns"), "Dead stock must return DataFrame"

alerts = get_low_stock_alerts()
if not alerts.empty:
    assert "urgency" in alerts.columns, "Missing urgency column"
    assert set(alerts["urgency"]).issubset({"CRITICAL","HIGH","MEDIUM"}), \
        "Invalid urgency values"

print("✅ All Day 7 checks passed.")