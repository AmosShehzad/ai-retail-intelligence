import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from analytics.engines import (
    get_revenue_by_period, get_category_margins,
    get_product_velocity, get_store_kpis
)

assert get_store_kpis()["total_revenue"] > 0, "Revenue should not be zero"
assert len(get_category_margins()) > 0, "Categories should exist"
assert "top_sellers" in get_product_velocity(), "Velocity dict structure broken"
print("✅ All Day 6 checks passed.")