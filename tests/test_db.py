import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.queries import (
    get_top_selling_products,
    get_low_stock_products,
    get_profit_margins,
    get_daily_revenue,
)

print("\n=== TOP 5 SELLING PRODUCTS ===")
for p in get_top_selling_products(5):
    print(f"  {p['product_name']}: {p['total_sold']} units | Rs.{p['total_revenue']}")

print("\n=== LOW STOCK ALERT (≤20 units) ===")
for p in get_low_stock_products(20):
    print(f"  {p['product_name']}: {p['stock']} remaining")

print("\n=== TOP 5 HIGHEST MARGIN PRODUCTS ===")
for p in get_profit_margins()[:5]:
    print(f"  {p['product_name']}: {p['margin_pct']}% margin")

print("\n=== LAST 7 DAYS REVENUE ===")
for d in get_daily_revenue(7):
    print(f"  {d['sale_date']}: Rs.{d['daily_revenue']}")