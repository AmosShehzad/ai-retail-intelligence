"""
Day 6: Business Analytics Engine
Computes revenue, margins, inventory velocity, and store KPIs
directly from the SQLite database.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import logging
from database.db_manager import get_connection

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _load_sales_with_products() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            s.sale_id,
            s.product_id,
            s.quantity,
            s.sale_date,
            s.sale_price,
            p.product_name,
            p.category,
            p.cost_price,
            p.selling_price
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        WHERE p.is_active = 1
    """, conn)
    conn.close()

    df["sale_date"] = pd.to_datetime(df["sale_date"])

    # Use sale_price (historical) not selling_price (current)
    # This ensures revenue doesn't change if you update a product's price
    df["revenue"] = df["quantity"] * df["sale_price"]
    df["cost"]    = df["quantity"] * df["cost_price"]
    df["profit"]  = df["revenue"] - df["cost"]
    return df

# ── 1. Periodic Sales Totals ──────────────────────────────────────────────────
def get_revenue_by_period(period: str = "D") -> pd.DataFrame:
    """
    Aggregates revenue, cost, profit by time period.
    period: 'D' = daily, 'W' = weekly, 'M' = monthly
    """
    df = _load_sales_with_products()

    grouped = df.resample(period, on="sale_date").agg(
        total_revenue   = ("revenue", "sum"),
        total_cost      = ("cost", "sum"),
        total_profit    = ("profit", "sum"),
        units_sold      = ("quantity", "sum"),
        transactions    = ("sale_id", "count"),
    ).reset_index()

    grouped["profit_margin_pct"] = (
        grouped["total_profit"] / grouped["total_revenue"] * 100
    ).round(2)

    return grouped.round(2)

# ── 2. Category Margins ───────────────────────────────────────────────────────
def get_category_margins() -> pd.DataFrame:
    """
    Returns profit margin %, revenue, and units sold per category.
    Answers: 'Which category contributes most to revenue?'
             'Which category has the best margins?'
    """
    df = _load_sales_with_products()

    grouped = df.groupby("category").agg(
        total_revenue = ("revenue", "sum"),
        total_profit  = ("profit", "sum"),
        units_sold    = ("quantity", "sum"),
    ).reset_index()

    grouped["margin_pct"] = (
        grouped["total_profit"] / grouped["total_revenue"] * 100
    ).round(2)

    grouped["revenue_share_pct"] = (
        grouped["total_revenue"] / grouped["total_revenue"].sum() * 100
    ).round(2)

    return grouped.sort_values("total_revenue", ascending=False).round(2)

# ── 3. Product-Level Velocity ─────────────────────────────────────────────────
def get_product_velocity(top_n: int = 10) -> dict:
    """
    Returns top-selling and slow-moving (dead stock candidate) products.
    'Velocity' = how fast a product sells (units/day over its sales history).
    """
    df = _load_sales_with_products()

    product_stats = df.groupby(["product_id", "product_name", "category"]).agg(
        total_units_sold = ("quantity", "sum"),
        total_revenue    = ("revenue", "sum"),
        days_active       = ("sale_date", lambda x: int((x.max() - x.min()).days) + 1),
    ).reset_index()

    # Force days_active to a clean numeric type — on some pandas versions,
    # this named-agg lambda over a datetime column can be inferred as a
    # datetime-typed array instead of plain integers, which breaks division.
    product_stats["days_active"] = pd.to_numeric(
        product_stats["days_active"], errors="coerce"
    ).fillna(1).astype(int)

    product_stats["units_per_day"] = (
        product_stats["total_units_sold"] / product_stats["days_active"]
    ).round(2)
    
    top_sellers = product_stats.sort_values("units_per_day", ascending=False).head(top_n)
    slow_movers = product_stats.sort_values("units_per_day", ascending=True).head(top_n)

    return {
        "top_sellers": top_sellers.round(2),
        "slow_movers": slow_movers.round(2),
    }

# ── 4. Store-Level KPIs (single summary dict) ─────────────────────────────────
def get_store_kpis() -> dict:
    """
    Returns a single dictionary of headline store metrics.
    This is what the dashboard's top KPI cards will show.
    """
    df = _load_sales_with_products()

    total_revenue = df["revenue"].sum()
    total_profit  = df["profit"].sum()
    total_units   = df["quantity"].sum()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT
        COUNT(*),
        SUM(stock),
        SUM(stock * selling_price)
    FROM products
    WHERE is_active = 1
    """)
    product_count, total_stock_units, inventory_value = cursor.fetchone()
    conn.close()

    return {
        "total_revenue"        : round(total_revenue, 2),
        "total_profit"         : round(total_profit, 2),
        "overall_margin_pct"   : round(total_profit / total_revenue * 100, 2) if total_revenue else 0,
        "total_units_sold"     : int(total_units),
        "total_products"       : int(product_count),
        "total_stock_units"    : int(total_stock_units or 0),
        "current_inventory_value": round(inventory_value or 0, 2),
        "avg_order_value"      : round(df["revenue"].sum() / df["sale_id"].nunique(), 2) if df["sale_id"].nunique() > 0 else 0.0,
    }

# ── Run all (for manual testing) ──────────────────────────────────────────────
def run_all_analytics():
    log.info("=" * 50)
    log.info("ANALYTICS ENGINE — TEST RUN")
    log.info("=" * 50)

    print("\n=== STORE KPIs ===")
    for k, v in get_store_kpis().items():
        print(f"  {k}: {v}")

    print("\n=== DAILY REVENUE (last 5 days) ===")
    print(get_revenue_by_period("D").tail(5).to_string(index=False))

    print("\n=== CATEGORY MARGINS ===")
    print(get_category_margins().to_string(index=False))

    velocity = get_product_velocity(5)
    print("\n=== TOP 5 SELLERS ===")
    print(velocity["top_sellers"][["product_name", "units_per_day", "total_revenue"]].to_string(index=False))

    print("\n=== TOP 5 SLOW MOVERS ===")
    print(velocity["slow_movers"][["product_name", "units_per_day", "total_revenue"]].to_string(index=False))


if __name__ == "__main__":
    run_all_analytics()

