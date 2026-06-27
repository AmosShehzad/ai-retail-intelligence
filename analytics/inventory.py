"""
Day 7: Advanced Inventory Intelligence Module

Three core engines:
1. Dead/Stagnant Stock Detector
2. Low Stock Alert System  
3. Restock Recommendation Engine

All functions read from SQLite directly.
Output is clean DataFrames ready for FastAPI + RAG.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import logging
from datetime import datetime, timedelta
from database.db_manager import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Shared loader ──────────────────────────────────────────────────────────────
def _load_inventory_base(days_window: int = 30) -> pd.DataFrame:
    """
    Loads all products joined with their sales data
    for the last N days into one DataFrame.

    Why: Every inventory function below needs current stock
    AND recent sales together. Load once, reuse everywhere.

    columns returned:
    product_id, product_name, category, stock,
    cost_price, selling_price, supplier,
    units_sold_in_window, revenue_in_window
    """
    conn = get_connection()

    cutoff_date = (datetime.now() - timedelta(days=days_window)).strftime("%Y-%m-%d")

    df = pd.read_sql_query("""
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        p.stock          AS current_stock,
        p.cost_price,
        p.selling_price,
        p.supplier,
        p.low_stock_threshold,
        COALESCE(SUM(s.quantity), 0)                   AS units_sold,
        COALESCE(SUM(s.quantity * p.selling_price), 0) AS revenue
    FROM products p
    LEFT JOIN sales s
        ON p.product_id = s.product_id
        AND s.sale_date >= ?
    WHERE p.is_active = 1
    GROUP BY p.product_id
""", conn, params=(cutoff_date,))

    conn.close()

    # Daily and weekly velocity — how many units sold per day/week on average
    # Why: "units sold in 30 days / 30" = daily burn rate
    df["daily_velocity"]  = (df["units_sold"] / days_window).round(4)
    df["weekly_velocity"] = (df["daily_velocity"] * 7).round(2)

    # Days of stock remaining at current burn rate
    # Why: stock=50, daily_velocity=5 → 10 days left before stockout
    df["days_of_stock_remaining"] = df.apply(
        lambda r: round(r["current_stock"] / r["daily_velocity"], 1)
        if r["daily_velocity"] > 0 else 999,  # 999 = infinite (not selling)
        axis=1
    )

    return df


# ── Engine 1: Dead / Stagnant Stock Detector ──────────────────────────────────
def get_dead_stock(
    days_window: int = 30,
    min_units_threshold: int = 3,
    min_stock: int = 1
) -> pd.DataFrame:
    """
    Identifies products that are NOT selling — dead stock.

    Logic:
    - Look at last 30 days of sales
    - If a product sold fewer than `min_units_threshold` units (default=3)
      AND still has stock sitting on the shelf → dead stock
    - min_stock=1 ensures we only flag products that actually have stock
      (no point flagging already-zero stock items)

    Why this threshold:
    - 3 units in 30 days = 0.1 units/day = basically not moving
    - Adjust based on category (spices may sell slower than tea)

    Returns DataFrame sorted by stock value locked in dead stock (worst first)
    """
    df = _load_inventory_base(days_window)

    dead = df[
        (df["units_sold"] <= min_units_threshold) &
        (df["current_stock"] >= min_stock)
    ].copy()

    # Capital locked = money sitting in unsold inventory
    # Why: shows the COST of inaction. Rs.50,000 locked in dead stock
    # is a real business number, not just "X items aren't selling"
    dead["capital_locked_pkr"] = (dead["current_stock"] * dead["cost_price"]).round(2)

    dead["status"] = "DEAD STOCK"

    dead = dead.sort_values("capital_locked_pkr", ascending=False)

    result = dead[[
        "product_id", "product_name", "category",
        "current_stock", "units_sold", "daily_velocity",
        "capital_locked_pkr", "supplier", "status"
    ]].reset_index(drop=True)

    log.info("Dead stock identified: %d products | Rs.%.0f capital locked",
             len(result), result["capital_locked_pkr"].sum())
    return result


# ── Engine 2: Low Stock Alert System ──────────────────────────────────────────
def get_low_stock_alerts(
    days_window: int = 30,
    alert_days_threshold: int = 7,
    absolute_low_threshold: int = 10
) -> pd.DataFrame:
    """
    Identifies products that WILL run out soon — low stock alerts.

    Two conditions trigger an alert (either one is enough):
    1. Velocity-based: days_of_stock_remaining <= alert_days_threshold (7 days)
       Example: stock=20, selling 4/day → 5 days left → ALERT
    2. Absolute-based: current_stock <= absolute_low_threshold (10 units)
       Example: stock=8 regardless of velocity → ALERT
       Why: catches slow sellers that are still dangerously low

    Urgency levels:
    - CRITICAL: <= 3 days of stock
    - HIGH:     <= 7 days of stock
    - MEDIUM:   absolute stock <= 10 but more than 7 days of stock
    """
    df = _load_inventory_base(days_window)

    # Only flag products that are actually selling
    # (no point alerting on dead stock — different problem)
    selling = df[df["daily_velocity"] > 0].copy()

    velocity_alert  = selling["days_of_stock_remaining"] <= alert_days_threshold
    absolute_alert = selling["current_stock"] <= selling["low_stock_threshold"]
    low_stock       = selling[velocity_alert | absolute_alert].copy()

    def assign_urgency(row):
        if row["days_of_stock_remaining"] <= 3:
            return "CRITICAL"
        elif row["days_of_stock_remaining"] <= 7:
            return "HIGH"
        else:
            return "MEDIUM"

    low_stock["urgency"] = low_stock.apply(assign_urgency, axis=1)

    # Suggested reorder quantity:
    # "Order enough to last 30 more days at current velocity"
    # minus what you already have
    low_stock["suggested_reorder_qty"] = (
        (low_stock["daily_velocity"] * 30) - low_stock["current_stock"]
    ).round().clip(lower=0).astype(int)

    low_stock["estimated_reorder_cost_pkr"] = (
        low_stock["suggested_reorder_qty"] * low_stock["cost_price"]
    ).round(2)

    low_stock = low_stock.sort_values(
        "days_of_stock_remaining", ascending=True
    )

    result = low_stock[[
        "product_id", "product_name", "category",
        "current_stock", "daily_velocity", "days_of_stock_remaining",
        "urgency", "suggested_reorder_qty",
        "estimated_reorder_cost_pkr", "supplier"
    ]].reset_index(drop=True)

    log.info(
        "Low stock alerts: %d products | CRITICAL=%d HIGH=%d MEDIUM=%d",
        len(result),
        len(result[result["urgency"] == "CRITICAL"]),
        len(result[result["urgency"] == "HIGH"]),
        len(result[result["urgency"] == "MEDIUM"])
    )
    return result


# ── Engine 3: Restock Recommendation Engine ───────────────────────────────────
def get_restock_recommendations(days_window: int = 30) -> pd.DataFrame:
    """
    Generates a prioritized restock shopping list for the store owner.

    Combines low stock alerts into an actionable purchase order:
    - What to buy
    - How much to buy
    - Who to buy from (supplier)
    - How much it will cost
    - Priority order (CRITICAL first)

    This is the direct answer to the RAG query:
    "What should I restock next week?"
    """
    alerts = get_low_stock_alerts(days_window)

    if alerts.empty:
        log.info("No restock needed — all products sufficiently stocked.")
        return pd.DataFrame()

    # Only recommend items where reorder qty > 0
    restock = alerts[alerts["suggested_reorder_qty"] > 0].copy()

    # Priority score for sorting:
    # CRITICAL=1, HIGH=2, MEDIUM=3
    urgency_order = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3}
    restock["priority_score"] = restock["urgency"].map(urgency_order)
    restock = restock.sort_values(["priority_score", "estimated_reorder_cost_pkr"],
                                   ascending=[True, False])

    result = restock[[
        "product_name", "category", "supplier",
        "current_stock", "suggested_reorder_qty",
        "estimated_reorder_cost_pkr", "urgency"
    ]].reset_index(drop=True)

    total_cost = result["estimated_reorder_cost_pkr"].sum()
    log.info("Restock list: %d items | Total estimated cost: Rs.%.0f",
             len(result), total_cost)
    return result


# ── Engine 4: Full Inventory Health Summary ───────────────────────────────────
def get_inventory_health_summary() -> dict:
    """
    Returns a single dictionary — the complete inventory health picture.
    Used by:
    - FastAPI /inventory/health endpoint
    - Streamlit dashboard KPI cards
    - RAG assistant for store overview queries

    Why a dict and not a DataFrame:
    These are scalar metrics (single numbers), not tabular data.
    A dict maps directly to JSON for the API response.
    """
    df           = _load_inventory_base(30)
    dead         = get_dead_stock()
    alerts       = get_low_stock_alerts()
    restock      = get_restock_recommendations()

    total_products      = len(df)
    total_stock_units   = int(df["current_stock"].sum())
    total_inventory_val = round((df["current_stock"] * df["cost_price"]).sum(), 2)
    dead_stock_count    = len(dead)
    dead_stock_value    = round(dead["capital_locked_pkr"].sum(), 2)
    critical_alerts     = len(alerts[alerts["urgency"] == "CRITICAL"]) if not alerts.empty else 0
    high_alerts         = len(alerts[alerts["urgency"] == "HIGH"]) if not alerts.empty else 0
    restock_cost        = round(restock["estimated_reorder_cost_pkr"].sum(), 2) if not restock.empty else 0

    summary = {
        "total_products"          : total_products,
        "total_stock_units"       : total_stock_units,
        "total_inventory_value_pkr": total_inventory_val,
        "dead_stock_products"     : dead_stock_count,
        "dead_stock_value_pkr"    : dead_stock_value,
        "low_stock_critical"      : critical_alerts,
        "low_stock_high"          : high_alerts,
        "restock_items_needed"    : len(restock) if not restock.empty else 0,
        "estimated_restock_cost_pkr": restock_cost,
        "generated_at"            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return summary


# ── Master test runner ────────────────────────────────────────────────────────
def run_inventory_intelligence():
    log.info("=" * 55)
    log.info("INVENTORY INTELLIGENCE ENGINE — FULL RUN")
    log.info("=" * 55)

    print("\n=== INVENTORY HEALTH SUMMARY ===")
    summary = get_inventory_health_summary()
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    print("\n=== DEAD STOCK (TOP 10) ===")
    dead = get_dead_stock()
    if dead.empty:
        print("  No dead stock detected.")
    else:
        print(dead.head(10)[[
            "product_name", "current_stock",
            "units_sold", "capital_locked_pkr"
        ]].to_string(index=False))

    print("\n=== LOW STOCK ALERTS ===")
    alerts = get_low_stock_alerts()
    if alerts.empty:
        print("  No low stock alerts.")
    else:
        print(alerts[[
            "product_name", "current_stock",
            "days_of_stock_remaining", "urgency",
            "suggested_reorder_qty"
        ]].to_string(index=False))

    print("\n=== RESTOCK RECOMMENDATION LIST ===")
    restock = get_restock_recommendations()
    if restock.empty:
        print("  Nothing to restock.")
    else:
        print(restock.to_string(index=False))
        print(f"\n  TOTAL ESTIMATED RESTOCK COST: "
              f"Rs.{restock['estimated_reorder_cost_pkr'].sum():,.0f}")


if __name__ == "__main__":
    run_inventory_intelligence()