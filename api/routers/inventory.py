"""
Day 9: Inventory API Endpoints

Exposes Day 7 inventory intelligence engine over HTTP.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from analytics.inventory import (
    get_dead_stock,
    get_low_stock_alerts,
    get_restock_recommendations,
    get_inventory_health_summary,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory", tags=["Inventory"])


# ── 1. Inventory Health Summary ───────────────────────────────────────────────
@router.get("/health")
async def inventory_health():
    """
    Returns complete inventory health picture in one call.
    
    Why: Streamlit dashboard INVENTORY CARD shows:
    - How many products are dead stock
    - How many are critically low
    - Total capital locked in dead stock
    - Estimated restock cost
    
    RAG uses this for:
    "What is the state of my inventory?"
    "How much money is stuck in dead stock?"
    """
    try:
        data = get_inventory_health_summary()
        return {
            "success": True,
            "data"   : data,
        }
    except Exception as e:
        log.exception("Failed to fetch inventory health")
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Dead Stock ─────────────────────────────────────────────────────────────
@router.get("/dead-stock")
async def dead_stock(
    days_window: int = Query(
        default=30,
        ge=7,
        le=365,
        description="Sales lookback window in days"
    ),
    min_units: int = Query(
        default=3,
        ge=0,
        description="Max units sold to qualify as dead stock"
    )
):
    """
    Returns products with zero or near-zero sales velocity.
    
    Why: Store owner needs to know what's wasting shelf space
    and locking up capital. This answers:
    "Which products should I stop buying / discount / return?"
    
    Parameterized so the frontend can let users adjust
    the definition of "dead" stock dynamically.
    """
    try:
        df   = get_dead_stock(
            days_window         = days_window,
            min_units_threshold = min_units
        )
        data = df.to_dict(orient="records")

        return {
            "success"             : True,
            "days_window"         : days_window,
            "dead_stock_count"    : len(data),
            "total_capital_locked": round(
                sum(r.get("capital_locked_pkr", 0) for r in data), 2
            ),
            "data"                : data,
        }
    except Exception as e:
        log.exception("Failed to fetch dead stock")
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Low Stock Alerts ───────────────────────────────────────────────────────
@router.get("/alerts")
async def low_stock_alerts(
    days_window: int = Query(default=30, ge=7, le=365),
    alert_threshold: int = Query(
        default=7,
        ge=1,
        le=30,
        description="Days of stock remaining to trigger alert"
    )
):
    """
    Returns products that will run out within alert_threshold days.
    
    Why: Critical for a kiryana store — running out of Tapal Tea
    or Surf Excel means immediate lost sales and unhappy customers.
    
    Urgency levels returned (CRITICAL / HIGH / MEDIUM) let the
    dashboard show color-coded alerts — red/orange/yellow.
    
    RAG uses this for:
    "What will I run out of this week?"
    "Which products need urgent restocking?"
    """
    try:
        df   = get_low_stock_alerts(
            days_window           = days_window,
            alert_days_threshold  = alert_threshold
        )

        if df.empty:
            return {
                "success": True,
                "message": "All products are sufficiently stocked.",
                "data"   : [],
            }

        data = df.to_dict(orient="records")

        # Group by urgency for quick frontend summary
        urgency_summary = {
            "CRITICAL": len([r for r in data if r["urgency"] == "CRITICAL"]),
            "HIGH"    : len([r for r in data if r["urgency"] == "HIGH"]),
            "MEDIUM"  : len([r for r in data if r["urgency"] == "MEDIUM"]),
        }

        return {
            "success"        : True,
            "days_window"    : days_window,
            "alert_threshold": alert_threshold,
            "urgency_summary": urgency_summary,
            "total_alerts"   : len(data),
            "data"           : data,
        }
    except Exception as e:
        log.exception("Failed to fetch low stock alerts")
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Restock Recommendations ────────────────────────────────────────────────
@router.get("/restock")
async def restock_recommendations(
    days_window: int = Query(default=30, ge=7, le=365)
):
    """
    Returns prioritized shopping list of what to restock.
    
    Why: This is the most actionable endpoint in the whole API.
    The store owner opens the dashboard Monday morning,
    clicks "Restock List", and gets exactly what to order
    from which supplier and how much it costs.
    
    RAG uses this for:
    "What should I buy this week?"
    "Give me a purchase order for tomorrow."
    """
    try:
        df = get_restock_recommendations(days_window)

        if df.empty:
            return {
                "success": True,
                "message": "No restocking needed right now.",
                "data"   : [],
            }

        data              = df.to_dict(orient="records")
        total_restock_cost = round(
            sum(r.get("estimated_reorder_cost_pkr", 0) for r in data), 2
        )

        return {
            "success"                  : True,
            "items_to_restock"         : len(data),
            "total_estimated_cost_pkr" : total_restock_cost,
            "data"                     : data,
        }
    except Exception as e:
        log.exception("Failed to fetch restock recommendations")
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Products list with full details ────────────────────────────────────────
@router.get("/products")
async def get_all_products(
    category: Optional[str] = Query(
        default=None,
        description="Filter by category name"
    ),
    low_stock_only: bool = Query(
        default=False,
        description="Return only low stock products"
    )
):
    """
    Returns all products with current stock levels.
    Optional filters: by category, or only low stock items.
    
    Why: Dashboard's product inventory table uses this.
    Also used by the competitor analysis page (Day scraper)
    to show our price vs competitor price side by side.
    """
    try:
        from database.db_manager import get_connection
        import pandas as pd

        conn = get_connection()
        df   = pd.read_sql_query("""
            SELECT
                product_id, product_name, category,
                cost_price, selling_price, stock, supplier,
                ROUND((selling_price - cost_price) * 100.0 / selling_price, 2)
                    AS margin_pct
            FROM products
            ORDER BY category, product_name
        """, conn)
        conn.close()

        # Apply filters
        if category:
            df = df[df["category"].str.lower() == category.lower()]

        if low_stock_only:
            df = df[df["stock"] <= 10]

        data = df.to_dict(orient="records")

        return {
            "success" : True,
            "count"   : len(data),
            "filters" : {
                "category"      : category,
                "low_stock_only": low_stock_only,
            },
            "data"    : data,
        }
    except Exception as e:
        log.exception("Failed to fetch products")
        raise HTTPException(status_code=500, detail=str(e))
