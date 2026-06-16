"""
Day 9: Analytics API Endpoints

Exposes Day 6 business analytics engine over HTTP.
All endpoints return JSON. All are async.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from analytics.engines import (
    get_revenue_by_period,
    get_category_margins,
    get_product_velocity,
    get_store_kpis,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ── 1. Store KPIs ─────────────────────────────────────────────────────────────
@router.get("/kpis")
async def store_kpis():
    """
    Returns headline store metrics in one call.
    
    Why: Streamlit dashboard's TOP CARD ROW shows these —
    total revenue, profit %, inventory value, units sold.
    One endpoint = one API call = fast dashboard load.
    
    RAG also calls this for queries like:
    "Give me a store overview" / "How is business doing?"
    """
    try:
        data = get_store_kpis()
        return {
            "success": True,
            "data"   : data,
        }
    except Exception as e:
        log.exception("Failed to fetch KPIs")
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Revenue by Period ──────────────────────────────────────────────────────
@router.get("/revenue")
async def revenue_by_period(
    period: str = Query(
        default="D",
        description="Aggregation period: D=daily, W=weekly, M=monthly"
    )
):
    """
    Returns revenue/profit/cost aggregated by time period.
    
    Why: Powers the revenue trend LINE CHART on the dashboard.
    - period=D → daily chart (last 60 days)
    - period=W → weekly chart
    - period=M → monthly summary
    
    Query param example: GET /api/v1/analytics/revenue?period=W
    """
    if period not in ("D", "W", "M"):
        raise HTTPException(
            status_code=400,
            detail="Invalid period. Use D (daily), W (weekly), or M (monthly)."
        )

    try:
        df   = get_revenue_by_period(period)

        # Convert DataFrame to list of dicts for JSON serialization
        # DataFrames cannot be sent as JSON directly
        data = df.to_dict(orient="records")

        # Convert date objects to strings
        # Why: Python date objects aren't JSON serializable
        for row in data:
            if "sale_date" in row and row["sale_date"] is not None:
                row["sale_date"] = str(row["sale_date"])

        return {
            "success": True,
            "period" : period,
            "count"  : len(data),
            "data"   : data,
        }
    except Exception as e:
        log.exception("Failed to fetch revenue for period=%s", period)
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Category Margins ───────────────────────────────────────────────────────
@router.get("/margins")
async def category_margins():
    """
    Returns profit margin % and revenue share per product category.
    
    Why: Powers the PIE CHART / BAR CHART on the dashboard showing
    which categories make the most money.
    
    RAG uses this for:
    "Which category has the best margins?"
    "What product type should I focus on?"
    """
    try:
        df   = get_category_margins()
        data = df.to_dict(orient="records")
        return {
            "success"       : True,
            "total_categories": len(data),
            "data"          : data,
        }
    except Exception as e:
        log.exception("Failed to fetch category margins")
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Product Velocity ───────────────────────────────────────────────────────
@router.get("/velocity")
async def product_velocity(
    top_n: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of top/slow products to return"
    )
):
    """
    Returns top-selling and slow-moving products.
    
    Why: Powers TWO dashboard tables:
    - "Best Sellers" table (top_sellers)
    - "Slow Movers" table (slow_movers)
    
    RAG uses this for:
    "What are my best selling products?"
    "Which products are not moving?"
    
    ge=1, le=100 → FastAPI auto-validates top_n is between 1 and 100
    No need for manual if/else checks.
    """
    try:
        result = get_product_velocity(top_n)

        return {
            "success"    : True,
            "top_n"      : top_n,
            "top_sellers": result["top_sellers"].to_dict(orient="records"),
            "slow_movers": result["slow_movers"].to_dict(orient="records"),
        }
    except Exception as e:
        log.exception("Failed to fetch product velocity")
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Health check (keeps Day 8 skeleton, now improved) ─────────────────────
@router.get("/status")
async def analytics_status():
    """
    Confirms analytics engine is functional.
    Runs a lightweight KPI call to verify DB + engine are working.
    """
    try:
        kpis = get_store_kpis()
        return {
            "status"        : "online",
            "products_found": kpis.get("total_products", 0),
            "engine"        : "analytics",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Analytics engine error: {e}")