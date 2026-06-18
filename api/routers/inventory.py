"""
Day 10 update: inventory router now uses Pydantic response_model.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from analytics.inventory import (
    get_dead_stock,
    get_low_stock_alerts,
    get_restock_recommendations,
    get_inventory_health_summary,
)
from api.schemas.inventory import (
    ProductSchema,
    DeadStockSchema,
    DeadStockResponse,
    LowStockAlertSchema,
    AlertResponse,
    UrgencySummary,
    RestockItemSchema,
    RestockResponse,
    InventoryHealthSchema,
)
from api.schemas.common import BaseResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.get("/health", response_model=BaseResponse[InventoryHealthSchema])
async def inventory_health():
    try:
        data = get_inventory_health_summary()
        return BaseResponse(success=True, data=InventoryHealthSchema(**data))
    except Exception as e:
        log.exception("Inventory health fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dead-stock", response_model=DeadStockResponse)
async def dead_stock(
    days_window : int = Query(default=30, ge=7, le=365),
    min_units   : int = Query(default=3, ge=0)
):
    try:
        df   = get_dead_stock(days_window=days_window, min_units_threshold=min_units)
        data = [DeadStockSchema(**r) for r in df.to_dict(orient="records")]
        return DeadStockResponse(
            success              = True,
            days_window          = days_window,
            dead_stock_count     = len(data),
            total_capital_locked = round(sum(r.capital_locked_pkr for r in data), 2),
            data                 = data,
        )
    except Exception as e:
        log.exception("Dead stock fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=AlertResponse)
async def low_stock_alerts(
    days_window     : int = Query(default=30, ge=7, le=365),
    alert_threshold : int = Query(default=7, ge=1, le=30)
):
    try:
        df = get_low_stock_alerts(
            days_window          = days_window,
            alert_days_threshold = alert_threshold
        )

        if df.empty:
            return AlertResponse(
                success         = True,
                days_window     = days_window,
                alert_threshold = alert_threshold,
                total_alerts    = 0,
                data            = [],
            )

        records = df.to_dict(orient="records")
        data    = [LowStockAlertSchema(**r) for r in records]

        return AlertResponse(
            success         = True,
            days_window     = days_window,
            alert_threshold = alert_threshold,
            urgency_summary = UrgencySummary(
                CRITICAL = len([r for r in data if r.urgency == "CRITICAL"]),
                HIGH     = len([r for r in data if r.urgency == "HIGH"]),
                MEDIUM   = len([r for r in data if r.urgency == "MEDIUM"]),
            ),
            total_alerts    = len(data),
            data            = data,
        )
    except Exception as e:
        log.exception("Alerts fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/restock", response_model=RestockResponse)
async def restock_recommendations(
    days_window: int = Query(default=30, ge=7, le=365)
):
    try:
        df = get_restock_recommendations(days_window)

        if df.empty:
            return RestockResponse(
                success                  = True,
                items_to_restock         = 0,
                total_estimated_cost_pkr = 0.0,
                data                     = [],
            )

        data = [RestockItemSchema(**r) for r in df.to_dict(orient="records")]
        return RestockResponse(
            success                  = True,
            items_to_restock         = len(data),
            total_estimated_cost_pkr = round(
                sum(r.estimated_reorder_cost_pkr for r in data), 2
            ),
            data                     = data,
        )
    except Exception as e:
        log.exception("Restock fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products", response_model=BaseResponse[List[ProductSchema]])
async def get_all_products(
    category       : Optional[str]  = Query(default=None),
    low_stock_only : bool            = Query(default=False)
):
    try:
        import pandas as pd
        from database.db_manager import get_connection

        conn = get_connection()
        df   = pd.read_sql_query("""
            SELECT product_id, product_name, category,
                   cost_price, selling_price, stock, supplier,
                   ROUND((selling_price - cost_price) * 100.0 / selling_price, 2)
                       AS margin_pct
            FROM products ORDER BY category, product_name
        """, conn)
        conn.close()

        if category:
            df = df[df["category"].str.lower() == category.lower()]
        if low_stock_only:
            df = df[df["stock"] <= 10]

        data = [ProductSchema(**r) for r in df.to_dict(orient="records")]
        return BaseResponse(success=True, data=data)
    except Exception as e:
        log.exception("Products fetch failed")
        raise HTTPException(status_code=500, detail=str(e))