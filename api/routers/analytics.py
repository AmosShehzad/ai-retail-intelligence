"""
Day 10 update: analytics router now uses Pydantic response_model.
response_model tells FastAPI to validate AND document the output shape.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Query

from analytics.engines import (
    get_revenue_by_period,
    get_category_margins,
    get_product_velocity,
    get_store_kpis,
)
from api.schemas.analytics import (
    StoreKPISchema,
    RevenuePeriodRow,
    CategoryMarginSchema,
    ProductVelocitySchema,
    VelocityResponse,
    RevenueResponse,
)
from api.schemas.common import BaseResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/kpis", response_model=BaseResponse[StoreKPISchema])
async def store_kpis():
    try:
        data = get_store_kpis()
        return BaseResponse(success=True, data=StoreKPISchema(**data))
    except Exception as e:
        log.exception("KPI fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/revenue", response_model=RevenueResponse)
async def revenue_by_period(
    period: str = Query(default="D", description="D=daily W=weekly M=monthly")
):
    if period not in ("D", "W", "M"):
        raise HTTPException(status_code=400, detail="period must be D, W, or M")
    try:
        df   = get_revenue_by_period(period)
        rows = []
        for row in df.to_dict(orient="records"):
            sale_date = row.get("sale_date")
            if sale_date is not None:
                row["sale_date"] = sale_date.date().isoformat() if hasattr(sale_date, "date") else str(sale_date)
            rows.append(RevenuePeriodRow(**row))
        return RevenueResponse(success=True, period=period, count=len(rows), data=rows)
    except Exception as e:
        log.exception("Revenue fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/margins", response_model=BaseResponse[List[CategoryMarginSchema]])
async def category_margins():
    try:
        df   = get_category_margins()
        data = [CategoryMarginSchema(**r) for r in df.to_dict(orient="records")]
        return BaseResponse(success=True, data=data)
    except Exception as e:
        log.exception("Margins fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/velocity", response_model=VelocityResponse)
async def product_velocity(
    top_n: int = Query(default=10, ge=1, le=100)
):
    try:
        result = get_product_velocity(top_n)
        return VelocityResponse(
            success     = True,
            top_n       = top_n,
            top_sellers = [ProductVelocitySchema(**r)
                           for r in result["top_sellers"].to_dict(orient="records")],
            slow_movers = [ProductVelocitySchema(**r)
                           for r in result["slow_movers"].to_dict(orient="records")],
        )
    except Exception as e:
        log.exception("Velocity fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def analytics_status():
    try:
        kpis = get_store_kpis()
        return {"status": "online", "products_found": kpis.get("total_products", 0)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))