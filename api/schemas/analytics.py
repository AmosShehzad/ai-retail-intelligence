"""
Pydantic schemas for analytics endpoints.

Each schema has two jobs:
1. INPUT validation — checks what the client sends is correct
2. OUTPUT validation — guarantees what the client receives is correct

Field(...) = required field (no default)
Field(default=...) = optional field with a default value
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import date


# ── Store KPI Schema ───────────────────────────────────────────────────────────
class StoreKPISchema(BaseModel):
    """
    Shape of the /analytics/kpis response data.
    
    Every field has a type. If get_store_kpis() returns
    a string for total_revenue, Pydantic raises an error
    immediately instead of letting it silently break the dashboard.
    """
    total_revenue            : float = Field(..., description="Total revenue in PKR", ge=0)
    total_profit             : float = Field(..., description="Total profit in PKR")
    overall_margin_pct       : float = Field(..., description="Overall profit margin percentage")
    total_units_sold         : int   = Field(..., description="Total units sold", ge=0)
    total_products           : int   = Field(..., description="Total products in catalog", ge=0)
    total_stock_units        : int   = Field(..., description="Total units currently in stock", ge=0)
    current_inventory_value  : float = Field(..., description="Current inventory value in PKR", ge=0)
    avg_order_value          : float = Field(..., description="Average order value in PKR", ge=0)

    class Config:
        from_attributes = True


# ── Revenue Period Schema ──────────────────────────────────────────────────────
class RevenuePeriodRow(BaseModel):
    """
    One row of revenue data (one day/week/month).
    
    Optional[date] because some rows might have null dates
    after aggregation — Pydantic handles this gracefully.
    """
    sale_date         : Optional[str]   = None
    total_revenue     : float           = Field(0.0, ge=0)
    total_cost        : float           = Field(0.0, ge=0)
    total_profit      : float           = Field(0.0)
    units_sold        : int             = Field(0, ge=0)
    transactions      : int             = Field(0, ge=0)
    profit_margin_pct : Optional[float] = None

    class Config:
        from_attributes = True


class RevenueResponse(BaseModel):
    """
    Full response shape for /analytics/revenue endpoint.
    """
    success : bool
    period  : str
    count   : int
    data    : List[RevenuePeriodRow]

    class Config:
        from_attributes = True


# ── Category Margin Schema ─────────────────────────────────────────────────────
class CategoryMarginSchema(BaseModel):
    """
    One category's margin and revenue data.
    Used by the pie/bar chart on the dashboard.
    """
    category          : str
    total_revenue     : float = Field(0.0, ge=0)
    total_profit      : float = Field(0.0)
    units_sold        : int   = Field(0, ge=0)
    margin_pct        : float = Field(0.0)
    revenue_share_pct : float = Field(0.0)

    class Config:
        from_attributes = True


# ── Product Velocity Schema ────────────────────────────────────────────────────
class ProductVelocitySchema(BaseModel):
    """
    One product's velocity metrics.
    Used by best-sellers and slow-movers tables.
    """
    product_id       : Optional[int]   = None
    product_name     : str
    category         : str
    total_units_sold : int             = Field(0, ge=0)
    total_revenue    : float           = Field(0.0, ge=0)
    days_active      : Optional[int]   = None
    units_per_day    : float           = Field(0.0, ge=0)

    class Config:
        from_attributes = True


class VelocityResponse(BaseModel):
    """Full response for /analytics/velocity"""
    success     : bool
    top_n       : int
    top_sellers : List[ProductVelocitySchema]
    slow_movers : List[ProductVelocitySchema]

    class Config:
        from_attributes = True


# ── Query parameter validators ─────────────────────────────────────────────────
class RevenuePeriodParams(BaseModel):
    """
    Validates the ?period= query parameter.
    
    @field_validator runs custom logic beyond basic type checking.
    Without this, someone could send period=INVALID and get a 500 error.
    With this, they get a clean 422 validation error with a helpful message.
    """
    period: str = "D"

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        allowed = {"D", "W", "M"}
        if v.upper() not in allowed:
            raise ValueError(f"period must be one of {allowed}. Got: '{v}'")
        return v.upper()