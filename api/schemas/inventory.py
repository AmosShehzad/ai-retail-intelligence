"""
Pydantic schemas for inventory endpoints.
Validates all inventory intelligence data going in and out.
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Product Schema ─────────────────────────────────────────────────────────────
class ProductSchema(BaseModel):
    """
    Full product record from the database.
    Used by /inventory/products endpoint.
    
    margin_pct is Optional because Kaggle-sourced products
    might not have a cost_price set correctly.
    """
    product_id    : int
    product_name  : str
    category      : str
    cost_price    : float = Field(..., ge=0)
    selling_price : float = Field(..., ge=0)
    stock         : int   = Field(..., ge=0)
    supplier      : Optional[str] = None
    margin_pct    : Optional[float] = None

    class Config:
        from_attributes = True


# ── Dead Stock Schema ──────────────────────────────────────────────────────────
class DeadStockSchema(BaseModel):
    """
    One dead stock product record.
    capital_locked_pkr is the key business metric —
    it shows the real cost of holding dead inventory.
    """
    product_id         : int
    product_name       : str
    category           : str
    current_stock      : int   = Field(..., ge=0)
    units_sold         : int   = Field(0, ge=0)
    daily_velocity     : float = Field(0.0, ge=0)
    capital_locked_pkr : float = Field(..., ge=0)
    supplier           : Optional[str] = None
    status             : str   = "DEAD STOCK"

    class Config:
        from_attributes = True


class DeadStockResponse(BaseModel):
    """Full response for /inventory/dead-stock"""
    success              : bool
    days_window          : int
    dead_stock_count     : int
    total_capital_locked : float
    data                 : List[DeadStockSchema]

    class Config:
        from_attributes = True


# ── Low Stock Alert Schema ─────────────────────────────────────────────────────
class LowStockAlertSchema(BaseModel):
    """
    One low stock alert record.
    
    Literal["CRITICAL", "HIGH", "MEDIUM"] means Pydantic will
    REJECT any urgency value that isn't one of these three strings.
    This guarantees the frontend's color-coding logic never breaks.
    """
    product_id                : int
    product_name              : str
    category                  : str
    current_stock             : int   = Field(..., ge=0)
    daily_velocity            : float = Field(..., ge=0)
    days_of_stock_remaining   : float = Field(..., ge=0)
    urgency                   : Literal["CRITICAL", "HIGH", "MEDIUM"]
    suggested_reorder_qty     : int   = Field(..., ge=0)
    estimated_reorder_cost_pkr: float = Field(..., ge=0)
    supplier                  : Optional[str] = None

    class Config:
        from_attributes = True


class UrgencySummary(BaseModel):
    CRITICAL : int = 0
    HIGH     : int = 0
    MEDIUM   : int = 0


class AlertResponse(BaseModel):
    """Full response for /inventory/alerts"""
    success         : bool
    days_window     : int
    alert_threshold : int
    urgency_summary : Optional[UrgencySummary] = None
    total_alerts    : int = 0
    data            : List[LowStockAlertSchema]

    class Config:
        from_attributes = True


# ── Restock Schema ────────────────────────────────────────────────────────────
class RestockItemSchema(BaseModel):
    """
    One item in the restock recommendation list.
    This is what the store owner actually uses to place orders.
    """
    product_name               : str
    category                   : str
    supplier                   : Optional[str] = None
    current_stock              : int   = Field(..., ge=0)
    suggested_reorder_qty      : int   = Field(..., ge=0)
    estimated_reorder_cost_pkr : float = Field(..., ge=0)
    urgency                    : Literal["CRITICAL", "HIGH", "MEDIUM"]

    class Config:
        from_attributes = True


class RestockResponse(BaseModel):
    """Full response for /inventory/restock"""
    success                   : bool
    items_to_restock          : int
    total_estimated_cost_pkr  : float
    data                      : List[RestockItemSchema]

    class Config:
        from_attributes = True


# ── Inventory Health Summary Schema ───────────────────────────────────────────
class InventoryHealthSchema(BaseModel):
    """
    Complete inventory health picture.
    Used by the dashboard's inventory summary card.
    """
    total_products             : int
    total_stock_units          : int
    total_inventory_value_pkr  : float
    dead_stock_products        : int
    dead_stock_value_pkr       : float
    low_stock_critical         : int
    low_stock_high             : int
    restock_items_needed       : int
    estimated_restock_cost_pkr : float
    generated_at               : str

    class Config:
        from_attributes = True