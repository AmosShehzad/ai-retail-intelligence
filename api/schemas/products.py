"""
Pydantic schemas for Products and Purchase Orders CRUD.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime


# ── Product Schemas ────────────────────────────────────────────────────────────

class ProductCreate(BaseModel):
    """POST /products/ — what the client sends."""
    product_name       : str   = Field(..., min_length=2, max_length=200)
    category           : str   = Field(..., min_length=2)
    cost_price         : float = Field(..., gt=0)
    selling_price      : float = Field(..., gt=0)
    stock              : int   = Field(default=0, ge=0)
    supplier           : Optional[str] = None
    low_stock_threshold: int   = Field(default=10, ge=1)

    @field_validator("selling_price")
    @classmethod
    def selling_must_exceed_cost(cls, v, info):
        if "cost_price" in info.data and v <= info.data["cost_price"]:
            raise ValueError("selling_price must be greater than cost_price")
        return v

    @field_validator("product_name")
    @classmethod
    def name_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("product_name cannot be blank")
        return v.strip().title()


class ProductUpdate(BaseModel):
    """PUT /products/{id} — all fields optional."""
    product_name       : Optional[str]   = Field(None, min_length=2)
    category           : Optional[str]   = None
    cost_price         : Optional[float] = Field(None, gt=0)
    selling_price      : Optional[float] = Field(None, gt=0)
    supplier           : Optional[str]   = None
    low_stock_threshold: Optional[int]   = Field(None, ge=1)


class StockUpdate(BaseModel):
    """PATCH /products/{id}/stock"""
    quantity : int = Field(..., description="Units to add (positive) or remove (negative)")
    operation: Literal["add", "set"] = "add"
    # add: stock = stock + quantity
    # set: stock = quantity (direct override)


class ProductResponse(BaseModel):
    product_id         : int
    product_name       : str
    category           : str
    cost_price         : float
    selling_price      : float
    stock              : int
    supplier           : Optional[str]
    low_stock_threshold: int
    is_active          : int
    created_at         : Optional[str]
    updated_at         : Optional[str]
    margin_pct         : Optional[float] = None

    class Config:
        from_attributes = True


# ── Purchase Order Schemas ─────────────────────────────────────────────────────

class PurchaseOrderCreate(BaseModel):
    """POST /orders/"""
    product_id      : int
    quantity_ordered: int   = Field(..., gt=0)
    cost_per_unit   : float = Field(..., gt=0)
    supplier        : str   = Field(..., min_length=2)


class PurchaseOrderResponse(BaseModel):
    id              : int
    product_id      : int
    product_name    : Optional[str] = None
    quantity_ordered: int
    cost_per_unit   : float
    total_cost      : float
    supplier        : str
    order_date      : Optional[str]
    status          : str
    received_date   : Optional[str]

    class Config:
        from_attributes = True