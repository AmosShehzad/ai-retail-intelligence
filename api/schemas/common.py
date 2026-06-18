"""
Common response schemas shared across all endpoints.

Every API response in this project follows this envelope:
{
    "success": true/false,
    "message": "optional info",
    "data": { ... }          ← actual payload
}

This consistency means the frontend handles all responses
the same way — check success, then read data.
"""

from __future__ import annotations
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

# T is a placeholder for any data type
# Generic[T] means this model works with any payload type
T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """
    Universal response envelope for all endpoints.
    
    Generic[T] means:
    - BaseResponse[StoreKPISchema] for KPI endpoint
    - BaseResponse[List[ProductSchema]] for product list
    T gets replaced with the actual data type at usage time.
    """
    success  : bool         = True
    message  : Optional[str] = None
    data     : Optional[T]  = None

    class Config:
        # Allows the model to work with ORM objects and dicts equally
        from_attributes = True


class ErrorResponse(BaseModel):
    """
    Standard error shape returned by all error handlers.
    Matches what you defined in error_handlers.py on Day 8.
    """
    success : bool        = False
    error   : str
    detail  : Optional[str] = None

    class Config:
        from_attributes = True


class PaginationMeta(BaseModel):
    """
    Metadata for paginated responses.
    Add this to any list endpoint that might return large datasets.
    """
    total  : int
    count  : int
    page   : int = 1