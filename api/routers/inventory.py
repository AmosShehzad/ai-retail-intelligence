"""
Inventory endpoints — filled on Day 9.
Dead stock, low stock alerts, restock recommendations.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.get("/health-check")
async def inventory_health():
    return {"status": "inventory router online"}