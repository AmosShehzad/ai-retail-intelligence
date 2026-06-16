"""
Analytics endpoints — filled on Day 9.
Revenue, margins, category performance, KPIs.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/health-check")
async def analytics_health():
    return {"status": "analytics router online"}