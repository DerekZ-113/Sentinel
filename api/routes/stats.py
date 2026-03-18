"""
Sentinel API — Stats Route

GET /api/stats          — aggregate stats across all types
GET /api/stats/{type}   — per-type breakdown
"""

import logging
from fastapi import APIRouter, Query, HTTPException

from api.models import StatsResponse, TypeStats, ModelHealthResponse

logger = logging.getLogger("sentinel.api")
router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(hours: int = Query(24, ge=1, le=720)):
    """Get aggregate prediction stats over a time window."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_stats(hours=hours)
        return StatsResponse(**result)
    except Exception as e:
        logger.exception("Failed to fetch stats")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/stats/model-health", response_model=ModelHealthResponse)
async def get_model_health(hours: int = Query(24, ge=1, le=720)):
    """Get model health metrics for monitoring panel."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_model_health(hours=hours)
        return ModelHealthResponse(**result)
    except Exception as e:
        logger.exception("Failed to fetch model health")
        raise HTTPException(status_code=500, detail=f"Failed to fetch model health: {str(e)}")


@router.get("/stats/{notification_type}", response_model=TypeStats)
async def get_stats_by_type(notification_type: str, hours: int = Query(24, ge=1, le=720)):
    """Get detailed stats for a specific notification type."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_stats_by_type(notification_type, hours=hours)
        return TypeStats(**result)
    except Exception as e:
        logger.exception(f"Failed to fetch stats for {notification_type}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")
