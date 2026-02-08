"""
Sentinel API — Stats Route

GET /api/stats          — aggregate stats across all types
GET /api/stats/{type}   — per-type breakdown
"""

from fastapi import APIRouter, Query

from api.models import StatsResponse, TypeStats, ModelHealthResponse

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(hours: int = Query(24, ge=1, le=720)):
    """Get aggregate prediction stats over a time window."""
    from api.main import get_db_service

    db_service = get_db_service()
    result = db_service.get_stats(hours=hours)
    return StatsResponse(**result)


@router.get("/stats/model-health", response_model=ModelHealthResponse)
async def get_model_health(hours: int = Query(24, ge=1, le=720)):
    """Get model health metrics for monitoring panel."""
    from api.main import get_db_service

    db_service = get_db_service()
    result = db_service.get_model_health(hours=hours)
    return ModelHealthResponse(**result)


@router.get("/stats/{notification_type}", response_model=TypeStats)
async def get_stats_by_type(notification_type: str, hours: int = Query(24, ge=1, le=720)):
    """Get detailed stats for a specific notification type."""
    from api.main import get_db_service

    db_service = get_db_service()
    result = db_service.get_stats_by_type(notification_type, hours=hours)
    return TypeStats(**result)
