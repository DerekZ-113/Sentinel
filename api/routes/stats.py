"""
Sentinel API — Stats Route

GET /api/stats          — aggregate stats across all types
GET /api/stats/{type}   — per-type breakdown
"""

import logging
from fastapi import APIRouter, Query, HTTPException

from api.models import StatsResponse, TypeStats, ModelHealthResponse, FPOverTimeResponse

logger = logging.getLogger("sentinel.api")
router = APIRouter()


# Plain `def` on all four handlers: sync DB work runs in FastAPI's
# threadpool instead of blocking the event loop (see predict.py)
@router.get("/stats", response_model=StatsResponse)
def get_stats(hours: int = Query(24, ge=1, le=720)):
    """Get aggregate prediction stats over a time window."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_stats(hours=hours)
        return StatsResponse(**result)
    except Exception:
        logger.exception("Failed to fetch stats")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@router.get("/stats/model-health", response_model=ModelHealthResponse)
def get_model_health(hours: int = Query(24, ge=1, le=720)):
    """Get model health metrics for monitoring panel."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_model_health(hours=hours)
        return ModelHealthResponse(**result)
    except Exception:
        logger.exception("Failed to fetch model health")
        raise HTTPException(status_code=500, detail="Failed to fetch model health")


@router.get("/stats/fp-over-time", response_model=FPOverTimeResponse)
def get_fp_over_time(
    hours: int = Query(24, ge=1, le=720),
    buckets: int = Query(12, ge=4, le=48),
):
    """Get FP rate bucketed over time."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_fp_over_time(hours=hours, buckets=buckets)
        return FPOverTimeResponse(**result)
    except Exception:
        logger.exception("Failed to fetch FP over time")
        raise HTTPException(status_code=500, detail="Failed to fetch FP over time")


@router.get("/stats/{notification_type}", response_model=TypeStats)
def get_stats_by_type(notification_type: str, hours: int = Query(24, ge=1, le=720)):
    """Get detailed stats for a specific notification type."""
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_stats_by_type(notification_type, hours=hours)
        return TypeStats(**result)
    except Exception:
        logger.exception(f"Failed to fetch stats for {notification_type}")
        # notification_type is client input, safe to echo; exception text is not
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats for {notification_type}")
