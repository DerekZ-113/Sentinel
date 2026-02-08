"""
Sentinel API — Alerts Route

GET /api/alerts
Returns recent alerts with predictions for the dashboard feed.
"""

from fastapi import APIRouter, Query
from typing import Optional

from api.models import AlertsResponse

router = APIRouter()


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    notification_type: Optional[str] = Query(None),
):
    """Get recent alerts with model predictions."""
    from api.main import get_db_service

    db_service = get_db_service()
    result = db_service.get_recent_alerts(
        limit=limit, offset=offset, notification_type=notification_type
    )
    return AlertsResponse(**result)
