"""
Sentinel API — Alerts Route

GET /api/alerts
Returns recent alerts with predictions for the dashboard feed.
"""

import logging
from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from api.models import AlertsResponse

logger = logging.getLogger("sentinel.api")
router = APIRouter()


# Plain `def`: sync DB work runs in FastAPI's threadpool (see predict.py)
@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    notification_type: Optional[str] = Query(None),
):
    """Get recent alerts with model predictions.

    `total` is capped at 10,000 — see DatabaseService.get_recent_alerts.
    """
    from api.main import get_db_service

    try:
        db_service = get_db_service()
        result = db_service.get_recent_alerts(
            limit=limit, offset=offset, notification_type=notification_type
        )
        return AlertsResponse(**result)
    except Exception:
        logger.exception("Failed to fetch alerts")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")
