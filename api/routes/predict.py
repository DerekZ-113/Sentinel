"""
Sentinel API — Predict Route

POST /api/predict
Takes notification payload, runs through model, stores result, returns prediction.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from api.models import NotificationPayload, PredictionResponse
from api.auth import verify_api_key

logger = logging.getLogger("sentinel.api")
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(payload: NotificationPayload):
    """Run a notification through the model and return prediction."""
    from api.main import get_model_service, get_db_service

    try:
        model_service = get_model_service()
        db_service = get_db_service()

        payload_dict = payload.model_dump()
        prediction = model_service.predict(payload_dict)
        db_service.store_prediction(payload_dict, prediction)

        return PredictionResponse(
            vehicle_id=payload.vehicle_id,
            notification_type=payload.notification_type,
            needs_intervention=prediction['needs_intervention'],
            confidence=prediction['confidence'],
            raw_score=prediction['raw_score'],
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
