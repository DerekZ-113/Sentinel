"""
Sentinel API — Predict Route

POST /api/predict
Takes notification payload, runs through model, stores result, returns prediction.
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from api.models import NotificationPayload, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: NotificationPayload):
    """Run a notification through the model and return prediction."""
    from api.main import get_model_service, get_db_service

    model_service = get_model_service()
    db_service = get_db_service()

    # Convert Pydantic model to dict for model service
    payload_dict = payload.model_dump()

    # Get prediction
    prediction = model_service.predict(payload_dict)

    # Store in DB
    db_service.store_prediction(payload_dict, prediction)

    return PredictionResponse(
        vehicle_id=payload.vehicle_id,
        notification_type=payload.notification_type,
        needs_intervention=prediction['needs_intervention'],
        confidence=prediction['confidence'],
        raw_score=prediction['raw_score'],
        timestamp=datetime.now(timezone.utc),
    )
