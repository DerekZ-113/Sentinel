"""
Sentinel API Models

Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class RoadType(str, Enum):
    highway = "highway"
    main_road = "main_road"
    residential = "residential"
    downtown = "downtown"
    school_zone = "school_zone"


class TrafficCondition(str, Enum):
    light = "light"
    moderate = "moderate"
    heavy = "heavy"
    standstill = "standstill"


class ConstructionZone(str, Enum):
    none = "none"
    temporary = "temporary"
    persistent = "persistent"
    flagger = "flagger"


class NotificationType(str, Enum):
    verification_request = "verification_request"
    emergency_vehicle_alert = "emergency_vehicle_alert"
    stuck = "stuck"
    speed_anomaly = "speed_anomaly"
    impact_l0 = "impact_l0"
    passenger_assist = "passenger_assist"


class NotificationSubtype(str, Enum):
    object_query = "object_query"
    traffic_signal_verify = "traffic_signal_verify"
    lane_mapping_verify = "lane_mapping_verify"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class NotificationPayload(BaseModel):
    """Incoming notification from fleet"""
    vehicle_id: str
    speed: float = Field(ge=0, description="Current speed in mph")
    expected_speed: float = Field(ge=0, description="Expected speed for context")
    road_type: RoadType
    traffic_condition: TrafficCondition
    construction_zone: ConstructionZone = ConstructionZone.none
    notification_type: NotificationType
    notification_subtype: Optional[NotificationSubtype] = None
    ev_distance: Optional[float] = Field(None, ge=0, description="Distance to EV in meters")
    pedestrian_density: float = Field(0.0, ge=0.0, le=1.0)
    object_in_path: bool = False
    time_since_stop: float = Field(0.0, ge=0)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)

    # Optional: ground truth for evaluation (seed script provides this)
    needs_intervention_actual: Optional[bool] = None

    model_config = {"use_enum_values": True}


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response from POST /predict"""
    vehicle_id: str
    notification_type: str
    needs_intervention: bool
    confidence: float
    raw_score: float
    timestamp: datetime


class AlertRecord(BaseModel):
    """Single alert record from GET /alerts"""
    id: int
    time: datetime
    vehicle_id: str
    notification_type: str
    notification_subtype: Optional[str] = None
    needs_intervention_predicted: bool
    needs_intervention_actual: Optional[bool] = None
    confidence: float
    speed: Optional[float] = None
    road_type: Optional[str] = None
    traffic_condition: Optional[str] = None


class AlertsResponse(BaseModel):
    """Response from GET /alerts"""
    alerts: list[AlertRecord]
    total: int
    limit: int
    offset: int


class TypeStats(BaseModel):
    """Stats for a single notification type"""
    notification_type: str
    total: int
    flagged: int
    suppressed: int
    fp_rate: Optional[float] = None
    accuracy: Optional[float] = None


class StatsResponse(BaseModel):
    """Response from GET /stats"""
    time_window_hours: int
    total_alerts: int
    total_flagged: int
    total_suppressed: int
    overall_fp_rate: Optional[float] = None
    by_type: list[TypeStats]


class ModelHealthResponse(BaseModel):
    """Response from GET /api/stats/model-health"""
    status: str  # healthy / warning / degraded
    total_predictions: int
    pct_flagged: float
    pct_suppressed: float
    avg_confidence: float | None
    accuracy: float | None
    confidence_buckets: dict  # {"high": n, "medium": n, "low": n}
    flagged_by_type: dict  # {type: count}
    suppressed_by_type: dict  # {type: count}


class FPBucket(BaseModel):
    """Single time bucket for FP rate over time."""
    time: str
    total: int
    flagged: int
    suppressed: int
    fp_rate: float | None
    accuracy: float | None


class FPOverTimeResponse(BaseModel):
    """Response from GET /api/stats/fp-over-time"""
    time_window_hours: int
    buckets: list[FPBucket]


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str
    model_loaded: bool
    db_connected: bool
    model_features: int
    model_threshold: float
    uptime_seconds: float
