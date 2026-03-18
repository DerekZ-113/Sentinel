"""
Tests for Pydantic models and enum validation in api/models.py.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from api.models import (
    RoadType, TrafficCondition, ConstructionZone,
    NotificationType, NotificationSubtype,
    NotificationPayload, PredictionResponse, AlertRecord,
    AlertsResponse, TypeStats, StatsResponse,
    ModelHealthResponse, HealthResponse,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEnums:

    def test_road_type_enum_values(self):
        expected = {"highway", "main_road", "residential", "downtown", "school_zone"}
        assert {e.value for e in RoadType} == expected

    def test_traffic_condition_enum_values(self):
        expected = {"light", "moderate", "heavy", "standstill"}
        assert {e.value for e in TrafficCondition} == expected

    def test_construction_zone_enum_values(self):
        expected = {"none", "temporary", "persistent", "flagger"}
        assert {e.value for e in ConstructionZone} == expected

    def test_notification_type_enum_values(self):
        expected = {
            "verification_request", "emergency_vehicle_alert", "stuck",
            "speed_anomaly", "impact_l0", "passenger_assist",
        }
        assert {e.value for e in NotificationType} == expected

    def test_notification_subtype_enum_values(self):
        expected = {"object_query", "traffic_signal_verify", "lane_mapping_verify"}
        assert {e.value for e in NotificationSubtype} == expected


# ============================================================================
# NOTIFICATION PAYLOAD VALIDATION
# ============================================================================

class TestNotificationPayload:

    def test_valid_minimal(self):
        """Only required fields; check defaults."""
        p = NotificationPayload(
            vehicle_id="v1",
            speed=30.0,
            expected_speed=35.0,
            road_type="highway",
            traffic_condition="light",
            notification_type="stuck",
        )
        assert p.construction_zone == "none"
        assert p.pedestrian_density == 0.0
        assert p.object_in_path is False
        assert p.time_since_stop == 0.0
        assert p.notification_subtype is None
        assert p.hour_of_day is None

    def test_valid_full(self):
        p = NotificationPayload(
            vehicle_id="v1",
            speed=30.0,
            expected_speed=35.0,
            road_type="downtown",
            traffic_condition="heavy",
            construction_zone="temporary",
            notification_type="verification_request",
            notification_subtype="object_query",
            ev_distance=100.0,
            pedestrian_density=0.6,
            object_in_path=True,
            time_since_stop=60.0,
            hour_of_day=14,
            needs_intervention_actual=True,
        )
        assert p.vehicle_id == "v1"

    def test_speed_negative_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=-1, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="stuck",
            )

    def test_pedestrian_density_over_1_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="stuck", pedestrian_density=1.5,
            )

    def test_pedestrian_density_negative_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="stuck", pedestrian_density=-0.1,
            )

    def test_hour_of_day_out_of_range(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="stuck", hour_of_day=24,
            )

    def test_invalid_road_type_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="invalid_road", traffic_condition="light",
                notification_type="stuck",
            )

    def test_invalid_notification_type_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="bogus",
            )

    def test_ev_distance_negative_rejected(self):
        with pytest.raises(ValidationError):
            NotificationPayload(
                vehicle_id="v1", speed=10, expected_speed=35.0,
                road_type="highway", traffic_condition="light",
                notification_type="stuck", ev_distance=-10,
            )

    def test_enum_values_serialized_as_strings(self):
        p = NotificationPayload(
            vehicle_id="v1", speed=10, expected_speed=35.0,
            road_type="highway", traffic_condition="light",
            notification_type="stuck",
        )
        d = p.model_dump()
        assert isinstance(d["road_type"], str)
        assert d["road_type"] == "highway"


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TestResponseModels:

    def test_prediction_response(self):
        r = PredictionResponse(
            vehicle_id="v1", notification_type="stuck",
            needs_intervention=True, confidence=0.85,
            raw_score=0.85, timestamp=datetime.now(timezone.utc),
        )
        assert r.needs_intervention is True

    def test_alert_record_optional_nulls(self):
        r = AlertRecord(
            id=1, time=datetime.now(timezone.utc),
            vehicle_id="v1", notification_type="stuck",
            needs_intervention_predicted=True, confidence=0.9,
        )
        assert r.needs_intervention_actual is None
        assert r.speed is None

    def test_alerts_response_empty_list(self):
        r = AlertsResponse(alerts=[], total=0, limit=50, offset=0)
        assert len(r.alerts) == 0

    def test_stats_response_with_by_type(self):
        ts = TypeStats(
            notification_type="stuck", total=100,
            flagged=30, suppressed=70,
        )
        r = StatsResponse(
            time_window_hours=24, total_alerts=100,
            total_flagged=30, total_suppressed=70,
            by_type=[ts],
        )
        assert len(r.by_type) == 1

    def test_model_health_response(self):
        r = ModelHealthResponse(
            status="healthy", total_predictions=100,
            pct_flagged=30.0, pct_suppressed=70.0,
            avg_confidence=0.85, accuracy=0.80,
            confidence_buckets={"high": 60, "medium": 30, "low": 10},
            flagged_by_type={"stuck": 10},
            suppressed_by_type={"stuck": 30},
        )
        assert r.status == "healthy"

    def test_health_response(self):
        r = HealthResponse(
            status="healthy", model_loaded=True, db_connected=True,
            model_features=28, model_threshold=0.5, uptime_seconds=100.0,
        )
        assert r.model_loaded is True
