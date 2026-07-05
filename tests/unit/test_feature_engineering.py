"""
Tests for feature engineering in api/services/model_service.py.

Tests all 28 features individually: speed, encoding, context, time, derived, interactions.
"""

import pytest
import numpy as np
from copy import deepcopy
from unittest.mock import patch
from datetime import datetime, timezone


def _make_payload(**overrides):
    """Build a payload with sensible defaults, applying overrides."""
    base = {
        "vehicle_id": "test_v",
        "speed": 30.0,
        "expected_speed": 35.0,
        "road_type": "main_road",
        "traffic_condition": "moderate",
        "construction_zone": "none",
        "notification_type": "stuck",
        "notification_subtype": None,
        "ev_distance": None,
        "pedestrian_density": 0.3,
        "object_in_path": False,
        "time_since_stop": 0.0,
        "hour_of_day": 12,
    }
    base.update(overrides)
    return base


# Feature column index map (matching model_service.py assembly order)
IDX = {
    "speed_ratio": 0, "speed_deviation": 1, "is_stopped": 2, "expected_stopped": 3,
    "road_type_encoded": 4, "traffic_encoded": 5, "construction_encoded": 6,
    "notification_type_encoded": 7, "notification_subtype_encoded": 8,
    "ev_distance_normalized": 9, "pedestrian_density": 10, "object_in_path": 11,
    "time_since_stop_normalized": 12,
    "hour_sin": 13, "hour_cos": 14,
    "high_traffic": 15, "high_pedestrian": 16,
    "stuck_in_traffic": 17, "stuck_in_construction": 18, "stuck_clear_road": 19,
    "object_query_high_ped": 20, "object_query_low_ped": 21, "object_query_moving": 22,
    "ev_far_away": 23, "ev_close": 24,
    "speed_anomaly_in_traffic": 25, "speed_anomaly_clear": 26,
    "impact_rough_road": 27,
}


def _feat(model_service, payload, feature_name):
    """Get a specific feature value from a payload."""
    features = model_service.engineer_features(payload)
    return features[0, IDX[feature_name]]


# ============================================================================
# SPEED FEATURES
# ============================================================================

class TestSpeedFeatures:

    def test_speed_ratio_calculation(self, model_service):
        f = _feat(model_service, _make_payload(speed=30, expected_speed=60), "speed_ratio")
        assert f == pytest.approx(30.0 / 61.0, abs=1e-6)

    def test_speed_ratio_zero_expected(self, model_service):
        f = _feat(model_service, _make_payload(speed=10, expected_speed=0), "speed_ratio")
        assert f == pytest.approx(10.0 / 1.0, abs=1e-6)

    def test_speed_deviation(self, model_service):
        f = _feat(model_service, _make_payload(speed=30, expected_speed=60), "speed_deviation")
        assert f == pytest.approx(-30.0, abs=1e-6)

    def test_is_stopped_below_5(self, model_service):
        assert _feat(model_service, _make_payload(speed=4.9), "is_stopped") == 1

    def test_is_stopped_at_5(self, model_service):
        assert _feat(model_service, _make_payload(speed=5.0), "is_stopped") == 0

    def test_expected_stopped(self, model_service):
        assert _feat(model_service, _make_payload(expected_speed=4.9), "expected_stopped") == 1


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================

class TestCategoricalEncoding:

    @pytest.mark.parametrize("road_type,expected", [
        ("highway", 0), ("main_road", 1), ("residential", 2),
        ("downtown", 3), ("school_zone", 4),
    ])
    def test_road_type_encoding(self, model_service, road_type, expected):
        f = _feat(model_service, _make_payload(road_type=road_type), "road_type_encoded")
        assert f == expected

    @pytest.mark.parametrize("traffic,expected", [
        ("light", 0), ("moderate", 1), ("heavy", 2), ("standstill", 3),
    ])
    def test_traffic_encoding(self, model_service, traffic, expected):
        f = _feat(model_service, _make_payload(traffic_condition=traffic), "traffic_encoded")
        assert f == expected

    @pytest.mark.parametrize("construction,expected", [
        ("none", 0), ("temporary", 1), ("persistent", 2), ("flagger", 3),
    ])
    def test_construction_encoding(self, model_service, construction, expected):
        f = _feat(model_service, _make_payload(construction_zone=construction), "construction_encoded")
        assert f == expected

    @pytest.mark.parametrize("ntype,expected", [
        ("verification_request", 1), ("emergency_vehicle_alert", 2),
        ("stuck", 3), ("speed_anomaly", 4), ("impact_l0", 5), ("passenger_assist", 6),
    ])
    def test_notification_type_encoding(self, model_service, ntype, expected):
        f = _feat(model_service, _make_payload(notification_type=ntype), "notification_type_encoded")
        assert f == expected

    @pytest.mark.parametrize("subtype,expected", [
        (None, 0), ("object_query", 1), ("traffic_signal_verify", 2), ("lane_mapping_verify", 3),
    ])
    def test_notification_subtype_encoding(self, model_service, subtype, expected):
        f = _feat(model_service, _make_payload(
            notification_type="verification_request", notification_subtype=subtype
        ), "notification_subtype_encoded")
        assert f == expected

    def test_unknown_road_type_defaults_0(self, model_service):
        f = _feat(model_service, _make_payload(road_type="nonexistent"), "road_type_encoded")
        assert f == 0

    def test_unknown_notification_type_defaults_0(self, model_service):
        f = _feat(model_service, _make_payload(notification_type="bogus"), "notification_type_encoded")
        assert f == 0


# ============================================================================
# CONTEXT FEATURES
# ============================================================================

class TestContextFeatures:

    def test_ev_distance_none_gives_max(self, model_service):
        f = _feat(model_service, _make_payload(ev_distance=None), "ev_distance_normalized")
        assert f == pytest.approx(999.0 / 500.0, abs=1e-4)

    def test_ev_distance_zero_normalized(self, model_service):
        f = _feat(model_service, _make_payload(ev_distance=0), "ev_distance_normalized")
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_ev_distance_250_normalized(self, model_service):
        f = _feat(model_service, _make_payload(ev_distance=250), "ev_distance_normalized")
        assert f == pytest.approx(0.5, abs=1e-6)

    def test_ev_distance_1500_clamped(self, model_service):
        f = _feat(model_service, _make_payload(ev_distance=1500), "ev_distance_normalized")
        assert f == pytest.approx(2.0, abs=1e-6)

    def test_pedestrian_density_passthrough(self, model_service):
        f = _feat(model_service, _make_payload(pedestrian_density=0.6), "pedestrian_density")
        assert f == pytest.approx(0.6, abs=1e-6)

    def test_time_since_stop_normalized(self, model_service):
        f = _feat(model_service, _make_payload(time_since_stop=300), "time_since_stop_normalized")
        assert f == pytest.approx(0.5, abs=1e-6)

    def test_time_since_stop_clamped(self, model_service):
        f = _feat(model_service, _make_payload(time_since_stop=1500), "time_since_stop_normalized")
        assert f == pytest.approx(2.0, abs=1e-6)

    def test_object_in_path_true(self, model_service):
        assert _feat(model_service, _make_payload(object_in_path=True), "object_in_path") == 1

    def test_object_in_path_false(self, model_service):
        assert _feat(model_service, _make_payload(object_in_path=False), "object_in_path") == 0


# ============================================================================
# TIME FEATURES
# ============================================================================

class TestTimeFeatures:

    def test_hour_sin_cos_midnight(self, model_service):
        p = _make_payload(hour_of_day=0)
        features = model_service.engineer_features(p)
        assert features[0, IDX["hour_sin"]] == pytest.approx(0.0, abs=1e-6)
        assert features[0, IDX["hour_cos"]] == pytest.approx(1.0, abs=1e-6)

    def test_hour_sin_cos_6am(self, model_service):
        p = _make_payload(hour_of_day=6)
        features = model_service.engineer_features(p)
        assert features[0, IDX["hour_sin"]] == pytest.approx(1.0, abs=1e-6)
        assert features[0, IDX["hour_cos"]] == pytest.approx(0.0, abs=1e-6)

    def test_hour_sin_cos_noon(self, model_service):
        p = _make_payload(hour_of_day=12)
        features = model_service.engineer_features(p)
        assert features[0, IDX["hour_sin"]] == pytest.approx(0.0, abs=1e-6)
        assert features[0, IDX["hour_cos"]] == pytest.approx(-1.0, abs=1e-6)

    def test_hour_sin_cos_6pm(self, model_service):
        p = _make_payload(hour_of_day=18)
        features = model_service.engineer_features(p)
        assert features[0, IDX["hour_sin"]] == pytest.approx(-1.0, abs=1e-6)
        assert features[0, IDX["hour_cos"]] == pytest.approx(0.0, abs=1e-6)

    def test_hour_none_uses_current_utc(self, model_service):
        """When hour_of_day is not provided, the fallback must be the
        current UTC hour — asserted exactly by freezing time. (A range
        check on sin/cos is true for any input and detects nothing.)"""
        frozen = datetime(2024, 12, 1, 8, 30, 0, tzinfo=timezone.utc)

        with patch("api.services.model_service.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            p = _make_payload()
            del p["hour_of_day"]
            features = model_service.engineer_features(p)

        mock_dt.now.assert_called_once_with(timezone.utc)
        expected_sin = np.sin(2 * np.pi * 8 / 24)
        expected_cos = np.cos(2 * np.pi * 8 / 24)
        assert features[0, IDX["hour_sin"]] == pytest.approx(expected_sin, abs=1e-9)
        assert features[0, IDX["hour_cos"]] == pytest.approx(expected_cos, abs=1e-9)


# ============================================================================
# DERIVED FEATURES
# ============================================================================

class TestDerivedFeatures:

    def test_high_traffic_heavy(self, model_service):
        f = _feat(model_service, _make_payload(traffic_condition="heavy", construction_zone="none"), "high_traffic")
        assert f == 1

    def test_high_traffic_construction(self, model_service):
        f = _feat(model_service, _make_payload(traffic_condition="light", construction_zone="temporary"), "high_traffic")
        assert f == 1

    def test_high_traffic_neither(self, model_service):
        f = _feat(model_service, _make_payload(traffic_condition="light", construction_zone="none"), "high_traffic")
        assert f == 0

    def test_high_pedestrian_above_05(self, model_service):
        f = _feat(model_service, _make_payload(pedestrian_density=0.6), "high_pedestrian")
        assert f == 1

    def test_high_pedestrian_at_05(self, model_service):
        f = _feat(model_service, _make_payload(pedestrian_density=0.5), "high_pedestrian")
        assert f == 0  # not > 0.5


# ============================================================================
# INTERACTION FEATURES
# ============================================================================

class TestInteractionFeatures:

    # STUCK
    def test_stuck_in_traffic_active(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="stuck", traffic_condition="heavy"
        ), "stuck_in_traffic")
        assert f == 1

    def test_stuck_in_traffic_wrong_type(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="speed_anomaly", traffic_condition="heavy"
        ), "stuck_in_traffic")
        assert f == 0

    def test_stuck_in_construction(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="stuck", construction_zone="temporary"
        ), "stuck_in_construction")
        assert f == 1

    def test_stuck_clear_road(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="stuck", traffic_condition="light", construction_zone="none"
        ), "stuck_clear_road")
        assert f == 1

    def test_stuck_clear_road_not_clear(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="stuck", traffic_condition="moderate", construction_zone="none"
        ), "stuck_clear_road")
        assert f == 0

    # OBJECT QUERY
    def test_object_query_high_ped(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", pedestrian_density=0.6
        ), "object_query_high_ped")
        assert f == 1

    def test_object_query_low_ped(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", pedestrian_density=0.2
        ), "object_query_low_ped")
        assert f == 1

    def test_object_query_low_ped_boundary(self, model_service):
        assert _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", pedestrian_density=0.3
        ), "object_query_low_ped") == 1
        assert _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", pedestrian_density=0.31
        ), "object_query_low_ped") == 0

    def test_object_query_moving(self, model_service):
        assert _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", speed=11
        ), "object_query_moving") == 1
        assert _feat(model_service, _make_payload(
            notification_type="verification_request",
            notification_subtype="object_query", speed=10
        ), "object_query_moving") == 0

    # EMERGENCY VEHICLE
    def test_ev_far_away(self, model_service):
        # ev_distance=250 => normalized = 0.5 > 0.4 => far_away=1
        f = _feat(model_service, _make_payload(
            notification_type="emergency_vehicle_alert", ev_distance=250
        ), "ev_far_away")
        assert f == 1

    def test_ev_close(self, model_service):
        # ev_distance=25 => normalized = 0.05 < 0.1 => close=1
        f = _feat(model_service, _make_payload(
            notification_type="emergency_vehicle_alert", ev_distance=25
        ), "ev_close")
        assert f == 1

    def test_ev_far_away_boundary(self, model_service):
        # ev_distance=200 => normalized = 0.4, not > 0.4 => 0
        f = _feat(model_service, _make_payload(
            notification_type="emergency_vehicle_alert", ev_distance=200
        ), "ev_far_away")
        assert f == 0

    # SPEED ANOMALY
    def test_speed_anomaly_in_traffic(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="speed_anomaly", traffic_condition="heavy"
        ), "speed_anomaly_in_traffic")
        assert f == 1

    def test_speed_anomaly_clear(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="speed_anomaly", traffic_condition="light"
        ), "speed_anomaly_clear")
        assert f == 1

    # IMPACT
    def test_impact_rough_road_residential(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="impact_l0", road_type="residential"
        ), "impact_rough_road")
        assert f == 1

    def test_impact_rough_road_downtown(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="impact_l0", road_type="downtown"
        ), "impact_rough_road")
        assert f == 1

    def test_impact_rough_road_highway(self, model_service):
        f = _feat(model_service, _make_payload(
            notification_type="impact_l0", road_type="highway"
        ), "impact_rough_road")
        assert f == 0


# ============================================================================
# FEATURE VECTOR SHAPE & ORDER
# ============================================================================

class TestFeatureVector:

    def test_shape(self, model_service):
        features = model_service.engineer_features(_make_payload())
        assert features.shape == (1, 28)

    def test_dtype(self, model_service):
        features = model_service.engineer_features(_make_payload())
        assert features.dtype == np.float64

    def test_feature_vector_column_order(self, model_service):
        """The 28 features must be in the same order as feature_columns."""
        expected_order = model_service.feature_columns
        assert len(expected_order) == 28
        # Verify our IDX map matches
        for name, idx in IDX.items():
            assert expected_order[idx] == name, f"Column order mismatch at index {idx}: expected {name}, got {expected_order[idx]}"
