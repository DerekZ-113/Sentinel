"""
Tests for ml/prepare_data.py — DataFrame-based feature engineering.

Uses synthetic DataFrames (no DB required).
"""

import pytest
import numpy as np
import pandas as pd

from ml.prepare_data import engineer_features


def _make_df(**overrides):
    """Build a single-row DataFrame with sensible defaults."""
    defaults = {
        "speed": 30.0,
        "expected_speed": 35.0,
        "road_type": "main_road",
        "traffic_condition": "moderate",
        "construction_zone": "none",
        "notification_type": "stuck",
        "notification_subtype": None,
        "needs_intervention": True,
        "ev_distance": np.nan,
        "pedestrian_density": 0.3,
        "object_in_path": False,
        "time_since_stop": 0.0,
        "hour_of_day": 12,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestEngineerFeaturesColumns:

    def test_all_28_columns_created(self):
        df = _make_df()
        result = engineer_features(df)
        expected_cols = [
            "speed_ratio", "speed_deviation", "is_stopped", "expected_stopped",
            "road_type_encoded", "traffic_encoded", "construction_encoded",
            "notification_type_encoded", "notification_subtype_encoded",
            "ev_distance_normalized", "pedestrian_density", "object_in_path",
            "time_since_stop_normalized",
            "hour_sin", "hour_cos",
            "high_traffic", "high_pedestrian",
            "stuck_in_traffic", "stuck_in_construction", "stuck_clear_road",
            "object_query_high_ped", "object_query_low_ped", "object_query_moving",
            "ev_far_away", "ev_close",
            "speed_anomaly_in_traffic", "speed_anomaly_clear",
            "impact_rough_road",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestSpeedFeatures:

    def test_speed_ratio(self):
        df = _make_df(speed=30, expected_speed=60)
        result = engineer_features(df)
        assert result["speed_ratio"].iloc[0] == pytest.approx(30.0 / 61.0, abs=1e-6)

    def test_is_stopped_threshold(self):
        df = pd.DataFrame([
            {"speed": 4.9, "expected_speed": 35.0, "road_type": "main_road",
             "traffic_condition": "light", "construction_zone": "none",
             "notification_type": "stuck", "notification_subtype": None,
             "ev_distance": np.nan, "pedestrian_density": 0.0,
             "object_in_path": False, "time_since_stop": 0.0, "hour_of_day": 12},
            {"speed": 5.0, "expected_speed": 35.0, "road_type": "main_road",
             "traffic_condition": "light", "construction_zone": "none",
             "notification_type": "stuck", "notification_subtype": None,
             "ev_distance": np.nan, "pedestrian_density": 0.0,
             "object_in_path": False, "time_since_stop": 0.0, "hour_of_day": 12},
        ])
        result = engineer_features(df)
        assert result["is_stopped"].iloc[0] == 1
        assert result["is_stopped"].iloc[1] == 0


class TestContextFeatures:

    def test_ev_distance_fillna(self):
        df = _make_df(ev_distance=np.nan)
        result = engineer_features(df)
        assert result["ev_distance_normalized"].iloc[0] == pytest.approx(999.0 / 500.0, abs=1e-3)

    def test_ev_distance_clip(self):
        df = _make_df(ev_distance=1500)
        result = engineer_features(df)
        assert result["ev_distance_normalized"].iloc[0] == pytest.approx(2.0, abs=1e-6)

    def test_object_in_path_fillna(self):
        df = _make_df(object_in_path=np.nan)
        result = engineer_features(df)
        assert result["object_in_path"].iloc[0] == 0

    def test_time_since_stop_normalization(self):
        df = _make_df(time_since_stop=600)
        result = engineer_features(df)
        assert result["time_since_stop_normalized"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_time_since_stop_clamped(self):
        df = _make_df(time_since_stop=1200)
        result = engineer_features(df)
        assert result["time_since_stop_normalized"].iloc[0] == pytest.approx(2.0, abs=1e-6)


class TestTimeFeatures:

    def test_hour_sin_cos_vectorized(self):
        df = pd.DataFrame([
            {**_make_df(hour_of_day=0).iloc[0].to_dict()},
            {**_make_df(hour_of_day=6).iloc[0].to_dict()},
            {**_make_df(hour_of_day=12).iloc[0].to_dict()},
        ])
        result = engineer_features(df)
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert result["hour_sin"].iloc[1] == pytest.approx(1.0, abs=1e-6)
        assert result["hour_sin"].iloc[2] == pytest.approx(0.0, abs=1e-6)
        assert result["hour_cos"].iloc[2] == pytest.approx(-1.0, abs=1e-6)


class TestInteractionFeatures:

    def test_stuck_in_traffic(self):
        df = _make_df(notification_type="stuck", traffic_condition="heavy")
        result = engineer_features(df)
        assert result["stuck_in_traffic"].iloc[0] == 1

    def test_stuck_in_traffic_wrong_type(self):
        df = _make_df(notification_type="speed_anomaly", traffic_condition="heavy")
        result = engineer_features(df)
        assert result["stuck_in_traffic"].iloc[0] == 0

    def test_object_query_high_ped(self):
        df = _make_df(
            notification_type="verification_request",
            notification_subtype="object_query",
            pedestrian_density=0.6,
        )
        result = engineer_features(df)
        assert result["object_query_high_ped"].iloc[0] == 1

    def test_impact_rough_road(self):
        df = _make_df(notification_type="impact_l0", road_type="residential")
        result = engineer_features(df)
        assert result["impact_rough_road"].iloc[0] == 1
