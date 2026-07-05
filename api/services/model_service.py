"""
Sentinel Model Service

Loads trained XGBoost model and serves predictions.
Handles feature engineering from raw notification data.

The model consumes raw engineered features — there is no scaler.
XGBoost is tree-based and scale-invariant; a scaling stage would only
reintroduce the training/serving-skew failure class this design removed.
"""

import json
import os
import logging
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone

logger = logging.getLogger("sentinel.model")


from ml.constants import (
    ROAD_TYPE_MAP, TRAFFIC_MAP, CONSTRUCTION_MAP,
    NOTIFICATION_TYPE_MAP, NOTIFICATION_SUBTYPE_MAP,
)


class ModelService:
    """
    Loads XGBoost model + config on init.
    Accepts raw notification data -> returns prediction + confidence.
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Default: look in ml/ relative to project root
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'ml'
            )

        model_path = os.path.join(model_dir, 'xgboost_model.json')
        config_path = os.path.join(model_dir, 'model_config.json')

        # Load model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load config (feature columns + threshold) — plain JSON, never pickle
        with open(config_path) as f:
            config = json.load(f)
        self.feature_columns = config['feature_columns']
        self.threshold = config.get('threshold', 0.5)
        self.best_iteration = config.get('best_iteration', 0)

        logger.info(f"Model loaded: {len(self.feature_columns)} features, "
                    f"threshold={self.threshold}, best_iter={self.best_iteration}")

    def engineer_features(self, payload: dict) -> np.ndarray:
        """
        Transform raw notification payload into the 28-feature vector.
        Mirrors the logic in prepare_data.py exactly.
        """
        # --- Extract raw values ---
        speed = payload['speed']
        expected_speed = payload['expected_speed']
        road_type = payload['road_type']
        traffic_condition = payload['traffic_condition']
        construction_zone = payload.get('construction_zone', 'none')
        notification_type = payload['notification_type']
        notification_subtype = payload.get('notification_subtype')
        ev_distance = payload.get('ev_distance')
        pedestrian_density = payload.get('pedestrian_density', 0.0)
        object_in_path = payload.get('object_in_path', False)
        time_since_stop = payload.get('time_since_stop', 0.0)
        hour_of_day = payload.get('hour_of_day')
        if hour_of_day is None:
            # UTC, not server-local: training timestamps are UTC, and a
            # server's TZ must not silently shift the hour feature
            hour_of_day = datetime.now(timezone.utc).hour

        # --- Speed features ---
        speed_ratio = speed / (expected_speed + 1)
        speed_deviation = speed - expected_speed
        is_stopped = int(speed < 5)
        expected_stopped = int(expected_speed < 5)

        # --- Encode categoricals ---
        road_type_encoded = ROAD_TYPE_MAP.get(road_type, 0)
        traffic_encoded = TRAFFIC_MAP.get(traffic_condition, 0)
        construction_encoded = CONSTRUCTION_MAP.get(construction_zone, 0)
        notification_type_encoded = NOTIFICATION_TYPE_MAP.get(notification_type, 0)
        notification_subtype_encoded = NOTIFICATION_SUBTYPE_MAP.get(notification_subtype, 0)

        # --- Context features ---
        ev_distance_for_model = ev_distance if ev_distance is not None else 999
        ev_distance_normalized = min(ev_distance_for_model / 500.0, 2.0)
        pedestrian_density = max(0.0, min(1.0, pedestrian_density))
        object_in_path_int = int(object_in_path)
        time_since_stop_normalized = min(time_since_stop / 600.0, 2.0)

        # --- Time encoding ---
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)

        # --- Derived features ---
        high_traffic = int(traffic_encoded >= 2 or construction_encoded >= 1)
        high_pedestrian = int(pedestrian_density > 0.5)

        # --- Interaction features ---
        is_stuck = (notification_type == 'stuck')
        is_object_query = (notification_subtype == 'object_query')
        is_ev = (notification_type == 'emergency_vehicle_alert')
        is_speed_anomaly = (notification_type == 'speed_anomaly')
        is_impact = (notification_type == 'impact_l0')

        stuck_in_traffic = int(is_stuck and traffic_encoded >= 2)
        stuck_in_construction = int(is_stuck and construction_encoded >= 1)
        stuck_clear_road = int(is_stuck and traffic_encoded == 0 and construction_encoded == 0)
        object_query_high_ped = int(is_object_query and pedestrian_density > 0.5)
        object_query_low_ped = int(is_object_query and pedestrian_density <= 0.3)
        object_query_moving = int(is_object_query and speed > 10)
        ev_far_away = int(is_ev and ev_distance_normalized > 0.4)
        ev_close = int(is_ev and ev_distance_normalized < 0.1)
        speed_anomaly_in_traffic = int(is_speed_anomaly and traffic_encoded >= 2)
        speed_anomaly_clear = int(is_speed_anomaly and traffic_encoded == 0)
        impact_rough_road = int(is_impact and road_type in ('residential', 'downtown'))

        # --- Assemble in correct column order ---
        features = np.array([[
            speed_ratio, speed_deviation, is_stopped, expected_stopped,
            road_type_encoded, traffic_encoded, construction_encoded,
            notification_type_encoded, notification_subtype_encoded,
            ev_distance_normalized, pedestrian_density, object_in_path_int,
            time_since_stop_normalized,
            hour_sin, hour_cos,
            high_traffic, high_pedestrian,
            stuck_in_traffic, stuck_in_construction, stuck_clear_road,
            object_query_high_ped, object_query_low_ped, object_query_moving,
            ev_far_away, ev_close,
            speed_anomaly_in_traffic, speed_anomaly_clear,
            impact_rough_road,
        ]])

        return features

    def predict(self, payload: dict) -> dict:
        """
        Full prediction pipeline:
        raw payload -> feature engineering -> predict -> result
        """
        # Step 1: Engineer features
        features = self.engineer_features(payload)

        # Step 2: Predict (raw features — the model is trained unscaled)
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_columns)
        raw_score = float(self.model.predict(dmatrix)[0])

        # Step 3: Apply threshold
        needs_intervention = raw_score >= self.threshold

        return {
            'needs_intervention': needs_intervention,
            'confidence': raw_score if needs_intervention else (1.0 - raw_score),
            'raw_score': raw_score,
        }
