"""
Sentinel Model Service

Loads trained XGBoost model and serves predictions.
Handles feature engineering from raw notification data.
"""

import os
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime


# ============================================================================
# ENCODING MAPS (must match prepare_data.py exactly)
# ============================================================================

ROAD_TYPE_MAP = {
    'highway': 0, 'main_road': 1, 'residential': 2,
    'downtown': 3, 'school_zone': 4
}

TRAFFIC_MAP = {
    'light': 0, 'moderate': 1, 'heavy': 2, 'standstill': 3
}

CONSTRUCTION_MAP = {
    'none': 0, 'temporary': 1, 'persistent': 2, 'flagger': 3
}

NOTIFICATION_TYPE_MAP = {
    'verification_request': 1, 'emergency_vehicle_alert': 2,
    'stuck': 3, 'speed_anomaly': 4, 'impact_l0': 5, 'passenger_assist': 6
}

NOTIFICATION_SUBTYPE_MAP = {
    None: 0, 'object_query': 1,
    'traffic_signal_verify': 2, 'lane_mapping_verify': 3
}


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
        config_path = os.path.join(model_dir, 'xgboost_config.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')

        # Load model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load config (feature columns + threshold)
        config = joblib.load(config_path)
        self.feature_columns = config['feature_columns']
        self.threshold = config.get('threshold', 0.5)
        self.best_iteration = config.get('best_iteration', 0)

        # Load scaler if available
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from {scaler_path}")
        else:
            self.scaler = self._build_fallback_scaler()
            print("⚠️  No scaler found — using reconstructed fallback")

        print(f"✅ Model loaded: {len(self.feature_columns)} features, "
              f"threshold={self.threshold}, best_iter={self.best_iteration}")

    def _build_fallback_scaler(self):
        """
        Reconstruct a MinMaxScaler from known feature ranges.
        Used when the original scaler file isn't available.
        
        These ranges are derived from the data generation code
        and feature engineering in prepare_data.py.
        """
        from sklearn.preprocessing import MinMaxScaler

        # Build a 2-row array with [min_values, max_values] for each feature
        # then fit the scaler on it — this gives the same result as fitting
        # on data with those min/max values
        feature_ranges = {
            'speed_ratio':                  (0.0, 3.0),
            'speed_deviation':              (-65.0, 15.0),
            'is_stopped':                   (0, 1),
            'expected_stopped':             (0, 1),
            'road_type_encoded':            (0, 4),
            'traffic_encoded':              (0, 3),
            'construction_encoded':         (0, 3),
            'notification_type_encoded':    (1, 6),    # Training data has no 0 (no-notification)
            'notification_subtype_encoded': (0, 3),
            'ev_distance_normalized':       (0.0, 2.0),
            'pedestrian_density':           (0.0, 1.0),
            'object_in_path':               (0, 1),
            'time_since_stop_normalized':   (0.0, 2.0),
            'hour_sin':                     (-1.0, 1.0),
            'hour_cos':                     (-1.0, 1.0),
            'high_traffic':                 (0, 1),
            'high_pedestrian':              (0, 1),
            'stuck_in_traffic':             (0, 1),
            'stuck_in_construction':        (0, 1),
            'stuck_clear_road':             (0, 1),
            'object_query_high_ped':        (0, 1),
            'object_query_low_ped':         (0, 1),
            'object_query_moving':          (0, 1),
            'ev_far_away':                  (0, 1),
            'ev_close':                     (0, 1),
            'speed_anomaly_in_traffic':     (0, 1),
            'speed_anomaly_clear':          (0, 1),
            'impact_rough_road':            (0, 1),
        }

        mins = [feature_ranges[col][0] for col in self.feature_columns]
        maxs = [feature_ranges[col][1] for col in self.feature_columns]

        scaler = MinMaxScaler()
        scaler.fit(np.array([mins, maxs]))
        return scaler

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
        hour_of_day = payload.get('hour_of_day') or datetime.now().hour

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
        ev_distance_normalized = min((ev_distance or 999) / 500.0, 2.0)
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
        raw payload -> feature engineering -> scale -> predict -> result
        """
        # Step 1: Engineer features
        features = self.engineer_features(payload)

        # Step 2: Scale
        features_scaled = self.scaler.transform(features)

        # Step 3: Predict
        dmatrix = xgb.DMatrix(features_scaled, feature_names=self.feature_columns)
        raw_score = float(self.model.predict(dmatrix)[0])

        # Step 4: Apply threshold
        needs_intervention = raw_score >= self.threshold

        return {
            'needs_intervention': needs_intervention,
            'confidence': raw_score if needs_intervention else (1.0 - raw_score),
            'raw_score': raw_score,
        }
