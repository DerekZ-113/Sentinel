"""
CRITICAL: Encoding parity tests.

Verifies that model_service.py and prepare_data.py both use the same encoding
maps from ml/constants.py, that the saved model_config.json matches, and that
the two feature-engineering implementations produce identical vectors.
"""

import json
import os
import numpy as np
import pandas as pd

from ml.constants import (
    ROAD_TYPE_MAP, TRAFFIC_MAP, CONSTRUCTION_MAP,
    NOTIFICATION_TYPE_MAP, NOTIFICATION_SUBTYPE_MAP,
    FEATURE_COLUMNS,
)
from api.services.model_service import (
    ROAD_TYPE_MAP as MS_ROAD_TYPE_MAP,
    TRAFFIC_MAP as MS_TRAFFIC_MAP,
    CONSTRUCTION_MAP as MS_CONSTRUCTION_MAP,
    NOTIFICATION_TYPE_MAP as MS_NOTIFICATION_TYPE_MAP,
    NOTIFICATION_SUBTYPE_MAP as MS_NOTIFICATION_SUBTYPE_MAP,
)

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")


def _load_config():
    with open(os.path.join(ML_DIR, "model_config.json")) as f:
        return json.load(f)


class TestEncodingParity:
    """Verify model_service imports the same maps as ml.constants."""

    def test_road_type_map_parity(self):
        assert MS_ROAD_TYPE_MAP is ROAD_TYPE_MAP

    def test_traffic_map_parity(self):
        assert MS_TRAFFIC_MAP is TRAFFIC_MAP

    def test_construction_map_parity(self):
        assert MS_CONSTRUCTION_MAP is CONSTRUCTION_MAP

    def test_notification_type_map_parity(self):
        assert MS_NOTIFICATION_TYPE_MAP is NOTIFICATION_TYPE_MAP

    def test_notification_subtype_map_parity(self):
        assert MS_NOTIFICATION_SUBTYPE_MAP is NOTIFICATION_SUBTYPE_MAP

    def test_feature_column_order_parity(self):
        """Config saved by train_classifier must match constants."""
        config = _load_config()
        assert config['feature_columns'] == FEATURE_COLUMNS

    def test_feature_count_is_28(self):
        assert len(FEATURE_COLUMNS) == 28
        config = _load_config()
        assert len(config['feature_columns']) == 28


class TestFeatureVectorParity:
    """The training path (prepare_data.engineer_features on a DataFrame) and
    the serving path (model_service.engineer_features on a payload) must
    produce the same 28-vector for the same notification. With no scaler
    between them, any drift here reaches the model directly."""

    PAYLOAD = {
        "vehicle_id": "v001",
        "speed": 3.2,
        "expected_speed": 35.0,
        "road_type": "downtown",
        "traffic_condition": "heavy",
        "construction_zone": "temporary",
        "notification_type": "verification_request",
        "notification_subtype": "object_query",
        "ev_distance": None,
        "pedestrian_density": 0.62,
        "object_in_path": True,
        "time_since_stop": 45.0,
        "hour_of_day": 17,
    }

    def test_training_and_serving_vectors_match(self, model_service):
        from ml.prepare_data import engineer_features

        serving_vec = model_service.engineer_features(self.PAYLOAD)[0]

        row = dict(self.PAYLOAD)
        row["needs_intervention"] = True  # label; not a feature
        df = engineer_features(pd.DataFrame([row]))
        training_vec = df[FEATURE_COLUMNS].values[0].astype(float)

        assert np.allclose(training_vec, serving_vec, atol=1e-9), (
            f"training={training_vec}\nserving={serving_vec}"
        )
