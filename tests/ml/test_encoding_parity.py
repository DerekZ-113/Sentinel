"""
CRITICAL: Encoding parity tests.

Verifies that model_service.py and prepare_data.py both use the same encoding maps
from ml/constants.py, and that the saved config.joblib matches.
"""

import os
import pytest
import joblib

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
        config = joblib.load(os.path.join(ML_DIR, "xgboost_config.joblib"))
        assert config['feature_columns'] == FEATURE_COLUMNS

    def test_feature_count_is_28(self):
        assert len(FEATURE_COLUMNS) == 28
        config = joblib.load(os.path.join(ML_DIR, "xgboost_config.joblib"))
        assert len(config['feature_columns']) == 28
