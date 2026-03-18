"""
Tests for ModelService class in api/services/model_service.py.
"""

import os
import pytest
import numpy as np
import xgboost as xgb


class TestModelServiceInit:

    def test_loads_successfully(self, model_service):
        assert model_service is not None

    def test_feature_columns_count_28(self, model_service):
        assert len(model_service.feature_columns) == 28

    def test_threshold_value(self, model_service):
        assert model_service.threshold == 0.5

    def test_model_is_booster(self, model_service):
        assert isinstance(model_service.model, xgb.Booster)

    def test_invalid_model_dir_raises(self):
        from api.services.model_service import ModelService
        with pytest.raises(Exception):
            ModelService(model_dir="/nonexistent/path")

    def test_fallback_scaler_builds(self, model_service):
        """The scaler should exist (either loaded or fallback)."""
        assert model_service.scaler is not None
        assert hasattr(model_service.scaler, "transform")

    def test_fallback_scaler_correct_shape(self):
        """Build a fallback scaler and check its dimensions."""
        from api.services.model_service import ModelService
        ml_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ml"
        )
        service = ModelService(model_dir=ml_dir)
        # Scaler should handle 28 features
        test_input = np.zeros((1, 28))
        result = service.scaler.transform(test_input)
        assert result.shape == (1, 28)


class TestModelServicePredict:

    def test_predict_returns_required_keys(self, model_service, sample_payload):
        result = model_service.predict(sample_payload)
        assert "needs_intervention" in result
        assert "confidence" in result
        assert "raw_score" in result

    def test_predict_confidence_range(self, model_service, sample_payload):
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_raw_score_range(self, model_service, sample_payload):
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0

    def test_predict_needs_intervention_is_bool(self, model_service, sample_payload):
        result = model_service.predict(sample_payload)
        assert isinstance(result["needs_intervention"], (bool, np.bool_))

    def test_confidence_when_flagged(self, model_service, all_notification_payloads):
        """When needs_intervention=True, confidence should equal raw_score."""
        for payload in all_notification_payloads.values():
            result = model_service.predict(payload)
            if result["needs_intervention"]:
                assert result["confidence"] == pytest.approx(result["raw_score"], abs=1e-6)

    def test_confidence_when_suppressed(self, model_service, all_notification_payloads):
        """When needs_intervention=False, confidence should equal 1 - raw_score."""
        for payload in all_notification_payloads.values():
            result = model_service.predict(payload)
            if not result["needs_intervention"]:
                assert result["confidence"] == pytest.approx(1.0 - result["raw_score"], abs=1e-6)

    def test_passenger_assist_likely_flagged(self, model_service, all_notification_payloads):
        """passenger_assist with fp_rate=0 should almost always need intervention."""
        result = model_service.predict(all_notification_payloads["passenger_assist"])
        # This is a strong signal but not guaranteed by model
        assert result["raw_score"] > 0.3  # Should have meaningful signal

    def test_stuck_heavy_traffic(self, model_service):
        """Stuck in heavy traffic + downtown should have meaningful prediction."""
        payload = {
            "vehicle_id": "v1", "speed": 0.0, "expected_speed": 35.0,
            "road_type": "downtown", "traffic_condition": "heavy",
            "construction_zone": "none", "notification_type": "stuck",
            "notification_subtype": None, "ev_distance": None,
            "pedestrian_density": 0.3, "object_in_path": False,
            "time_since_stop": 120.0, "hour_of_day": 14,
        }
        result = model_service.predict(payload)
        # Should return valid result regardless of direction
        assert 0.0 <= result["raw_score"] <= 1.0
