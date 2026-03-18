"""
End-to-end model pipeline tests — payload → features → predict with real model.
"""

import pytest


class TestAllNotificationTypes:

    def test_all_types_produce_predictions(self, model_service, all_notification_payloads):
        for ntype, payload in all_notification_payloads.items():
            result = model_service.predict(payload)
            assert "needs_intervention" in result, f"Failed for {ntype}"
            assert 0.0 <= result["raw_score"] <= 1.0, f"Score out of range for {ntype}"

    @pytest.mark.parametrize("road_type", [
        "highway", "main_road", "residential", "downtown", "school_zone"
    ])
    def test_all_road_types(self, model_service, sample_payload, road_type):
        sample_payload["road_type"] = road_type
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0

    @pytest.mark.parametrize("traffic", ["light", "moderate", "heavy", "standstill"])
    def test_all_traffic_conditions(self, model_service, sample_payload, traffic):
        sample_payload["traffic_condition"] = traffic
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0

    @pytest.mark.parametrize("construction", ["none", "temporary", "persistent", "flagger"])
    def test_all_construction_zones(self, model_service, sample_payload, construction):
        sample_payload["construction_zone"] = construction
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0


class TestPredictionBehavior:

    def test_deterministic(self, model_service, sample_payload):
        r1 = model_service.predict(sample_payload)
        r2 = model_service.predict(sample_payload)
        assert r1["raw_score"] == r2["raw_score"]
        assert r1["needs_intervention"] == r2["needs_intervention"]

    def test_extreme_speed_zero(self, model_service, sample_payload):
        sample_payload["speed"] = 0.0
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0

    def test_extreme_speed_high(self, model_service, sample_payload):
        sample_payload["speed"] = 200.0
        result = model_service.predict(sample_payload)
        assert 0.0 <= result["raw_score"] <= 1.0

    def test_ev_close_vs_far(self, model_service):
        base = {
            "vehicle_id": "v1", "speed": 30.0, "expected_speed": 35.0,
            "road_type": "main_road", "traffic_condition": "moderate",
            "construction_zone": "none", "notification_type": "emergency_vehicle_alert",
            "notification_subtype": None, "pedestrian_density": 0.3,
            "object_in_path": False, "time_since_stop": 0.0, "hour_of_day": 14,
        }
        close = {**base, "ev_distance": 20.0}
        far = {**base, "ev_distance": 400.0}

        r_close = model_service.predict(close)
        r_far = model_service.predict(far)
        # Close EV should have higher raw_score (more likely needs intervention)
        assert r_close["raw_score"] > r_far["raw_score"]

    def test_stuck_traffic_vs_clear(self, model_service):
        base = {
            "vehicle_id": "v1", "speed": 0.0, "expected_speed": 35.0,
            "road_type": "main_road", "construction_zone": "none",
            "notification_type": "stuck", "notification_subtype": None,
            "ev_distance": None, "pedestrian_density": 0.3,
            "object_in_path": False, "time_since_stop": 120.0, "hour_of_day": 14,
        }
        heavy = {**base, "traffic_condition": "heavy"}
        light = {**base, "traffic_condition": "light"}

        r_heavy = model_service.predict(heavy)
        r_light = model_service.predict(light)
        # Stuck in heavy traffic is more likely FP → lower raw_score
        assert r_heavy["raw_score"] < r_light["raw_score"]

    def test_object_query_with_object(self, model_service):
        base = {
            "vehicle_id": "v1", "speed": 15.0, "expected_speed": 35.0,
            "road_type": "downtown", "traffic_condition": "moderate",
            "construction_zone": "none",
            "notification_type": "verification_request",
            "notification_subtype": "object_query",
            "ev_distance": None, "pedestrian_density": 0.3,
            "time_since_stop": 0.0, "hour_of_day": 14,
        }
        with_obj = {**base, "object_in_path": True}
        without_obj = {**base, "object_in_path": False}

        r_with = model_service.predict(with_obj)
        r_without = model_service.predict(without_obj)
        # With object in path should have higher raw_score
        assert r_with["raw_score"] > r_without["raw_score"]
