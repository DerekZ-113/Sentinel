"""
Integration tests for POST /api/predict endpoint.
"""

import pytest


class TestPredictEndpoint:

    def test_valid_stuck_payload(self, client, sample_payload):
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["vehicle_id"] == "vehicle_001"
        assert data["notification_type"] == "stuck"
        assert "needs_intervention" in data
        assert "confidence" in data
        assert "raw_score" in data
        assert "timestamp" in data

    def test_verification_request_with_subtype(self, client, all_notification_payloads):
        resp = client.post("/api/predict", json=all_notification_payloads["verification_request"])
        assert resp.status_code == 200
        assert resp.json()["notification_type"] == "verification_request"

    def test_emergency_vehicle_alert(self, client, all_notification_payloads):
        resp = client.post("/api/predict", json=all_notification_payloads["emergency_vehicle_alert"])
        assert resp.status_code == 200

    def test_impact_l0(self, client, all_notification_payloads):
        resp = client.post("/api/predict", json=all_notification_payloads["impact_l0"])
        assert resp.status_code == 200

    def test_passenger_assist(self, client, all_notification_payloads):
        resp = client.post("/api/predict", json=all_notification_payloads["passenger_assist"])
        assert resp.status_code == 200

    def test_stores_in_db(self, client, mock_db_service, sample_payload):
        client.post("/api/predict", json=sample_payload)
        mock_db_service.store_prediction.assert_called_once()

    def test_missing_required_field(self, client):
        resp = client.post("/api/predict", json={"speed": 10, "expected_speed": 35})
        assert resp.status_code == 422

    def test_invalid_speed(self, client, sample_payload):
        sample_payload["speed"] = -1
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 422

    def test_invalid_road_type(self, client, sample_payload):
        sample_payload["road_type"] = "invalid_road"
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 422

    def test_ped_density_boundary(self, client, sample_payload):
        sample_payload["pedestrian_density"] = 1.0
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 200

        sample_payload["pedestrian_density"] = 1.01
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 422

    def test_timestamp_format(self, client, sample_payload):
        resp = client.post("/api/predict", json=sample_payload)
        data = resp.json()
        assert "T" in data["timestamp"]  # ISO format


class TestPredictAuth:

    def test_auth_skipped_when_no_key(self, client, sample_payload):
        """When API_KEY env var is empty, auth is not enforced."""
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 200

    def test_rejected_with_wrong_key(self, client, sample_payload, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-secret-123")
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 401

    def test_accepted_with_correct_key(self, client, sample_payload, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-secret-123")
        resp = client.post(
            "/api/predict", json=sample_payload,
            headers={"X-API-Key": "test-secret-123"}
        )
        assert resp.status_code == 200
