"""
Integration tests for GET /health endpoint.
"""

import pytest


class TestHealthEndpoint:

    def test_healthy(self, client, mock_db_service):
        mock_db_service.health_check.return_value = True
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["db_connected"] is True

    def test_degraded_no_db(self, client, mock_db_service):
        mock_db_service.health_check.return_value = False
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["db_connected"] is False

    def test_model_features_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_features"] == 28

    def test_model_threshold(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_threshold"] == 0.5

    def test_uptime_positive(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] > 0
