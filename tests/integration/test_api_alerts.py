"""
Integration tests for GET /api/alerts endpoint.
"""

import pytest


class TestAlertsEndpoint:

    def test_default_params(self, client, mock_db_service):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        mock_db_service.get_recent_alerts.assert_called_once_with(
            limit=50, offset=0, notification_type=None
        )

    def test_custom_limit(self, client, mock_db_service):
        resp = client.get("/api/alerts?limit=10")
        assert resp.status_code == 200
        mock_db_service.get_recent_alerts.assert_called_once_with(
            limit=10, offset=0, notification_type=None
        )

    def test_custom_offset(self, client, mock_db_service):
        resp = client.get("/api/alerts?offset=5")
        assert resp.status_code == 200
        mock_db_service.get_recent_alerts.assert_called_once_with(
            limit=50, offset=5, notification_type=None
        )

    def test_limit_max_boundary(self, client):
        resp = client.get("/api/alerts?limit=500")
        assert resp.status_code == 200

        resp = client.get("/api/alerts?limit=501")
        assert resp.status_code == 422

    def test_limit_min_boundary(self, client):
        resp = client.get("/api/alerts?limit=1")
        assert resp.status_code == 200

        resp = client.get("/api/alerts?limit=0")
        assert resp.status_code == 422

    def test_offset_negative(self, client):
        resp = client.get("/api/alerts?offset=-1")
        assert resp.status_code == 422

    def test_type_filter(self, client, mock_db_service):
        resp = client.get("/api/alerts?notification_type=stuck")
        assert resp.status_code == 200
        mock_db_service.get_recent_alerts.assert_called_once_with(
            limit=50, offset=0, notification_type="stuck"
        )

    def test_response_structure(self, client):
        resp = client.get("/api/alerts")
        data = resp.json()
        assert "alerts" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
