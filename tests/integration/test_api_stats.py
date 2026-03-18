"""
Integration tests for stats endpoints:
  GET /api/stats
  GET /api/stats/model-health
  GET /api/stats/{notification_type}
"""

import pytest


class TestStatsEndpoint:

    def test_default_hours(self, client, mock_db_service):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        mock_db_service.get_stats.assert_called_once_with(hours=24)

    def test_custom_hours(self, client, mock_db_service):
        resp = client.get("/api/stats?hours=72")
        assert resp.status_code == 200
        mock_db_service.get_stats.assert_called_once_with(hours=72)

    def test_hours_min_boundary(self, client):
        resp = client.get("/api/stats?hours=1")
        assert resp.status_code == 200

        resp = client.get("/api/stats?hours=0")
        assert resp.status_code == 422

    def test_hours_max_boundary(self, client):
        resp = client.get("/api/stats?hours=720")
        assert resp.status_code == 200

        resp = client.get("/api/stats?hours=721")
        assert resp.status_code == 422

    def test_response_keys(self, client):
        resp = client.get("/api/stats")
        data = resp.json()
        assert "time_window_hours" in data
        assert "total_alerts" in data
        assert "total_flagged" in data
        assert "total_suppressed" in data
        assert "overall_fp_rate" in data
        assert "by_type" in data


class TestModelHealthEndpoint:

    def test_endpoint(self, client):
        resp = client.get("/api/stats/model-health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "total_predictions" in data
        assert "confidence_buckets" in data
        assert "flagged_by_type" in data

    def test_custom_hours(self, client, mock_db_service):
        resp = client.get("/api/stats/model-health?hours=48")
        assert resp.status_code == 200
        mock_db_service.get_model_health.assert_called_once_with(hours=48)


class TestStatsByTypeEndpoint:

    def test_valid_type(self, client, mock_db_service):
        resp = client.get("/api/stats/stuck")
        assert resp.status_code == 200
        data = resp.json()
        assert data["notification_type"] == "stuck"

    def test_unknown_type_returns_200(self, client, mock_db_service):
        """Unknown type still returns 200 with 0 counts, not 404."""
        mock_db_service.get_stats_by_type.return_value = {
            "notification_type": "nonexistent",
            "total": 0, "flagged": 0, "suppressed": 0,
            "avg_confidence": None, "accuracy": None,
        }
        resp = client.get("/api/stats/nonexistent")
        assert resp.status_code == 200
