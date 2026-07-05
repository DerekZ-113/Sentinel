"""
Tests for DatabaseService in api/services/db_service.py.

All database interactions are mocked — no real DB needed.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone
import json


# ============================================================================
# HELPERS
# ============================================================================

def _create_db_service():
    """Create DatabaseService with fully mocked psycopg2."""
    with patch("api.services.db_service.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn

        from api.services.db_service import DatabaseService
        service = DatabaseService(host="testhost", port=5432, database="testdb",
                                   user="testuser", password="testpass")

        return service, mock_pool, mock_conn, mock_cursor


# ============================================================================
# INIT
# ============================================================================

class TestDatabaseServiceInit:

    def test_init_creates_pool(self):
        with patch("api.services.db_service.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool
            mock_conn = MagicMock()
            mock_pool.getconn.return_value = mock_conn

            from api.services.db_service import DatabaseService
            DatabaseService(host="h", port=5432, database="d", user="u", password="p")

            mock_pool_cls.assert_called_once()
            args, kwargs = mock_pool_cls.call_args
            assert kwargs["host"] == "h"
            assert kwargs["password"] == "p"

    def test_init_ensures_table(self):
        service, _, _, mock_cursor = _create_db_service()
        # _ensure_predictions_table should have executed CREATE TABLE
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS predictions" in c for c in calls)

    def test_init_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "envhost")
        monkeypatch.setenv("DB_PORT", "9999")

        with patch("api.services.db_service.pool.ThreadedConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool
            mock_conn = MagicMock()
            mock_pool.getconn.return_value = mock_conn

            from api.services.db_service import DatabaseService
            DatabaseService()

            _, kwargs = mock_pool_cls.call_args
            assert kwargs["host"] == "envhost"
            assert kwargs["port"] == 9999


# ============================================================================
# STORE PREDICTION
# ============================================================================

class TestStorePrediction:

    def test_returns_id(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.fetchone.return_value = (42,)

        payload = {"vehicle_id": "v1", "notification_type": "stuck"}
        prediction = {"needs_intervention": True, "confidence": 0.9, "raw_score": 0.9}

        result = service.store_prediction(payload, prediction)
        assert result == 42

    def test_insert_sql(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.fetchone.return_value = (1,)

        payload = {"vehicle_id": "v1", "notification_type": "stuck"}
        prediction = {"needs_intervention": True, "confidence": 0.9, "raw_score": 0.9}

        service.store_prediction(payload, prediction)

        # Find the INSERT call (skip the CREATE TABLE call from init)
        insert_calls = [c for c in mock_cursor.execute.call_args_list
                        if "INSERT INTO predictions" in str(c)]
        assert len(insert_calls) == 1

    def test_commits(self):
        service, _, mock_conn, mock_cursor = _create_db_service()
        mock_cursor.fetchone.return_value = (1,)

        payload = {"vehicle_id": "v1", "notification_type": "stuck"}
        prediction = {"needs_intervention": True, "confidence": 0.9, "raw_score": 0.9}

        service.store_prediction(payload, prediction)
        mock_conn.commit.assert_called()

    def test_returns_connection_to_pool(self):
        service, mock_pool, mock_conn, mock_cursor = _create_db_service()
        mock_cursor.fetchone.return_value = (1,)

        payload = {"vehicle_id": "v1", "notification_type": "stuck"}
        prediction = {"needs_intervention": True, "confidence": 0.9, "raw_score": 0.9}

        service.store_prediction(payload, prediction)
        mock_pool.putconn.assert_called_with(mock_conn)


# ============================================================================
# GET RECENT ALERTS
# ============================================================================

class TestGetRecentAlerts:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_default_params(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 0}
        mock_cursor.fetchall.return_value = []

        result = service.get_recent_alerts()
        assert result["limit"] == 50
        assert result["offset"] == 0

    def test_with_type_filter(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 0}
        mock_cursor.fetchall.return_value = []

        service.get_recent_alerts(notification_type="stuck")

        # One of the execute calls should contain WHERE notification_type
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("WHERE notification_type" in c for c in calls)

    def test_without_filter(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 5}
        mock_cursor.fetchall.return_value = []

        result = service.get_recent_alerts(notification_type=None)
        assert result["total"] == 5

    def test_pagination(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 100}
        mock_cursor.fetchall.return_value = []

        result = service.get_recent_alerts(limit=10, offset=20)
        assert result["limit"] == 10
        assert result["offset"] == 20


# ============================================================================
# GET STATS
# ============================================================================

class TestGetStats:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_required_keys(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.side_effect = [
            {"total": 100, "flagged": 30, "suppressed": 70, "correct": 80, "with_ground_truth": 90},
            {"false_positives": 3, "predicted_positive": 10},
        ]
        mock_cursor.fetchall.return_value = []

        result = service.get_stats(hours=24)
        assert "time_window_hours" in result
        assert "total_alerts" in result
        assert "total_flagged" in result
        assert "total_suppressed" in result
        assert "overall_fp_rate" in result
        assert "by_type" in result

    def test_fp_rate_calculation(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.side_effect = [
            {"total": 100, "flagged": 30, "suppressed": 70, "correct": 80, "with_ground_truth": 90},
            {"false_positives": 3, "predicted_positive": 10},
        ]
        mock_cursor.fetchall.return_value = []

        result = service.get_stats()
        assert result["overall_fp_rate"] == pytest.approx(0.3, abs=1e-6)

    def test_fp_rate_none_when_no_positives(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.side_effect = [
            {"total": 100, "flagged": 0, "suppressed": 100, "correct": 0, "with_ground_truth": 0},
            {"false_positives": 0, "predicted_positive": 0},
        ]
        mock_cursor.fetchall.return_value = []

        result = service.get_stats()
        assert result["overall_fp_rate"] is None


# ============================================================================
# GET STATS BY TYPE
# ============================================================================

class TestGetStatsByType:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    ROW = {
        "total": 50, "flagged": 15, "suppressed": 35,
        "avg_confidence": 0.85, "correct": 8, "with_ground_truth": 10,
        "false_positives": 3, "flagged_with_truth": 12,
    }

    def test_required_keys(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = dict(self.ROW)

        result = service.get_stats_by_type("stuck")
        assert result["notification_type"] == "stuck"
        assert "total" in result
        assert "flagged" in result
        assert "accuracy" in result
        assert "fp_rate" in result
        assert "avg_confidence" in result

    def test_accuracy_calculation(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = dict(self.ROW)

        result = service.get_stats_by_type("stuck")
        assert result["accuracy"] == pytest.approx(0.8, abs=1e-6)

    def test_fp_rate_calculation(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = dict(self.ROW)

        result = service.get_stats_by_type("stuck")
        assert result["fp_rate"] == pytest.approx(0.25, abs=1e-6)

    def test_accuracy_none_no_ground_truth(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            **self.ROW, "correct": 0, "with_ground_truth": 0,
            "false_positives": 0, "flagged_with_truth": 0,
        }

        result = service.get_stats_by_type("stuck")
        assert result["accuracy"] is None
        assert result["fp_rate"] is None


# ============================================================================
# GET MODEL HEALTH
# ============================================================================

class TestGetModelHealth:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_status_degraded(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 100, "flagged": 30, "suppressed": 70,
            "avg_confidence": 0.85, "correct": 50, "with_ground_truth": 100,
            "high_conf": 60, "med_conf": 30, "low_conf": 10,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["status"] == "degraded"  # accuracy=0.5 < 0.6

    def test_status_warning_low_confidence(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 100, "flagged": 30, "suppressed": 70,
            "avg_confidence": 0.65, "correct": 80, "with_ground_truth": 100,
            "high_conf": 20, "med_conf": 30, "low_conf": 50,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["status"] == "warning"

    def test_status_warning_no_data(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 0, "flagged": 0, "suppressed": 0,
            "avg_confidence": None, "correct": 0, "with_ground_truth": 0,
            "high_conf": 0, "med_conf": 0, "low_conf": 0,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["status"] == "warning"

    def test_status_healthy(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 100, "flagged": 30, "suppressed": 70,
            "avg_confidence": 0.85, "correct": 80, "with_ground_truth": 100,
            "high_conf": 60, "med_conf": 30, "low_conf": 10,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["status"] == "healthy"

    def test_confidence_buckets(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 100, "flagged": 30, "suppressed": 70,
            "avg_confidence": 0.85, "correct": 80, "with_ground_truth": 100,
            "high_conf": 60, "med_conf": 30, "low_conf": 10,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["confidence_buckets"] == {"high": 60, "medium": 30, "low": 10}

    def test_pct_calculations(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 100, "flagged": 30, "suppressed": 70,
            "avg_confidence": 0.85, "correct": 80, "with_ground_truth": 100,
            "high_conf": 60, "med_conf": 30, "low_conf": 10,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["pct_flagged"] == pytest.approx(30.0, abs=0.1)
        assert result["pct_suppressed"] == pytest.approx(70.0, abs=0.1)

    def test_zero_total(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {
            "total": 0, "flagged": 0, "suppressed": 0,
            "avg_confidence": None, "correct": 0, "with_ground_truth": 0,
            "high_conf": 0, "med_conf": 0, "low_conf": 0,
        }
        mock_cursor.fetchall.return_value = []

        result = service.get_model_health()
        assert result["pct_flagged"] == 0
        assert result["pct_suppressed"] == 0


# ============================================================================
# HEALTH CHECK
# ============================================================================

class TestHealthCheck:

    def test_success(self):
        service, _, _, mock_cursor = _create_db_service()
        assert service.health_check() is True

    def test_exception_returns_false(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.execute.side_effect = Exception("connection lost")
        # health_check is called after init, so we need to reset side_effect
        # after init has already run
        mock_cursor.execute.side_effect = Exception("connection lost")
        assert service.health_check() is False

    def test_getconn_failure_returns_false(self):
        """During a real outage getconn() itself raises (pool reconnects to
        a dead server). /health must see False, not an unhandled exception."""
        service, mock_pool, _, _ = _create_db_service()
        mock_pool.getconn.side_effect = Exception("could not connect to server")
        assert service.health_check() is False

    def test_getconn_failure_does_not_return_none_to_pool(self):
        service, mock_pool, _, _ = _create_db_service()
        mock_pool.getconn.side_effect = Exception("could not connect to server")
        mock_pool.putconn.reset_mock()
        service.health_check()
        mock_pool.putconn.assert_not_called()


# ============================================================================
# COUNT CAP (B11)
# ============================================================================

class TestAlertsCountCap:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_count_query_is_bounded(self):
        """The total count must not be an unbounded COUNT(*) full scan."""
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 3}
        mock_cursor.fetchall.return_value = []

        service.get_recent_alerts()

        count_calls = [c for c in mock_cursor.execute.call_args_list
                       if "COUNT(*)" in str(c)]
        assert count_calls, "expected a count query"
        assert "LIMIT" in str(count_calls[-1])

    def test_total_reports_cap_when_saturated(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.return_value = {"cnt": 10_000}
        mock_cursor.fetchall.return_value = []

        result = service.get_recent_alerts()
        assert result["total"] == 10_000


# ============================================================================
# GET STATS — PER-TYPE FP RATE / ACCURACY (H2)
# ============================================================================

class TestGetStatsByTypeBreakdown:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_by_type_includes_fp_rate_and_accuracy(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.side_effect = [
            {"total": 100, "flagged": 30, "suppressed": 70, "correct": 80, "with_ground_truth": 90},
            {"false_positives": 3, "predicted_positive": 10},
        ]
        mock_cursor.fetchall.return_value = [{
            "notification_type": "stuck", "total": 40, "flagged": 10,
            "suppressed": 30, "false_positives": 2, "flagged_with_truth": 8,
            "correct": 30, "with_ground_truth": 36,
        }]

        result = service.get_stats()
        entry = result["by_type"][0]
        assert entry["fp_rate"] == pytest.approx(0.25, abs=1e-6)
        assert entry["accuracy"] == pytest.approx(30 / 36, abs=1e-4)

    def test_by_type_rates_none_without_ground_truth(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchone.side_effect = [
            {"total": 100, "flagged": 30, "suppressed": 70, "correct": 0, "with_ground_truth": 0},
            {"false_positives": 0, "predicted_positive": 0},
        ]
        mock_cursor.fetchall.return_value = [{
            "notification_type": "stuck", "total": 40, "flagged": 10,
            "suppressed": 30, "false_positives": 0, "flagged_with_truth": 0,
            "correct": 0, "with_ground_truth": 0,
        }]

        result = service.get_stats()
        entry = result["by_type"][0]
        assert entry["fp_rate"] is None
        assert entry["accuracy"] is None


# ============================================================================
# FP OVER TIME (B7 — SQL-side bucketing)
# ============================================================================

class TestGetFPOverTime:

    def _setup(self):
        service, _, _, mock_cursor = _create_db_service()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        return service, mock_cursor

    def test_returns_requested_bucket_count_with_empty_table(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchall.return_value = []

        result = service.get_fp_over_time(hours=24, buckets=12)
        assert result["time_window_hours"] == 24
        assert len(result["buckets"]) == 12
        for b in result["buckets"]:
            assert b["total"] == 0
            assert b["flagged"] == 0
            assert b["suppressed"] == 0
            assert b["fp_rate"] is None
            assert b["accuracy"] is None

    def test_bucket_metrics_from_sql_aggregates(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchall.return_value = [
            {"bucket_idx": 0, "total": 20, "flagged": 10, "false_positives": 2,
             "flagged_with_truth": 8, "correct": 15, "with_truth": 18},
            {"bucket_idx": 11, "total": 5, "flagged": 5, "false_positives": 0,
             "flagged_with_truth": 0, "correct": 0, "with_truth": 0},
        ]

        result = service.get_fp_over_time(hours=24, buckets=12)
        first, last = result["buckets"][0], result["buckets"][11]

        assert first["total"] == 20
        assert first["suppressed"] == 10
        assert first["fp_rate"] == pytest.approx(0.25, abs=1e-6)
        assert first["accuracy"] == pytest.approx(15 / 18, abs=1e-4)
        # Bucket with flagged rows but no ground truth: rates are None
        assert last["fp_rate"] is None
        assert last["accuracy"] is None
        # Buckets with no rows at all are zero-filled
        assert result["buckets"][5]["total"] == 0

    def test_bucketing_happens_in_sql(self):
        """The query must aggregate (GROUP BY), not fetch raw rows."""
        service, mock_cursor = self._setup()
        mock_cursor.fetchall.return_value = []

        service.get_fp_over_time()

        fp_calls = [str(c) for c in mock_cursor.execute.call_args_list
                    if "bucket_idx" in str(c)]
        assert fp_calls, "expected the bucketing query"
        assert "GROUP BY" in fp_calls[-1]

    def test_bucket_times_are_evenly_spaced(self):
        service, mock_cursor = self._setup()
        mock_cursor.fetchall.return_value = []

        result = service.get_fp_over_time(hours=6, buckets=12)
        times = [datetime.fromisoformat(b["time"]) for b in result["buckets"]]
        gaps = {(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)}
        assert gaps == {1800.0}  # 6h / 12 buckets = 30 min


# ============================================================================
# CLOSE
# ============================================================================

class TestClose:

    def test_close_calls_closeall(self):
        service, mock_pool, _, _ = _create_db_service()
        service.close()
        mock_pool.closeall.assert_called_once()
