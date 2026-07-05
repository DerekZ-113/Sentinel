"""
Real-SQL integration tests for DatabaseService.

Every other DB test in this suite mocks at the pool level — these run the
actual SQL against a real PostgreSQL/TimescaleDB (a typo in the stats SQL
passes every mocked test green).

Opt-in: set SENTINEL_DB_TESTS=1 with DB_* env vars pointing at a scratch
database (CI provides a service container; locally `docker-compose up db`).
The predictions table is TRUNCATED — never point this at data you care about.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SENTINEL_DB_TESTS") != "1",
    reason="real-database tests are opt-in (SENTINEL_DB_TESTS=1)",
)


def _payload(vehicle_id="v001", ntype="stuck", actual=None):
    return {
        "vehicle_id": vehicle_id,
        "notification_type": ntype,
        "notification_subtype": None,
        "speed": 0.0,
        "expected_speed": 35.0,
        "road_type": "downtown",
        "traffic_condition": "heavy",
        "construction_zone": "none",
        "pedestrian_density": 0.3,
        "ev_distance": None,
        "object_in_path": False,
        "time_since_stop": 60.0,
        "needs_intervention_actual": actual,
    }


def _prediction(flag=True, confidence=0.9):
    return {
        "needs_intervention": flag,
        "confidence": confidence,
        "raw_score": confidence if flag else 1 - confidence,
    }


@pytest.fixture(scope="module")
def db():
    from api.services.db_service import DatabaseService
    service = DatabaseService()
    yield service
    service.close()


@pytest.fixture(autouse=True)
def clean_predictions(db):
    conn = db._get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE predictions RESTART IDENTITY")
        conn.commit()
        cursor.close()
    finally:
        db._put_conn(conn)


def _insert_at(db, at, ntype="stuck", predicted=True, actual=None, confidence=0.9):
    """Insert directly with a controlled timestamp (store_prediction uses NOW())."""
    conn = db._get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                time, vehicle_id, notification_type,
                needs_intervention_predicted, needs_intervention_actual,
                confidence, raw_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (at, "v001", ntype, predicted, actual, confidence, confidence))
        conn.commit()
        cursor.close()
    finally:
        db._put_conn(conn)


class TestHealthAndRoundTrip:

    def test_health_check(self, db):
        assert db.health_check() is True

    def test_store_and_fetch_round_trip(self, db):
        row_id = db.store_prediction(_payload(), _prediction())
        assert isinstance(row_id, int)

        result = db.get_recent_alerts()
        assert result["total"] == 1
        alert = result["alerts"][0]
        assert alert["vehicle_id"] == "v001"
        assert alert["notification_type"] == "stuck"
        assert alert["needs_intervention_predicted"] is True

    def test_pagination_and_filter(self, db):
        for i in range(5):
            db.store_prediction(_payload(vehicle_id=f"v{i:03d}"), _prediction())
        db.store_prediction(_payload(ntype="speed_anomaly"), _prediction(flag=False))

        page = db.get_recent_alerts(limit=2, offset=2)
        assert page["total"] == 6
        assert len(page["alerts"]) == 2

        filtered = db.get_recent_alerts(notification_type="speed_anomaly")
        assert filtered["total"] == 1
        assert filtered["alerts"][0]["notification_type"] == "speed_anomaly"


class TestStatsRealSQL:

    def test_get_stats_counts_and_rates(self, db):
        # 4 stuck: 3 flagged (1 an actual FP), 1 suppressed; all with truth
        db.store_prediction(_payload(actual=True), _prediction(True))
        db.store_prediction(_payload(actual=True), _prediction(True))
        db.store_prediction(_payload(actual=False), _prediction(True))   # false positive
        db.store_prediction(_payload(actual=False), _prediction(False))  # correct suppress
        # 1 speed_anomaly without ground truth
        db.store_prediction(_payload(ntype="speed_anomaly"), _prediction(True))

        stats = db.get_stats(hours=24)
        assert stats["total_alerts"] == 5
        assert stats["total_flagged"] == 4
        assert stats["total_suppressed"] == 1
        # FP rate among flagged-with-truth: 1/3
        assert stats["overall_fp_rate"] == pytest.approx(1 / 3, abs=1e-6)

        by_type = {t["notification_type"]: t for t in stats["by_type"]}
        stuck = by_type["stuck"]
        assert stuck["total"] == 4
        assert stuck["fp_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert stuck["accuracy"] == pytest.approx(3 / 4, abs=1e-4)
        # No ground truth → rates are None, not 0
        anomaly = by_type["speed_anomaly"]
        assert anomaly["fp_rate"] is None
        assert anomaly["accuracy"] is None

    def test_get_stats_by_type(self, db):
        db.store_prediction(_payload(actual=True), _prediction(True, 0.8))
        db.store_prediction(_payload(actual=False), _prediction(True, 0.6))

        result = db.get_stats_by_type("stuck", hours=24)
        assert result["total"] == 2
        assert result["flagged"] == 2
        assert result["fp_rate"] == pytest.approx(0.5, abs=1e-6)
        assert result["accuracy"] == pytest.approx(0.5, abs=1e-6)
        assert result["avg_confidence"] == pytest.approx(0.7, abs=1e-6)


class TestFPOverTimeRealSQL:
    """Hand-computed bucket expectations for the SQL bucketing (B7/T2)."""

    def test_bucket_assignment_and_rates(self, db):
        now = datetime.now(timezone.utc)
        # hours=1, buckets=4 → 15-minute buckets. Mid-bucket offsets so
        # wall-clock drift between insert and query can't cross a boundary.
        # bucket 0: two flagged, one an actual FP
        _insert_at(db, now - timedelta(minutes=52.5), predicted=True, actual=False)
        _insert_at(db, now - timedelta(minutes=52.0), predicted=True, actual=True)
        # bucket 2: one suppressed row, correct, with truth
        _insert_at(db, now - timedelta(minutes=22.5), predicted=False, actual=False)
        # bucket 3: one flagged, no ground truth
        _insert_at(db, now - timedelta(minutes=7.5), predicted=True, actual=None)

        result = db.get_fp_over_time(hours=1, buckets=4)
        assert result["time_window_hours"] == 1
        b = result["buckets"]
        assert len(b) == 4

        assert b[0]["total"] == 2
        assert b[0]["flagged"] == 2
        assert b[0]["fp_rate"] == pytest.approx(0.5, abs=1e-6)
        assert b[0]["accuracy"] == pytest.approx(0.5, abs=1e-6)

        assert b[1]["total"] == 0
        assert b[1]["fp_rate"] is None

        assert b[2]["total"] == 1
        assert b[2]["suppressed"] == 1
        assert b[2]["accuracy"] == pytest.approx(1.0, abs=1e-6)

        assert b[3]["total"] == 1
        assert b[3]["flagged"] == 1
        assert b[3]["fp_rate"] is None  # flagged but no ground truth
        assert b[3]["accuracy"] is None

    def test_rows_outside_window_excluded(self, db):
        now = datetime.now(timezone.utc)
        _insert_at(db, now - timedelta(hours=3))
        result = db.get_fp_over_time(hours=1, buckets=4)
        assert all(bucket["total"] == 0 for bucket in result["buckets"])
