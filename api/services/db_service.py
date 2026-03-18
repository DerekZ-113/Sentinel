"""
Sentinel Database Service

Handles all database operations for the prediction API.
Uses psycopg2 with connection pooling.
"""

import os
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from typing import Optional
import json


logger = logging.getLogger("sentinel.db")


class DatabaseService:
    """Manages DB connections and queries for Sentinel API."""

    def __init__(self, host=None, port=None, database=None,
                 user=None, password=None, min_conn=2, max_conn=10) -> None:
        host = host or os.environ.get('DB_HOST', 'localhost')
        port = port or int(os.environ.get('DB_PORT', '5432'))
        database = database or os.environ.get('DB_NAME', 'postgres')
        user = user or os.environ.get('DB_USER', 'postgres')
        password = password or os.environ.get('DB_PASSWORD', 'password')
        self.connection_pool = pool.ThreadedConnectionPool(
            min_conn, max_conn,
            host=host, port=port, database=database,
            user=user, password=password
        )
        self._ensure_predictions_table()
        logger.info("Database service initialized")

    def _get_conn(self):
        return self.connection_pool.getconn()

    def _put_conn(self, conn) -> None:
        self.connection_pool.putconn(conn)

    def _ensure_predictions_table(self) -> None:
        """Create predictions table if it doesn't exist."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id                          SERIAL PRIMARY KEY,
                    time                        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    vehicle_id                  TEXT NOT NULL,
                    notification_type           TEXT NOT NULL,
                    notification_subtype        TEXT,
                    needs_intervention_predicted BOOLEAN NOT NULL,
                    needs_intervention_actual    BOOLEAN,
                    confidence                  FLOAT NOT NULL,
                    raw_score                   FLOAT NOT NULL,
                    speed                       FLOAT,
                    expected_speed              FLOAT,
                    road_type                   TEXT,
                    traffic_condition           TEXT,
                    construction_zone           TEXT,
                    pedestrian_density          FLOAT,
                    ev_distance                 FLOAT,
                    object_in_path              BOOLEAN,
                    time_since_stop             FLOAT,
                    payload_json                JSONB
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_time
                    ON predictions (time DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_type
                    ON predictions (notification_type, time DESC);
            """)
            conn.commit()
            cursor.close()
        finally:
            self._put_conn(conn)

    # ========================================================================
    # WRITE
    # ========================================================================

    def store_prediction(self, payload: dict, prediction: dict) -> int:
        """
        Store a prediction result in the DB.
        Returns the new row ID.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    vehicle_id, notification_type, notification_subtype,
                    needs_intervention_predicted, needs_intervention_actual,
                    confidence, raw_score,
                    speed, expected_speed, road_type, traffic_condition,
                    construction_zone, pedestrian_density, ev_distance,
                    object_in_path, time_since_stop, payload_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                payload['vehicle_id'],
                payload['notification_type'],
                payload.get('notification_subtype'),
                prediction['needs_intervention'],
                payload.get('needs_intervention_actual'),
                prediction['confidence'],
                prediction['raw_score'],
                payload.get('speed'),
                payload.get('expected_speed'),
                payload.get('road_type'),
                payload.get('traffic_condition'),
                payload.get('construction_zone'),
                payload.get('pedestrian_density'),
                payload.get('ev_distance'),
                payload.get('object_in_path'),
                payload.get('time_since_stop'),
                json.dumps(payload, default=str),
            ))
            row_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            return row_id
        finally:
            self._put_conn(conn)

    # ========================================================================
    # READ
    # ========================================================================

    def get_recent_alerts(self, limit: int = 50, offset: int = 0,
                          notification_type: Optional[str] = None) -> dict:
        """Get recent predictions with optional type filter."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            where_clause = ""
            params = []
            if notification_type:
                where_clause = "WHERE notification_type = %s"
                params.append(notification_type)

            # Get total count
            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM predictions {where_clause}",
                params
            )
            total = cursor.fetchone()['cnt']

            # Get page
            cursor.execute(f"""
                SELECT id, time, vehicle_id, notification_type,
                       notification_subtype, needs_intervention_predicted,
                       needs_intervention_actual, confidence, speed,
                       road_type, traffic_condition
                FROM predictions
                {where_clause}
                ORDER BY time DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])

            alerts = [dict(row) for row in cursor.fetchall()]
            cursor.close()

            return {
                'alerts': alerts,
                'total': total,
                'limit': limit,
                'offset': offset,
            }
        finally:
            self._put_conn(conn)

    def get_stats(self, hours: int = 24) -> dict:
        """Get aggregate stats over a time window."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = true) as flagged,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = false) as suppressed,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                        AND needs_intervention_predicted = needs_intervention_actual
                    ) as correct,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                    ) as with_ground_truth
                FROM predictions
                WHERE time >= %s
            """, (since,))
            overall = dict(cursor.fetchone())

            # FP rate: among predicted positives, how many were actually negative
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual = false
                    ) as false_positives,
                    COUNT(*) as predicted_positive
                FROM predictions
                WHERE time >= %s
                  AND needs_intervention_predicted = true
                  AND needs_intervention_actual IS NOT NULL
            """, (since,))
            fp_row = dict(cursor.fetchone())
            fp_rate = None
            if fp_row['predicted_positive'] and fp_row['predicted_positive'] > 0:
                fp_rate = fp_row['false_positives'] / fp_row['predicted_positive']

            # Per-type breakdown
            cursor.execute("""
                SELECT
                    notification_type,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = true) as flagged,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = false) as suppressed
                FROM predictions
                WHERE time >= %s
                GROUP BY notification_type
                ORDER BY total DESC
            """, (since,))
            by_type = [dict(row) for row in cursor.fetchall()]

            cursor.close()

            return {
                'time_window_hours': hours,
                'total_alerts': overall['total'],
                'total_flagged': overall['flagged'],
                'total_suppressed': overall['suppressed'],
                'overall_fp_rate': fp_rate,
                'by_type': [
                    {
                        'notification_type': t['notification_type'],
                        'total': t['total'],
                        'flagged': t['flagged'],
                        'suppressed': t['suppressed'],
                    }
                    for t in by_type
                ],
            }
        finally:
            self._put_conn(conn)

    def get_stats_by_type(self, notification_type: str, hours: int = 24) -> dict:
        """Get detailed stats for a specific notification type."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = true) as flagged,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = false) as suppressed,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                        AND needs_intervention_predicted = needs_intervention_actual
                    ) as correct,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                    ) as with_ground_truth
                FROM predictions
                WHERE time >= %s AND notification_type = %s
            """, (since, notification_type))

            row = dict(cursor.fetchone())
            cursor.close()

            accuracy = None
            if row['with_ground_truth'] and row['with_ground_truth'] > 0:
                accuracy = row['correct'] / row['with_ground_truth']

            return {
                'notification_type': notification_type,
                'total': row['total'],
                'flagged': row['flagged'],
                'suppressed': row['suppressed'],
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else None,
                'accuracy': accuracy,
            }
        finally:
            self._put_conn(conn)

    # ========================================================================
    # MODEL HEALTH
    # ========================================================================

    def get_model_health(self, hours: int = 24) -> dict:
        """Compute model health metrics for monitoring panel."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            # Core metrics
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = true) as flagged,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = false) as suppressed,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                        AND needs_intervention_predicted = needs_intervention_actual
                    ) as correct,
                    COUNT(*) FILTER (
                        WHERE needs_intervention_actual IS NOT NULL
                    ) as with_ground_truth,
                    COUNT(*) FILTER (WHERE confidence >= 0.9) as high_conf,
                    COUNT(*) FILTER (WHERE confidence >= 0.7 AND confidence < 0.9) as med_conf,
                    COUNT(*) FILTER (WHERE confidence < 0.7) as low_conf
                FROM predictions
                WHERE time >= %s
            """, (since,))
            row = dict(cursor.fetchone())

            total = row['total'] or 0
            flagged = row['flagged'] or 0
            suppressed = row['suppressed'] or 0

            accuracy = None
            if row['with_ground_truth'] and row['with_ground_truth'] > 0:
                accuracy = row['correct'] / row['with_ground_truth']

            avg_conf = float(row['avg_confidence']) if row['avg_confidence'] else None

            # Per-type flagged/suppressed
            cursor.execute("""
                SELECT
                    notification_type,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = true) as flagged,
                    COUNT(*) FILTER (WHERE needs_intervention_predicted = false) as suppressed
                FROM predictions
                WHERE time >= %s
                GROUP BY notification_type
            """, (since,))

            flagged_by_type = {}
            suppressed_by_type = {}
            for t in cursor.fetchall():
                flagged_by_type[t['notification_type']] = t['flagged']
                suppressed_by_type[t['notification_type']] = t['suppressed']

            cursor.close()

            # Determine status
            if accuracy is not None and accuracy < 0.6:
                status = 'degraded'
            elif avg_conf is not None and avg_conf < 0.7:
                status = 'warning'
            elif total == 0:
                status = 'warning'
            else:
                status = 'healthy'

            return {
                'status': status,
                'total_predictions': total,
                'pct_flagged': (flagged / total * 100) if total > 0 else 0,
                'pct_suppressed': (suppressed / total * 100) if total > 0 else 0,
                'avg_confidence': round(avg_conf, 4) if avg_conf else None,
                'accuracy': round(accuracy, 4) if accuracy else None,
                'confidence_buckets': {
                    'high': row['high_conf'] or 0,
                    'medium': row['med_conf'] or 0,
                    'low': row['low_conf'] or 0,
                },
                'flagged_by_type': flagged_by_type,
                'suppressed_by_type': suppressed_by_type,
            }
        finally:
            self._put_conn(conn)

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health_check(self) -> bool:
        """Simple connectivity check."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """Close all connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
