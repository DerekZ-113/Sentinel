"""
Sentinel Database Setup

Creates TimescaleDB schema for AV fleet notification triage system.
Supports 500-vehicle fleet with 6 notification types.

Idempotent by default (CREATE IF NOT EXISTS) — safe to run on every
container start. Destructive reset only with --reset.

Usage:
    python setup_database.py [--reset]
"""

import argparse
import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from api.services.schema import (
    VEHICLE_METRICS_DDL, VEHICLE_METRICS_INDEXES, VEHICLE_METRICS_HYPERTABLE,
    PREDICTIONS_DDL, SEED_STATUS_DDL,
)

logger = logging.getLogger("sentinel.setup")


def setup_database(reset: bool = False):
    """Initialize the database schema for Sentinel"""

    conn_params = {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': int(os.environ.get('DB_PORT', '5432')),
        'database': os.environ.get('DB_NAME', 'postgres'),
        'user': os.environ.get('DB_USER', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', 'password'),
    }

    try:
        logger.info("Connecting to TimescaleDB...")
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        if reset:
            # Destructive: drops the training-data telemetry table.
            # Never run implicitly — an API container restart must not
            # wipe a persistent volume.
            logger.warning("--reset: dropping vehicle_metrics...")
            cursor.execute("DROP TABLE IF EXISTS vehicle_metrics;")

        logger.info("Ensuring tables (idempotent)...")
        cursor.execute(VEHICLE_METRICS_DDL)
        cursor.execute(PREDICTIONS_DDL)
        cursor.execute(SEED_STATUS_DDL)

        # Hypertable conversion is TimescaleDB-specific; skip gracefully on
        # plain PostgreSQL (e.g. the CI service container)
        try:
            cursor.execute(VEHICLE_METRICS_HYPERTABLE)
            logger.info("vehicle_metrics is a hypertable")
        except psycopg2.Error as e:
            logger.warning(f"Hypertable conversion skipped: {e}")

        logger.info("Creating indexes...")
        cursor.execute(VEHICLE_METRICS_INDEXES)

        logger.info("Database setup complete")
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'vehicle_metrics'
            ORDER BY ordinal_position;
        """)
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]}")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    parser = argparse.ArgumentParser(description="Provision the Sentinel database schema")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate vehicle_metrics (DESTRUCTIVE)")
    args = parser.parse_args()
    setup_database(reset=args.reset)
