"""
Sentinel database schema — single source of truth.

Both setup_database.py (provisioning) and DatabaseService (runtime
ensure-exists) execute these statements. Keep every CREATE here; duplicated
DDL is how the two paths drifted apart historically.

All statements are idempotent (IF NOT EXISTS) — destructive resets live
only behind setup_database.py --reset.
"""

VEHICLE_METRICS_DDL = """
    CREATE TABLE IF NOT EXISTS vehicle_metrics (
        -- Timestamp and vehicle ID
        time                    TIMESTAMPTZ NOT NULL,
        vehicle_id              TEXT NOT NULL,

        -- Vehicle state
        speed                   FLOAT NOT NULL,
        latitude                FLOAT NOT NULL,
        longitude               FLOAT NOT NULL,
        status                  TEXT NOT NULL,

        -- Road context
        road_type               TEXT NOT NULL,
        traffic_condition       TEXT NOT NULL,
        construction_zone       TEXT,
        expected_speed          FLOAT NOT NULL,

        -- Notification info
        notification_type       TEXT,
        notification_subtype    TEXT,
        needs_intervention      BOOLEAN DEFAULT FALSE,

        -- Context for specific notification types
        ev_distance             FLOAT,
        pedestrian_density      FLOAT,
        object_in_path          BOOLEAN,
        time_since_stop         FLOAT
    );
"""

VEHICLE_METRICS_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_vehicle_time
        ON vehicle_metrics (vehicle_id, time DESC);
    CREATE INDEX IF NOT EXISTS idx_notification
        ON vehicle_metrics (notification_type, time DESC);
    CREATE INDEX IF NOT EXISTS idx_intervention
        ON vehicle_metrics (needs_intervention, time DESC);
    CREATE INDEX IF NOT EXISTS idx_context
        ON vehicle_metrics (road_type, traffic_condition, time DESC);
"""

# Best-effort TimescaleDB hypertable conversion for the telemetry table.
# On plain PostgreSQL (no timescaledb extension) callers skip this.
VEHICLE_METRICS_HYPERTABLE = """
    SELECT create_hypertable(
        'vehicle_metrics',
        'time',
        if_not_exists => TRUE
    );
"""

PREDICTIONS_DDL = """
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
"""

# One-row bookkeeping table: entrypoint.sh marks demo seeding complete here.
# Checking a count>0 instead conflated "empty" with "interrupted mid-seed"
# (and with "probe failed") — see review findings H10/H20.
SEED_STATUS_DDL = """
    CREATE TABLE IF NOT EXISTS seed_status (
        seeded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        note        TEXT
    );
"""
