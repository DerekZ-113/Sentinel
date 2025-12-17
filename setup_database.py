"""
Sentinel Database Setup

Creates TimescaleDB schema for AV fleet notification triage system.
Supports 500-vehicle fleet with 6 notification types.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def setup_database():
    """Initialize the database schema for Sentinel"""
    
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': 'password'
    }
    
    try:
        print("Connecting to TimescaleDB...")
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Drop old table for fresh start
        print("Dropping old table if exists...")
        cursor.execute("DROP TABLE IF EXISTS vehicle_metrics;")
        
        # Create table with all context columns
        print("Creating vehicle_metrics table...")
        cursor.execute("""
            CREATE TABLE vehicle_metrics (
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
        """)
        
        # Convert to hypertable
        print("Converting to hypertable...")
        cursor.execute("""
            SELECT create_hypertable(
                'vehicle_metrics',
                'time',
                if_not_exists => TRUE
            );
        """)
        
        # Create indexes
        print("Creating indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_time 
            ON vehicle_metrics (vehicle_id, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_notification 
            ON vehicle_metrics (notification_type, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_intervention 
            ON vehicle_metrics (needs_intervention, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_context 
            ON vehicle_metrics (road_type, traffic_condition, time DESC);
        """)
        
        print("\n✅ Database setup complete!")
        print("\nTable structure:")
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'vehicle_metrics'
            ORDER BY ordinal_position;
        """)
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise


if __name__ == "__main__":
    setup_database()
