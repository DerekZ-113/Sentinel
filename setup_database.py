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
                time                TIMESTAMPTZ NOT NULL,
                vehicle_id          TEXT NOT NULL,
                speed               FLOAT NOT NULL,
                latitude            FLOAT NOT NULL,
                longitude           FLOAT NOT NULL,
                status              TEXT NOT NULL,
                road_type           TEXT NOT NULL,
                traffic_condition   TEXT NOT NULL,
                construction_zone   TEXT,
                expected_speed      FLOAT NOT NULL,
                is_anomaly          BOOLEAN DEFAULT FALSE,
                anomaly_type        TEXT
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
            CREATE INDEX IF NOT EXISTS idx_vehicle_id 
            ON vehicle_metrics (vehicle_id, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_anomaly 
            ON vehicle_metrics (is_anomaly, time DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_context 
            ON vehicle_metrics (road_type, traffic_condition, time DESC);
        """)
        
        print("✅ Database setup complete!")
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