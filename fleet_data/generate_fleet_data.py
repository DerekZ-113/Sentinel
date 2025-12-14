"""
Sentinel Fleet Data Generator

Simulates an autonomous vehicle fleet with:
- 200 vehicles operating over 7 days
- Realistic traffic patterns (rush hour, midday, nighttime)
- Construction zones and traffic conditions
- Injected anomalies (stuck, wrong_speed) with ground truth labels

Output: ~10.8M records in TimescaleDB
"""

import random
import psycopg2
from datetime import datetime, timedelta
import signal
import sys

# ============================================================================
# CONSTANTS
# ============================================================================

ROAD_TYPE = {
    'highway':     {'base_speed': 65, 'weight': 0.05},
    'main_road':   {'base_speed': 45, 'weight': 0.40},
    'residential': {'base_speed': 30, 'weight': 0.30},
    'downtown':    {'base_speed': 35, 'weight': 0.20},
    'school_zone': {'base_speed': 15, 'weight': 0.05},
}

TRAFFIC_CONDITION = {
    'light':      {'speed_modifier': 1.0,  'weight': 0.40},
    'moderate':   {'speed_modifier': 0.7,  'weight': 0.25},
    'heavy':      {'speed_modifier': 0.3,  'weight': 0.30},
    'standstill': {'speed_modifier': 0.05, 'weight': 0.05},
}

CONSTRUCTION_ZONE = {
    'none':       {'speed_modifier': 1.0, 'weight': 0.6},
    'temporary':  {'speed_modifier': 0.6, 'weight': 0.2},
    'persistent': {'speed_modifier': 0.5, 'weight': 0.1},
    'flagger':    {'speed_modifier': 0.0, 'weight': 0.1},
}

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database():
    """Connect to TimescaleDB"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='postgres',
            password='password'
        )
        print("✅ Connected to TimescaleDB")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

def insert_batch(cursor, batch_data):
    """Insert a batch of vehicle metrics into the database"""
    try:
        cursor.executemany("""
            INSERT INTO vehicle_metrics (
                time, vehicle_id, speed, latitude, longitude, status,
                road_type, traffic_condition, construction_zone,
                expected_speed, is_anomaly, anomaly_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, batch_data)
        return True
    except Exception as e:
        print(f"❌ Error inserting batch: {e}")
        return False

# ============================================================================
# TRAFFIC PATTERNS
# ============================================================================

def get_traffic_weights(sim_time):
    """Returns traffic condition weights based on time of day"""
    hour = sim_time.hour

    if hour in [6, 7, 8, 16, 17, 18]:  # Rush hour
        return {
            'light': 0.05,
            'moderate': 0.3,
            'heavy': 0.5,
            'standstill': 0.15
        }
    elif hour in [5, 9, 10, 11, 12, 13, 14, 15, 19, 20]:  # Midday
        return {
            'light': 0.35,
            'moderate': 0.35,
            'heavy': 0.2,
            'standstill': 0.1
        }
    else:  # Night/early morning
        return {
            'light': 0.7,
            'moderate': 0.1,
            'heavy': 0.1,
            'standstill': 0.1
        }

# ============================================================================
# VEHICLE CLASS
# ============================================================================

class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id

        # Location (Bay Area coordinates)
        self.latitude = None
        self.longitude = None

        # Speed
        self.speed = 0.0
        self.target_speed = 0.0
        self.expected_speed = 0.0

        # Status
        self.status = 'stopped'
        self.stopped_duration = 0

        # Context
        self.road_type = None
        self.traffic_condition = None
        self.construction_zone = None

        # Anomaly tracking
        self.is_anomaly = False
        self.anomaly_type = None
        self.anomaly_duration = 0.0
        self.anomaly_remaining = 0.0

        # Context timing
        self.seconds_until_traffic_change = None
        self.construction_zone_ends_at = None

    def calculate_expected_speed(self):
        """Calculate expected speed based on road type, traffic, and construction"""
        base_speed = ROAD_TYPE[self.road_type]['base_speed']
        traffic_modifier = TRAFFIC_CONDITION[self.traffic_condition]['speed_modifier']
        construction_modifier = CONSTRUCTION_ZONE[self.construction_zone]['speed_modifier']
        self.expected_speed = base_speed * traffic_modifier * construction_modifier
        return self.expected_speed
        
    def assign_initial_context(self):
        """Assign random initial context to vehicle"""
        self.road_type = random.choices(
            list(ROAD_TYPE.keys()), 
            weights=[v['weight'] for v in ROAD_TYPE.values()]
        )[0]
        self.traffic_condition = random.choices(
            list(TRAFFIC_CONDITION.keys()), 
            weights=[v['weight'] for v in TRAFFIC_CONDITION.values()]
        )[0]
        self.construction_zone = random.choices(
            list(CONSTRUCTION_ZONE.keys()), 
            weights=[v['weight'] for v in CONSTRUCTION_ZONE.values()]
        )[0]
        
        self.expected_speed = self.calculate_expected_speed()
        
        # Bay Area coordinate range
        self.latitude = random.uniform(37.3, 37.8)   
        self.longitude = random.uniform(-122.5, -122.0)

        self.seconds_until_traffic_change = random.randint(300, 900)

        if self.construction_zone != 'none':
            self.construction_zone_ends_at = random.randint(600, 1800)
        else:
            self.construction_zone_ends_at = None

    def maybe_trigger_anomaly(self):
        """
        Randomly trigger an anomaly (0.1% chance per tick).
        
        Anomaly types:
        - stuck: Vehicle stops completely when it should be moving
        - wrong_speed: Vehicle moves at 30-50% of expected speed
        """
        if self.is_anomaly:
            return
        
        if random.random() > 0.001:
            return

        # Can only be "stuck" if we should be moving
        can_be_stuck = self.target_speed > 10

        if can_be_stuck:
            self.anomaly_type = random.choice(['stuck', 'wrong_speed'])
        else:
            self.anomaly_type = 'wrong_speed'

        # Set anomaly duration
        if self.anomaly_type == 'stuck':
            self.anomaly_duration = random.randint(120, 600)  # 2-10 minutes
        else:
            self.anomaly_duration = random.randint(60, 300)   # 1-5 minutes

        self.is_anomaly = True
        self.anomaly_remaining = self.anomaly_duration

    def update(self, sim_time):
        """Update vehicle state for one simulation tick (1 second)"""
        
        # Check for new anomaly
        self.maybe_trigger_anomaly()

        if self.is_anomaly:
            # Anomaly behavior
            self.anomaly_remaining -= 1

            if self.anomaly_type == 'stuck':
                self.speed = 0
            elif self.anomaly_type == 'wrong_speed':
                self.speed = self.expected_speed * random.uniform(0.3, 0.5)
        
            # End anomaly when duration expires
            if self.anomaly_remaining <= 0:
                self.is_anomaly = False
                self.anomaly_type = None
                self.anomaly_duration = 0
                self.anomaly_remaining = 0
        
        else:
            # Normal behavior
            
            # Update traffic condition periodically
            self.seconds_until_traffic_change -= 1
            if self.seconds_until_traffic_change <= 0:
                traffic_weights = get_traffic_weights(sim_time)
                self.traffic_condition = random.choices(
                    list(traffic_weights.keys()),
                    weights=list(traffic_weights.values())
                )[0]
                self.seconds_until_traffic_change = random.randint(300, 900)
                self.expected_speed = self.calculate_expected_speed()

            # End construction zone
            if self.construction_zone_ends_at is not None:
                self.construction_zone_ends_at -= 1
                if self.construction_zone_ends_at <= 0:
                    self.construction_zone = 'none'
                    self.construction_zone_ends_at = None
                    self.expected_speed = self.calculate_expected_speed()

            # Smooth speed transition
            self.target_speed = self.expected_speed
            speed_diff = self.target_speed - self.speed
            self.speed += speed_diff * 0.15

            # Add realistic speed variation
            self.speed += random.uniform(-1, 1)

            # Update location (small random movement)
            self.latitude += random.uniform(-0.0001, 0.0001)
            self.longitude += random.uniform(-0.0001, 0.0001)

            # Update status
            if self.speed < 5:
                self.status = 'stopped'
                self.stopped_duration += 1
            else:
                self.status = 'moving'
                self.stopped_duration = 0

            # Clamp speed to non-negative
            if self.speed < 0:
                self.speed = 0

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    conn = connect_to_database()
    cursor = conn.cursor()
    
    # Graceful shutdown handler
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down gracefully...")
        cursor.close()
        conn.close()
        print("✅ Database connection closed")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize fleet
    num_vehicles = 200
    num_days = 7
    vehicles = [Vehicle(f"vehicle_{i:03d}") for i in range(num_vehicles)]

    for vehicle in vehicles:
        vehicle.assign_initial_context()

    sim_time = datetime(2024, 12, 1, 7, 30, 0)
    end_time = datetime(2024, 12, 1 + num_days, 7, 30, 0)
    
    print(f"🚗 Simulating {num_vehicles} vehicles for {num_days} days")
    print(f"📊 Expected records: ~{num_vehicles * num_days * 24 * 60 * 12:,}")
    print("Press Ctrl+C to stop\n")
    
    batch_data = []
    iteration = 0
    
    while sim_time < end_time:
        iteration += 1
        
        for vehicle in vehicles:
            vehicle.update(sim_time)
        
        # Collect data every 5 simulated seconds
        if sim_time.second % 5 == 0:
            for vehicle in vehicles:
                batch_data.append((
                    sim_time,
                    vehicle.vehicle_id,
                    vehicle.speed,
                    vehicle.latitude,
                    vehicle.longitude,
                    vehicle.status,
                    vehicle.road_type,
                    vehicle.traffic_condition,
                    vehicle.construction_zone,
                    vehicle.expected_speed,
                    vehicle.is_anomaly,
                    vehicle.anomaly_type
                ))

        # Insert when batch reaches 1000 records
        if len(batch_data) >= 1000:
            if insert_batch(cursor, batch_data):
                conn.commit()
                batch_data = []
        
        sim_time += timedelta(seconds=1)

        # Print progress every simulated hour
        if sim_time.minute == 0 and sim_time.second == 0:
            speeds = [v.speed for v in vehicles]
            avg_speed = sum(speeds) / len(speeds)
            stopped_count = sum(1 for s in speeds if s < 5)
            print(f"[{sim_time.strftime('%a %I:%M %p')}] "
                  f"Avg: {avg_speed:.1f} mph | Stopped: {stopped_count}")

    # Insert remaining data
    if batch_data:
        if insert_batch(cursor, batch_data):
            conn.commit()
            print(f"✅ Inserted final {len(batch_data)} records")
    
    cursor.close()
    conn.close()
    
    print(f"\n✅ Simulation complete!")
    print(f"   Vehicles: {num_vehicles}")
    print(f"   Duration: {num_days} days")

if __name__ == "__main__":
    main()
