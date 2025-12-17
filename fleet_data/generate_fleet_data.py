"""
Sentinel Fleet Data Generator v2.0

Simulates an autonomous vehicle fleet notification system with:
- 500 vehicles operating over 7 days
- 6 notification types (verification_request, emergency_vehicle_alert, stuck, etc.)
- Context-aware false positive labeling
- Realistic traffic patterns

Output: ~18M+ records in TimescaleDB
"""

import random
import psycopg2
from datetime import datetime, timedelta
import signal
import sys

# ============================================================================
# CONSTANTS - ROAD CONTEXT
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
# CONSTANTS - NOTIFICATION TYPES
# ============================================================================

NOTIFICATION_TYPES = {
    'verification_request': {
        'frequency': 0.003,
        'duration': (30, 120),
        'subtypes': {
            'object_query':          {'share': 0.75, 'fp_rate': 0.90},
            'traffic_signal_verify': {'share': 0.15, 'fp_rate': 0.10},
            'lane_mapping_verify':   {'share': 0.10, 'fp_rate': 0.30},
        },
    },
    'emergency_vehicle_alert': {
        'frequency': 0.001,
        'fp_rate': 0.70,
        'duration': (20, 60),
    },
    'stuck': {
        'frequency': 0.001,
        'fp_rate': 0.65,
        'duration': (120, 600),
    },
    'speed_anomaly': {
        'frequency': 0.0008,
        'fp_rate': 0.50,
        'duration': (60, 300),
    },
    'impact_l0': {
        'frequency': 0.0002,
        'fp_rate': 0.40,
        'duration': (30, 90),
    },
    'passenger_assist': {
        'frequency': 0.0001,
        'fp_rate': 0.0,  # Always needs intervention
        'duration': (60, 300),
    },
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
                road_type, traffic_condition, construction_zone, expected_speed,
                notification_type, notification_subtype, needs_intervention,
                ev_distance, pedestrian_density, object_in_path, time_since_stop
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def get_pedestrian_density(road_type, hour):
    """Returns pedestrian density based on location and time"""
    base_density = {
        'highway': 0.0,
        'main_road': 0.3,
        'residential': 0.4,
        'downtown': 0.7,
        'school_zone': 0.5,
    }
    
    # Adjust for time of day
    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
        time_modifier = 1.3
    elif 10 <= hour <= 15:  # Midday
        time_modifier = 1.0
    elif 19 <= hour <= 22:  # Evening
        time_modifier = 0.7
    else:  # Night
        time_modifier = 0.2
    
    density = base_density[road_type] * time_modifier
    # Add randomness
    density += random.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, density))


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
        self.status = 'moving'
        self.time_since_stop = 0.0

        # Context
        self.road_type = None
        self.traffic_condition = None
        self.construction_zone = None

        # Notification state
        self.active_notification = None
        self.notification_subtype = None
        self.notification_remaining = 0
        self.needs_intervention = False
        
        # Notification context
        self.ev_distance = None
        self.pedestrian_density = 0.0
        self.object_in_path = False

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

    def maybe_trigger_notification(self, sim_time):
        """
        Randomly trigger a notification based on frequencies.
        Determines if it needs intervention based on context.
        """
        if self.active_notification:
            return
        
        # Check each notification type
        for notif_type, config in NOTIFICATION_TYPES.items():
            if random.random() < config['frequency']:
                self._trigger_notification(notif_type, config, sim_time)
                return

    def _trigger_notification(self, notif_type, config, sim_time):
        """Set up a notification with appropriate context"""
        self.active_notification = notif_type
        self.notification_remaining = random.randint(*config['duration'])
        
        # Handle subtypes for verification_request
        if notif_type == 'verification_request':
            subtypes = config['subtypes']
            subtype_names = list(subtypes.keys())
            subtype_weights = [subtypes[s]['share'] for s in subtype_names]
            self.notification_subtype = random.choices(subtype_names, weights=subtype_weights)[0]
            fp_rate = subtypes[self.notification_subtype]['fp_rate']
        else:
            self.notification_subtype = None
            fp_rate = config['fp_rate']
        
        # Determine needs_intervention based on FP rate and context
        self.needs_intervention = self._determine_intervention(notif_type, fp_rate, sim_time)
        
        # Set context-specific fields
        self._set_notification_context(notif_type, sim_time)

    def _determine_intervention(self, notif_type, base_fp_rate, sim_time):
        """
        Determine if notification needs intervention.
        Context can shift the FP rate.
        """
        # Start with base FP rate
        fp_rate = base_fp_rate
        
        # Context adjustments
        if notif_type == 'verification_request' and self.notification_subtype == 'object_query':
            # Higher pedestrian density = more likely FP (just someone walking by)
            if self.pedestrian_density > 0.5:
                fp_rate = min(0.95, fp_rate + 0.05)
            # If vehicle is moving, more likely real obstruction
            if self.speed > 10:
                fp_rate = max(0.70, fp_rate - 0.10)
                
        elif notif_type == 'emergency_vehicle_alert':
            # Will be set properly in context
            pass
            
        elif notif_type == 'stuck':
            # If there's traffic/construction, more likely FP
            if self.traffic_condition in ['heavy', 'standstill']:
                fp_rate = min(0.85, fp_rate + 0.15)
            if self.construction_zone != 'none':
                fp_rate = min(0.90, fp_rate + 0.20)
            # If clear conditions, more likely real
            if self.traffic_condition == 'light' and self.construction_zone == 'none':
                fp_rate = max(0.30, fp_rate - 0.25)
                
        elif notif_type == 'speed_anomaly':
            # Similar to stuck
            if self.traffic_condition in ['heavy', 'standstill']:
                fp_rate = min(0.80, fp_rate + 0.20)
            if self.construction_zone != 'none':
                fp_rate = min(0.85, fp_rate + 0.25)
                
        elif notif_type == 'impact_l0':
            # Rough roads more likely FP
            if self.road_type in ['residential', 'downtown']:
                fp_rate = min(0.60, fp_rate + 0.15)
        
        # Roll the dice
        is_false_positive = random.random() < fp_rate
        return not is_false_positive  # needs_intervention = NOT a false positive

    def _set_notification_context(self, notif_type, sim_time):
        """Set context fields based on notification type"""
        hour = sim_time.hour
        
        # Update pedestrian density
        self.pedestrian_density = get_pedestrian_density(self.road_type, hour)
        
        if notif_type == 'emergency_vehicle_alert':
            # EV distance affects whether it's relevant
            if self.needs_intervention:
                self.ev_distance = random.uniform(10, 100)  # Close = real
            else:
                self.ev_distance = random.uniform(150, 500)  # Far = FP
                
        elif notif_type == 'verification_request' and self.notification_subtype == 'object_query':
            # Object in path affects intervention need
            self.object_in_path = self.needs_intervention
            
        else:
            self.ev_distance = None
            self.object_in_path = False

    def update(self, sim_time):
        """Update vehicle state for one simulation tick (1 second)"""
        
        # Check for new notification
        self.maybe_trigger_notification(sim_time)

        if self.active_notification:
            # Handle active notification
            self.notification_remaining -= 1
            
            # Notification affects vehicle behavior
            if self.active_notification == 'stuck':
                self.speed = 0
            elif self.active_notification == 'speed_anomaly':
                if self.needs_intervention:
                    self.speed = self.expected_speed * random.uniform(0.2, 0.4)
                else:
                    self.speed = self.expected_speed * random.uniform(0.4, 0.6)
            elif self.active_notification == 'impact_l0':
                self.speed = max(0, self.speed - 5)  # Slow down after impact
            elif self.active_notification == 'passenger_assist':
                self.speed = 0  # Stop for passenger
            
            # End notification when duration expires
            if self.notification_remaining <= 0:
                self._clear_notification()
        
        else:
            # Normal behavior
            self._update_context(sim_time)
            self._update_speed()
        
        # Update location
        self._update_location()
        
        # Update stop timer
        if self.speed < 5:
            self.status = 'stopped'
            self.time_since_stop += 1
        else:
            self.status = 'moving'
            self.time_since_stop = 0

    def _clear_notification(self):
        """Clear active notification state"""
        self.active_notification = None
        self.notification_subtype = None
        self.notification_remaining = 0
        self.needs_intervention = False
        self.ev_distance = None
        self.object_in_path = False

    def _update_context(self, sim_time):
        """Update road context periodically"""
        # Update traffic condition
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

    def _update_speed(self):
        """Smoothly adjust speed toward target"""
        self.target_speed = self.expected_speed
        speed_diff = self.target_speed - self.speed
        self.speed += speed_diff * 0.15
        
        # Add realistic variation
        self.speed += random.uniform(-1, 1)
        
        # Clamp to non-negative
        if self.speed < 0:
            self.speed = 0

    def _update_location(self):
        """Update GPS coordinates with small movement"""
        self.latitude += random.uniform(-0.0001, 0.0001)
        self.longitude += random.uniform(-0.0001, 0.0001)


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
    num_vehicles = 500
    num_days = 7
    vehicles = [Vehicle(f"vehicle_{i:03d}") for i in range(num_vehicles)]

    for vehicle in vehicles:
        vehicle.assign_initial_context()

    sim_time = datetime(2024, 12, 1, 7, 30, 0)
    end_time = datetime(2024, 12, 1 + num_days, 7, 30, 0)
    
    # Calculate expected records
    total_seconds = num_days * 24 * 60 * 60
    records_per_sample = num_vehicles
    sample_interval = 5
    expected_records = (total_seconds // sample_interval) * records_per_sample
    
    print(f"🚗 Simulating {num_vehicles} vehicles for {num_days} days")
    print(f"📊 Expected records: ~{expected_records:,}")
    print("Press Ctrl+C to stop\n")
    
    batch_data = []
    records_inserted = 0
    
    while sim_time < end_time:
        
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
                    vehicle.active_notification,
                    vehicle.notification_subtype,
                    vehicle.needs_intervention if vehicle.active_notification else None,
                    vehicle.ev_distance,
                    vehicle.pedestrian_density,
                    vehicle.object_in_path if vehicle.active_notification else None,
                    vehicle.time_since_stop if vehicle.speed < 5 else None
                ))

        # Insert when batch reaches 5000 records
        if len(batch_data) >= 5000:
            if insert_batch(cursor, batch_data):
                conn.commit()
                records_inserted += len(batch_data)
                batch_data = []
        
        sim_time += timedelta(seconds=1)

        # Print progress every simulated hour
        if sim_time.minute == 0 and sim_time.second == 0:
            speeds = [v.speed for v in vehicles]
            avg_speed = sum(speeds) / len(speeds)
            active_notifs = sum(1 for v in vehicles if v.active_notification)
            print(f"[{sim_time.strftime('%a %I:%M %p')}] "
                  f"Avg: {avg_speed:.1f} mph | "
                  f"Active notifications: {active_notifs} | "
                  f"Records: {records_inserted:,}")

    # Insert remaining data
    if batch_data:
        if insert_batch(cursor, batch_data):
            conn.commit()
            records_inserted += len(batch_data)
            print(f"✅ Inserted final {len(batch_data)} records")
    
    cursor.close()
    conn.close()
    
    print(f"\n✅ Simulation complete!")
    print(f"   Vehicles: {num_vehicles}")
    print(f"   Duration: {num_days} days")
    print(f"   Total records: {records_inserted:,}")


if __name__ == "__main__":
    main()
