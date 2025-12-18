"""
Sentinel Data Preparation v2.1

Feature engineering for notification triage system.
Transforms raw telemetry + notification data into ML-ready features.

v2.1: Added interaction features for better FP/real separation
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_database_engine():
    """Create SQLAlchemy engine for TimescaleDB"""
    connection_string = 'postgresql://postgres:password@localhost:5432/postgres'
    engine = create_engine(connection_string)
    return engine

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all vehicle metrics from database into a pandas DataFrame"""
    engine = get_database_engine()
    
    query = """
        SELECT 
            time,
            vehicle_id,
            speed,
            expected_speed,
            road_type,
            traffic_condition,
            construction_zone,
            notification_type,
            notification_subtype,
            needs_intervention,
            ev_distance,
            pedestrian_density,
            object_in_path,
            time_since_stop,
            EXTRACT(HOUR FROM time) as hour_of_day
        FROM vehicle_metrics
        ORDER BY time
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    print(f"✅ Loaded {len(df):,} records from database")
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Transform raw data into features for notification triage.
    
    The model needs to learn: "Given this notification and context,
    does it actually need operator intervention?"
    """
    
    df = df.copy()
    
    # ========================================
    # SPEED FEATURES
    # ========================================
    
    # Speed ratio (actual / expected)
    df['speed_ratio'] = df['speed'] / (df['expected_speed'] + 1)
    
    # Speed deviation (actual - expected)
    df['speed_deviation'] = df['speed'] - df['expected_speed']
    
    # Binary flags
    df['is_stopped'] = (df['speed'] < 5).astype(int)
    df['expected_stopped'] = (df['expected_speed'] < 5).astype(int)
    
    # ========================================
    # ROAD CONTEXT ENCODING
    # ========================================
    
    road_type_map = {
        'highway': 0, 
        'main_road': 1, 
        'residential': 2, 
        'downtown': 3, 
        'school_zone': 4
    }
    traffic_map = {
        'light': 0, 
        'moderate': 1, 
        'heavy': 2, 
        'standstill': 3
    }
    construction_map = {
        'none': 0, 
        'temporary': 1, 
        'persistent': 2, 
        'flagger': 3
    }
    
    df['road_type_encoded'] = df['road_type'].map(road_type_map)
    df['traffic_encoded'] = df['traffic_condition'].map(traffic_map)
    df['construction_encoded'] = df['construction_zone'].map(construction_map)
    
    # ========================================
    # NOTIFICATION TYPE ENCODING
    # ========================================
    
    notification_type_map = {
        None: 0,  # No notification (normal operation)
        'verification_request': 1,
        'emergency_vehicle_alert': 2,
        'stuck': 3,
        'speed_anomaly': 4,
        'impact_l0': 5,
        'passenger_assist': 6,
    }
    
    notification_subtype_map = {
        None: 0,
        'object_query': 1,
        'traffic_signal_verify': 2,
        'lane_mapping_verify': 3,
    }
    
    df['notification_type_encoded'] = df['notification_type'].map(notification_type_map)
    df['notification_subtype_encoded'] = df['notification_subtype'].map(notification_subtype_map)
    
    # ========================================
    # CONTEXT FEATURES
    # ========================================
    
    # EV distance (normalized, fill missing with max distance = not relevant)
    df['ev_distance_normalized'] = df['ev_distance'].fillna(999) / 500.0
    df['ev_distance_normalized'] = df['ev_distance_normalized'].clip(0, 2)
    
    # Pedestrian density (already 0-1, fill missing with 0)
    df['pedestrian_density'] = df['pedestrian_density'].fillna(0)
    
    # Object in path (binary, fill missing with 0)
    df['object_in_path'] = df['object_in_path'].fillna(False).astype(int)
    
    # Time since stop (normalized, fill missing with 0)
    df['time_since_stop_normalized'] = df['time_since_stop'].fillna(0) / 600.0
    df['time_since_stop_normalized'] = df['time_since_stop_normalized'].clip(0, 2)
    
    # Hour of day (cyclical encoding for time patterns)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # ========================================
    # DERIVED FEATURES
    # ========================================
    
    # Is this a high-traffic situation? (explains many FPs)
    df['high_traffic'] = ((df['traffic_encoded'] >= 2) | 
                          (df['construction_encoded'] >= 1)).astype(int)
    
    # Is this a pedestrian-heavy area?
    df['high_pedestrian'] = (df['pedestrian_density'] > 0.5).astype(int)
    
    # ========================================
    # INTERACTION FEATURES (Key for separation!)
    # ========================================
    # These encode domain knowledge about what makes a FP vs real intervention
    
    # STUCK notifications
    is_stuck = (df['notification_type'] == 'stuck')
    df['stuck_in_traffic'] = (is_stuck & (df['traffic_encoded'] >= 2)).astype(int)
    df['stuck_in_construction'] = (is_stuck & (df['construction_encoded'] >= 1)).astype(int)
    df['stuck_clear_road'] = (is_stuck & 
                              (df['traffic_encoded'] == 0) & 
                              (df['construction_encoded'] == 0)).astype(int)
    
    # OBJECT QUERY notifications
    is_object_query = (df['notification_subtype'] == 'object_query')
    df['object_query_high_ped'] = (is_object_query & (df['pedestrian_density'] > 0.5)).astype(int)
    df['object_query_low_ped'] = (is_object_query & (df['pedestrian_density'] <= 0.3)).astype(int)
    df['object_query_moving'] = (is_object_query & (df['speed'] > 10)).astype(int)
    
    # EMERGENCY VEHICLE notifications
    is_ev = (df['notification_type'] == 'emergency_vehicle_alert')
    df['ev_far_away'] = (is_ev & (df['ev_distance_normalized'] > 0.4)).astype(int)  # >200m
    df['ev_close'] = (is_ev & (df['ev_distance_normalized'] < 0.1)).astype(int)     # <50m
    
    # SPEED ANOMALY notifications
    is_speed_anomaly = (df['notification_type'] == 'speed_anomaly')
    df['speed_anomaly_in_traffic'] = (is_speed_anomaly & (df['traffic_encoded'] >= 2)).astype(int)
    df['speed_anomaly_clear'] = (is_speed_anomaly & (df['traffic_encoded'] == 0)).astype(int)
    
    # IMPACT notifications
    is_impact = (df['notification_type'] == 'impact_l0')
    df['impact_rough_road'] = (is_impact & 
                               (df['road_type'].isin(['residential', 'downtown']))).astype(int)
    
    print(f"✅ Engineered {len(df.columns)} features")
    
    return df

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(df):
    """
    Prepare data for ML training.
    
    Saves X_all (all notification features) and y_all (needs_intervention labels)
    for supervised classification, plus X_train (FPs only) for VAE experiments.
    """
    
    # Features for the model
    feature_columns = [
        # Speed context
        'speed_ratio',
        'speed_deviation', 
        'is_stopped',
        'expected_stopped',
        
        # Road context
        'road_type_encoded',
        'traffic_encoded',
        'construction_encoded',
        
        # Notification info
        'notification_type_encoded',
        'notification_subtype_encoded',
        
        # Situational context
        'ev_distance_normalized',
        'pedestrian_density',
        'object_in_path',
        'time_since_stop_normalized',
        
        # Time patterns
        'hour_sin',
        'hour_cos',
        
        # Derived
        'high_traffic',
        'high_pedestrian',
        
        # Interaction features (domain knowledge encoded)
        'stuck_in_traffic',
        'stuck_in_construction',
        'stuck_clear_road',
        'object_query_high_ped',
        'object_query_low_ped',
        'object_query_moving',
        'ev_far_away',
        'ev_close',
        'speed_anomaly_in_traffic',
        'speed_anomaly_clear',
        'impact_rough_road',
    ]
    
    # Filter to only notification records (we're triaging notifications)
    notification_df = df[df['notification_type'].notna()].copy()
    
    # Split by intervention needed
    no_intervention = notification_df[notification_df['needs_intervention'] == False]
    needs_intervention = notification_df[notification_df['needs_intervention'] == True]
    
    print(f"\n📊 Notification Records:")
    print(f"   Total notifications: {len(notification_df):,}")
    print(f"   No intervention needed (FP): {len(no_intervention):,}")
    print(f"   Intervention needed (Real): {len(needs_intervention):,}")
    print(f"   Baseline FP rate: {len(no_intervention)/len(notification_df)*100:.1f}%")
    
    # Extract features
    X_no_intervention = no_intervention[feature_columns].values
    X_all = notification_df[feature_columns].values
    y_all = notification_df['needs_intervention'].values
    
    # Store notification types for per-type evaluation later
    notif_types = notification_df[['notification_type', 'notification_subtype']].values
    
    # Normalize features to 0-1 range
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_no_intervention)
    X_all_scaled = scaler.transform(X_all)
    
    print(f"\n✅ Prepared training data:")
    print(f"   Training samples (FP only): {X_train_scaled.shape}")
    print(f"   Evaluation samples (all): {X_all_scaled.shape}")
    print(f"   Features: {len(feature_columns)}")
    
    return X_train_scaled, X_all_scaled, y_all, notif_types, scaler, feature_columns

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL DATA PREPARATION v2.1")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 Loading data from database...")
    df = load_data()
    
    # Step 2: Engineer features
    print("\n🔧 Engineering features...")
    df = engineer_features(df)
    
    # Step 3: Prepare training data
    print("\n📦 Preparing training data...")
    X_train, X_all, y_all, notif_types, scaler, feature_cols = prepare_training_data(df)
    
    # Step 4: Save processed data
    print("\n💾 Saving processed data...")
    np.save('X_train.npy', X_train)
    np.save('X_all.npy', X_all)
    np.save('y_all.npy', y_all)
    np.save('notif_types.npy', notif_types)
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Saved: X_train.npy ({X_train.shape})")
    print(f"   Saved: X_all.npy ({X_all.shape})")
    print(f"   Saved: y_all.npy ({y_all.shape})")
    print(f"   Saved: notif_types.npy ({notif_types.shape})")
    
    # Preview features
    print(f"\n📋 Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"   {i}: {col}")
