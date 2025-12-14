import pandas as pd
import numpy as np
from sqlalchemy import create_engine

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
            is_anomaly,
            anomaly_type
        FROM vehicle_metrics
        ORDER BY time
    """
    
    # pandas can read directly from SQLAlchemy engine
    df = pd.read_sql(query, engine)
    engine.dispose()  # Clean up connection pool
    
    print(f"✅ Loaded {len(df):,} records from database")
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Transform raw data into features that capture 'speed relative to context'
    
    The VAE needs to learn: "Is this behavior normal given the context?"
    Raw speed alone doesn't tell us that. We need relative features.
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # FEATURE 1: Speed ratio (actual / expected)
    # - Value of 1.0 = exactly as expected
    # - Value of 0.0 = stopped when should be moving
    # - We add 1 to denominator to avoid division by zero
    df['speed_ratio'] = df['speed'] / (df['expected_speed'] + 1)
    
    # FEATURE 2: Speed deviation (actual - expected)
    # - Negative = going slower than expected
    # - Positive = going faster than expected
    # - Zero = exactly as expected
    df['speed_deviation'] = df['speed'] - df['expected_speed']
    
    # FEATURE 3: Binary flags
    df['is_stopped'] = (df['speed'] < 5).astype(int)
    df['expected_stopped'] = (df['expected_speed'] < 5).astype(int)
    
    # FEATURE 4: Encode categorical variables as numbers
    # We'll use simple label encoding for now
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
    
    print(f"✅ Engineered {len(df.columns)} features")
    
    return df

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(df):
    """
    Prepare data for VAE training.
    
    KEY INSIGHT: We only train on NORMAL data (is_anomaly = False)
    The VAE learns what "normal" looks like. Then anomalies have high reconstruction error.
    """
    
    # Select only the features we want to feed to the VAE
    feature_columns = [
        'speed_ratio',
        'speed_deviation', 
        'is_stopped',
        'expected_stopped',
        'road_type_encoded',
        'traffic_encoded',
        'construction_encoded'
    ]
    
    # Split into normal and anomaly data
    normal_data = df[df['is_anomaly'] == False].copy()
    anomaly_data = df[df['is_anomaly'] == True].copy()
    
    print(f"📊 Normal records: {len(normal_data):,}")
    print(f"📊 Anomaly records: {len(anomaly_data):,}")
    
    # Extract features for training (normal data only)
    X_normal = normal_data[feature_columns].values
    
    # Extract features for testing (all data - we'll evaluate on both)
    X_all = df[feature_columns].values
    y_all = df['is_anomaly'].values  # Ground truth labels
    
    # Normalize features to 0-1 range (important for neural networks!)
    # We fit the scaler on normal data only
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled = scaler.transform(X_all)
    
    print(f"✅ Normalized features to 0-1 range")
    print(f"   Training data shape: {X_normal_scaled.shape}")
    print(f"   Full data shape: {X_all_scaled.shape}")
    
    return X_normal_scaled, X_all_scaled, y_all, scaler, feature_columns

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 3A: DATA PREPARATION")
    print("=" * 50)
    
    # Step 1: Load data from database
    print("\n📥 Loading data from database...")
    df = load_data()
    
    # Step 2: Engineer features
    print("\n🔧 Engineering features...")
    df = engineer_features(df)
    
    # Step 3: Prepare training data
    print("\n📦 Preparing training data...")
    X_train, X_all, y_all, scaler, feature_cols = prepare_training_data(df)
    
    # Step 4: Save processed data for later use
    print("\n💾 Saving processed data...")
    np.save('X_train.npy', X_train)
    np.save('X_all.npy', X_all)
    np.save('y_all.npy', y_all)
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Saved: X_train.npy ({X_train.shape})")
    print(f"   Saved: X_all.npy ({X_all.shape})")
    print(f"   Saved: y_all.npy ({y_all.shape})")
    
    # Preview the features
    print(f"\n📋 Feature columns: {feature_cols}")
    print(f"\n📊 Sample of training data (first 5 rows):")
    print(X_train[:5])
