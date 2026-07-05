"""
Sentinel Data Preparation v3.0

Feature engineering for notification triage system.
Transforms raw telemetry + notification data into ML-ready features.

v2.1: Added interaction features for better FP/real separation
v3.0: Event-grouped training data (leakage-free splits), no scaler,
      notification-only SQL, chunked loading

Usage:
    python -m ml.prepare_data [--max-events 200000]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, URL

logger = logging.getLogger("sentinel.ml")
from ml.constants import (
    ROAD_TYPE_MAP, TRAFFIC_MAP, CONSTRUCTION_MAP,
    NOTIFICATION_TYPE_MAP, NOTIFICATION_SUBTYPE_MAP,
    FEATURE_COLUMNS,
)

# Artifacts always land next to this file, regardless of cwd
ML_DIR = Path(__file__).resolve().parent

# Fixed vocabularies (from the encoding maps) so per-chunk categoricals
# share dtype and concat cheaply
_CATEGORY_DTYPES = {
    'road_type': pd.CategoricalDtype([k for k in ROAD_TYPE_MAP]),
    'traffic_condition': pd.CategoricalDtype([k for k in TRAFFIC_MAP]),
    'construction_zone': pd.CategoricalDtype([k for k in CONSTRUCTION_MAP]),
    'notification_type': pd.CategoricalDtype([k for k in NOTIFICATION_TYPE_MAP if k]),
    'notification_subtype': pd.CategoricalDtype([k for k in NOTIFICATION_SUBTYPE_MAP if k]),
}
_FLOAT32_COLUMNS = ['speed', 'expected_speed', 'ev_distance',
                    'pedestrian_density', 'time_since_stop', 'hour_of_day']

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_database_engine():
    """Create SQLAlchemy engine for TimescaleDB"""
    import os
    url = URL.create(
        "postgresql",
        username=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'password'),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=int(os.environ.get('DB_PORT', '5432')),
        database=os.environ.get('DB_NAME', 'postgres'),
    )
    return create_engine(url)

# ============================================================================
# DATA LOADING
# ============================================================================

def _shrink_chunk(chunk):
    """Downcast a raw chunk in place: float32 numerics, fixed-vocabulary
    categoricals. 27M rows of object-dtype strings would exceed 10 GB."""
    for col, dtype in _CATEGORY_DTYPES.items():
        chunk[col] = chunk[col].astype(dtype)
    for col in _FLOAT32_COLUMNS:
        chunk[col] = chunk[col].astype(np.float32)
    chunk['needs_intervention'] = chunk['needs_intervention'].astype(bool)
    chunk['object_in_path'] = chunk['object_in_path'].astype(bool)
    return chunk


def load_data(chunksize=2_000_000):
    """Load notification rows from the database, ordered for event
    reconstruction (vehicle, then time). Only notification rows are
    fetched — training never uses normal-operation telemetry."""
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
        WHERE notification_type IS NOT NULL
        ORDER BY vehicle_id, time
    """

    chunks = []
    # stream_results → server-side cursor; without it the full result set
    # materializes client-side and chunksize is cosmetic
    with engine.connect().execution_options(stream_results=True) as conn:
        for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):
            chunks.append(_shrink_chunk(chunk))
            logger.info(f"  ...loaded {sum(len(c) for c in chunks):,} rows")
    engine.dispose()

    df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
    logger.info(f"Loaded {len(df):,} notification records from database")
    return df


# ============================================================================
# EVENT RECONSTRUCTION
# ============================================================================

def assign_event_ids(df):
    """Reconstruct notification events: an event is a contiguous run
    (samples ≤5 s apart) of the same (vehicle, type, subtype).

    One event yields ~4-120 near-identical rows sharing a label; splitting
    train/test by event_id is what keeps sibling rows from leaking across
    the split. Requires df sorted by (vehicle_id, time).
    """
    same_vehicle = df['vehicle_id'].eq(df['vehicle_id'].shift())
    contiguous = df['time'].diff().le(pd.Timedelta(seconds=5))
    same_type = df['notification_type'].eq(df['notification_type'].shift())
    subtype = df['notification_subtype']
    prev_subtype = subtype.shift()
    same_subtype = (subtype == prev_subtype) | (subtype.isna() & prev_subtype.isna())

    df['event_id'] = (~(same_vehicle & contiguous & same_type & same_subtype)).cumsum()
    return df


def subsample_events(df, max_events, seed=42):
    """Cap the training set by sampling whole events (never splitting one),
    so the cap cannot reintroduce leakage. max_events <= 0 keeps everything."""
    n_events = df['event_id'].nunique()
    if max_events <= 0 or n_events <= max_events:
        return df

    rng = np.random.default_rng(seed)
    keep = rng.choice(df['event_id'].unique(), size=max_events, replace=False)
    sampled = df[df['event_id'].isin(keep)].copy()
    logger.info(f"Subsampled {max_events:,}/{n_events:,} events "
                f"({len(sampled):,}/{len(df):,} rows)")
    return sampled


def restore_object_dtypes(df):
    """Convert the fixed-vocabulary categoricals back to object with None
    for missing — engineer_features' encoding maps key on None (not NaN),
    and by this point the frame is small enough that memory doesn't matter."""
    for col in _CATEGORY_DTYPES:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
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

    df['road_type_encoded'] = df['road_type'].map(ROAD_TYPE_MAP)
    df['traffic_encoded'] = df['traffic_condition'].map(TRAFFIC_MAP)
    df['construction_encoded'] = df['construction_zone'].map(CONSTRUCTION_MAP)

    # ========================================
    # NOTIFICATION TYPE ENCODING
    # ========================================

    df['notification_type_encoded'] = df['notification_type'].map(NOTIFICATION_TYPE_MAP)
    df['notification_subtype_encoded'] = df['notification_subtype'].map(NOTIFICATION_SUBTYPE_MAP)
    
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
    
    logger.info(f"Engineered {len(df.columns)} features")
    
    return df

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(df):
    """
    Prepare data for ML training.

    Extracts X_all (notification features, raw — XGBoost needs no scaling),
    y_all (needs_intervention labels), notif_types (for per-type evaluation),
    and groups (event_id per row, for leakage-free grouped splits).
    Feature order comes from ml.constants.FEATURE_COLUMNS — the single
    source of truth shared with inference.
    """

    # Filter to only notification records (we're triaging notifications)
    notification_df = df[df['notification_type'].notna()].copy()

    no_intervention = notification_df[notification_df['needs_intervention'] == False]
    needs_intervention = notification_df[notification_df['needs_intervention'] == True]

    logger.info("Notification Records:")
    logger.info(f"  Total notifications: {len(notification_df):,}")
    logger.info(f"  No intervention needed (FP): {len(no_intervention):,}")
    logger.info(f"  Intervention needed (Real): {len(needs_intervention):,}")
    logger.info(f"  Baseline FP rate: {len(no_intervention)/len(notification_df)*100:.1f}%")

    X_all = notification_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_all = notification_df['needs_intervention'].values
    notif_types = notification_df[['notification_type', 'notification_subtype']].values
    groups = notification_df['event_id'].values

    logger.info(f"Prepared training data:")
    logger.info(f"  Samples: {X_all.shape}")
    logger.info(f"  Events: {notification_df['event_id'].nunique():,}")
    logger.info(f"  Features: {len(FEATURE_COLUMNS)}")

    return X_all, y_all, notif_types, groups

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    parser = argparse.ArgumentParser(description="Prepare Sentinel training data")
    parser.add_argument("--max-events", type=int, default=200_000,
                        help="Cap training data at N whole events (0 = keep all)")
    args = parser.parse_args()

    logger.info("SENTINEL DATA PREPARATION v3.0")

    # Step 1: Load notification rows
    logger.info("Loading data from database...")
    df = load_data()

    # Step 2: Reconstruct events and record full-dataset totals — the
    # per-day math in training/README uses these, not the training sample
    logger.info("Reconstructing notification events...")
    df = assign_event_ids(df)
    sim_days = max(1, round((df['time'].max() - df['time'].min()).total_seconds() / 86400))
    dataset_meta = {
        'full_rows': int(len(df)),
        'full_positives': int(df['needs_intervention'].sum()),
        'full_events': int(df['event_id'].nunique()),
        'sim_days': int(sim_days),
        'sim_start': str(df['time'].min()),
        'sim_end': str(df['time'].max()),
    }
    logger.info(f"Dataset totals: {dataset_meta['full_rows']:,} rows, "
                f"{dataset_meta['full_events']:,} events over {sim_days} days")

    # Step 3: Cap by whole events, then restore dtypes for feature engineering
    df = subsample_events(df, args.max_events)
    df = restore_object_dtypes(df)

    # Step 4: Engineer features
    logger.info("Engineering features...")
    df = engineer_features(df)

    # Step 5: Extract arrays
    logger.info("Preparing training data...")
    X_all, y_all, notif_types, groups = prepare_training_data(df)

    # Step 6: Save processed data (anchored to ml/, cwd-independent)
    logger.info("Saving processed data...")
    np.save(ML_DIR / 'X_all.npy', X_all)
    np.save(ML_DIR / 'y_all.npy', y_all)
    np.save(ML_DIR / 'notif_types.npy', notif_types)
    np.save(ML_DIR / 'groups.npy', groups)
    with open(ML_DIR / 'dataset_meta.json', 'w') as f:
        json.dump(dataset_meta, f, indent=2)

    logger.info("Data preparation complete!")
    logger.info(f"  Saved: X_all.npy ({X_all.shape})")
    logger.info(f"  Saved: y_all.npy ({y_all.shape})")
    logger.info(f"  Saved: notif_types.npy ({notif_types.shape})")
    logger.info(f"  Saved: groups.npy ({groups.shape})")
    logger.info(f"  Saved: dataset_meta.json {dataset_meta}")

    logger.info(f"Feature columns ({len(FEATURE_COLUMNS)}):")
    for i, col in enumerate(FEATURE_COLUMNS):
        logger.info(f"  {i}: {col}")
