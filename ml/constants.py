"""
Shared encoding maps for feature engineering.
Single source of truth — used by both training (prepare_data.py) and inference (model_service.py).
"""

ROAD_TYPE_MAP = {
    'highway': 0, 'main_road': 1, 'residential': 2,
    'downtown': 3, 'school_zone': 4
}

TRAFFIC_MAP = {
    'light': 0, 'moderate': 1, 'heavy': 2, 'standstill': 3
}

CONSTRUCTION_MAP = {
    'none': 0, 'temporary': 1, 'persistent': 2, 'flagger': 3
}

NOTIFICATION_TYPE_MAP = {
    None: 0,  # No notification (normal operation) — used in training data only
    'verification_request': 1, 'emergency_vehicle_alert': 2,
    'stuck': 3, 'speed_anomaly': 4, 'impact_l0': 5, 'passenger_assist': 6
}

NOTIFICATION_SUBTYPE_MAP = {
    None: 0, 'object_query': 1,
    'traffic_signal_verify': 2, 'lane_mapping_verify': 3
}

FEATURE_COLUMNS = [
    'speed_ratio', 'speed_deviation', 'is_stopped', 'expected_stopped',
    'road_type_encoded', 'traffic_encoded', 'construction_encoded',
    'notification_type_encoded', 'notification_subtype_encoded',
    'ev_distance_normalized', 'pedestrian_density', 'object_in_path',
    'time_since_stop_normalized',
    'hour_sin', 'hour_cos',
    'high_traffic', 'high_pedestrian',
    'stuck_in_traffic', 'stuck_in_construction', 'stuck_clear_road',
    'object_query_high_ped', 'object_query_low_ped', 'object_query_moving',
    'ev_far_away', 'ev_close',
    'speed_anomaly_in_traffic', 'speed_anomaly_clear',
    'impact_rough_road',
]
