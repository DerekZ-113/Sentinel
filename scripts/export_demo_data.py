#!/usr/bin/env python
"""
Sentinel Demo Data Export

Generates realistic demo data using the fleet simulation and XGBoost model,
then writes JSON files for the static Vercel deployment.

No Docker, no database, no API server needed — runs entirely locally.

Usage:
    python -m scripts.export_demo_data
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("sentinel.export")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fleet_data.generate_fleet_data import Vehicle
from api.services.model_service import ModelService

TARGET_NOTIFICATIONS = 1000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "src", "data")


def run_simulation(num_vehicles=50, hours=12):
    """Run a mini fleet simulation and collect notification events."""
    random.seed(42)  # Reproducible data

    vehicles = [Vehicle(f"vehicle_{i:03d}") for i in range(num_vehicles)]
    for v in vehicles:
        v.assign_initial_context()

    sim_time = datetime(2024, 12, 1, 6, 0, 0)
    end_time = sim_time + timedelta(hours=hours)
    notifications = []

    while sim_time < end_time:
        for vehicle in vehicles:
            vehicle.update(sim_time)

            if sim_time.second % 5 == 0 and vehicle.active_notification:
                notifications.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'speed': round(vehicle.speed, 1),
                    'expected_speed': round(vehicle.expected_speed, 1),
                    'road_type': vehicle.road_type,
                    'traffic_condition': vehicle.traffic_condition,
                    'construction_zone': vehicle.construction_zone or 'none',
                    'notification_type': vehicle.active_notification,
                    'notification_subtype': vehicle.notification_subtype,
                    'ev_distance': round(vehicle.ev_distance, 1) if vehicle.ev_distance else None,
                    'pedestrian_density': round(vehicle.pedestrian_density, 2),
                    'object_in_path': vehicle.object_in_path if vehicle.active_notification else False,
                    'time_since_stop': round(vehicle.time_since_stop, 1) if vehicle.speed < 5 else 0,
                    'hour_of_day': sim_time.hour,
                    'needs_intervention_actual': vehicle.needs_intervention,
                    '_sim_time': sim_time.isoformat() + 'Z',
                })

        sim_time += timedelta(seconds=1)
        if len(notifications) >= TARGET_NOTIFICATIONS * 1.5:
            break

    return notifications[:TARGET_NOTIFICATIONS]


def predict_all(notifications, model_service):
    """Run each notification through the XGBoost model."""
    alerts = []
    for i, notif in enumerate(notifications):
        payload = {k: v for k, v in notif.items() if not k.startswith('_')}
        prediction = model_service.predict(payload)

        alerts.append({
            'id': i + 1,
            'time': notif['_sim_time'],
            'vehicle_id': notif['vehicle_id'],
            'notification_type': notif['notification_type'],
            'notification_subtype': notif['notification_subtype'],
            'needs_intervention_predicted': bool(prediction['needs_intervention']),
            'needs_intervention_actual': notif['needs_intervention_actual'],
            'confidence': round(prediction['confidence'], 4),
            'speed': notif['speed'],
            'road_type': notif['road_type'],
            'traffic_condition': notif['traffic_condition'],
        })

    # Sort by time descending (most recent first)
    alerts.sort(key=lambda a: a['time'], reverse=True)
    # Re-assign IDs after sorting
    for i, alert in enumerate(alerts):
        alert['id'] = i + 1

    return alerts


def compute_stats(alerts):
    """Compute aggregate stats matching StatsResponse shape."""
    total = len(alerts)
    flagged = sum(1 for a in alerts if a['needs_intervention_predicted'])
    suppressed = total - flagged

    # FP rate: among predicted positives with ground truth, how many were actually negative
    predicted_pos_with_truth = [
        a for a in alerts
        if a['needs_intervention_predicted'] and a['needs_intervention_actual'] is not None
    ]
    if predicted_pos_with_truth:
        false_positives = sum(1 for a in predicted_pos_with_truth if not a['needs_intervention_actual'])
        fp_rate = round(false_positives / len(predicted_pos_with_truth), 4)
    else:
        fp_rate = None

    # Per-type breakdown
    by_type_dict = {}
    for a in alerts:
        t = a['notification_type']
        if t not in by_type_dict:
            by_type_dict[t] = {'notification_type': t, 'total': 0, 'flagged': 0, 'suppressed': 0}
        by_type_dict[t]['total'] += 1
        if a['needs_intervention_predicted']:
            by_type_dict[t]['flagged'] += 1
        else:
            by_type_dict[t]['suppressed'] += 1

    by_type = sorted(by_type_dict.values(), key=lambda x: x['total'], reverse=True)

    return {
        'time_window_hours': 24,
        'total_alerts': total,
        'total_flagged': flagged,
        'total_suppressed': suppressed,
        'overall_fp_rate': fp_rate,
        'by_type': by_type,
    }


def compute_model_health(alerts):
    """Compute model health metrics matching ModelHealthResponse shape."""
    total = len(alerts)
    flagged = sum(1 for a in alerts if a['needs_intervention_predicted'])
    suppressed = total - flagged

    confidences = [a['confidence'] for a in alerts]
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None

    # Accuracy where ground truth available
    with_truth = [a for a in alerts if a['needs_intervention_actual'] is not None]
    if with_truth:
        correct = sum(1 for a in with_truth if a['needs_intervention_predicted'] == a['needs_intervention_actual'])
        accuracy = round(correct / len(with_truth), 4)
    else:
        accuracy = None

    # Confidence buckets
    high = sum(1 for c in confidences if c >= 0.9)
    medium = sum(1 for c in confidences if 0.7 <= c < 0.9)
    low = sum(1 for c in confidences if c < 0.7)

    # Per-type flagged/suppressed
    flagged_by_type = {}
    suppressed_by_type = {}
    for a in alerts:
        t = a['notification_type']
        if a['needs_intervention_predicted']:
            flagged_by_type[t] = flagged_by_type.get(t, 0) + 1
        else:
            suppressed_by_type[t] = suppressed_by_type.get(t, 0) + 1

    status = 'healthy'
    if accuracy is not None and accuracy < 0.6:
        status = 'degraded'
    elif avg_confidence is not None and avg_confidence < 0.7:
        status = 'warning'

    return {
        'status': status,
        'total_predictions': total,
        'pct_flagged': round(flagged / total * 100, 1) if total > 0 else 0,
        'pct_suppressed': round(suppressed / total * 100, 1) if total > 0 else 0,
        'avg_confidence': avg_confidence,
        'accuracy': accuracy,
        'confidence_buckets': {'high': high, 'medium': medium, 'low': low},
        'flagged_by_type': flagged_by_type,
        'suppressed_by_type': suppressed_by_type,
    }


def main():
    logger.info("Sentinel Demo Data Export")

    # Step 1: Simulate
    logger.info("Running fleet simulation (50 vehicles, 12 hours)...")
    notifications = run_simulation()
    logger.info(f"Collected {len(notifications)} notifications")

    # Step 2: Load model and predict
    logger.info("Loading XGBoost model...")
    model_service = ModelService()
    logger.info("Running predictions on all notifications...")
    alerts = predict_all(notifications, model_service)

    # Step 3: Compute aggregates
    stats = compute_stats(alerts)
    model_health = compute_model_health(alerts)
    health = {
        'status': 'healthy',
        'model_loaded': True,
        'db_connected': True,
        'model_features': 28,
        'model_threshold': 0.5,
        'uptime_seconds': 86400.0,
    }

    # Step 4: Write JSON files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alerts_data = {
        'alerts': alerts,
        'total': len(alerts),
        'limit': len(alerts),
        'offset': 0,
    }

    files = {
        'alerts.json': alerts_data,
        'stats.json': stats,
        'model-health.json': model_health,
        'health.json': health,
    }

    for filename, data in files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Wrote {path}")

    # Summary
    logger.info(f"Export complete!")
    logger.info(f"  Alerts: {len(alerts)}")
    logger.info(f"  Flagged: {stats['total_flagged']}")
    logger.info(f"  Suppressed: {stats['total_suppressed']}")
    logger.info(f"  FP Rate: {stats['overall_fp_rate']}")
    logger.info(f"  Types: {[t['notification_type'] for t in stats['by_type']]}")


if __name__ == "__main__":
    main()
