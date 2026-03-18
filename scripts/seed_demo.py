#!/usr/bin/env python
"""
Sentinel Seed Script

Generates realistic demo data and fires it through the API.
Populates the predictions table so the dashboard has data on first load.

Usage:
    # Make sure API is running first:
    # uvicorn api.main:app --port 8000

    python -m scripts.seed_demo
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger("sentinel.seed")

# Add project root to path
sys.path.insert(0, '.')

from fleet_data.generate_fleet_data import (
    Vehicle, NOTIFICATION_TYPES, get_traffic_weights, get_pedestrian_density
)


API_BASE = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY", "")
TARGET_NOTIFICATIONS = 1000


def check_api():
    """Verify the API is running."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        data = resp.json()
        if data.get('model_loaded') and data.get('db_connected'):
            return True
        logger.warning(f"API is up but not fully ready: {data}")
        return False
    except requests.ConnectionError:
        return False


def run_mini_simulation(num_vehicles=50, hours=12):
    """
    Run a mini fleet simulation and collect notification events.
    Reuses the Vehicle class from generate_fleet_data.py.
    """
    vehicles = [Vehicle(f"vehicle_{i:03d}") for i in range(num_vehicles)]
    for v in vehicles:
        v.assign_initial_context()

    sim_time = datetime(2024, 12, 1, 6, 0, 0)  # Start at 6 AM
    end_time = sim_time + timedelta(hours=hours)

    notifications = []

    while sim_time < end_time:
        for vehicle in vehicles:
            vehicle.update(sim_time)

            # Collect notification events (sample every 5 seconds)
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
                    # Ground truth from simulation
                    'needs_intervention_actual': vehicle.needs_intervention,
                })

        sim_time += timedelta(seconds=1)

        # Early exit if we have enough
        if len(notifications) >= TARGET_NOTIFICATIONS * 1.5:
            break

    return notifications


def post_predictions(notifications):
    """Send notifications to the API and track results."""
    results = {
        'success': 0,
        'failed': 0,
        'by_type': {},
    }

    total = len(notifications)
    start = time.time()

    for i, notif in enumerate(notifications):
        try:
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["X-API-Key"] = API_KEY
            resp = requests.post(
                f"{API_BASE}/api/predict",
                json=notif,
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                results['success'] += 1
                data = resp.json()

                # Track by type
                ntype = notif['notification_type']
                if ntype not in results['by_type']:
                    results['by_type'][ntype] = {
                        'total': 0, 'fp': 0, 'real': 0,
                        'predicted_fp': 0, 'predicted_real': 0,
                    }
                t = results['by_type'][ntype]
                t['total'] += 1
                if notif.get('needs_intervention_actual'):
                    t['real'] += 1
                else:
                    t['fp'] += 1
                if data.get('needs_intervention'):
                    t['predicted_real'] += 1
                else:
                    t['predicted_fp'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            results['failed'] += 1

        # Progress bar
        if (i + 1) % 50 == 0 or i == total - 1:
            pct = (i + 1) / total
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\r├── {bar} {i+1}/{total}  ({rate:.0f}/s)", end="", flush=True)

    print()  # newline after progress bar
    results['elapsed'] = time.time() - start
    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger.info("Sentinel Seed Script")

    # Step 1: Check API
    logger.info("Checking API...")
    if not check_api():
        logger.error(f"API not reachable at {API_BASE}. Start it first: uvicorn api.main:app --port 8000")
        sys.exit(1)
    logger.info("API is healthy")

    # Step 2: Simulate
    logger.info("Simulating fleet (50 vehicles, 12 hours)...")
    notifications = run_mini_simulation(num_vehicles=50, hours=12)

    # Trim to target
    if len(notifications) > TARGET_NOTIFICATIONS:
        notifications = notifications[:TARGET_NOTIFICATIONS]

    logger.info(f"Collected {len(notifications)} notifications")

    # Step 3: Fire predictions
    logger.info("Sending to /api/predict...")
    results = post_predictions(notifications)

    # Step 4: Summary
    logger.info(f"Seed complete! Sent: {results['success']}  |  Failed: {results['failed']}  |  Time: {results['elapsed']:.1f}s")
    for ntype in sorted(results['by_type'].keys()):
        t = results['by_type'][ntype]
        logger.info(f"  {ntype:<30} total={t['total']}  fp={t['fp']}  real={t['real']}")
    logger.info(f"Dashboard ready at http://localhost:3000")
    logger.info(f"API docs at {API_BASE}/docs")


if __name__ == "__main__":
    main()
