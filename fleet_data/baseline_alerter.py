from functools import total_ordering
import psycopg2
from collections import defaultdict

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database():
    """Connect to TimescaleDB"""
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='postgres',
        user='postgres',
        password='password'
    )
    return conn

# ============================================================================
# BASELINE ALERTER
# ============================================================================

# Thresholds (in number of records, each record = 5 seconds)
STUCK_THRESHOLD = 60      # 5 minutes at speed = 0
SLOW_THRESHOLD = 120      # 10 minutes at speed < 5

def run_baseline_alerter():
    """
    Run the baseline (dumb) alerter that only looks at speed thresholds.
    Returns list of alerts and comparison with ground truth.
    """
    conn = connect_to_database()
    cursor = conn.cursor()
    
    # TODO: Query data and apply rules
    # Query all data ordered by vehicle, then time
    cursor.execute("""
        SELECT vehicle_id, time, speed, is_anomaly, anomaly_type
        FROM vehicle_metrics
        ORDER BY vehicle_id, time
    """)
    
    rows = cursor.fetchall()

    alerts = []

    current_vehicle = None
    consecutive_stopped = 0
    alert_fired = False # Don't fire multiple alerts for same stop event

    for row in rows:
        vehicle_id, time, speed, is_anomaly, anomaly_type = row

        # Reset when switching to new vehicle
        if vehicle_id != current_vehicle:
            current_vehicle = vehicle_id
            consecutive_stopped = 0
            alert_fired = False
        
        # Check if stopped
        if speed < 5:
            consecutive_stopped += 1

            # Fire alert if threshold exceeded (and haven't already fired for this event)
            if consecutive_stopped >= STUCK_THRESHOLD and not alert_fired:
                alerts.append((vehicle_id, time, 'STUCK', is_anomaly, anomaly_type))
                alert_fired = True
        
        else:
            # Vehicle moving - reset counter
            consecutive_stopped = 0
            alert_fired = False

    cursor.close()
    conn.close()

    return alerts

def calculate_metrics(alerts):
    true_positive = 0
    false_positive = 0

    for alert in alerts:
        vehicle_id, time, alert_type, is_anomaly, anomaly_type = alert

        if is_anomaly:
            true_positive += 1
        else:
            false_positive += 1

    total_alerts = len(alerts)

    if total_alerts > 0:
        fp_rate = (false_positive / total_alerts) * 100
        precision = (true_positive / total_alerts) * 100
    else:
        fp_rate = 0
        precision = 0

    return {
        'total_alerts': total_alerts,
        'true_positives': true_positive,
        'false_positives': false_positive,
        'fp_rate': fp_rate,
        'precision': precision
    }

if __name__ == "__main__":
    print("Running baseline alerter...")
    print("=" * 50)
    
    alerts = run_baseline_alerter()
    metrics = calculate_metrics(alerts)
    
    print(f"\n📊 BASELINE ALERTER RESULTS")
    print("=" * 50)
    print(f"Total alerts fired:    {metrics['total_alerts']}")
    print(f"True positives:        {metrics['true_positives']}")
    print(f"False positives:       {metrics['false_positives']}")
    print(f"")
    print(f"False positive rate:   {metrics['fp_rate']:.1f}%")
    print(f"Precision:             {metrics['precision']:.1f}%")
    print("=" * 50)
    
    # Show some example alerts
    print(f"\n📋 Sample Alerts (first 10):")
    print("-" * 50)
    for alert in alerts[:10]:
        vehicle_id, time, alert_type, is_anomaly, anomaly_type = alert
        status = "✓ TRUE" if is_anomaly else "✗ FALSE"
        print(f"{time} | {vehicle_id} | {alert_type} | {status} | {anomaly_type or 'normal'}")