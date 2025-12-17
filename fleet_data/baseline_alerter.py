"""
Sentinel Baseline Alerter

Simple rule-based alerting that treats ALL notifications as needing intervention.
This is the "before" state - what operators deal with without ML triage.

This establishes the baseline false positive rate we're trying to beat.
"""

import psycopg2


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


def analyze_baseline():
    """
    Analyze what happens if we treat every notification as needing intervention.
    This is the baseline that operators currently experience.
    """
    conn = connect_to_database()
    cursor = conn.cursor()
    
    print("=" * 70)
    print("BASELINE ALERTER ANALYSIS")
    print("Strategy: Treat ALL notifications as needing intervention")
    print("=" * 70)
    
    # Overall stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as total_notifications,
            COUNT(*) FILTER (WHERE needs_intervention = true) as real_interventions,
            COUNT(*) FILTER (WHERE notification_type IS NOT NULL AND needs_intervention = false) as false_positives
        FROM vehicle_metrics
    """)
    
    row = cursor.fetchone()
    total_records = row[0]
    total_notifications = row[1]
    real_interventions = row[2]
    false_positives = row[3]
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total records: {total_records:,}")
    print(f"   Total notifications: {total_notifications:,}")
    print(f"   Real interventions needed: {real_interventions:,}")
    print(f"   False positives: {false_positives:,}")
    
    if total_notifications > 0:
        fp_rate = false_positives / total_notifications * 100
        precision = real_interventions / total_notifications * 100
        print(f"\n   📈 Baseline False Positive Rate: {fp_rate:.1f}%")
        print(f"   📈 Baseline Precision: {precision:.1f}%")
    
    # Per notification type breakdown
    print(f"\n{'=' * 70}")
    print("BREAKDOWN BY NOTIFICATION TYPE")
    print(f"{'=' * 70}")
    
    cursor.execute("""
        SELECT 
            notification_type,
            notification_subtype,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE needs_intervention = true) as real,
            COUNT(*) FILTER (WHERE needs_intervention = false) as fp
        FROM vehicle_metrics
        WHERE notification_type IS NOT NULL
        GROUP BY notification_type, notification_subtype
        ORDER BY total DESC
    """)
    
    print(f"\n{'Type':<30} {'Subtype':<25} {'Total':>10} {'Real':>10} {'FP':>10} {'FP Rate':>10}")
    print("-" * 95)
    
    for row in cursor.fetchall():
        notif_type = row[0]
        subtype = row[1] or '-'
        total = row[2]
        real = row[3]
        fp = row[4]
        fp_rate = (fp / total * 100) if total > 0 else 0
        
        print(f"{notif_type:<30} {subtype:<25} {total:>10,} {real:>10,} {fp:>10,} {fp_rate:>9.1f}%")
    
    # Summary by main type only
    print(f"\n{'=' * 70}")
    print("SUMMARY BY MAIN TYPE")
    print(f"{'=' * 70}")
    
    cursor.execute("""
        SELECT 
            notification_type,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE needs_intervention = true) as real,
            COUNT(*) FILTER (WHERE needs_intervention = false) as fp
        FROM vehicle_metrics
        WHERE notification_type IS NOT NULL
        GROUP BY notification_type
        ORDER BY total DESC
    """)
    
    print(f"\n{'Type':<35} {'Total':>12} {'Real':>12} {'FP':>12} {'FP Rate':>12}")
    print("-" * 85)
    
    type_stats = {}
    for row in cursor.fetchall():
        notif_type = row[0]
        total = row[1]
        real = row[2]
        fp = row[3]
        fp_rate = (fp / total * 100) if total > 0 else 0
        
        type_stats[notif_type] = {'total': total, 'real': real, 'fp': fp, 'fp_rate': fp_rate}
        print(f"{notif_type:<35} {total:>12,} {real:>12,} {fp:>12,} {fp_rate:>11.1f}%")
    
    # Operator workload analysis
    print(f"\n{'=' * 70}")
    print("OPERATOR WORKLOAD ANALYSIS")
    print(f"{'=' * 70}")
    
    if total_notifications > 0:
        # Assuming 7 days of data
        notifs_per_day = total_notifications / 7
        fps_per_day = false_positives / 7
        real_per_day = real_interventions / 7
        
        print(f"\n   Notifications per day: {notifs_per_day:,.0f}")
        print(f"   False positives per day: {fps_per_day:,.0f}")
        print(f"   Real interventions per day: {real_per_day:,.0f}")
        print(f"\n   ⚠️  Operators waste time on {fps_per_day:,.0f} false alarms daily")
        print(f"   ⚠️  Only {real_per_day/notifs_per_day*100:.1f}% of notifications need action")
    
    cursor.close()
    conn.close()
    
    return type_stats


if __name__ == "__main__":
    analyze_baseline()
