-- ==================================================================
-- SENTINEL - Useful Queries for Notification Triage Analysis
-- ==================================================================

-- 1. DATASET OVERVIEW
-- -------------------
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT vehicle_id) as num_vehicles,
    MIN(time) as start_time,
    MAX(time) as end_time,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as notification_records,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real_interventions,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL AND needs_intervention = false) as false_positives
FROM vehicle_metrics;


-- 2. NOTIFICATION TYPE BREAKDOWN
-- ------------------------------
SELECT 
    notification_type,
    notification_subtype,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    COUNT(*) FILTER (WHERE needs_intervention = false) as fp,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / COUNT(*), 1) as fp_rate
FROM vehicle_metrics 
WHERE notification_type IS NOT NULL
GROUP BY notification_type, notification_subtype
ORDER BY total DESC;


-- 3. SUMMARY BY MAIN NOTIFICATION TYPE
-- ------------------------------------
SELECT 
    notification_type,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    COUNT(*) FILTER (WHERE needs_intervention = false) as fp,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / COUNT(*), 1) as fp_rate
FROM vehicle_metrics
WHERE notification_type IS NOT NULL
GROUP BY notification_type
ORDER BY total DESC;


-- 4. HOURLY NOTIFICATION PATTERN
-- ------------------------------
SELECT 
    EXTRACT(HOUR FROM time) as hour,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as notifications,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = true) / 
          NULLIF(COUNT(*) FILTER (WHERE notification_type IS NOT NULL), 0), 1) as intervention_rate
FROM vehicle_metrics
GROUP BY EXTRACT(HOUR FROM time)
ORDER BY hour;


-- 5. OBJECT QUERY ANALYSIS
-- ------------------------
-- The highest volume, highest FP notification
SELECT 
    road_type,
    traffic_condition,
    ROUND(AVG(pedestrian_density)::numeric, 2) as avg_pedestrian_density,
    COUNT(*) as count,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / COUNT(*), 1) as fp_rate
FROM vehicle_metrics
WHERE notification_subtype = 'object_query'
GROUP BY road_type, traffic_condition
ORDER BY count DESC;


-- 6. EMERGENCY VEHICLE ALERT ANALYSIS
-- -----------------------------------
SELECT 
    CASE 
        WHEN ev_distance < 50 THEN 'very_close (<50m)'
        WHEN ev_distance < 100 THEN 'close (50-100m)'
        WHEN ev_distance < 200 THEN 'medium (100-200m)'
        ELSE 'far (>200m)'
    END as distance_category,
    COUNT(*) as count,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / COUNT(*), 1) as fp_rate
FROM vehicle_metrics
WHERE notification_type = 'emergency_vehicle_alert'
GROUP BY 1
ORDER BY count DESC;


-- 7. STUCK NOTIFICATION ANALYSIS
-- ------------------------------
SELECT 
    traffic_condition,
    construction_zone,
    COUNT(*) as count,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / COUNT(*), 1) as fp_rate
FROM vehicle_metrics
WHERE notification_type = 'stuck'
GROUP BY traffic_condition, construction_zone
ORDER BY count DESC;


-- 8. DAILY NOTIFICATION VOLUME
-- ----------------------------
SELECT 
    DATE(time) as date,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as notifications,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real_interventions,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL AND needs_intervention = false) as false_positives
FROM vehicle_metrics
GROUP BY DATE(time)
ORDER BY date;


-- 9. TRAFFIC CONDITION IMPACT
-- ---------------------------
SELECT 
    traffic_condition,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as notifications,
    COUNT(*) FILTER (WHERE needs_intervention = true) as real,
    ROUND(100.0 * COUNT(*) FILTER (WHERE needs_intervention = false) / 
          NULLIF(COUNT(*) FILTER (WHERE notification_type IS NOT NULL), 0), 1) as fp_rate
FROM vehicle_metrics
GROUP BY traffic_condition
ORDER BY notifications DESC;


-- 10. SAMPLE NOTIFICATION SEQUENCE
-- --------------------------------
SELECT 
    time,
    vehicle_id,
    speed,
    expected_speed,
    notification_type,
    notification_subtype,
    needs_intervention,
    ev_distance,
    pedestrian_density
FROM vehicle_metrics
WHERE notification_type IS NOT NULL
ORDER BY time DESC
LIMIT 50;


-- 11. OPERATOR WORKLOAD SIMULATION
-- --------------------------------
-- What operators would see per hour
SELECT 
    DATE(time) as date,
    EXTRACT(HOUR FROM time) as hour,
    COUNT(*) FILTER (WHERE notification_type IS NOT NULL) as total_notifications,
    COUNT(*) FILTER (WHERE notification_type = 'verification_request') as verification_requests,
    COUNT(*) FILTER (WHERE notification_type = 'stuck') as stuck,
    COUNT(*) FILTER (WHERE notification_type = 'emergency_vehicle_alert') as ev_alerts
FROM vehicle_metrics
GROUP BY DATE(time), EXTRACT(HOUR FROM time)
ORDER BY date, hour;


-- 12. CHECK INDEXES
-- -----------------
SELECT 
    indexname, 
    indexdef 
FROM pg_indexes 
WHERE tablename = 'vehicle_metrics';


-- 13. CLEAR TABLE (FOR REGENERATION)
-- ----------------------------------
-- TRUNCATE TABLE vehicle_metrics;
