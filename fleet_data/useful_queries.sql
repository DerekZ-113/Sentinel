-- ==================================================================
-- SENTINEL - Useful Queries for Fleet Anomaly Detection
-- ==================================================================

-- 1. DATASET OVERVIEW
-- -------------------
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT vehicle_id) as num_vehicles,
    MIN(time) as start_time,
    MAX(time) as end_time,
    COUNT(*) FILTER (WHERE is_anomaly = true) as anomaly_records,
    COUNT(*) FILTER (WHERE is_anomaly = false) as normal_records
FROM vehicle_metrics;


-- 2. ANOMALY BREAKDOWN BY TYPE
-- ----------------------------
SELECT 
    anomaly_type,
    COUNT(*) as count,
    ROUND(AVG(speed)::numeric, 2) as avg_speed,
    ROUND(AVG(expected_speed)::numeric, 2) as avg_expected
FROM vehicle_metrics 
WHERE is_anomaly = true
GROUP BY anomaly_type
ORDER BY count DESC;


-- 3. TRAFFIC CONDITION DISTRIBUTION
-- ---------------------------------
SELECT 
    traffic_condition,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage,
    ROUND(AVG(speed)::numeric, 2) as avg_speed
FROM vehicle_metrics
GROUP BY traffic_condition
ORDER BY count DESC;


-- 4. HOURLY ANOMALY PATTERN
-- -------------------------
SELECT 
    EXTRACT(HOUR FROM time) as hour,
    COUNT(*) FILTER (WHERE is_anomaly = true) as anomalies,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_anomaly = true) / COUNT(*), 2) as anomaly_rate
FROM vehicle_metrics
GROUP BY EXTRACT(HOUR FROM time)
ORDER BY hour;


-- 5. FIND LONG STUCK EVENTS
-- -------------------------
-- Vehicles stopped for 5+ minutes when expected to be moving
SELECT 
    vehicle_id,
    time,
    speed,
    expected_speed,
    is_anomaly,
    anomaly_type
FROM vehicle_metrics
WHERE speed < 5 
  AND expected_speed > 10
ORDER BY time DESC
LIMIT 100;


-- 6. SPEED RATIO ANALYSIS
-- -----------------------
-- Shows distribution of actual vs expected speed
SELECT 
    CASE 
        WHEN expected_speed < 5 THEN 'expected_stopped'
        WHEN speed / NULLIF(expected_speed, 0) > 0.9 THEN 'on_target'
        WHEN speed / NULLIF(expected_speed, 0) > 0.5 THEN 'slow'
        WHEN speed / NULLIF(expected_speed, 0) > 0.1 THEN 'very_slow'
        ELSE 'stopped'
    END as speed_category,
    COUNT(*) as count,
    COUNT(*) FILTER (WHERE is_anomaly = true) as anomalies
FROM vehicle_metrics
GROUP BY 1
ORDER BY count DESC;


-- 7. CONSTRUCTION ZONE IMPACT
-- ---------------------------
SELECT 
    construction_zone,
    ROUND(AVG(speed)::numeric, 2) as avg_speed,
    ROUND(AVG(expected_speed)::numeric, 2) as avg_expected,
    COUNT(*) FILTER (WHERE speed < 5) as stopped_count,
    COUNT(*) as total
FROM vehicle_metrics
GROUP BY construction_zone
ORDER BY avg_speed;


-- 8. SAMPLE DATA FOR ONE VEHICLE
-- ------------------------------
SELECT 
    time,
    speed,
    expected_speed,
    traffic_condition,
    construction_zone,
    is_anomaly,
    anomaly_type
FROM vehicle_metrics
WHERE vehicle_id = 'vehicle_000'
ORDER BY time
LIMIT 50;


-- 9. CLEAR TABLE (FOR REGENERATION)
-- ---------------------------------
-- TRUNCATE TABLE vehicle_metrics;


-- 10. CHECK INDEXES
-- -----------------
SELECT 
    indexname, 
    indexdef 
FROM pg_indexes 
WHERE tablename = 'vehicle_metrics';
