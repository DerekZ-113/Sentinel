-- ==================================================================
-- SENTINEL - Useful Queries
-- ==================================================================

-- 1. COUNT TOTAL ROWS
SELECT COUNT(*) FROM vehicle_metrics;

-- 2. SEE MOST RECENT DATA (any vehicle)
SELECT * FROM vehicle_metrics 
ORDER BY time DESC 
LIMIT 20;

-- 3. TRACK ONE VEHICLE OVER TIME
SELECT time, vehicle_id, speed, status 
FROM vehicle_metrics 
WHERE vehicle_id = 'vehicle_000'  -- Change vehicle ID here
ORDER BY time DESC 
LIMIT 50;

-- 4. FIND TRAFFIC JAMS
SELECT vehicle_id, status, COUNT(*) as count
FROM vehicle_metrics 
WHERE status = 'traffic_jam'  -- Change status here
GROUP BY vehicle_id, status
ORDER BY count DESC;

-- 5. FIND SLOWEST/FASTEST SPEEDS
-- For slowest, use: ORDER BY speed ASC
-- For fastest, use: ORDER BY speed DESC
SELECT time, vehicle_id, speed, status
FROM vehicle_metrics
ORDER BY speed ASC
LIMIT 10;