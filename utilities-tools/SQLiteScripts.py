# Weekly Training Load (distance, duration, calories per sport)

SELECT 
    sport,
    strftime('%Y-%W', start_time) AS week,
    SUM(distance) AS total_distance_m,
    SUM(duration) AS total_duration_s,
    SUM(calories) AS total_calories
FROM activities
GROUP BY sport, week
ORDER BY week DESC, sport;



# Cardio Efficiency (average pace vs. average HR)

SELECT 
    activity_id,
    sport,
    distance / duration AS pace_m_per_s,
    avg_hr
FROM activities
WHERE duration > 0 AND avg_hr IS NOT NULL;
-- Note: pace is calculated as meters per second (m/s)

# Pacing Consistency (lap pace variance per activity)

SELECT 
    activity_id,
    AVG(lap_distance / lap_elapsed_time) AS avg_lap_pace,
    (MAX(lap_distance / lap_elapsed_time) - MIN(lap_distance / lap_elapsed_time)) AS pace_variance
FROM activity_laps
WHERE lap_elapsed_time > 0
GROUP BY activity_id;
-- Note: pace is calculated as meters per second (m/s)

# Heart Rate Drift (compare first vs last 25% of activity)

WITH quartiles AS (
    SELECT 
        activity_id,
        MIN(timestamp) AS start_ts,
        MAX(timestamp) AS end_ts
    FROM activity_records
    GROUP BY activity_id
)
SELECT 
    r.activity_id,
    AVG(CASE WHEN r.timestamp < q.start_ts + 0.25 * (q.end_ts - q.start_ts) THEN r.heart_rate END) AS early_avg_hr,
    AVG(CASE WHEN r.timestamp > q.start_ts + 0.75 * (q.end_ts - q.start_ts) THEN r.heart_rate END) AS late_avg_hr,
    (AVG(CASE WHEN r.timestamp > q.start_ts + 0.75 * (q.end_ts - q.start_ts) THEN r.heart_rate END) -
     AVG(CASE WHEN r.timestamp < q.start_ts + 0.25 * (q.end_ts - q.start_ts) THEN r.heart_rate END)) AS hr_drift
FROM activity_records r
JOIN quartiles q ON r.activity_id = q.activity_id
GROUP BY r.activity_id;
-- Note: This assumes activity_records has a timestamp column indicating the time of each record.

# Gait Stability (stride length variability)

SELECT 
    activity_id,
    AVG(stride_length) AS avg_stride_length,
    (MAX(stride_length) - MIN(stride_length)) AS stride_range
FROM steps_activities
WHERE stride_length IS NOT NULL
GROUP BY activity_id;
-- Note: This assumes steps_activities has a stride_length column.

# Fatigue Indicators (ground contact time increase across activity)

SELECT 
    activity_id,
    AVG(ground_contact_time) AS avg_gct,
    MAX(ground_contact_time) - MIN(ground_contact_time) AS gct_drift
FROM steps_activities
WHERE ground_contact_time IS NOT NULL
GROUP BY activity_id;
-- Note: This assumes steps_activities has a ground_contact_time column.




