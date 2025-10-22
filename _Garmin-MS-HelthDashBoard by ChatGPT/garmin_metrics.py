#!/usr/bin/env python3
"""
garmin_metrics.py

Query garmin_activities.db, save CSVs, produce Matplotlib plots,
and run alerts (alerts.py) to detect baseline breaches.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# try to import alerts module (must be in same folder)
try:
    import alerts
except Exception:
    alerts = None

# ------------------------------
# Configuration
# ------------------------------
DB_PATH = os.environ.get("GARMIN_ACTIVITIES_DB", "garmin_activities.db")
OUTPUT_DIR = os.environ.get("GARMIN_OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------
# SQL Queries (same as before)
# ------------------------------
QUERIES = {
    "weekly_training_load": '''
        SELECT sport,
               strftime('%Y-%W', start_time) AS week,
               SUM(distance) AS total_distance_m,
               SUM(duration) AS total_duration_s,
               SUM(calories) AS total_calories
        FROM activities
        GROUP BY sport, week
        ORDER BY week DESC, sport;
    ''',
    "cardio_efficiency": '''
        SELECT activity_id, sport,
               CASE WHEN duration > 0 THEN (distance / duration) ELSE NULL END AS pace_m_per_s,
               avg_hr
        FROM activities
        WHERE duration > 0 AND avg_hr IS NOT NULL AND distance IS NOT NULL;
    ''',
    "pacing_consistency": '''
        SELECT activity_id,
               AVG(CASE WHEN lap_elapsed_time > 0 THEN lap_distance / lap_elapsed_time END) AS avg_lap_pace,
               (MAX(CASE WHEN lap_elapsed_time > 0 THEN lap_distance / lap_elapsed_time END) - 
                MIN(CASE WHEN lap_elapsed_time > 0 THEN lap_distance / lap_elapsed_time END)) AS pace_variance
        FROM activity_laps
        GROUP BY activity_id;
    ''',
    "heart_rate_drift": '''
        WITH quartiles AS (
            SELECT activity_id,
                   MIN(timestamp) AS start_ts,
                   MAX(timestamp) AS end_ts
            FROM activity_records
            GROUP BY activity_id
        )
        SELECT r.activity_id,
               AVG(CASE WHEN r.timestamp < q.start_ts + 0.25 * (q.end_ts - q.start_ts) THEN r.heart_rate END) AS early_avg_hr,
               AVG(CASE WHEN r.timestamp > q.start_ts + 0.75 * (q.end_ts - q.start_ts) THEN r.heart_rate END) AS late_avg_hr,
               (AVG(CASE WHEN r.timestamp > q.start_ts + 0.75 * (q.end_ts - q.start_ts) THEN r.heart_rate END) -
                AVG(CASE WHEN r.timestamp < q.start_ts + 0.25 * (q.end_ts - q.start_ts) THEN r.heart_rate END)) AS hr_drift
        FROM activity_records r
        JOIN quartiles q ON r.activity_id = q.activity_id
        GROUP BY r.activity_id;
    ''',
    "gait_stability": '''
        SELECT activity_id,
               AVG(stride_length) AS avg_stride_length,
               (MAX(stride_length) - MIN(stride_length)) AS stride_range
        FROM steps_activities
        WHERE stride_length IS NOT NULL
        GROUP BY activity_id;
    ''',
    "fatigue_indicators": '''
        SELECT activity_id,
               AVG(ground_contact_time) AS avg_gct,
               (MAX(ground_contact_time) - MIN(ground_contact_time)) AS gct_drift
        FROM steps_activities
        WHERE ground_contact_time IS NOT NULL
        GROUP BY activity_id;
    '''
}

# ------------------------------
# Helpers (same as earlier)
# ------------------------------
def fetch_df(conn, sql):
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        print(f"[WARN] Query failed: {e}")
        return pd.DataFrame()

def save_csv(df, name):
    if df is None or df.empty:
        print(f"[INFO] No data for {name}; skipping CSV save.")
        return
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[OK] Saved: {path}")

def plot_and_save(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[OK] Plot: {path}")

# Visualization functions (same as previous code)
def plot_weekly_training_load(df):
    if df.empty:
        return
    d = df.copy()
    d["total_distance_km"] = d["total_distance_m"] / 1000.0
    pivot_dist = d.pivot_table(index="week", columns="sport", values="total_distance_km", aggfunc="sum").fillna(0)
    pivot_dur = d.pivot_table(index="week", columns="sport", values="total_duration_s", aggfunc="sum").fillna(0)
    pivot_cal = d.pivot_table(index="week", columns="sport", values="total_calories", aggfunc="sum").fillna(0)
    for metric_name, pivot in [("weekly_distance_km_by_sport", pivot_dist),
                               ("weekly_duration_s_by_sport", pivot_dur),
                               ("weekly_calories_by_sport", pivot_cal)]:
        fig = plt.figure()
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(metric_name.replace("_", " ").title())
        plt.xlabel("Week")
        plt.ylabel(metric_name.replace("_", " "))
        plt.xticks(rotation=45, ha="right")
        plot_and_save(fig, metric_name)

def plot_cardio_efficiency(df):
    if df.empty:
        return
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df["avg_hr"], df["pace_m_per_s"])
    plt.title("Cardio Efficiency: Pace (m/s) vs Avg HR")
    plt.xlabel("Avg HR (bpm)")
    plt.ylabel("Pace (m/s)")
    plot_and_save(fig, "cardio_efficiency_scatter")

def plot_pacing_consistency(df):
    if df.empty:
        return
    d = df.dropna(subset=["pace_variance"]).copy()
    if d.empty:
        return
    d = d.sort_values("pace_variance", ascending=False).head(30)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["pace_variance"])
    plt.title("Pacing Consistency: Lap Pace Variance (Top 30)")
    plt.xlabel("Activity ID")
    plt.ylabel("Pace Variance (m/s)")
    plt.xticks(rotation=90)
    plot_and_save(fig, "pacing_consistency_variance_top30")

def plot_heart_rate_drift(df):
    if df.empty:
        return
    d = df.dropna(subset=["hr_drift"]).copy()
    if d.empty:
        return
    d = d.sort_values("hr_drift", ascending=False).head(30)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["hr_drift"])
    plt.title("Heart Rate Drift (Top 30 by Drift)")
    plt.xlabel("Activity ID")
    plt.ylabel("HR Drift (bpm)")
    plt.xticks(rotation=90)
    plot_and_save(fig, "heart_rate_drift_top30")

def plot_gait_stability(df):
    if df.empty:
        return
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df["avg_stride_length"], df["stride_range"])
    plt.title("Gait Stability: Stride Range vs Avg Stride Length")
    plt.xlabel("Avg Stride Length (m)")
    plt.ylabel("Stride Range (m)")
    plot_and_save(fig, "gait_stability_scatter")

def plot_fatigue_indicators(df):
    if df.empty:
        return
    d = df.dropna(subset=["gct_drift"]).copy()
    if d.empty:
        return
    d = d.sort_values("gct_drift", ascending=False).head(30)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["gct_drift"])
    plt.title("Fatigue Indicator: Ground Contact Time Drift (Top 30)")
    plt.xlabel("Activity ID")
    plt.ylabel("GCT Drift (ms)")
    plt.xticks(rotation=90)
    plot_and_save(fig, "fatigue_gct_drift_top30")

def main():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    results = {}
    for name, sql in QUERIES.items():
        df = fetch_df(conn, sql)
        results[name] = df
        save_csv(df, name)
    conn.close()

    # Visualizations
    plot_weekly_training_load(results.get("weekly_training_load", pd.DataFrame()))
    plot_cardio_efficiency(results.get("cardio_efficiency", pd.DataFrame()))
    plot_pacing_consistency(results.get("pacing_consistency", pd.DataFrame()))
    plot_heart_rate_drift(results.get("heart_rate_drift", pd.DataFrame()))
    plot_gait_stability(results.get("gait_stability", pd.DataFrame()))
    plot_fatigue_indicators(results.get("fatigue_indicators", pd.DataFrame()))

    print("[DONE] Metrics computed and plots saved to:", PLOTS_DIR)

    # Run alerts if available
    try:
        if alerts is not None:
            print("[INFO] Running alerts...")
            sent = alerts.alert_on_metrics(metrics_dir=OUTPUT_DIR)
            if sent:
                print("[INFO] Alerts sent:", sent)
            else:
                print("[INFO] No alerts triggered.")
        else:
            print("[INFO] alerts module not available; skipping alerts.")
    except Exception as e:
        print("[ERROR] Exception while running alerts:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
