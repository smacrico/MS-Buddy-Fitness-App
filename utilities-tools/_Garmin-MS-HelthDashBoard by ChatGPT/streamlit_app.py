#!/usr/bin/env python3
"""
streamlit_app.py

Interactive Streamlit dashboard for garmin_activities.db.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Garmin Activities Dashboard", layout="wide")

st.title("Garmin Activities Dashboard")
st.markdown("Interactive analytics for **garmin_activities.db**.")

# Inputs
db_path = st.text_input("Path to garmin_activities.db", value="garmin_activities.db")
hr_zones = st.text_area(
    "HR Zones (comma-separated, bpm thresholds; e.g., 100,120,140,160,180)",
    value="100,120,140,160,180"
)

def parse_zones(s):
    try:
        z = [int(x.strip()) for x in s.split(",") if x.strip()]
        z = sorted(z)
        return z
    except Exception:
        return []

zones = parse_zones(hr_zones)

# Queries
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
    ''',
    "activities_list": '''
        SELECT activity_id, sport, start_time, distance, duration, avg_hr
        FROM activities
        ORDER BY start_time DESC
        LIMIT 500;
    '''
}

def run_query(conn, sql, params=None):
    try:
        if params:
            return pd.read_sql_query(sql, conn, params=params)
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()

if not os.path.exists(db_path):
    st.error(f"Database not found at {db_path}")
    st.stop()

conn = sqlite3.connect(db_path)

# Sidebar filters
act_list = run_query(conn, QUERIES["activities_list"])
with st.sidebar:
    st.header("Filters")
    sports = sorted([s for s in act_list["sport"].dropna().unique()]) if not act_list.empty else []
    sport_sel = st.multiselect("Sport", sports, default=sports)
    st.caption("Filters apply to charts where relevant.")

# Load data
weekly = run_query(conn, QUERIES["weekly_training_load"])
eff = run_query(conn, QUERIES["cardio_efficiency"])
pacevar = run_query(conn, QUERIES["pacing_consistency"])
drift = run_query(conn, QUERIES["heart_rate_drift"])
gait = run_query(conn, QUERIES["gait_stability"])
fat = run_query(conn, QUERIES["fatigue_indicators"])

# Apply filters
if not weekly.empty and sport_sel:
    weekly = weekly[weekly["sport"].isin(sport_sel)]
if not eff.empty and sport_sel:
    eff = eff[eff["sport"].isin(sport_sel)]

# Layout
col1, col2 = st.columns(2)

# Weekly load charts
with col1:
    st.subheader("Weekly Training Load")
    if not weekly.empty:
        d = weekly.copy()
        d["total_distance_km"] = d["total_distance_m"] / 1000.0
        pivot_dist = d.pivot_table(index="week", columns="sport", values="total_distance_km", aggfunc="sum").fillna(0)
        fig = plt.figure()
        pivot_dist.plot(kind="bar", ax=plt.gca())
        plt.title("Weekly Distance (km) by Sport")
        plt.xlabel("Week")
        plt.ylabel("Distance (km)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    if not weekly.empty:
        pivot_dur = weekly.pivot_table(index="week", columns="sport", values="total_duration_s", aggfunc="sum").fillna(0)
        fig2 = plt.figure()
        pivot_dur.plot(kind="bar", ax=plt.gca())
        plt.title("Weekly Duration (s) by Sport")
        plt.xlabel("Week")
        plt.ylabel("Duration (s)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

# Cardio efficiency
with col2:
    st.subheader("Cardio Efficiency")
    if not eff.empty:
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(eff["avg_hr"], eff["pace_m_per_s"])
        plt.title("Pace (m/s) vs Avg HR")
        plt.xlabel("Avg HR (bpm)")
        plt.ylabel("Pace (m/s)")
        st.pyplot(fig)

# Pacing consistency
st.subheader("Pacing Consistency")
if not pacevar.empty:
    d = pacevar.dropna(subset=["pace_variance"]).copy().sort_values("pace_variance", ascending=False).head(50)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["pace_variance"])
    plt.title("Lap Pace Variance (Top 50)")
    plt.xlabel("Activity ID")
    plt.ylabel("Pace Variance (m/s)")
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Heart rate drift
st.subheader("Heart Rate Drift")
if not drift.empty:
    d = drift.dropna(subset=["hr_drift"]).copy().sort_values("hr_drift", ascending=False).head(50)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["hr_drift"])
    plt.title("HR Drift (Top 50 by Drift)")
    plt.xlabel("Activity ID")
    plt.ylabel("HR Drift (bpm)")
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Gait stability
st.subheader("Gait Stability")
if not gait.empty:
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(gait["avg_stride_length"], gait["stride_range"])
    plt.title("Stride Range vs Avg Stride Length")
    plt.xlabel("Avg Stride Length (m)")
    plt.ylabel("Stride Range (m)")
    st.pyplot(fig)

# Fatigue indicators
st.subheader("Fatigue Indicators: GCT Drift")
if not fat.empty:
    d = fat.dropna(subset=["gct_drift"]).copy().sort_values("gct_drift", ascending=False).head(50)
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(d["activity_id"].astype(str), d["gct_drift"])
    plt.title("Ground Contact Time Drift (Top 50)")
    plt.xlabel("Activity ID")
    plt.ylabel("GCT Drift (ms)")
    plt.xticks(rotation=90)
    st.pyplot(fig)

# HR Zones per selected activity
st.subheader("HR Zones (per-activity, interactive)")
act_options = act_list["activity_id"].astype(str).tolist() if not act_list.empty else []
act_choice = st.selectbox("Choose an activity for zone breakdown", options=act_options)

if act_choice:
    try:
        act_id = int(act_choice)
    except ValueError:
        act_id = None

    if act_id is not None and zones:
        rec = run_query(conn,
                         "SELECT timestamp, heart_rate FROM activity_records WHERE activity_id = ? AND heart_rate IS NOT NULL ORDER BY timestamp ASC;",
                         params=(act_id,))
        if not rec.empty:
            rec = rec.sort_values("timestamp")
            if "timestamp" in rec.columns and len(rec) > 1:
                dt = rec["timestamp"].diff().fillna(0)
                dt = dt.where(dt > 0, 1)
            else:
                dt = pd.Series([1]*len(rec))

            z_edges = [-10**9] + zones + [10**9]
            z_labels = []
            for i in range(len(z_edges)-1):
                if i == 0:
                    z_labels.append(f"<= {z_edges[i+1]} bpm")
                else:
                    z_labels.append(f"{z_edges[i]}–{z_edges[i+1]} bpm" if i < len(z_edges)-2 else f">{z_edges[i]} bpm")

            rec["zone"] = pd.cut(rec["heart_rate"], bins=z_edges, labels=z_labels, include_lowest=True)
            rec["dt"] = dt.values

            zone_time = rec.groupby("zone")["dt"].sum().reset_index(name="seconds")
            zone_time = zone_time.sort_values("seconds", ascending=False)

            fig = plt.figure()
            ax = plt.gca()
            ax.bar(zone_time["zone"].astype(str), zone_time["seconds"])
            plt.title(f"Time in HR Zones (Activity {act_id})")
            plt.xlabel("Zone")
            plt.ylabel("Seconds")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            st.dataframe(zone_time)

conn.close()
    st.success("Data loaded and visualizations generated.")