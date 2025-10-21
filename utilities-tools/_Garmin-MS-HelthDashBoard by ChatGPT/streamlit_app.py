#!/usr/bin/env python3
"""
streamlit_app.py

Interactive Streamlit dashboard for garmin_activities.db with Alert History tab.
Includes full metrics + Alert History with acknowledge button functionality.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Garmin Activities Dashboard", layout="wide")

st.title("Garmin Activities Dashboard")
st.markdown("Interactive analytics for **garmin_activities.db**. Includes Alert History with acknowledge feature.")

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
                AVG(CASE WHEN r.timestamp < 0.25 * (q.end_ts - q.start_ts) THEN r.heart_rate END)) AS hr_drift
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
    ''',
    "alert_logs": '''
        SELECT id, timestamp, metric, value, threshold, message, channel, status, 
               COALESCE(acknowledged,0) AS acknowledged, ack_by, ack_time
        FROM alert_logs
        ORDER BY timestamp DESC
        LIMIT 1000;
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

# Tabs
tab = st.tabs(["Dashboard", "Alert History"])

# ========== Dashboard tab ==========
with tab[0]:
    st.subheader("Weekly Training Load")
    weekly = run_query(conn, QUERIES["weekly_training_load"])
    if not weekly.empty and sport_sel:
        weekly = weekly[weekly["sport"].isin(sport_sel)]
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

        pivot_dur = d.pivot_table(index="week", columns="sport", values="total_duration_s", aggfunc="sum").fillna(0)
        fig2 = plt.figure()
        pivot_dur.plot(kind="bar", ax=plt.gca())
        plt.title("Weekly Duration (s) by Sport")
        plt.xlabel("Week")
        plt.ylabel("Duration (s)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

    st.subheader("Cardio Efficiency")
    eff = run_query(conn, QUERIES["cardio_efficiency"])
    if not eff.empty and sport_sel:
        eff = eff[eff["sport"].isin(sport_sel)]
        fig = plt.figure()
        plt.scatter(eff["avg_hr"], eff["pace_m_per_s"])
        plt.title("Pace (m/s) vs Avg HR")
        plt.xlabel("Avg HR (bpm)")
        plt.ylabel("Pace (m/s)")
        st.pyplot(fig)

    st.subheader("Pacing Consistency")
    pacevar = run_query(conn, QUERIES["pacing_consistency"])
    if not pacevar.empty:
        d = pacevar.dropna(subset=["pace_variance"]).copy().sort_values("pace_variance", ascending=False).head(50)
        fig = plt.figure()
        plt.bar(d["activity_id"].astype(str), d["pace_variance"])
        plt.title("Lap Pace Variance (Top 50)")
        plt.xlabel("Activity ID")
        plt.ylabel("Pace Variance (m/s)")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    st.subheader("Heart Rate Drift")
    drift = run_query(conn, QUERIES["heart_rate_drift"])
    if not drift.empty:
        d = drift.dropna(subset=["hr_drift"]).copy().sort_values("hr_drift", ascending=False).head(50)
        fig = plt.figure()
        plt.bar(d["activity_id"].astype(str), d["hr_drift"])
        plt.title("HR Drift (Top 50 by Drift)")
        plt.xlabel("Activity ID")
        plt.ylabel("HR Drift (bpm)")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    st.subheader("Gait Stability")
    gait = run_query(conn, QUERIES["gait_stability"])
    if not gait.empty:
        plt.figure()
        plt.scatter(gait["avg_stride_length"], gait["stride_range"])
        plt.title("Stride Range vs Avg Stride Length")
        plt.xlabel("Avg Stride Length (m)")
        plt.ylabel("Stride Range (m)")
        st.pyplot(plt.gcf())

    st.subheader("Fatigue Indicators: GCT Drift")
    fat = run_query(conn, QUERIES["fatigue_indicators"])
    if not fat.empty:
        d = fat.dropna(subset=["gct_drift"]).copy().sort_values("gct_drift", ascending=False).head(50)
        plt.figure()
        plt.bar(d["activity_id"].astype(str), d["gct_drift"])
        plt.title("Ground Contact Time Drift (Top 50)")
        plt.xlabel("Activity ID")
        plt.ylabel("GCT Drift (ms)")
        plt.xticks(rotation=90)
        st.pyplot(plt.gcf())

# ========== Alert History tab ==========
with tab[1]:
    st.subheader("Alert History (from alert_logs table)")
    try:
        alerts_df = run_query(conn, QUERIES["alert_logs"])
    except Exception as e:
        alerts_df = pd.DataFrame()
        st.error(f"Could not read alert_logs: {e}")

    if alerts_df.empty:
        st.info("No alert logs found in database. Alerts not fired yet or alert_logs table missing.")
    else:
        # Filter by metric
        metrics = alerts_df["metric"].dropna().unique().tolist()
        metric_sel = st.multiselect("Filter metrics", options=sorted(metrics), default=sorted(metrics))
        df_filtered = alerts_df[alerts_df["metric"].isin(metric_sel)] if metric_sel else alerts_df

        # Filter by date range
        df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
        min_date = df_filtered["timestamp"].min()
        max_date = df_filtered["timestamp"].max()
        date_range = st.date_input("Date range", value=(min_date.date() if pd.notnull(min_date) else None,
                                                        max_date.date() if pd.notnull(max_date) else None))
        if date_range and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            df_filtered = df_filtered[(df_filtered["timestamp"] >= start_dt) & (df_filtered["timestamp"] < end_dt)]

        # Display table with acknowledge button
        st.markdown("### Alerts Table")
        for idx, row in df_filtered.iterrows():
            col1, col2, col3 = st.columns([6, 1, 1])
            with col1:
                st.markdown(f"**{row['metric']}** | {row['timestamp']} | value: {row['value']} | status: {row['status']}")
            with col2:
                if row.get("acknowledged", 0):
                    st.success(f"Ack by {row.get('ack_by')} at {row.get('ack_time')}")
                else:
                    if st.button(f"Acknowledge {row['id']}", key=f"ack_{row['id']}"):
                        try:
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE alert_logs
                                SET acknowledged=1, ack_by=?, ack_time=?
                                WHERE id=?
                            """, ("User", datetime.utcnow().isoformat(), row["id"]))
                            conn.commit()
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to acknowledge: {e}")
            with col3:
                st.write("")

        # Plot: counts by metric
        st.markdown("### Alerts count by metric")
        counts = df_filtered.groupby("metric").size().reset_index(name="count").sort_values("count", ascending=False)
        if not counts.empty:
            fig = plt.figure()
            plt.bar(counts["metric"].astype(str), counts["count"])
            plt.title("Alert counts by metric")
            plt.xlabel("Metric")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        # Plot: daily alert counts
        st.markdown("### Alerts over time (daily)")
        df_filtered["date"] = df_filtered["timestamp"].dt.date
        time_series = df_filtered.groupby("date").size().reset_index(name="alerts")
        if not time_series.empty:
            plt.figure()
            plt.plot(time_series["date"].astype(str), time_series["alerts"], marker='o')
            plt.title("Daily alert counts")
            plt.xlabel("Date")
            plt.ylabel("Alerts")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(plt.gcf())

conn.close()
