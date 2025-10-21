#!/usr/bin/env python3
"""
alerts.py (updated)

- Computes baselines and alerts (gait, HR drift, GCT drift, heat sensitivity).
- Sends notifications to Slack and/or Email.
- Persists alert history to:
    1) alerts_state.json (cooldown/history, backward compatible)
    2) sqlite table alert_logs inside configured DB (ALERT_DB_PATH or GARMIN_ACTIVITIES_DB)

Environment variables (summary):
- ALERT_LOOKBACK (int) default 30
- ALERT_K (float) default 2.0
- ALERT_COOLDOWN_HOURS (int) default 24
- SLACK_WEBHOOK_URL (str) optional
- ALERT_SMTP_* for email (optional)
- ALERT_FROM_EMAIL, ALERT_TO_EMAIL
- ALERT_STATE_PATH default ./alerts_state.json
- ALERT_DB_PATH (optional) path to sqlite DB to store alert_logs; falls back to GARMIN_ACTIVITIES_DB or ./garmin_activities.db
"""

import os
import time
import json
from datetime import datetime
import sqlite3
import math
import smtplib
from email.message import EmailMessage
import traceback

import pandas as pd
import numpy as np
import requests

# Config (env overrides)
LOOKBACK = int(os.environ.get("ALERT_LOOKBACK", "30"))
K = float(os.environ.get("ALERT_K", "2.0"))
COOLDOWN_HOURS = int(os.environ.get("ALERT_COOLDOWN_HOURS", "24"))
STATE_PATH = os.environ.get("ALERT_STATE_PATH", "alerts_state.json")

SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL", None)

SMTP_HOST = os.environ.get("ALERT_SMTP_HOST")
SMTP_PORT = int(os.environ.get("ALERT_SMTP_PORT", "587")) if os.environ.get("ALERT_SMTP_PORT") else None
SMTP_USER = os.environ.get("ALERT_SMTP_USER")
SMTP_PASS = os.environ.get("ALERT_SMTP_PASS")
ALERT_FROM = os.environ.get("ALERT_FROM_EMAIL")
ALERT_TO = os.environ.get("ALERT_TO_EMAIL")  # comma separated

MIN_SAMPLES = int(os.environ.get("ALERT_MIN_SAMPLES", "8"))

# DB path for logging alerts: prefer explicit ALERT_DB_PATH, else GARMIN_ACTIVITIES_DB, else default
ALERT_DB_PATH = os.environ.get("ALERT_DB_PATH") or os.environ.get("GARMIN_ACTIVITIES_DB") or "garmin_activities.db"

# create alert_logs table if not exists
def ensure_alert_table(db_path=ALERT_DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS alert_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL,
            threshold REAL,
            message TEXT,
            channel TEXT,
            status TEXT
        );
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print("[ERROR] Could not ensure alert_logs table:", e)

# state persistence helpers
def load_state(path=STATE_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_state(state, path=STATE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, default=str, indent=2)

# notification helpers
def send_slack_message(text):
    if not SLACK_WEBHOOK:
        print("[ALERT] Slack webhook not configured; skipping Slack delivery.")
        return False
    try:
        payload = {"text": text}
        resp = requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
        resp.raise_for_status()
        print("[ALERT] Slack message sent.")
        return True
    except Exception as e:
        print("[ERROR] Slack send failed:", e)
        return False

def send_email(subject, body):
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and ALERT_FROM and ALERT_TO):
        print("[ALERT] SMTP/email not configured; skipping email delivery.")
        return False
    try:
        msg = EmailMessage()
        msg["From"] = ALERT_FROM
        msg["To"] = ALERT_TO
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("[ALERT] Email sent.")
        return True
    except Exception as e:
        print("[ERROR] Email send failed:", e)
        return False

# robust stats
def median_mad(series):
    series = series.dropna()
    if len(series) == 0:
        return None, None
    med = series.median()
    mad = (series - med).abs().median()
    return float(med), float(mad)

def mean_std(series):
    series = series.dropna()
    if len(series) == 0:
        return None, None
    return float(series.mean()), float(series.std())

def compute_baseline(series):
    s = series.dropna()
    if len(s) >= MIN_SAMPLES:
        med, mad = median_mad(s)
        return {"method":"median_mad", "center":med, "scale":mad}
    else:
        m, sd = mean_std(s)
        return {"method":"mean_std", "center":m, "scale":sd}

def days_since_iso(iso_ts):
    try:
        t = datetime.fromisoformat(iso_ts)
        return (datetime.utcnow() - t).total_seconds() / 3600.0
    except Exception:
        return 9999.0

def should_alert(state, metric_name, cooldown=COOLDOWN_HOURS):
    last = state.get("last_alert", {}).get(metric_name)
    if not last:
        return True
    hrs = days_since_iso(last)
    return hrs >= cooldown

def update_alert_time(state, metric_name):
    state.setdefault("last_alert", {})[metric_name] = datetime.utcnow().isoformat()

# log alert into sqlite table
def log_alert_to_db(metric, value, threshold, message, channel, status, db_path=ALERT_DB_PATH):
    try:
        ensure_alert_table(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO alert_logs (timestamp, metric, value, threshold, message, channel, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), metric, value, threshold, message, channel, status))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("[ERROR] Failed to write alert log to DB:", e)
        return False

# main alerting function (reads CSVs in outputs dir as before)
def alert_on_metrics(metrics_dir="outputs", lookback=LOOKBACK, k=K):
    """
    Expects CSVs to exist in metrics_dir:
      - gait_stability.csv -> stride_range
      - heart_rate_drift.csv -> hr_drift
      - fatigue_indicators.csv -> gct_drift
    Optionally activity_temps.csv for avg_temp per activity to compute heat sensitivity.
    """
    state = load_state()
    alerts_sent = []

    def load_csv(name):
        path = os.path.join(metrics_dir, f"{name}.csv")
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()

    gait = load_csv("gait_stability")
    hr = load_csv("heart_rate_drift")
    fat = load_csv("fatigue_indicators")
    temps = load_csv("activity_temps")  # optional {activity_id, avg_temp}

    def check_metric(df, col, name, min_samples=MIN_SAMPLES, absolute_min=None, extra_text=""):
        if df.empty or col not in df.columns:
            print(f"[INFO] Metric {name} not available.")
            return False

        s_all = df[col].dropna().astype(float)
        if s_all.empty or len(s_all) < 2:
            print(f"[INFO] Not enough samples for {name}.")
            return False

        # use the last `lookback` samples for baseline
        baseline_series = s_all.tail(lookback)
        baseline = compute_baseline(baseline_series)

        latest_value = float(s_all.tail(1).iloc[0])
        center = baseline.get("center")
        scale = baseline.get("scale") if baseline.get("scale") is not None else 0.0
        scale_nonzero = scale if scale and scale > 0 else (max(abs(center or 0)*0.01, 1e-6))

        z = (latest_value - center) / scale_nonzero if center is not None else None
        breached = False
        if z is not None:
            breached = z >= k
        elif absolute_min is not None:
            breached = latest_value >= absolute_min

        if breached:
            if should_alert(state, name):
                msg = (
                    f"ALERT: {name} breached\n"
                    f"- value: {latest_value}\n"
                    f"- baseline (method={baseline.get('method')}): center={center}, scale={scale}\n"
                    f"- z: {z}\n{extra_text}"
                )
                slack_ok = send_slack_message(msg) if SLACK_WEBHOOK else False
                email_ok = False
                try:
                    email_ok = send_email(f"[Garmin Alert] {name} breach", msg)
                except Exception:
                    email_ok = False

                # log into sqlite
                channels = []
                if slack_ok: channels.append("slack")
                if email_ok: channels.append("email")
                channel_str = ",".join(channels) if channels else "none"

                status = "sent" if slack_ok or email_ok else "failed"
                db_ok = log_alert_to_db(name, latest_value, scale, msg, channel_str, status)

                update_alert_time(state, name)
                state.setdefault("history", {}).setdefault(name, []).append({
                    "time": datetime.utcnow().isoformat(),
                    "value": latest_value,
                    "z": z,
                    "sent_slack": slack_ok,
                    "sent_email": email_ok,
                    "db_logged": db_ok
                })
                save_state(state)
                alerts_sent.append({"metric": name, "value": latest_value, "z": z, "slack": slack_ok, "email": email_ok})
                print(f"[ALERT] {name} triggered. Slack:{slack_ok} Email:{email_ok} DB:{db_ok}")
            else:
                print(f"[INFO] {name} breached but cooldown active; skipping.")
        else:
            print(f"[INFO] {name} within baseline (z={z:.2f} if computed).")
        return breached

    # gait_striderange
    check_metric(gait, "stride_range", "gait_stride_range", extra_text="Gait stability deviation detected.")

    # heart rate drift
    check_metric(hr, "hr_drift", "heart_rate_drift", extra_text="HR drift late vs early in activity.")

    # gct drift
    check_metric(fat, "gct_drift", "gct_drift", extra_text="Ground contact time (GCT) drift detected.")

    # heat sensitivity: need hr and temps merged with activity ids
    if not hr.empty:
        # Merge hr (expects activity_id, hr_drift) with temps (activity_id, avg_temp) if available
        merged = hr.copy()
        if not temps.empty and "activity_id" in temps.columns and "avg_temp" in temps.columns:
            merged = merged.merge(temps[["activity_id", "avg_temp"]], on="activity_id", how="inner")
        else:
            merged["avg_temp"] = np.nan

        m = merged.dropna(subset=["hr_drift", "avg_temp"])
        if len(m) >= MIN_SAMPLES:
            x = m["avg_temp"].astype(float).values
            y = m["hr_drift"].astype(float).values
            try:
                slope = np.polyfit(x, y, 1)[0]
            except Exception:
                slope = None
            slope_threshold = float(os.environ.get("ALERT_HEAT_SLOPE", "0.3"))
            name = "heat_sensitivity"
            if slope is not None and slope >= slope_threshold:
                if should_alert(state, name):
                    extra = f"Heat sensitivity slope (hr_drift vs temp) = {slope:.3f} >= threshold {slope_threshold}"
                    msg = (
                        f"ALERT: {name} breached\n"
                        f"- slope: {slope}\n"
                        f"- threshold: {slope_threshold}\n{extra}"
                    )
                    slack_ok = send_slack_message(msg) if SLACK_WEBHOOK else False
                    email_ok = send_email(f"[Garmin Alert] {name} breach", msg)
                    channels = []
                    if slack_ok: channels.append("slack")
                    if email_ok: channels.append("email")
                    channel_str = ",".join(channels) if channels else "none"
                    status = "sent" if slack_ok or email_ok else "failed"
                    db_ok = log_alert_to_db(name, slope, slope_threshold, msg, channel_str, status)
                    update_alert_time(state, name)
                    state.setdefault("history", {}).setdefault(name, []).append({
                        "time": datetime.utcnow().isoformat(),
                        "value": slope,
                        "threshold": slope_threshold,
                        "sent_slack": slack_ok,
                        "sent_email": email_ok,
                        "db_logged": db_ok
                    })
                    save_state(state)
                    alerts_sent.append({"metric": name, "value": slope, "slack": slack_ok, "email": email_ok})
                    print(f"[ALERT] {name} triggered. Slack:{slack_ok} Email:{email_ok} DB:{db_ok}")
                else:
                    print("[INFO] heat_sensitivity breached but cooldown active; skipping.")
            else:
                print(f"[INFO] heat_sensitivity slope {slope} below threshold {slope_threshold}.")
        else:
            print("[INFO] Not enough samples for heat sensitivity check.")
    else:
        print("[INFO] HR drift data not available; skipping heat sensitivity check.")

    return alerts_sent
# -------------------------