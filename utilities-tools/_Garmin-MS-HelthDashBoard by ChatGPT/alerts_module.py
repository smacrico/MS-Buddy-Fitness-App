import os
import sqlite3
import pandas as pd
import smtplib
import requests
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# -------------------------
# Load ENV VARS
# -------------------------
DB_PATH = os.getenv("DB_PATH", "./garmin_activities.db")

# Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Email
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "").split(",")

# Thresholds (base values; baseline-adjusted later)
ALERT_THRESHOLD_GAIT = float(os.getenv("ALERT_THRESHOLD_GAIT", "0.15"))   # stride variance (m)
ALERT_THRESHOLD_HRDRIFT = float(os.getenv("ALERT_THRESHOLD_HRDRIFT", "10"))  # bpm
ALERT_THRESHOLD_HEAT = float(os.getenv("ALERT_THRESHOLD_HEAT", "0.25"))   # bpm per °C

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))  # rolling baseline window

# -------------------------
# Helper functions
# -------------------------
def send_slack_alert(message: str):
    if not SLACK_WEBHOOK_URL:
        print("Slack webhook not set; skipping Slack alert.")
        return
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        if response.status_code != 200:
            print(f"Slack error: {response.text}")
    except Exception as e:
        print(f"Slack exception: {e}")


def send_email_alert(subject: str, message: str):
    if not EMAIL_ENABLED or not EMAIL_SENDER or not EMAIL_RECIPIENTS:
        print("Email not configured; skipping email alert.")
        return
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = ", ".join(EMAIL_RECIPIENTS)

        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
    except Exception as e:
        print(f"Email error: {e}")


def fetch_dataframe(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# -------------------------
# Metrics & Baselines
# -------------------------
def compute_baselines():
    """Compute rolling baselines for gait, HR drift, heat sensitivity."""
    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).isoformat()

    # Example queries (tables may differ slightly depending on Garmin DB version)
    gait_df = fetch_dataframe(f"""
        SELECT activity_id, stride_length, start_time
        FROM steps_activities sa
        JOIN activities a ON sa.activity_id = a.activity_id
        WHERE a.start_time > '{cutoff}'
    """)
    hr_drift_df = fetch_dataframe(f"""
        SELECT activity_id, avg_hr, duration, start_time
        FROM activities
        WHERE start_time > '{cutoff}'
    """)
    heat_df = fetch_dataframe(f"""
        SELECT ar.activity_id, ar.heart_rate, ar.temperature, a.start_time
        FROM activity_records ar
        JOIN activities a ON ar.activity_id = a.activity_id
        WHERE a.start_time > '{cutoff}'
    """)

    baselines = {}
    if not gait_df.empty:
        baselines["gait"] = gait_df["stride_length"].std()
    if not hr_drift_df.empty:
        baselines["hr_drift"] = hr_drift_df["avg_hr"].mean()
    if not heat_df.empty:
        corr = heat_df[["heart_rate", "temperature"]].corr().iloc[0, 1]
        baselines["heat"] = corr if not pd.isna(corr) else 0
    return baselines


def check_alerts(baselines):
    alerts = []

    # Gait drift
    if "gait" in baselines and baselines["gait"] > ALERT_THRESHOLD_GAIT:
        alerts.append(f"⚠️ Gait variability exceeded: {baselines['gait']:.2f}m")

    # HR drift
    if "hr_drift" in baselines and baselines["hr_drift"] > ALERT_THRESHOLD_HRDRIFT:
        alerts.append(f"⚠️ HR drift baseline exceeded: {baselines['hr_drift']:.1f} bpm")

    # Heat sensitivity
    if "heat" in baselines and baselines["heat"] > ALERT_THRESHOLD_HEAT:
        alerts.append(f"⚠️ Heat sensitivity detected (HR/temp corr = {baselines['heat']:.2f})")

    return alerts


# -------------------------
# Main
# -------------------------
def main():
    baselines = compute_baselines()
    alerts = check_alerts(baselines)

    if not alerts:
        print("✅ No alerts triggered today.")
        return

    message = "\n".join(alerts)
    print(message)

    # Send notifications
    send_slack_alert(message)
    send_email_alert("MS Health Alert", message)


if __name__ == "__main__":
    main()
