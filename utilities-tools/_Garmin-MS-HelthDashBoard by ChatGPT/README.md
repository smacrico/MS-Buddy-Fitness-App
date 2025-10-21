# Garmin MS Health Dashboard

Interactive Streamlit dashboard for Garmin Forerunner 245 activities with automated alert monitoring and MS-specific health tracking.

## Features

- **Activity Metrics**
  - Weekly training load (distance, duration, calories) by sport.
  - Cardio efficiency (pace vs HR).
  - Pacing consistency (lap pace variance).
  - Heart rate drift analysis.
  - Gait stability and stride range.
  - Fatigue indicators (Ground Contact Time drift).
  
- **Automated Alerts**
  - MS-specific flags: gait drift, HR drift, heat sensitivity.
  - Configurable thresholds (automated baseline calculation).
  - Slack/Email notifications for triggered alerts.
  
- **Alert History & Management**
  - Table view of alerts from `alert_logs` SQLite table.
  - **Acknowledge button** for each alert (updates `acknowledged`, `ack_by`, `ack_time`).
  - Interactive filtering by metric and date range.
  - Visualizations:
    - Alert counts by metric.
    - Daily alert counts over time.
    
- **Logging**
  - Full persistence in `alert_logs` table.
  - Tracks acknowledged alerts with user and timestamp.

## Requirements

- Python 3.9+
- SQLite
- Streamlit
- Pandas
- Matplotlib
- Slack SDK or smtplib for email alerts (optional)

See `requirements.txt` for full package list.

## Installation

```bash
git clone <your-github-repo-url>
cd garmin-ms-dashboard
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
