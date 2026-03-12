# Garmin MS Health Dashboard - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Components](#components)
4. [Data Flow](#data-flow)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [Docker Deployment](#docker-deployment)
9. [Alert System](#alert-system)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

The **Garmin MS Health Dashboard** is a comprehensive health monitoring system designed to track, analyze, and alert on fitness metrics from Garmin wearable devices. It's particularly tailored for Multiple Sclerosis (MS) patients to detect early warning signs of health deterioration through continuous activity monitoring.

### Key Features
- ✅ **Interactive Web Dashboard** - Streamlit-based UI for data visualization
- ✅ **Advanced Metrics Analysis** - Track gait, HR drift, fatigue indicators
- ✅ **Statistical Alerting System** - Z-score based anomaly detection
- ✅ **Multi-Channel Notifications** - Slack and Email alerts
- ✅ **Alert Management** - Acknowledgement system with history tracking
- ✅ **Docker Deployment** - Containerized for easy deployment
- ✅ **Configurable Thresholds** - Environment-based configuration

### Health Metrics Monitored

1. **Gait Stability** - Stride length variance to detect mobility issues
2. **Heart Rate Drift** - HR changes during activities (early fatigue indicator)
3. **Ground Contact Time (GCT)** - Running biomechanics and fatigue
4. **Heat Sensitivity** - Correlation between temperature and HR drift
5. **Training Load** - Weekly distance, duration, and calorie expenditure
6. **Cardio Efficiency** - Pace vs. heart rate analysis
7. **Pacing Consistency** - Lap-to-lap pace variance

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Garmin Connect API                          │
│              (External Data Source)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              garmin_activities.db                            │
│         (SQLite Database - Central Data Store)              │
│  Tables: activities, activity_records, steps_activities,    │
│          activity_laps, alert_logs                           │
└────────┬───────────────────────────────────────────┬────────┘
         │                                            │
         ▼                                            ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  garmin_metrics.py   │                  │   streamlit_app.py   │
│  ─────────────────   │                  │   ───────────────    │
│  • Query database    │                  │  • Web Dashboard     │
│  • Compute metrics   │                  │  • Data Viz          │
│  • Generate CSVs     │                  │  • Alert History UI  │
│  • Create plots      │                  │  • User Interaction  │
│  • Trigger alerts.py │                  │                      │
└──────────┬───────────┘                  └──────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                      alerts.py                               │
│  ─────────────────────────────────────────────────────       │
│  • Load CSV metrics                                          │
│  • Compute statistical baselines (median/MAD, mean/std)     │
│  • Detect anomalies (z-score >= k)                          │
│  • Check cooldown period                                     │
│  • Send notifications                                        │
│  • Log to database (alert_logs table)                       │
│  • Persist state (alerts_state.json)                        │
└──────┬──────────────────────────────┬────────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│  Slack Webhook   │          │  Email (SMTP)    │
│  ───────────     │          │  ──────────      │
│  • POST alerts   │          │  • Send email    │
│  • JSON payload  │          │  • SMTP server   │
└──────────────────┘          └──────────────────┘
```

### Additional Components

- **alerts_module.py** - Simplified alerting alternative (not integrated into main flow)
- **Docker containers** - Streamlit app and metrics processor services
- **outputs/** - Generated CSV files and plots directory

---

## Components

### 1. garmin_metrics.py
**Purpose**: Core analytics engine that processes Garmin activity data

**Functionality**:
- Connects to `garmin_activities.db` SQLite database
- Executes 7 analytical SQL queries covering different health dimensions
- Generates CSV output files for each metric type
- Creates matplotlib visualizations
- Triggers the alerting system via `alerts.alert_on_metrics()`
- Saves results to configurable output directory

**Key Queries**:
```python
QUERIES = {
    "weekly_training_load": "...",     # Distance, duration, calories by sport/week
    "cardio_efficiency": "...",         # Pace vs. heart rate correlation
    "pacing_consistency": "...",        # Lap pace variance
    "heart_rate_drift": "...",          # HR change early vs late in activity
    "gait_stability": "...",            # Stride length range
    "fatigue_indicators": "...",        # GCT drift
    # Additional: activity_temps for heat sensitivity
}
```

**Configuration**:
```python
DB_PATH = os.environ.get("GARMIN_ACTIVITIES_DB", r"C:\Users\XP222SP\myHealthData\DBs\garmin_activities.db")
OUTPUT_DIR = os.environ.get("GARMIN_OUTPUT_DIR", r"c:\temp")
```

### 2. streamlit_app.py
**Purpose**: Interactive web dashboard for visualization and alert management

**Features**:
- **Two-Tab Interface**:
  1. **Dashboard Tab**: 
     - Weekly training load charts (distance/duration by sport)
     - Cardio efficiency scatter plot (HR vs pace)
     - Pacing consistency analysis
     - HR drift visualization
     - Gait stability plots
     - GCT fatigue indicators
  2. **Alert History Tab**:
     - Alert log table with filtering (metric, date range)
     - Acknowledge button for each alert
     - Alert count charts by metric
     - Time-series alert visualization

- **Filters**:
  - Sport selection (multi-select)
  - Date range picker
  - Metric type filter

- **Database Integration**:
  - Direct SQLite queries
  - Real-time alert acknowledgement updates
  - User/timestamp tracking for acknowledgements

**UI Components**:
```python
st.set_page_config(page_title="Garmin Activities Dashboard", layout="wide")
db_path = st.text_input("Path to garmin_activities.db", ...)
hr_zones = st.text_area("HR Zones (comma-separated)", ...)
```

### 3. alerts.py
**Purpose**: Sophisticated statistical alerting system with multi-channel notifications

**Core Algorithm**:
1. **Baseline Computation**:
   - Uses last N samples (default 30) for baseline
   - ≥8 samples: Median + MAD (Median Absolute Deviation) - robust to outliers
   - <8 samples: Mean + Standard Deviation
   
2. **Anomaly Detection**:
   ```python
   z_score = (latest_value - baseline_center) / baseline_scale
   if z_score >= K (default 2.0):
       trigger_alert()
   ```

3. **Cooldown Management**:
   - Prevents alert spam
   - Default 24-hour cooldown per metric
   - Tracks last alert time in `alerts_state.json`

4. **Notification Channels**:
   - **Slack**: Webhook POST with formatted message
   - **Email**: SMTP with TLS encryption
   - Logs channel success/failure

5. **Persistence**:
   - **JSON state file** (`alerts_state.json`): cooldown tracking, history
   - **SQLite `alert_logs` table**: permanent alert record with metadata

**Monitored Metrics**:
- `gait_stride_range` - from gait_stability.csv
- `heart_rate_drift` - from heart_rate_drift.csv
- `gct_drift` - from fatigue_indicators.csv
- `heat_sensitivity` - computed slope (hr_drift vs avg_temp)

**Configuration Variables**:
```python
LOOKBACK = int(os.environ.get("ALERT_LOOKBACK", "30"))
K = float(os.environ.get("ALERT_K", "2.0"))
COOLDOWN_HOURS = int(os.environ.get("ALERT_COOLDOWN_HOURS", "24"))
MIN_SAMPLES = int(os.environ.get("ALERT_MIN_SAMPLES", "8"))
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")
SMTP_HOST = os.environ.get("ALERT_SMTP_HOST")
ALERT_DB_PATH = os.environ.get("ALERT_DB_PATH") or os.environ.get("GARMIN_ACTIVITIES_DB")
```

### 4. alerts_module.py
**Purpose**: Simplified alternative alerting system (standalone, not integrated)

**Differences from alerts.py**:
- Queries database directly (no CSV intermediaries)
- Fixed threshold comparison (no z-score statistics)
- No cooldown mechanism
- No state persistence
- Simpler baseline computation (std, mean, correlation)

**Use Case**: Quick prototype or simple threshold-based alerts without statistical sophistication.

### 5. Docker Configuration

**DockerFile**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY garmin_metrics.py streamlit_app.py alerts.py requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "/app/streamlit_app.py", 
            "--server.headless", "true", 
            "--server.port", "8501", 
            "--server.address", "0.0.0.0"]
```

**Docker-compose.yaml**:
```yaml
version: "3.8"
services:
  streamlit-app:
    build: .
    ports: ["8501:8501"]
    environment:
      - GARMIN_ACTIVITIES_DB=/data/garmin_activities.db
      - GARMIN_OUTPUT_DIR=/data/outputs
      # Alert configs...
    volumes:
      - ./data:/data   # Local data folder mount
  
  metrics-processor:
    build: .
    command: ["python", "/app/garmin_metrics.py"]
    environment:
      - GARMIN_ACTIVITIES_DB=/data/garmin_activities.db
    volumes:
      - ./data:/data
```

---

## Data Flow

### Detailed Execution Flow

1. **Data Collection** (External - not in this codebase):
   - Garmin device syncs to Garmin Connect
   - Separate tool (e.g., GarminDB) exports data to `garmin_activities.db`

2. **Metrics Processing** (`garmin_metrics.py`):
   ```
   Connect to DB → Execute Queries → Generate DataFrames → 
   Save CSVs → Create Plots → Trigger Alerts
   ```

3. **Alert Analysis** (`alerts.py`):
   ```
   Load CSVs → Compute Baselines → Calculate Z-scores → 
   Check Thresholds → Verify Cooldown → Send Notifications → 
   Log to DB → Update State
   ```

4. **Visualization** (`streamlit_app.py`):
   ```
   User Opens Dashboard → Query DB → Render Charts → 
   Display Alert History → Handle Acknowledgements
   ```

### Database Schema (Key Tables)

**activities**:
```sql
activity_id, sport, start_time, distance, duration, 
avg_hr, calories, ...
```

**activity_records**:
```sql
activity_id, timestamp, heart_rate, temperature, 
cadence, ...
```

**steps_activities**:
```sql
activity_id, stride_length, ground_contact_time, ...
```

**activity_laps**:
```sql
activity_id, lap_distance, lap_elapsed_time, ...
```

**alert_logs** (created by alerts.py):
```sql
id, timestamp, metric, value, threshold, message, 
channel, status, acknowledged, ack_by, ack_time
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- SQLite database from Garmin device (`garmin_activities.db`)
- Optional: Docker & Docker Compose

### Local Installation

1. **Clone/Navigate to Directory**:
   ```bash
   cd "c:\smakryko\MS-Buddy-Fitness-App\_Garmin-MS-HelthDashBoard by ChatGPT"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - streamlit >= 1.25.0
   - pandas >= 2.1.0
   - numpy >= 1.26.0
   - matplotlib >= 3.8.0
   - pysqlite3-binary >= 0.4.7
   - slack_sdk >= 3.26.0
   - email-validator >= 2.1.0
   - python-dateutil >= 2.9.0
   - streamlit-aggrid >= 0.5.0
   - requests (for Slack webhooks)

3. **Configure Database Path**:
   Edit `garmin_metrics.py` or set environment variable:
   ```python
   DB_PATH = r"C:\Users\XP222SP\myHealthData\DBs\garmin_activities.db"
   OUTPUT_DIR = r"c:\temp"
   ```

4. **Set Up Notifications** (Optional):
   ```bash
   # Slack
   set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   
   # Email
   set ALERT_SMTP_HOST=smtp.gmail.com
   set ALERT_SMTP_PORT=587
   set ALERT_SMTP_USER=your-email@gmail.com
   set ALERT_SMTP_PASS=your-app-password
   set ALERT_FROM_EMAIL=your-email@gmail.com
   set ALERT_TO_EMAIL=recipient@example.com
   ```

### Docker Installation

1. **Prepare Data Directory**:
   ```bash
   mkdir data
   # Copy garmin_activities.db to ./data/
   ```

2. **Configure docker-compose.yaml**:
   - Edit environment variables for your setup
   - Uncomment notification settings if needed

3. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

4. **Access Dashboard**:
   - Open browser: `http://localhost:8501`

---

## Configuration

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Database** |
| `GARMIN_ACTIVITIES_DB` | `garmin_activities.db` | Path to SQLite database |
| `GARMIN_OUTPUT_DIR` | `outputs` (local) or `c:\temp` | CSV/plot output directory |
| **Alert Thresholds** |
| `ALERT_LOOKBACK` | `30` | Number of samples for baseline |
| `ALERT_K` | `2.0` | Z-score threshold (std deviations) |
| `ALERT_COOLDOWN_HOURS` | `24` | Hours between repeat alerts |
| `ALERT_MIN_SAMPLES` | `8` | Min samples for median/MAD method |
| `ALERT_HEAT_SLOPE` | `0.3` | Heat sensitivity slope threshold |
| **Slack Notifications** |
| `SLACK_WEBHOOK_URL` | None | Slack incoming webhook URL |
| **Email Notifications** |
| `ALERT_SMTP_HOST` | None | SMTP server hostname |
| `ALERT_SMTP_PORT` | `587` | SMTP server port (TLS) |
| `ALERT_SMTP_USER` | None | SMTP username |
| `ALERT_SMTP_PASS` | None | SMTP password |
| `ALERT_FROM_EMAIL` | None | Sender email address |
| `ALERT_TO_EMAIL` | None | Recipient email (comma-separated) |
| **Persistence** |
| `ALERT_STATE_PATH` | `alerts_state.json` | Alert state file location |
| `ALERT_DB_PATH` | Uses `GARMIN_ACTIVITIES_DB` | Alert logs database path |

### Customizing Thresholds

**Adjust sensitivity** (lower K = more sensitive):
```bash
set ALERT_K=1.5  # More alerts, detects smaller deviations
set ALERT_K=3.0  # Fewer alerts, only major deviations
```

**Change baseline window**:
```bash
set ALERT_LOOKBACK=60  # Use last 60 samples for baseline
```

**Modify cooldown**:
```bash
set ALERT_COOLDOWN_HOURS=12  # Alert every 12 hours if threshold breached
```

---

## Usage Guide

### Running Locally

**1. Generate Metrics and Check Alerts**:
```bash
python garmin_metrics.py
```
- Reads from database
- Creates CSVs in `OUTPUT_DIR`
- Generates plots in `OUTPUT_DIR/plots/`
- Triggers alert system
- Output: Console logs + CSV files

**2. Launch Dashboard**:
```bash
streamlit run streamlit_app.py
```
- Opens browser automatically
- Dashboard URL: `http://localhost:8501`

**3. Manual Alert Check**:
```python
python -c "from alerts import alert_on_metrics; alert_on_metrics()"
```

### Using the Dashboard

**Dashboard Tab**:
1. Enter database path or use default
2. Configure HR zones (optional)
3. Use sidebar to filter by sport
4. Scroll through metric visualizations:
   - Weekly training loads
   - Cardio efficiency
   - Pacing consistency
   - HR drift analysis
   - Gait stability
   - Fatigue indicators (GCT)

**Alert History Tab**:
1. View all triggered alerts
2. Filter by:
   - Metric type (multi-select)
   - Date range (date picker)
3. Acknowledge alerts:
   - Click "Acknowledge" button next to alert
   - Records your username and timestamp
   - Alert marked as acknowledged
4. Analyze patterns:
   - View alert counts by metric
   - Time-series chart of daily alerts

### Understanding Alerts

**Alert Message Format**:
```
ALERT: heart_rate_drift breached
- value: 15.3
- baseline (method=median_mad): center=8.2, scale=3.1
- z: 2.29
HR drift late vs early in activity.
```

**Interpreting Z-Score**:
- `z < 2.0`: Within normal range
- `2.0 ≤ z < 3.0`: Mild deviation (default alert)
- `z ≥ 3.0`: Significant deviation
- Negative z: Below baseline (less common for alerts)

**Heat Sensitivity Alert**:
```
ALERT: heat_sensitivity breached
- slope: 0.45
- threshold: 0.3
Heat sensitivity slope (hr_drift vs temp) = 0.45 >= threshold 0.3
```
- Indicates HR increases more than expected with temperature
- Potential heat intolerance - MS symptom concern

---

## Docker Deployment

### Production Deployment Steps

1. **Configure Environment**:
   Create `.env` file:
   ```env
   GARMIN_ACTIVITIES_DB=/data/garmin_activities.db
   GARMIN_OUTPUT_DIR=/data/outputs
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
   ALERT_SMTP_HOST=smtp.gmail.com
   ALERT_SMTP_PORT=587
   ALERT_SMTP_USER=alerts@yourdomain.com
   ALERT_SMTP_PASS=your-password
   ALERT_FROM_EMAIL=alerts@yourdomain.com
   ALERT_TO_EMAIL=user@example.com
   ```

2. **Update docker-compose.yaml**:
   ```yaml
   services:
     streamlit-app:
       env_file: .env
       restart: always
   ```

3. **Deploy**:
   ```bash
   docker-compose up -d  # Detached mode
   ```

4. **Monitor Logs**:
   ```bash
   docker-compose logs -f streamlit-app
   docker-compose logs -f metrics-processor
   ```

5. **Schedule Metrics Processing**:
   Add to cron or Windows Task Scheduler:
   ```bash
   0 */6 * * * docker-compose exec metrics-processor python /app/garmin_metrics.py
   ```

### Container Management

**Stop services**:
```bash
docker-compose down
```

**Restart services**:
```bash
docker-compose restart
```

**Update code**:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**View running containers**:
```bash
docker-compose ps
```

---

## Alert System

### Alert Lifecycle

1. **Detection**: `garmin_metrics.py` generates CSVs → `alerts.py` analyzes
2. **Baseline Computation**: Statistical baseline from recent history
3. **Threshold Check**: Compare latest value to baseline (z-score)
4. **Cooldown Verification**: Check if alert sent recently
5. **Notification**: Send to Slack/Email if conditions met
6. **Logging**: Write to database and JSON state file
7. **Dashboard Display**: View in Alert History tab
8. **Acknowledgement**: User marks alert as reviewed

### alert_logs Table Schema

```sql
CREATE TABLE alert_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO format
    metric TEXT NOT NULL,              -- e.g., "heart_rate_drift"
    value REAL,                        -- Actual metric value
    threshold REAL,                    -- Baseline scale/threshold
    message TEXT,                      -- Full alert text
    channel TEXT,                      -- "slack,email" or "none"
    status TEXT,                       -- "sent" or "failed"
    acknowledged INTEGER DEFAULT 0,    -- 0 or 1
    ack_by TEXT,                       -- Username who acknowledged
    ack_time TEXT                      -- ISO timestamp of acknowledgement
);
```

### alerts_state.json Structure

```json
{
  "last_alert": {
    "gait_stride_range": "2026-03-10T14:23:45.123456",
    "heart_rate_drift": "2026-03-11T08:15:30.789012"
  },
  "history": {
    "heart_rate_drift": [
      {
        "time": "2026-03-11T08:15:30.789012",
        "value": 15.3,
        "z": 2.29,
        "sent_slack": true,
        "sent_email": true,
        "db_logged": true
      }
    ]
  }
}
```

### Notification Best Practices

**Slack Setup**:
1. Create Slack App: https://api.slack.com/apps
2. Enable Incoming Webhooks
3. Add webhook to workspace
4. Copy webhook URL to `SLACK_WEBHOOK_URL`

**Email Setup (Gmail)**:
1. Enable 2FA on Gmail account
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Use app password (not account password) in `ALERT_SMTP_PASS`
4. Enable "Less secure app access" if using regular password (not recommended)

**Testing Notifications**:
```python
# Test Slack
import requests
requests.post(
    os.environ["SLACK_WEBHOOK_URL"], 
    json={"text": "Test alert from Garmin Dashboard"}
)

# Test Email
from alerts import send_email
send_email("Test Alert", "This is a test message.")
```

---

## Troubleshooting

### Common Issues

**1. Database Not Found**
```
Error: Database not found at garmin_activities.db
```
**Solution**: 
- Verify `DB_PATH` environment variable
- Check file path is correct
- Ensure database file exists

**2. Permission Denied (Windows)**
```
PermissionError: [Errno 13] Permission denied: 'c:\\temp\\outputs'
```
**Solution**:
- Run as Administrator
- Or change `OUTPUT_DIR` to user-writable location

**3. No Alerts Triggered**
```
[INFO] gait_stride_range within baseline (z=1.23)
```
**Solution**:
- Expected behavior if metrics are normal
- Lower `ALERT_K` to increase sensitivity
- Check if enough samples exist (need ≥2)

**4. Slack Webhook Fails**
```
[ERROR] Slack send failed: 404 Client Error
```
**Solution**:
- Verify webhook URL is correct
- Check Slack app is installed to workspace
- Test webhook URL with curl:
  ```bash
  curl -X POST -H 'Content-type: application/json' \
       --data '{"text":"Test"}' \
       YOUR_WEBHOOK_URL
  ```

**5. Email Authentication Error**
```
[ERROR] Email send failed: (535, b'5.7.8 Username and Password not accepted')
```
**Solution**:
- Use app password, not account password (Gmail)
- Enable SMTP access in email provider settings
- Check username is full email address

**6. Streamlit Won't Start**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**:
```bash
pip install -r requirements.txt
# Or
pip install streamlit
```

**7. CSV Files Not Found**
```
[WARN] Could not read outputs/gait_stability.csv
```
**Solution**:
- Run `garmin_metrics.py` first to generate CSVs
- Check `GARMIN_OUTPUT_DIR` path is correct
- Ensure write permissions on output directory

**8. Alert Cooldown Too Long**
```
[INFO] heart_rate_drift breached but cooldown active; skipping.
```
**Solution**:
- Reduce `ALERT_COOLDOWN_HOURS`:
  ```bash
  set ALERT_COOLDOWN_HOURS=1
  ```
- Or delete `alerts_state.json` to reset cooldowns

**9. Docker Container Exits Immediately**
```bash
docker-compose ps
# Shows container with status "Exit 1"
```
**Solution**:
```bash
docker-compose logs streamlit-app
# Check error message, usually missing database or env vars
```

**10. Database Locked Error**
```
sqlite3.OperationalError: database is locked
```
**Solution**:
- Close other connections to database
- In Docker: ensure only one metrics-processor container runs
- Add timeout:
  ```python
  conn = sqlite3.connect(db_path, timeout=30)
  ```

### Debug Mode

**Enable verbose logging**:
```python
# Add to top of garmin_metrics.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check alert state**:
```python
from alerts import load_state
state = load_state()
print(state)
```

**Manually trigger alert**:
```python
from alerts import alert_on_metrics
alerts = alert_on_metrics(metrics_dir=r"c:\temp", lookback=30, k=1.5)
print(alerts)
```

---

## Advanced Topics

### Custom Metrics

To add new metrics:

1. **Add SQL query** to `garmin_metrics.py`:
   ```python
   "my_custom_metric": '''
       SELECT activity_id, custom_calculation
       FROM my_table
       WHERE conditions
   '''
   ```

2. **Generate CSV**:
   ```python
   custom = run_query(conn, QUERIES["my_custom_metric"])
   custom.to_csv(os.path.join(OUTPUT_DIR, "my_custom_metric.csv"), index=False)
   ```

3. **Add alert check** in `alerts.py`:
   ```python
   custom = load_csv("my_custom_metric")
   check_metric(custom, "my_value", "my_custom_metric", extra_text="Custom alert description.")
   ```

4. **Add visualization** to `streamlit_app.py`:
   ```python
   st.subheader("My Custom Metric")
   custom = run_query(conn, QUERIES["my_custom_metric"])
   # Create plot...
   ```

### Baseline Algorithms

**Current: Robust Statistics**
- Median + MAD: Resistant to outliers
- Mean + Std: For small samples

**Alternative: Machine Learning**
For more advanced anomaly detection:
```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.1)
model.fit(historical_data)
anomaly = model.predict(latest_value)
# -1 = anomaly, 1 = normal
```

### Integration with External Systems

**Webhook for other services**:
```python
def send_custom_webhook(metric, value):
    requests.post(
        "https://your-api.com/alerts",
        json={"metric": metric, "value": value, "source": "garmin_dashboard"}
    )
```

**Export to Azure CosmosDB** (for cloud storage):
```python
from azure.cosmos import CosmosClient
client = CosmosClient(url, key)
database = client.get_database_client("health_data")
container = database.get_container_client("alerts")
container.create_item(alert_dict)
```

---

## Maintenance

### Regular Tasks

**Weekly**:
- Review alert history in dashboard
- Acknowledge all reviewed alerts
- Check for false positives (adjust thresholds if needed)

**Monthly**:
- Archive old alert logs:
  ```sql
  DELETE FROM alert_logs WHERE timestamp < date('now', '-90 days');
  ```
- Backup database:
  ```bash
  sqlite3 garmin_activities.db ".backup 'backup_YYYY-MM-DD.db'"
  ```
- Review alert patterns and adjust `ALERT_K` if needed

**Quarterly**:
- Update dependencies:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- Rebuild Docker containers:
  ```bash
  docker-compose build --no-cache
  ```

### Performance Optimization

**For large databases**:
- Add indexes:
  ```sql
  CREATE INDEX idx_activities_start_time ON activities(start_time);
  CREATE INDEX idx_activity_records_timestamp ON activity_records(activity_id, timestamp);
  ```
- Limit query ranges in SQL (add `WHERE start_time > date('now', '-365 days')`)
- Increase Docker memory limits if needed

---

## Security Considerations

1. **Credentials**: Never commit API keys, passwords, or webhook URLs to version control
2. **Environment Files**: Add `.env` to `.gitignore`
3. **Database Access**: Restrict file permissions on `garmin_activities.db`
4. **Network**: Use HTTPS for webhook URLs
5. **Email**: Use app passwords, not account passwords
6. **Docker**: Don't expose ports to public internet without authentication

---

## Project Structure Summary

```
_Garmin-MS-HelthDashBoard by ChatGPT/
│
├── garmin_metrics.py          # Core analytics engine
├── alerts.py                  # Statistical alerting system
├── alerts_module.py           # Simplified alerting (standalone)
├── streamlit_app.py           # Web dashboard UI
│
├── requirements.txt           # Python dependencies
├── DockerFile                 # Container image definition
├── Docker-compose.yaml        # Multi-container orchestration
│
├── README.md                  # Project overview
├── DOCUMENTATION.md           # This file (complete docs)
│
├── data/                      # Docker volume mount
│   ├── garmin_activities.db   # SQLite database (user-provided)
│   └── outputs/               # Generated CSVs and plots
│
├── alerts_state.json          # Alert cooldown state (auto-generated)
└── __pycache__/               # Python bytecode cache
```

---

## Contributing

To extend or modify this system:

1. **Fork/branch** the repository
2. **Test changes** locally before Docker deployment
3. **Document** new environment variables in this file
4. **Maintain compatibility** with existing database schema
5. **Update** both README.md and DOCUMENTATION.md

---

## Support & Contact

- **Issues**: Check Troubleshooting section above
- **Feature Requests**: Document in project issues
- **Health Concerns**: Always consult healthcare provider for medical decisions

---

## License

[Add your license information here]

---

## Acknowledgements

- Built for MS health monitoring
- Uses Garmin Connect API data (via external export tools)
- Streamlit for dashboard framework
- Matplotlib for visualizations
- SQLite for data storage

---

**Last Updated**: March 11, 2026  
**Version**: 1.0  
**Maintained By**: [Your Name/Team]
