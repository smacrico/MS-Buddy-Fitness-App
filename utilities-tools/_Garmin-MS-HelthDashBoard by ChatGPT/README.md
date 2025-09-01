# Garmin Activities Metrics & Dashboard

This package contains:
- `garmin_metrics.py` — computes metrics from `garmin_activities.db`, exports CSVs to `outputs/` and plots to `outputs/plots/`.
- `streamlit_app.py` — an interactive Streamlit dashboard for exploration.
- `requirements.txt` — minimal Python dependencies.

## Quick Start

1. Ensure you have `garmin_activities.db` (from your GarminDB workflow).
2. (Optional) Set environment variables:
   - `GARMIN_ACTIVITIES_DB` -> path to `garmin_activities.db` (default: `garmin_activities.db`)
   - `GARMIN_OUTPUT_DIR` -> output directory (default: `./outputs`)

### Run the batch metrics + plots
```bash
python garmin_metrics.py



### Instructions

# Launch the interactive dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py


## Notes & tips

The scripts assume tables exist in garmin_activities.db with common columns (activities, activity_records, activity_laps, steps_activities). GarminDB versions or device differences may omit some fields; the scripts handle missing data gracefully but some charts may be empty.

HR drift and the timestamp math expect activity_records.timestamp to be numeric (UNIX epoch) or other numeric values where subtraction returns duration. If your timestamps are text (ISO), convert them to epoch before running or adapt the SQL accordingly.

To make the pipeline robust in production:

Add a startup check that PRAGMA table_info(table) returns required columns; log and notify missing columns.

Persist a unified metrics.parquet for downstream tools (Power BI / Grafana).

If you want, I can:

Add a Dockerfile for one-command deployment.

Add a scheduled runner (cron) to refresh metrics automatically.

Add email/Slack alerting for threshold breaches (e.g., sudden high HR drift).#

