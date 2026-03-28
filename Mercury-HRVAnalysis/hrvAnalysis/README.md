Here is a **complete, production-style `README.md`** documenting your current HRV platform, including all the changes we made (API, ingestion, Artemis, plots, dashboard, recalculation, etc.).

You can drop this directly into your repo root as `README.md`.

---

# HRV Platform

A production-ready HRV (Heart Rate Variability) analytics platform with:

* ✅ Artemis data ingestion
* ✅ Centralized database (`hrv_platform.db`)
* ✅ Regression-based trend analysis
* ✅ MS recovery scoring
* ✅ Baseline & anomaly detection
* ✅ Real-time API + WebSocket updates
* ✅ Interactive dashboard
* ✅ PNG export (legacy-compatible)

---

## 📦 Architecture Overview

```
Artemis DB / View
        ↓
ArtemisSource (normalize + validate)
        ↓
hrv_measurements (raw data)
        ↓
RecalculationService
        ↓
---------------------------------------
| hrv_baselines   (aggregates)         |
| hrv_trends      (regression)         |
| hrv_alerts      (threshold alerts)   |
| hrv_anomalies   (z-score detection)  |
---------------------------------------
        ↓
FastAPI API + Dashboard + Plots
```

---

## 🗄️ Database

SQLite database created locally:

```
hrv_platform.db
```

### Tables

| Table              | Description             |
| ------------------ | ----------------------- |
| `hrv_measurements` | Raw HRV data            |
| `hrv_baselines`    | Aggregated averages/std |
| `hrv_trends`       | Regression-based trends |
| `hrv_alerts`       | Threshold-based alerts  |
| `hrv_anomalies`    | Statistical anomalies   |

---

## 🚀 Getting Started

### 1. Initialize DB

```bash
python -m hrv_platform.cli init-db
```

---

### 2. Sync Data from Artemis

```bash
python -m hrv_platform.cli sync-artemis
```

This:

* Reads Artemis view
* Normalizes data
* Stores into `hrv_measurements`
* Triggers full recalculation

---

### 3. Start API Server

```bash
python -m hrv_platform.cli serve
```

Open:

```
http://127.0.0.1:8000/dashboard
```

---

## 📊 Dashboard

### URL

```
/dashboard
```

### Features

* Data points count
* MS recovery score
* Date range
* Alerts count
* Current values
* Baselines
* Trends table
* Anomalies table
* Live updates via WebSocket

---

## 🔌 API Endpoints

### Health

```
GET /api/health
```

---

### Summary

```
GET /api/summary?source_name=MyHRV_import
```

Returns:

```json
{
  "data_points": 90,
  "date_range": {...},
  "current_values": {...},
  "baselines": {...},
  "recovery_scores": {...},
  "alerts": [...]
}
```

---

### Trends

```
GET /api/trends?source_name=MyHRV_import
```

Returns regression-based trends:

* slope
* correlation (R)
* direction
* strength

---

### Anomalies

```
GET /api/anomalies?source_name=MyHRV_import
```

Returns z-score anomalies.

---

### Artemis Import

```
POST /api/import/artemis
```

Triggers sync + recalculation + live update.

---

### Artemis Preview

```
GET /api/import/artemis/preview
```

---

### WebSocket (Real-time)

```
/ws/live
```

Events:

* `artemis_synced`
* `summary_updated`
* `measurement_ingested`

---

## 📥 Data Source (Artemis)

Configured via:

```python
settings.artemis_db_path
settings.artemis_source_view
```

### Requirements

Artemis view must include:

* Date column
* Metrics:

  ```
  SD1, SD2, sdnn, rmssd, pNN50, VLF, LF, HF
  ```

### Validation

* No missing metrics
* No negative values
* Safe SQL identifiers only

---

## 🧠 Analytics Engine

### 1. Baselines

* Mean + std per metric
* Stored in `hrv_baselines`

---

### 2. Trends (Regression)

* Linear regression via `numpy.polyfit`
* Correlation strength classification:

|       |          |
| ----- | -------- |
| ≥ 0.7 | strong   |
| ≥ 0.3 | moderate |
| < 0.3 | weak     |

---

### 3. MS Recovery Score

Computed from HRV metrics:

```python
compute_ms_recovery_score(...)
```

Normalized to 0–100.

---

### 4. Anomaly Detection

* Z-score based
* Stored in `hrv_anomalies`

---

### 5. Alerts

* Threshold-based deviations from baseline

---

## 🔄 Recalculation Engine

Triggered automatically on sync:

```python
RecalculationService.recompute_all()
```

Updates:

* baselines
* trends
* alerts
* anomalies

---

## 📈 Plot Export (PNG)

### Command

```bash
python -m hrv_platform.cli export-plots
```

### Output Folder

```
C:\temp\logsFitnessApp\HRV_DashBoards
```

### Generated Files

* `HRV_TimeTrends_*.png`
* `HRV_Histograms_*.png`
* `HRV_BaselineProfile_*.png`
* `HRV_RadarChart_*.png`
* `HRV_MSScore_*.png`
* `HRV_TrendSummary_*.png`

---

## 🔁 Hybrid Mode (Recommended)

You now use:

| Purpose       | Source          |
| ------------- | --------------- |
| Raw ingestion | Artemis         |
| Analytics     | hrv_platform DB |
| Dashboard     | API             |
| PNG export    | hrv_platform DB |

👉 Artemis is only needed for ingestion, not analytics.

---

## 🧪 Debug Endpoints

```
/api/debug/source-names
/api/debug/measurements
```

Useful for:

* checking data presence
* verifying source_name

---

## ⚠️ Common Issues

### 1. Dashboard stuck on "connecting..."

➡ You opened HTML directly.

✅ Use:

```
http://127.0.0.1:8000/dashboard
```

---

### 2. Connection refused

➡ Server not running

```bash
python -m hrv_platform.cli serve
```

---

### 3. Empty results

➡ Wrong `source_name`

Check:

```
/api/debug/source-names
```

---

### 4. Artemis errors

* missing columns
* invalid view name
* wrong DB path

---

## 🧱 Project Structure

```
hrv_platform/
│
├── api/
│   ├── app.py
│   ├── routes.py
│   ├── summary.py
│   ├── trends.py
│   ├── anomalies.py
│
├── sources/
│   └── artemis.py
│
├── models.py
├── repository.py
├── recalc.py
├── scoring.py
├── trends.py
├── anomalies.py
├── plots.py
├── live.py
├── cli.py
├── db.py
├── config.py
│
└── templates/
    └── dashboard.html
```

---

## 🔮 Next Steps (Optional Enhancements)

* Dockerize the app
* Add authentication
* Add multi-user support
* Replace SQLite with Postgres
* Add scheduled ingestion (cron / Celery)
* Add ML-based anomaly detection

---

## ✅ Summary

You now have:

✔ Central HRV analytics platform
✔ Artemis ingestion pipeline
✔ Clean API + dashboard
✔ Real-time updates
✔ Regression trends
✔ Anomaly detection
✔ Exportable plots

---

If you want next step, I’d strongly recommend:

👉 **Docker + Postgres + scheduled ingestion (production mode)**









# HRV Platform

A production-ready refactor of the original HRV analytics prototype into a modular Python package with:

- validated configuration
- input validation with Pydantic
- normalized MS recovery score
- regression-based trends
- anomaly detection
- SQLite persistence via SQLAlchemy
- API-driven dashboard (FastAPI + HTML)
- real-time ingestion and live updates via WebSocket

## Package layout

```text
src/hrv_platform/
  api/             FastAPI app, routes, dashboard
  anomalies.py     z-score and EWMA anomaly detection
  cli.py           local entry point
  config.py        environment-driven settings
  db.py            SQLAlchemy engine/session helpers
  ingest.py        ingestion service
  models.py        ORM models
  repository.py    persistence layer
  schemas.py       API/data validation schemas
  scoring.py       MS score normalization
  services.py      analytics orchestration
  trends.py        regression-based trend analysis
```

## Run locally

```bash
pip install -e ".[dev]"
hrv-platform init-db
hrv-platform seed-demo
hrv-platform serve
```

Then open `http://127.0.0.1:8000/dashboard`.

## Environment variables

```bash
export HRV_DB_URL=sqlite:///./hrv_platform.db
export HRV_ALLOWED_SOURCE_VIEWS=myHRV_view
export HRV_ALERT_DEVIATION_THRESHOLD=0.25
export HRV_INGESTION_ANOMALY_ZSCORE=2.5
```

## Key refactor changes

### 1) MS score normalization fixed

The original implementation clamped a weighted raw sum to `[0, 100]`. This package computes a bounded normalized score by comparing the weighted metric ratio against an expected reference profile:

- each metric is scaled to a comparable unit
- weighted ratios are summed
- score is normalized around a baseline ratio of `1.0`
- score is clipped to `[0, 100]`

### 2) Regression-based trend analysis

Linear regression is used instead of raw day-index correlation alone. The API returns:

- slope
- r-value
- p-value
- trend direction
- trend strength

### 3) API-driven dashboard

The frontend pulls JSON from `/api/summary`, `/api/series`, `/api/trends`, and `/api/anomalies` instead of reading PNG files.

### 4) Real-time ingestion

- `POST /api/ingest`
- `POST /api/ingest/batch`
- `WS /ws/live`

Clients connected to the WebSocket receive new measurements, summary refreshes, and anomaly events.

## Notes

This is still a health analytics engineering starter, not a medical device. Clinical thresholds and domain assumptions should be reviewed by qualified professionals before use in care workflows.


## Artemis source integration

This package can sync HRV measurements directly from the Artemis SQLite database.

Environment variables:
- `HRV_ARTEMIS_DB_PATH` defaults to `c:/smakrykoDBs/Artemis.db`
- `HRV_ARTEMIS_VIEW` is controlled by `HRV_ARTEMIS_SOURCE_VIEW` and defaults to `myHRV_view`
- `HRV_ALLOWED_SOURCE_VIEWS` should include `myHRV_view`

CLI:
- `python -m hrv_platform.cli preview-artemis`
- `python -m hrv_platform.cli sync-artemis`
- `python -m hrv_platform.cli watch-artemis`

API:
- `GET /api/import/artemis/preview`
- `POST /api/import/artemis`

Expected Artemis columns:
- `date`
- `SD1`, `SD2`, `sdnn`, `rmssd`, `pNN50`, `VLF`, `LF`, `HF`
- optional `name` column for source naming
