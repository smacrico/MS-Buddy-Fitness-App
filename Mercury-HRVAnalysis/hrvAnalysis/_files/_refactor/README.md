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
