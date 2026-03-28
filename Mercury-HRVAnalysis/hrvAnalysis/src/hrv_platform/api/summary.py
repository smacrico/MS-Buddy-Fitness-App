from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from ..db import session_scope
from ..models import HRVAlert, HRVBaseline, HRVMeasurement
from ..repository import METRICS
from ..scoring import compute_ms_recovery_score

router = APIRouter()


@router.get("/api/summary")
def get_summary(
    source_name: str = Query(default="MyHRV_import"),
    days_back: int = Query(default=90, ge=1, le=3650),
) -> dict:
    try:
        with session_scope() as session:
            measurements = session.execute(
                select(HRVMeasurement)
                .where(HRVMeasurement.source_name == source_name)
                .order_by(desc(HRVMeasurement.measurement_date))
                .limit(days_back)
            ).scalars().all()

            baseline = session.execute(
                select(HRVBaseline)
                .where(HRVBaseline.source_name == source_name)
                .order_by(desc(HRVBaseline.analysis_date))
                .limit(1)
            ).scalar_one_or_none()

            alerts = session.execute(
                select(HRVAlert)
                .where(HRVAlert.source_name == source_name)
                .order_by(desc(HRVAlert.alert_date), desc(HRVAlert.id))
                .limit(50)
            ).scalars().all()

            if not measurements:
                return {
                    "data_points": 0,
                    "date_range": {"start": None, "end": None},
                    "current_values": {},
                    "recovery_scores": {"ms": 0.0},
                    "baselines": {},
                    "alerts": [],
                }

            latest = measurements[0]
            oldest = measurements[-1]

            current_values = {
                metric: float(getattr(latest, metric) or 0.0)
                for metric in METRICS
            }

            baselines: dict[str, float] = {}
            if baseline is not None:
                for metric in METRICS:
                    baselines[f"avg_{metric}"] = float(getattr(baseline, f"avg_{metric}", 0.0) or 0.0)
                    baselines[f"std_{metric}"] = float(getattr(baseline, f"std_{metric}", 0.0) or 0.0)

            return {
                "data_points": len(measurements),
                "date_range": {
                    "start": oldest.measurement_date.isoformat() if oldest.measurement_date else None,
                    "end": latest.measurement_date.isoformat() if latest.measurement_date else None,
                },
                "current_values": current_values,
                "recovery_scores": {"ms": compute_ms_recovery_score(current_values)},
                "baselines": baselines,
                "alerts": [
                    {
                        "alert_date": row.alert_date.isoformat() if row.alert_date else None,
                        "source_name": row.source_name,
                        "metric": row.metric,
                        "current_value": float(row.current_value or 0.0),
                        "baseline_value": float(row.baseline_value or 0.0),
                        "deviation_pct": float(row.deviation_pct or 0.0),
                        "alert_type": row.alert_type,
                        "alert_message": row.alert_message,
                    }
                    for row in alerts
                ],
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"summary failed: {type(exc).__name__}: {exc}")