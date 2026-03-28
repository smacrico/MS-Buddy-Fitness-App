from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc, select

from ..db import session_scope
from ..models import HRVAnomaly

router = APIRouter()


@router.get("/api/anomalies")
def get_anomalies(
    source_name: str = Query(default="MyHRV_import"),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    try:
        with session_scope() as session:
            rows = session.execute(
                select(HRVAnomaly)
                .where(HRVAnomaly.source_name == source_name)
                .order_by(desc(HRVAnomaly.measurement_date), desc(HRVAnomaly.id))
                .limit(limit)
            ).scalars().all()

            return [
                {
                    "detected_at": row.detected_at.isoformat() if row.detected_at else None,
                    "measurement_date": row.measurement_date.isoformat() if row.measurement_date else None,
                    "source_name": row.source_name,
                    "metric": row.metric,
                    "value": float(row.value or 0.0),
                    "baseline_mean": float(row.baseline_mean or 0.0),
                    "baseline_std": float(row.baseline_std or 0.0),
                    "z_score": float(row.z_score or 0.0),
                    "detector": row.detector,
                    "message": row.message,
                }
                for row in rows
            ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"anomalies failed: {type(exc).__name__}: {exc}")