from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import asc, desc, select

from ..db import session_scope
from ..models import HRVTrend

router = APIRouter()


@router.get("/api/trends")
def get_trends(
    source_name: str = Query(default="MyHRV_import"),
    latest_only: bool = Query(default=True),
) -> list[dict]:
    try:
        with session_scope() as session:
            if latest_only:
                latest_analysis_date = session.execute(
                    select(HRVTrend.analysis_date)
                    .where(HRVTrend.source_name == source_name)
                    .order_by(desc(HRVTrend.analysis_date))
                    .limit(1)
                ).scalar_one_or_none()

                if latest_analysis_date is None:
                    return []

                rows = session.execute(
                    select(HRVTrend)
                    .where(
                        HRVTrend.source_name == source_name,
                        HRVTrend.analysis_date == latest_analysis_date,
                    )
                    .order_by(asc(HRVTrend.metric))
                ).scalars().all()
            else:
                rows = session.execute(
                    select(HRVTrend)
                    .where(HRVTrend.source_name == source_name)
                    .order_by(desc(HRVTrend.analysis_date), asc(HRVTrend.metric))
                ).scalars().all()

            return [
                {
                    "analysis_date": row.analysis_date.isoformat() if row.analysis_date else None,
                    "source_name": row.source_name,
                    "metric": row.metric,
                    "slope": float(row.slope or 0.0),
                    "r_value": float(row.r_value or 0.0),
                    "p_value": float(row.p_value or 1.0),
                    "trend_direction": row.trend_direction,
                    "trend_strength": row.trend_strength,
                    "mean": float(row.mean or 0.0),
                    "std": float(row.std or 0.0),
                    "min": float(row.min or 0.0),
                    "max": float(row.max or 0.0),
                    "latest_ms_score": float(row.latest_ms_score or 0.0),
                }
                for row in rows
            ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trends failed: {type(exc).__name__}: {exc}")