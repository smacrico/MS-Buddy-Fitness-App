from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import select, desc, func
from sqlalchemy.orm import Session

from .models import HRVMeasurement, HRVBaseline, HRVAlert


METRICS = ["SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"]


@dataclass
class HRVRepository:
    session: Session

    def upsert_measurement(self, payload: dict[str, Any]) -> HRVMeasurement:
        measurement_date = payload["measurement_date"]
        source_name = payload["source_name"]

        existing = self.session.execute(
            select(HRVMeasurement).where(
                HRVMeasurement.measurement_date == measurement_date,
                HRVMeasurement.source_name == source_name,
            )
        ).scalar_one_or_none()

        if existing is None:
            existing = HRVMeasurement(
                measurement_date=measurement_date,
                source_name=source_name,
            )
            self.session.add(existing)

        for metric in METRICS:
            setattr(existing, metric, float(payload.get(metric, 0.0) or 0.0))

        self.session.flush()
        return existing

    def get_recent_measurements(self, source_name: str, limit: int = 90) -> list[HRVMeasurement]:
        stmt = (
            select(HRVMeasurement)
            .where(HRVMeasurement.source_name == source_name)
            .order_by(desc(HRVMeasurement.measurement_date))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_baselines(self, source_name: str) -> dict[str, float]:
        row = self.session.execute(
            select(
                func.avg(HRVMeasurement.SD1),
                func.avg(HRVMeasurement.SD2),
                func.avg(HRVMeasurement.sdnn),
                func.avg(HRVMeasurement.rmssd),
                func.avg(HRVMeasurement.pNN50),
                func.avg(HRVMeasurement.VLF),
                func.avg(HRVMeasurement.LF),
                func.avg(HRVMeasurement.HF),
            ).where(HRVMeasurement.source_name == source_name)
        ).one()

        return {
            "avg_SD1": float(row[0] or 0.0),
            "avg_SD2": float(row[1] or 0.0),
            "avg_sdnn": float(row[2] or 0.0),
            "avg_rmssd": float(row[3] or 0.0),
            "avg_pNN50": float(row[4] or 0.0),
            "avg_VLF": float(row[5] or 0.0),
            "avg_LF": float(row[6] or 0.0),
            "avg_HF": float(row[7] or 0.0),

            # placeholder std values so anomaly code does not crash
            "std_SD1": 0.0,
            "std_SD2": 0.0,
            "std_sdnn": 0.0,
            "std_rmssd": 0.0,
            "std_pNN50": 0.0,
            "std_VLF": 0.0,
            "std_LF": 0.0,
            "std_HF": 0.0,
        }

    def get_recent_alerts(self, source_name: str, limit: int = 20) -> list[dict[str, Any]]:
        stmt = (
            select(HRVAlert)
            .where(HRVAlert.source_name == source_name)
            .order_by(desc(HRVAlert.alert_date))
            .limit(limit)
        )

        alerts = self.session.execute(stmt).scalars().all()

        return [
            {
                "alert_date": a.alert_date.isoformat() if a.alert_date else None,
                "source_name": a.source_name,
                "metric": a.metric,
                "current_value": float(a.current_value or 0.0),
                "baseline_value": float(a.baseline_value or 0.0),
                "deviation_pct": float(a.deviation_pct or 0.0),
                "alert_message": a.alert_message,
            }
            for a in alerts
        ]