from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .models import HRVAlert, HRVAnomaly, HRVBaseline, HRVMeasurement, HRVTrend


METRICS = ["SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"]


class HRVRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_measurement(self, payload: dict[str, Any]) -> HRVMeasurement:
        stmt = select(HRVMeasurement).where(
            HRVMeasurement.measurement_date == payload["measurement_date"],
            HRVMeasurement.source_name == payload["source_name"],
        )
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
            return existing
        item = HRVMeasurement(**payload)
        self.session.add(item)
        self.session.flush()
        return item

    def insert_baseline(self, source_name: str, values: dict[str, float]) -> HRVBaseline:
        item = HRVBaseline(
            analysis_date=date.today(),
            source_name=source_name,
            **{f"avg_{k}": float(v) for k, v in values.items()},
        )
        self.session.add(item)
        self.session.flush()
        return item

    def replace_trends(self, source_name: str, trends: dict[str, dict], latest_ms_score: float) -> None:
        self.session.execute(
            delete(HRVTrend).where(
                HRVTrend.source_name == source_name,
                HRVTrend.analysis_date == date.today(),
            )
        )
        for metric, trend in trends.items():
            self.session.add(
                HRVTrend(
                    analysis_date=date.today(),
                    source_name=source_name,
                    metric=metric,
                    slope=float(trend["slope"]),
                    r_value=float(trend["r_value"]),
                    p_value=float(trend["p_value"]),
                    trend_direction=str(trend["trend_direction"]),
                    trend_strength=str(trend["trend_strength"]),
                    mean=float(trend["mean"]),
                    std=float(trend["std"]),
                    min=float(trend["min"]),
                    max=float(trend["max"]),
                    latest_ms_score=float(latest_ms_score),
                )
            )
        self.session.flush()

    def replace_alerts(self, source_name: str, alerts: list[dict]) -> None:
        self.session.execute(delete(HRVAlert).where(HRVAlert.source_name == source_name, HRVAlert.alert_date == date.today()))
        for item in alerts:
            self.session.add(
                HRVAlert(
                    alert_date=date.today(),
                    source_name=source_name,
                    metric=item["metric"],
                    current_value=item["current_value"],
                    baseline_value=item["baseline_value"],
                    deviation_pct=item["deviation_pct"],
                    alert_type=item["alert_type"],
                    alert_message=item["alert_message"],
                )
            )
        self.session.flush()

    def save_anomalies(self, anomalies: list[dict]) -> None:
        for item in anomalies:
            self.session.add(HRVAnomaly(**item))
        self.session.flush()

    def get_measurements_df(self, source_name: str, days_back: int) -> pd.DataFrame:
        start_date = date.today() - timedelta(days=days_back)
        stmt = (
            select(HRVMeasurement)
            .where(HRVMeasurement.source_name == source_name, HRVMeasurement.measurement_date >= start_date)
            .order_by(HRVMeasurement.measurement_date.asc())
        )
        rows = [row[0] for row in self.session.execute(stmt).all()]
        if not rows:
            return pd.DataFrame(columns=["measurement_date", "source_name", *METRICS])

        records = []
        for row in rows:
            record = {"measurement_date": row.measurement_date, "source_name": row.source_name}
            for metric in METRICS:
                record[metric] = getattr(row, metric)
            records.append(record)
        return pd.DataFrame(records)

    def get_latest_measurements(self, source_name: str) -> HRVMeasurement | None:
        stmt = (
            select(HRVMeasurement)
            .where(HRVMeasurement.source_name == source_name)
            .order_by(HRVMeasurement.measurement_date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_baseline_averages(self, source_name: str, days_back: int) -> dict[str, float]:
        start_date = date.today() - timedelta(days=days_back)
        cols = [func.avg(getattr(HRVMeasurement, metric)).label(metric) for metric in METRICS]
        stmt = select(*cols).where(
            HRVMeasurement.source_name == source_name,
            HRVMeasurement.measurement_date >= start_date,
        )
        row = self.session.execute(stmt).one()
        return {metric: float(getattr(row, metric) or 0.0) for metric in METRICS}

    def get_recent_anomalies(self, source_name: str, limit: int = 20) -> list[HRVAnomaly]:
        stmt = (
            select(HRVAnomaly)
            .where(HRVAnomaly.source_name == source_name)
            .order_by(HRVAnomaly.detected_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars())

    def get_recent_alerts(self, source_name: str, limit: int = 20) -> list[HRVAlert]:
        stmt = (
            select(HRVAlert)
            .where(HRVAlert.source_name == source_name)
            .order_by(HRVAlert.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars())

    def get_trends(self, source_name: str) -> list[HRVTrend]:
        stmt = (
            select(HRVTrend)
            .where(HRVTrend.source_name == source_name, HRVTrend.analysis_date == date.today())
            .order_by(HRVTrend.metric.asc())
        )
        return list(self.session.execute(stmt).scalars())
