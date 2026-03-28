from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt, erf
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sqlalchemy import delete, select, desc
from sqlalchemy.orm import Session

from .models import HRVMeasurement, HRVBaseline, HRVTrend, HRVAlert, HRVAnomaly
from .repository import METRICS
from .scoring import compute_ms_recovery_score


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _two_sided_p_from_z(z: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))


@dataclass
class RecalculationService:
    session: Session

    def recompute_all(self, source_name: str) -> dict[str, int]:
        measurements = self._get_measurements(source_name)

        self._replace_baselines(source_name, measurements)
        self._replace_trends(source_name, measurements)
        self._replace_alerts(source_name, measurements)
        self._replace_anomalies(source_name, measurements)

        self.session.flush()

        return {
            "measurements": len(measurements),
            "baselines": 1 if measurements else 0,
            "trends": len(METRICS) if measurements else 0,
        }

    def _get_measurements(self, source_name: str) -> list[HRVMeasurement]:
        stmt = (
            select(HRVMeasurement)
            .where(HRVMeasurement.source_name == source_name)
            .order_by(desc(HRVMeasurement.measurement_date))
        )
        return list(self.session.execute(stmt).scalars().all())

    def _replace_baselines(self, source_name: str, measurements: list[HRVMeasurement]) -> None:
        self.session.execute(delete(HRVBaseline).where(HRVBaseline.source_name == source_name))

        if not measurements:
            return

        values = {
            metric: [float(getattr(m, metric) or 0.0) for m in measurements]
            for metric in METRICS
        }

        baseline = HRVBaseline(
            source_name=source_name,
            analysis_date=measurements[0].measurement_date,
            avg_SD1=mean(values["SD1"]),
            avg_SD2=mean(values["SD2"]),
            avg_sdnn=mean(values["sdnn"]),
            avg_rmssd=mean(values["rmssd"]),
            avg_pNN50=mean(values["pNN50"]),
            avg_VLF=mean(values["VLF"]),
            avg_LF=mean(values["LF"]),
            avg_HF=mean(values["HF"]),
            std_SD1=pstdev(values["SD1"]) if len(values["SD1"]) > 1 else 0.0,
            std_SD2=pstdev(values["SD2"]) if len(values["SD2"]) > 1 else 0.0,
            std_sdnn=pstdev(values["sdnn"]) if len(values["sdnn"]) > 1 else 0.0,
            std_rmssd=pstdev(values["rmssd"]) if len(values["rmssd"]) > 1 else 0.0,
            std_pNN50=pstdev(values["pNN50"]) if len(values["pNN50"]) > 1 else 0.0,
            std_VLF=pstdev(values["VLF"]) if len(values["VLF"]) > 1 else 0.0,
            std_LF=pstdev(values["LF"]) if len(values["LF"]) > 1 else 0.0,
            std_HF=pstdev(values["HF"]) if len(values["HF"]) > 1 else 0.0,
        )
        self.session.add(baseline)

    def _replace_trends(self, source_name: str, measurements: list[HRVMeasurement]) -> None:
        self.session.execute(delete(HRVTrend).where(HRVTrend.source_name == source_name))

        if not measurements:
            return

        ordered = list(reversed(measurements))
        x = np.arange(len(ordered), dtype=float)

        latest = ordered[-1]
        current_values = {metric: float(getattr(latest, metric) or 0.0) for metric in METRICS}
        latest_ms_score = compute_ms_recovery_score(current_values)

        for metric in METRICS:
            y = np.array([float(getattr(row, metric) or 0.0) for row in ordered], dtype=float)

            if len(y) < 2:
                slope = 0.0
                r_value = 0.0
                p_value = 1.0
            else:
                slope = float(np.polyfit(x, y, 1)[0])
                r_value = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r_value):
                    r_value = 0.0
                z_like = abs(r_value) * np.sqrt(len(y))
                p_value = _two_sided_p_from_z(z_like)

            abs_r = abs(r_value)
            if abs_r >= 0.7:
                strength = "strong"
            elif abs_r >= 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"

            self.session.add(
                HRVTrend(
                    source_name=source_name,
                    analysis_date=latest.measurement_date,
                    metric=metric,
                    slope=slope,
                    r_value=r_value,
                    p_value=p_value,
                    trend_direction=direction,
                    trend_strength=strength,
                    mean=float(np.mean(y)),
                    std=float(np.std(y)),
                    min=float(np.min(y)),
                    max=float(np.max(y)),
                    latest_ms_score=latest_ms_score,
                )
            )

    def _replace_alerts(self, source_name: str, measurements: list[HRVMeasurement]) -> None:
        self.session.execute(delete(HRVAlert).where(HRVAlert.source_name == source_name))

        if not measurements:
            return

        latest = measurements[0]
        baseline = self.session.execute(
            select(HRVBaseline).where(HRVBaseline.source_name == source_name)
        ).scalar_one_or_none()

        if baseline is None:
            return

        for metric in METRICS:
            current_value = float(getattr(latest, metric) or 0.0)
            baseline_value = float(getattr(baseline, f"avg_{metric}") or 0.0)

            if baseline_value <= 0:
                continue

            deviation_pct = ((current_value - baseline_value) / baseline_value) * 100.0

            if abs(deviation_pct) >= 25.0:
                self.session.add(
                    HRVAlert(
                        alert_date=latest.measurement_date,
                        source_name=source_name,
                        metric=metric,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        deviation_pct=deviation_pct,
                        alert_message=(
                            f"{metric} deviation: current={current_value:.2f}, "
                            f"baseline={baseline_value:.2f}, deviation={deviation_pct:.1f}%"
                        ),
                    )
                )

    def _replace_anomalies(self, source_name: str, measurements: list[HRVMeasurement]) -> None:
        self.session.execute(delete(HRVAnomaly).where(HRVAnomaly.source_name == source_name))

        if not measurements:
            return

        baseline = self.session.execute(
            select(HRVBaseline).where(HRVBaseline.source_name == source_name)
        ).scalar_one_or_none()

        if baseline is None:
            return

        for row in measurements[:30]:
            for metric in METRICS:
                value = float(getattr(row, metric) or 0.0)
                baseline_mean = float(getattr(baseline, f"avg_{metric}") or 0.0)
                baseline_std = float(getattr(baseline, f"std_{metric}") or 0.0)

                if baseline_std <= 0:
                    continue

                z_score = (value - baseline_mean) / baseline_std

                if abs(z_score) >= 2.5:
                    self.session.add(
                        HRVAnomaly(
                            measurement_date=row.measurement_date,
                            source_name=source_name,
                            metric=metric,
                            value=value,
                            baseline_mean=baseline_mean,
                            baseline_std=baseline_std,
                            z_score=z_score,
                            detector="z_score",
                            message=f"{metric} z-score {z_score:.2f} exceeds threshold 2.50",
                        )
                    )