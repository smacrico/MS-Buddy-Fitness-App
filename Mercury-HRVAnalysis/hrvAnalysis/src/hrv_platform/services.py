from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .anomalies import detect_point_anomalies
from .repository import HRVRepository
from .scoring import compute_ms_recovery_score
from .trends import compute_regression_trends


@dataclass
class AnalyticsService:
    repository: HRVRepository

    def build_summary(self, source_name: str = "MyHRV_import", limit: int = 90) -> dict[str, Any]:
        measurements = self.repository.get_recent_measurements(
            source_name=source_name,
            limit=limit,
        )

        if not measurements:
            return {
                "data_points": 0,
                "date_range": {"start": None, "end": None},
                "current_values": {},
                "recovery_scores": {"ms": 0.0},
                "baselines": {},
                "alerts": [],
                "anomalies": [],
                "trends": {},
            }

        latest = measurements[0]
        oldest = measurements[-1]

        current_values = {
            "SD1": float(getattr(latest, "SD1", 0.0) or 0.0),
            "SD2": float(getattr(latest, "SD2", 0.0) or 0.0),
            "sdnn": float(getattr(latest, "sdnn", 0.0) or 0.0),
            "rmssd": float(getattr(latest, "rmssd", 0.0) or 0.0),
            "pNN50": float(getattr(latest, "pNN50", 0.0) or 0.0),
            "VLF": float(getattr(latest, "VLF", 0.0) or 0.0),
            "LF": float(getattr(latest, "LF", 0.0) or 0.0),
            "HF": float(getattr(latest, "HF", 0.0) or 0.0),
        }

        baselines = self.repository.get_baselines(source_name=source_name)
        alerts = self.repository.get_recent_alerts(source_name=source_name)
        anomalies = detect_point_anomalies(measurements, baselines)
        trends = compute_regression_trends(measurements)
        ms_score = compute_ms_recovery_score(current_values)

        start_date = getattr(oldest, "measurement_date", None)
        end_date = getattr(latest, "measurement_date", None)

        return {
            "data_points": len(measurements),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "current_values": current_values,
            "recovery_scores": {"ms": ms_score},
            "baselines": baselines,
            "alerts": alerts,
            "anomalies": anomalies,
            "trends": trends,
        }