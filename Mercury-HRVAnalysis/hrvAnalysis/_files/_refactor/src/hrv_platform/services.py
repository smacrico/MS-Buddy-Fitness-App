from __future__ import annotations

from dataclasses import dataclass

from .anomalies import detect_point_anomalies
from .config import settings
from .repository import HRVRepository, METRICS
from .scoring import compute_ms_recovery_score
from .trends import build_trend_stats


@dataclass
class AnalysisResult:
    data_points: int
    date_range: dict[str, str | None]
    current_values: dict[str, float]
    recovery_scores: dict[str, float]
    baselines: dict[str, float]
    trends: dict[str, dict]
    alerts: list[dict]
    anomalies: list[dict]


class AnalyticsService:
    def __init__(self, repository: HRVRepository) -> None:
        self.repository = repository

    def _build_alerts(
        self,
        current_values: dict[str, float],
        baselines: dict[str, float],
        source_name: str,
    ) -> list[dict]:
        alerts: list[dict] = []
        threshold = settings.alert_deviation_threshold
        for metric in METRICS:
            baseline = float(baselines.get(metric, 0.0))
            current = float(current_values.get(metric, 0.0))
            if baseline <= 0:
                continue
            deviation = (current - baseline) / baseline
            if abs(deviation) >= threshold:
                alerts.append(
                    {
                        "metric": metric,
                        "current_value": current,
                        "baseline_value": baseline,
                        "deviation_pct": round(deviation * 100, 2),
                        "alert_type": "threshold",
                        "alert_message": f"{metric} deviated by {deviation * 100:.1f}% from baseline",
                    }
                )
        self.repository.replace_alerts(source_name, alerts)
        return alerts

    def analyze(self, source_name: str, days_back: int | None = None) -> AnalysisResult:
        days = days_back or settings.analysis_window_days
        df = self.repository.get_measurements_df(source_name=source_name, days_back=days)
        baselines = self.repository.get_baseline_averages(source_name, settings.baseline_window_days)

        if df.empty:
            return AnalysisResult(
                data_points=0,
                date_range={"start": None, "end": None},
                current_values={metric: 0.0 for metric in METRICS},
                recovery_scores={"ms": 0.0},
                baselines=baselines,
                trends={},
                alerts=[],
                anomalies=[],
            )

        df["ms_recovery"] = df.apply(lambda row: compute_ms_recovery_score(row.to_dict()), axis=1)
        current_row = df.iloc[-1]
        current_values = {metric: float(current_row[metric]) for metric in METRICS}
        trends = build_trend_stats(df, [*METRICS, "ms_recovery"])
        alerts = self._build_alerts(current_values=current_values, baselines=baselines, source_name=source_name)
        anomalies = detect_point_anomalies(df=df, source_name=source_name)

        if anomalies:
            self.repository.save_anomalies(anomalies)

        self.repository.insert_baseline(source_name, baselines)
        self.repository.replace_trends(source_name, trends, float(current_row["ms_recovery"]))

        return AnalysisResult(
            data_points=len(df),
            date_range={
                "start": str(df["measurement_date"].min()),
                "end": str(df["measurement_date"].max()),
            },
            current_values=current_values,
            recovery_scores={"ms": float(current_row["ms_recovery"])},
            baselines=baselines,
            trends=trends,
            alerts=alerts,
            anomalies=anomalies,
        )
