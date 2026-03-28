from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .models import HRVBaseline, HRVMeasurement
from .repository import METRICS
from .scoring import compute_ms_recovery_score


@dataclass
class PlotExportResult:
    output_dir: str
    files: list[str]


class PlotService:
    def __init__(self, session: Session, output_dir: str | None = None) -> None:
        self.session = session
        self.output_dir = Path(output_dir or r"C:\temp\logsFitnessApp\HRV_DashBoards")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, source_name: str = "MyHRV_import", days_back: int = 90) -> PlotExportResult:
        measurements = self._get_measurements(source_name=source_name, days_back=days_back)
        if not measurements:
            raise ValueError(f"No measurements found for source_name='{source_name}'")

        baseline = self._get_latest_baseline(source_name=source_name)

        files: list[str] = []
        files.append(self.plot_time_trends(measurements))
        files.append(self.plot_histograms(measurements))
        if baseline is not None:
            files.append(self.plot_baseline_bar(baseline))
            files.append(self.plot_radar_chart(measurements[0], baseline))
        files.append(self.plot_ms_score(measurements))
        files.append(self.plot_trend_summary(measurements))

        return PlotExportResult(output_dir=str(self.output_dir), files=files)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_measurements(self, source_name: str, days_back: int) -> list[HRVMeasurement]:
        stmt = (
            select(HRVMeasurement)
            .where(HRVMeasurement.source_name == source_name)
            .order_by(desc(HRVMeasurement.measurement_date))
            .limit(days_back)
        )
        return list(self.session.execute(stmt).scalars().all())

    def _get_latest_baseline(self, source_name: str) -> HRVBaseline | None:
        stmt = (
            select(HRVBaseline)
            .where(HRVBaseline.source_name == source_name)
            .order_by(desc(HRVBaseline.analysis_date), desc(HRVBaseline.id))
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def _measurement_series(self, measurements: list[HRVMeasurement], metric: str) -> list[float]:
        ordered = list(reversed(measurements))
        return [float(getattr(row, metric) or 0.0) for row in ordered]

    def _dates(self, measurements: list[HRVMeasurement]) -> list:
        ordered = list(reversed(measurements))
        return [row.measurement_date for row in ordered]

    def plot_time_trends(self, measurements: list[HRVMeasurement]) -> str:
        dates = self._dates(measurements)

        plt.figure(figsize=(14, 8))
        for metric in METRICS:
            plt.plot(dates, self._measurement_series(measurements, metric), marker="o", label=metric.upper())

        plt.title("HRV Metrics Time Trends")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = self.output_dir / f"HRV_TimeTrends_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)

    def plot_histograms(self, measurements: list[HRVMeasurement]) -> str:
        plt.figure(figsize=(16, 10))

        for i, metric in enumerate(METRICS, 1):
            plt.subplot(2, 4, i)
            values = self._measurement_series(measurements, metric)
            plt.hist(values, bins=15, edgecolor="black")
            plt.title(metric.upper())
            plt.axvline(np.mean(values), linestyle="--")

        plt.suptitle("HRV Metrics Distributions")
        plt.tight_layout()

        filename = self.output_dir / f"HRV_Histograms_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)

    def plot_baseline_bar(self, baseline: HRVBaseline) -> str:
        labels = [metric.upper() for metric in METRICS]
        values = [float(getattr(baseline, f"avg_{metric}") or 0.0) for metric in METRICS]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.title("90-Day HRV Baseline Profile")
        plt.ylabel("Baseline Value")
        plt.grid(axis="y")
        plt.tight_layout()

        filename = self.output_dir / f"HRV_BaselineProfile_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)

    def plot_radar_chart(self, latest: HRVMeasurement, baseline: HRVBaseline) -> str:
        import math

        categories = [metric.upper() for metric in METRICS]
        n = len(categories)

        latest_values = [float(getattr(latest, metric) or 0.0) for metric in METRICS]
        baseline_values = [float(getattr(baseline, f"avg_{metric}") or 0.0) for metric in METRICS]

        angles = [i / float(n) * 2 * math.pi for i in range(n)]
        angles += angles[:1]
        latest_values += latest_values[:1]
        baseline_values += baseline_values[:1]

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7))
        ax.plot(angles, latest_values, linewidth=2, label="Latest")
        ax.fill(angles, latest_values, alpha=0.20)
        ax.plot(angles, baseline_values, linewidth=2, linestyle="--", label="Baseline")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title("Latest vs Baseline (Radar)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        plt.tight_layout()

        filename = self.output_dir / f"HRV_RadarChart_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)

    def plot_ms_score(self, measurements: list[HRVMeasurement]) -> str:
        ordered = list(reversed(measurements))
        dates = [row.measurement_date for row in ordered]
        scores = [
            compute_ms_recovery_score(
                {metric: float(getattr(row, metric) or 0.0) for metric in METRICS}
            )
            for row in ordered
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, scores, marker="d", label="MS-Score")
        plt.title("MS-Aware HRV Health Score (Trend)")
        plt.ylabel("Score (0-100)")
        plt.xlabel("Date")
        plt.axhline(50, linestyle="--")
        plt.legend()
        plt.tight_layout()

        filename = self.output_dir / f"HRV_MSScore_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)

    def plot_trend_summary(self, measurements: list[HRVMeasurement]) -> str:
        ordered = list(reversed(measurements))
        x = np.arange(len(ordered), dtype=float)

        metrics = []
        corr_vals = []
        directions = []

        for metric in METRICS:
            y = np.array([float(getattr(row, metric) or 0.0) for row in ordered], dtype=float)
            if len(y) < 2:
                slope = 0.0
                corr = 0.0
            else:
                slope = float(np.polyfit(x, y, 1)[0])
                corr = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
            metrics.append(metric)
            corr_vals.append(corr)
            directions.append(direction)

        plt.figure(figsize=(10, 5))
        bars = plt.bar(metrics, corr_vals)
        plt.title("Trend Statistics Summary")
        plt.ylabel("Correlation coefficient")
        plt.ylim(-1, 1)
        plt.axhline(0, linewidth=0.8)

        for bar, direction in zip(bars, directions):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                direction,
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=40)
        plt.grid(axis="y")
        plt.tight_layout()

        filename = self.output_dir / f"HRV_TrendSummary_{self._timestamp()}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        return str(filename)