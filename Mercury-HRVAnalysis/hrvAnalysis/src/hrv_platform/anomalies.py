from __future__ import annotations

from typing import Any


METRICS = ["SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"]


def detect_point_anomalies(
    measurements: list[Any],
    baselines: dict[str, float] | None = None,
    z_threshold: float = 2.5,
) -> list[dict]:
    if not measurements:
        return []

    baselines = baselines or {}
    anomalies: list[dict] = []

    for row in measurements[:10]:
        for metric in METRICS:
            baseline_mean = float(baselines.get(f"avg_{metric}", 0.0) or 0.0)
            baseline_std = float(baselines.get(f"std_{metric}", 0.0) or 0.0)
            value = float(getattr(row, metric, 0.0) or 0.0)

            if baseline_std <= 0:
                continue

            z = (value - baseline_mean) / baseline_std
            if abs(z) >= z_threshold:
                measurement_date = getattr(row, "measurement_date", None)
                source_name = getattr(row, "source_name", "unknown")

                anomalies.append(
                    {
                        "measurement_date": measurement_date.isoformat() if measurement_date else None,
                        "source_name": source_name,
                        "metric": metric,
                        "value": value,
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                        "z_score": round(z, 4),
                        "detector": "z_score",
                        "message": f"{metric} z-score {z:.2f} exceeds threshold {z_threshold:.2f}",
                    }
                )

    return anomalies