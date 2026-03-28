from __future__ import annotations

import pandas as pd

from .config import settings


def detect_point_anomalies(
    df: pd.DataFrame,
    source_name: str,
    zscore_threshold: float | None = None,
) -> list[dict]:
    threshold = zscore_threshold or settings.ingestion_anomaly_zscore
    anomalies: list[dict] = []

    for metric in [c for c in df.columns if c not in {"measurement_date", "source_name", "ms_recovery"}]:
        series = df[metric].dropna()
        if len(series) < 5:
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        if std <= 0:
            continue

        latest_idx = series.index[-1]
        latest_value = float(series.loc[latest_idx])
        z = (latest_value - mean) / std
        if abs(z) >= threshold:
            anomalies.append(
                {
                    "measurement_date": df.loc[latest_idx, "measurement_date"],
                    "source_name": source_name,
                    "metric": metric,
                    "value": latest_value,
                    "baseline_mean": mean,
                    "baseline_std": std,
                    "z_score": float(z),
                    "detector": "zscore",
                    "message": f"{metric} z-score anomaly detected: z={z:.2f}",
                }
            )
    return anomalies
