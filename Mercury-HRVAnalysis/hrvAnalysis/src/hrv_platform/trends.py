from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

import numpy as np


METRICS = ["SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"]


def compute_regression_trends(measurements: list[Any]) -> dict[str, dict[str, float | str]]:
    if not measurements:
        return {}

    ordered = list(reversed(measurements))
    x = np.arange(len(ordered), dtype=float)

    results: dict[str, dict[str, float | str]] = {}

    for metric in METRICS:
        y = np.array([float(getattr(row, metric, 0.0) or 0.0) for row in ordered], dtype=float)

        if len(y) < 2:
            slope = 0.0
            corr = 0.0
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            corr = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(corr):
                corr = 0.0

        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"

        results[metric] = {
            "slope": round(slope, 6),
            "correlation": round(corr, 6),
            "trend_direction": direction,
            "trend_strength": strength,
            "mean": round(mean(y.tolist()), 4),
            "std": round(pstdev(y.tolist()) if len(y) > 1 else 0.0, 4),
            "min": round(float(np.min(y)), 4),
            "max": round(float(np.max(y)), 4),
        }

    return results