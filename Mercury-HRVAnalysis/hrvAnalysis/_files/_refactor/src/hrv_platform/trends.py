from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import linregress


def classify_trend(r_value: float, slope: float) -> tuple[str, str]:
    abs_r = abs(r_value)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    if abs(slope) < 1e-12:
        direction = "stable"
    else:
        direction = "improving" if slope > 0 else "declining"

    return direction, strength


def regression_stats(df: pd.DataFrame, metric: str) -> dict[str, float | str]:
    working = df[["measurement_date", metric]].dropna().copy()
    working = working.sort_values("measurement_date")
    working["day_index"] = np.arange(len(working))
    if len(working) < 2:
        return {
            "metric": metric,
            "slope": 0.0,
            "r_value": 0.0,
            "p_value": 1.0,
            "trend_direction": "stable",
            "trend_strength": "weak",
            "mean": float(working[metric].mean() if len(working) else 0.0),
            "std": float(working[metric].std() if len(working) else 0.0),
            "min": float(working[metric].min() if len(working) else 0.0),
            "max": float(working[metric].max() if len(working) else 0.0),
        }

    result = linregress(working["day_index"], working[metric])
    direction, strength = classify_trend(float(result.rvalue), float(result.slope))
    return {
        "metric": metric,
        "slope": float(result.slope),
        "r_value": float(result.rvalue),
        "p_value": float(result.pvalue),
        "trend_direction": direction,
        "trend_strength": strength,
        "mean": float(working[metric].mean()),
        "std": float(working[metric].std(ddof=1)),
        "min": float(working[metric].min()),
        "max": float(working[metric].max()),
    }


def build_trend_stats(df: pd.DataFrame, metrics: Iterable[str]) -> dict[str, dict[str, float | str]]:
    return {metric: regression_stats(df, metric) for metric in metrics if metric in df.columns}
