from __future__ import annotations

from typing import Mapping


DEFAULT_WEIGHTS: dict[str, float] = {
    "rmssd": 0.18,
    "sdnn": 0.24,
    "pNN50": 0.14,
    "SD1": 0.15,
    "SD2": 0.13,
    "LF": 0.08,
    "HF": 0.08,
}

DEFAULT_REFERENCE_RANGES: dict[str, float] = {
    "rmssd": 100.0,
    "sdnn": 100.0,
    "pNN50": 100.0,
    "SD1": 100.0,
    "SD2": 100.0,
    "LF": 2000.0,
    "HF": 2000.0,
}


def compute_ms_recovery_score(
    values: Mapping[str, float | int | None],
    weights: dict[str, float] | None = None,
    reference_ranges: dict[str, float] | None = None,
) -> float:
    weights = weights or DEFAULT_WEIGHTS
    reference_ranges = reference_ranges or DEFAULT_REFERENCE_RANGES

    total = 0.0

    for metric, weight in weights.items():
        raw_value = values.get(metric, 0.0)
        value = float(raw_value or 0.0)
        denom = float(reference_ranges.get(metric, 100.0))

        if denom <= 0:
            continue

        metric_score = max(0.0, min(1.0, value / denom))
        total += metric_score * weight

    return round(max(0.0, min(100.0, total * 100.0)), 2)