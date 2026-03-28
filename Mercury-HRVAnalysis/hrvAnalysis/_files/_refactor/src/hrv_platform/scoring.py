from __future__ import annotations

from .config import settings


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def compute_ms_recovery_score(metrics: dict[str, float]) -> float:
    """
    Normalize the weighted HRV profile to a 0-100 score.

    Method:
    - compare each supported metric to a reference value
    - calculate weighted ratio
    - ratio 1.0 maps to score 50
    - ratio 2.0 maps to 100
    - ratio 0.0 maps to 0
    """
    numerator = 0.0
    denominator = 0.0

    for metric, weight in settings.ms_weights.items():
        reference = settings.metric_reference_values[metric]
        value = float(metrics.get(metric, 0.0))
        ratio = 0.0 if reference <= 0 else value / reference
        numerator += ratio * weight
        denominator += weight

    if denominator == 0:
        return 0.0

    weighted_ratio = numerator / denominator
    score = weighted_ratio * 50.0
    return round(clamp(score), 2)
