"""
Activity and strain metrics computation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence
from .config import (
    TRIMP_MULTIPLIER_MALE,
    TRIMP_MULTIPLIER_FEMALE,
)


def compute_activity_trimp_simple(
    hr_series_bpm: Sequence[float],
    duration_min: float,
    hr_rest: float,
    hr_max: float,
    multiplier_male: float = TRIMP_MULTIPLIER_MALE,
    multiplier_female: float = TRIMP_MULTIPLIER_FEMALE,
    sex: str = 'male'
) -> float:
    """
    Compute simplified TRIMP (Training Impulse) for activity load.
    
    TRIMP = duration * intensity * sex_multiplier
    where intensity = (avg_HR - HR_rest) / (HR_max - HR_rest)
    
    Args:
        hr_series_bpm: Heart rate samples during activity (bpm)
        duration_min: Duration of activity (minutes)
        hr_rest: Resting heart rate (bpm)
        hr_max: Maximum heart rate (bpm)
        multiplier_male: Male sex multiplier (default from config)
        multiplier_female: Female sex multiplier (default from config)
        sex: 'male' or 'female'
    
    Returns:
        TRIMP score (non-negative float representing strain)
    """
    hr = np.asarray(hr_series_bpm, dtype=float)
    
    if hr.size == 0 or duration_min <= 0:
        return 0.0
    
    avg_hr = float(np.nanmean(hr))
    denom = hr_max - hr_rest if hr_max > hr_rest else 1.0
    intensity = max(0.0, (avg_hr - hr_rest) / denom)
    
    mult = multiplier_male if sex.lower().startswith('m') else multiplier_female
    
    return float(duration_min * intensity * mult)


def compute_activity_zones(
    hr_series_bpm: Sequence[float],
    hr_rest: float,
    hr_max: float
) -> dict:
    """
    Compute time spent in different heart rate zones.
    
    Zones (% of HR reserve):
    - Zone 1 (Easy): 50-60%
    - Zone 2 (Moderate): 60-70%
    - Zone 3 (Hard): 70-80%
    - Zone 4 (Very Hard): 80-90%
    - Zone 5 (Maximum): 90-100%
    
    Args:
        hr_series_bpm: Heart rate samples (bpm)
        hr_rest: Resting heart rate (bpm)
        hr_max: Maximum heart rate (bpm)
    
    Returns:
        Dictionary with time (count) in each zone
    """
    hr = np.asarray(hr_series_bpm, dtype=float)
    
    if hr.size == 0:
        return {f'zone_{i}': 0 for i in range(1, 6)}
    
    # Calculate HR reserve percentages
    hr_reserve = hr_max - hr_rest
    hr_pct = (hr - hr_rest) / hr_reserve if hr_reserve > 0 else np.zeros_like(hr)
    
    zones = {
        'zone_1': int(np.sum((hr_pct >= 0.50) & (hr_pct < 0.60))),
        'zone_2': int(np.sum((hr_pct >= 0.60) & (hr_pct < 0.70))),
        'zone_3': int(np.sum((hr_pct >= 0.70) & (hr_pct < 0.80))),
        'zone_4': int(np.sum((hr_pct >= 0.80) & (hr_pct < 0.90))),
        'zone_5': int(np.sum(hr_pct >= 0.90)),
    }
    
    return zones


def compute_activity_summary(
    hr_series_bpm: Sequence[float],
    duration_min: float,
    hr_rest: float,
    hr_max: float,
    sex: str = 'male'
) -> dict:
    """
    Compute comprehensive activity metrics.
    
    Args:
        hr_series_bpm: Heart rate samples (bpm)
        duration_min: Duration (minutes)
        hr_rest: Resting heart rate (bpm)
        hr_max: Maximum heart rate (bpm)
        sex: 'male' or 'female'
    
    Returns:
        Dictionary with all activity metrics
    """
    hr = np.asarray(hr_series_bpm, dtype=float)
    
    return {
        'trimp': compute_activity_trimp_simple(hr, duration_min, hr_rest, hr_max, sex=sex),
        'duration_min': duration_min,
        'avg_hr': float(np.nanmean(hr)) if hr.size > 0 else float('nan'),
        'max_hr': float(np.nanmax(hr)) if hr.size > 0 else float('nan'),
        'min_hr': float(np.nanmin(hr)) if hr.size > 0 else float('nan'),
        'zones': compute_activity_zones(hr, hr_rest, hr_max),
    }
