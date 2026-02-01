"""
Utility functions for longevity metrics computation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple
from . import config


def zscore(value: float, baseline_mean: float, baseline_std: float) -> float:
    """
    Calculate z-score for a value relative to baseline.
    
    Args:
        value: Value to score
        baseline_mean: Mean of baseline distribution
        baseline_std: Standard deviation of baseline distribution
        
    Returns:
        Z-score, or 0.0 if std is zero/NaN
    """
    if baseline_std is None or baseline_std == 0 or np.isnan(baseline_std):
        return 0.0
    return (value - baseline_mean) / baseline_std


def clamp(x: float, lo: float = config.SCORE_MIN, hi: float = config.SCORE_MAX) -> float:
    """
    Clamp value between minimum and maximum bounds.
    
    Args:
        x: Value to clamp
        lo: Lower bound (default from config)
        hi: Upper bound (default from config)
        
    Returns:
        Clamped value
    """
    return float(max(lo, min(hi, x)))


def rolling_baseline(series: Sequence[float], window: int = config.BASELINE_WINDOW_DAYS) -> Tuple[float, float]:
    """
    Compute baseline mean and std from recent window values.
    
    Args:
        series: Time series data
        window: Rolling window size (days)
        
    Returns:
        Tuple of (mean, std) ignoring NaN values
    """
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float('nan'), float('nan')
    if valid.size < 2:
        return float(np.nanmean(valid)), float(np.nanstd(valid, ddof=0))
    last = valid[-window:] if valid.size > window else valid
    return float(np.nanmean(last)), float(np.nanstd(last, ddof=0))


def normalize(x: float, low: float, high: float) -> float:
    """
    Normalize value from [low, high] to [0, 100].
    
    Args:
        x: Value to normalize
        low: Lower bound of input range
        high: Upper bound of input range
        
    Returns:
        Normalized value (0-100), or 50.0 for NaN
    """
    if np.isnan(x):
        return 50.0
    x = float(x)
    if x <= low:
        return 0.0
    if x >= high:
        return 100.0
    return 100.0 * ((x - low) / (high - low))


def normalize_inverse(x: float, low: float, high: float) -> float:
    """
    Inverse normalize: map [low, high] to [100, 0] (lower is better).
    
    Args:
        x: Value to normalize
        low: Lower bound of input range (best)
        high: Upper bound of input range (worst)
        
    Returns:
        Inverse normalized value (0-100), or 50.0 for NaN
    """
    if np.isnan(x):
        return 50.0
    x = float(x)
    if x <= low:
        return 100.0
    if x >= high:
        return 0.0
    return 100.0 * (1.0 - (x - low) / (high - low))


def validate_heart_rate(hr: float) -> bool:
    """
    Validate heart rate is within physiological range.
    
    Args:
        hr: Heart rate in bpm
        
    Returns:
        True if valid, False otherwise
    """
    if np.isnan(hr):
        return False
    return config.HEART_RATE_MIN_BPM <= hr <= config.HEART_RATE_MAX_BPM


def validate_spo2(spo2: float) -> bool:
    """
    Validate SpO2 is within physiological range.
    
    Args:
        spo2: SpO2 percentage
        
    Returns:
        True if valid, False otherwise
    """
    if np.isnan(spo2):
        return False
    return config.SPO2_MIN_PERCENT <= spo2 <= config.SPO2_MAX_PERCENT
