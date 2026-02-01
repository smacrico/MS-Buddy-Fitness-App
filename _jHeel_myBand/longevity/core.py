"""
Core composite metrics: recovery, capacity, momentum, biological age.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple
from .config import (
    RECOVERY_HRV_WEIGHT,
    RECOVERY_SLEEP_WEIGHT,
    RECOVERY_STRAIN_WEIGHT,
    RECOVERY_BASE_SCORE,
    CAPACITY_HRV_WEIGHT,
    CAPACITY_RHR_WEIGHT,
    CAPACITY_SLEEP_SCALE,
    CAPACITY_DEBT_SCALE,
    CAPACITY_DEBT_BASELINE,
    CAPACITY_BASE_SCORE,
    RECOVERY_DEBT_WINDOW_DAYS,
    MOMENTUM_WINDOW_DAYS,
    MOMENTUM_MIN_POINTS,
    CARDIO_RHR_WEIGHT,
    CARDIO_SDNN_WEIGHT,
    CARDIO_SPO2_WEIGHT,
    CARDIO_HR_RECOVERY_WEIGHT,
    CARDIO_RHR_BEST,
    CARDIO_RHR_WORST,
    CARDIO_SDNN_LOW,
    CARDIO_SDNN_HIGH,
    CARDIO_SPO2_LOW,
    CARDIO_SPO2_HIGH,
    CARDIO_HR_RECOVERY_LOW,
    CARDIO_HR_RECOVERY_HIGH,
    BIO_AGE_RHR_COEFF,
    BIO_AGE_RHR_BASELINE,
    BIO_AGE_RMSSD_COEFF,
    BIO_AGE_RMSSD_BASELINE,
    BIO_AGE_SLEEP_COEFF,
    BIO_AGE_SLEEP_BASELINE,
    BIO_AGE_ACTIVITY_COEFF,
    BIO_AGE_ACTIVITY_BASELINE,
    FORECAST_DEFAULT_DAYS,
    FORECAST_FIT_WINDOW_DAYS,
    FORECAST_MIN_POINTS,
)
from .utils import zscore, clamp, normalize_range, normalize_inverse


def compute_recovery_score(
    rmssd_today: float,
    rmssd_baseline_mean: float,
    rmssd_baseline_std: float,
    sleep_quality_score: float,
    strain_today: float,
    baseline_strain_mean: float,
    baseline_strain_std: float
) -> float:
    """
    Compute composite recovery score (0-100).
    
    Combines:
    - HRV (RMSSD) relative to baseline (positive contribution)
    - Sleep quality (positive contribution)
    - Strain relative to baseline (negative contribution)
    
    Args:
        rmssd_today: Today's RMSSD (ms)
        rmssd_baseline_mean: Baseline RMSSD mean
        rmssd_baseline_std: Baseline RMSSD std
        sleep_quality_score: Sleep quality (0-100)
        strain_today: Today's strain/TRIMP
        baseline_strain_mean: Baseline strain mean
        baseline_strain_std: Baseline strain std
    
    Returns:
        Recovery score (0-100)
    """
    z_hrv = zscore(rmssd_today, rmssd_baseline_mean, rmssd_baseline_std)
    z_strain = zscore(strain_today, baseline_strain_mean, baseline_strain_std)
    
    hrv_contrib = RECOVERY_HRV_WEIGHT * z_hrv
    sleep_contrib = RECOVERY_SLEEP_WEIGHT * sleep_quality_score
    strain_contrib = RECOVERY_STRAIN_WEIGHT * z_strain
    
    score = RECOVERY_BASE_SCORE + hrv_contrib + sleep_contrib + strain_contrib
    
    return clamp(score, 0.0, 100.0)


def compute_recovery_debt(
    strain_series: Sequence[float],
    recovery_credit_series: Sequence[float],
    window_days: int = RECOVERY_DEBT_WINDOW_DAYS
) -> float:
    """
    Compute recovery debt over rolling window.
    
    debt = max(0, sum(strain) - sum(recovery_credit))
    
    Args:
        strain_series: Daily strain values
        recovery_credit_series: Daily recovery credits
        window_days: Rolling window size (days)
    
    Returns:
        Recovery debt (non-negative, higher is worse)
    """
    s = np.asarray(strain_series, dtype=float)
    r = np.asarray(recovery_credit_series, dtype=float)
    
    n = min(s.size, r.size)
    if n == 0:
        return 0.0
    
    s = s[-window_days:]
    r = r[-window_days:]
    
    debt = float(np.sum(s) - np.sum(r))
    
    return max(0.0, debt)


def compute_metabolic_capacity(
    rmssd_today: float,
    rmssd_baseline_mean: float,
    rmssd_baseline_std: float,
    rhr_today: float,
    rhr_baseline_mean: float,
    rhr_baseline_std: float,
    sleep_quality_score: float,
    recovery_debt: float
) -> float:
    """
    Compute Metabolic Capacity score (0-100).
    
    Composite of:
    - HRV (positive)
    - Resting HR (negative)
    - Sleep quality (positive)
    - Recovery debt (negative)
    
    Args:
        rmssd_today: Today's RMSSD (ms)
        rmssd_baseline_mean: Baseline RMSSD mean
        rmssd_baseline_std: Baseline RMSSD std
        rhr_today: Today's resting HR (bpm)
        rhr_baseline_mean: Baseline resting HR mean
        rhr_baseline_std: Baseline resting HR std
        sleep_quality_score: Sleep quality (0-100)
        recovery_debt: Current recovery debt
    
    Returns:
        Metabolic capacity score (0-100)
    """
    z_hrv = zscore(rmssd_today, rmssd_baseline_mean, rmssd_baseline_std)
    z_rhr = zscore(rhr_today, rhr_baseline_mean, rhr_baseline_std)
    
    hrv_pts = CAPACITY_HRV_WEIGHT * z_hrv
    rhr_pts = CAPACITY_RHR_WEIGHT * z_rhr
    sleep_pts = CAPACITY_SLEEP_SCALE * sleep_quality_score
    debt_scale = CAPACITY_DEBT_SCALE * (recovery_debt / max(1.0, CAPACITY_DEBT_BASELINE))
    
    raw_score = CAPACITY_BASE_SCORE + hrv_pts + rhr_pts + sleep_pts + debt_scale
    
    return clamp(raw_score, 0.0, 100.0)


def compute_metabolic_momentum(
    capacity_series: Sequence[float],
    window_days: int = MOMENTUM_WINDOW_DAYS
) -> float:
    """
    Compute metabolic momentum as trend slope (capacity points per week).
    
    Args:
        capacity_series: Historical capacity scores
        window_days: Rolling window for trend calculation
    
    Returns:
        Momentum (capacity points/week, positive = improving)
    """
    arr = np.asarray(capacity_series, dtype=float)
    
    if arr.size < MOMENTUM_MIN_POINTS:
        return 0.0
    
    arr = arr[-window_days:]
    x = np.arange(arr.size)
    
    # Remove NaNs
    valid = ~np.isnan(arr)
    if valid.sum() < MOMENTUM_MIN_POINTS:
        return 0.0
    
    coeffs = np.polyfit(x[valid], arr[valid], 1)
    slope_per_day = coeffs[0]
    slope_per_week = slope_per_day * 7.0
    
    return float(slope_per_week)


def compute_cardiovascular_health(
    resting_hr: float,
    sdnn_24h: float,
    spo2: float = None,
    hr_recovery_1min: float = None
) -> float:
    """
    Compute cardiovascular health composite score (0-100).
    
    Components:
    - Resting HR: lower is better (30%)
    - SDNN: higher is better (40%)
    - SpO2: higher is better (15%)
    - HR recovery: higher is better (15%)
    
    Args:
        resting_hr: Resting heart rate (bpm)
        sdnn_24h: 24-hour SDNN (ms)
        spo2: Oxygen saturation (%, optional)
        hr_recovery_1min: HR drop after 1 min (bpm, optional)
    
    Returns:
        Cardiovascular health score (0-100)
    """
    rhr_score = normalize_inverse(resting_hr, CARDIO_RHR_BEST, CARDIO_RHR_WORST)
    sdnn_score = normalize_range(sdnn_24h, CARDIO_SDNN_LOW, CARDIO_SDNN_HIGH)
    spo2_score = normalize_range(spo2 if spo2 is not None else np.nan, 
                                  CARDIO_SPO2_LOW, CARDIO_SPO2_HIGH)
    hrrec_score = normalize_range(hr_recovery_1min if hr_recovery_1min is not None else np.nan,
                                   CARDIO_HR_RECOVERY_LOW, CARDIO_HR_RECOVERY_HIGH)
    
    score = (
        CARDIO_RHR_WEIGHT * rhr_score +
        CARDIO_SDNN_WEIGHT * sdnn_score +
        CARDIO_SPO2_WEIGHT * spo2_score +
        CARDIO_HR_RECOVERY_WEIGHT * hrrec_score
    )
    
    return clamp(score, 0.0, 100.0)


def compute_biological_age(
    chronological_age: float,
    resting_hr: float,
    rmssd_baseline: float,
    sleep_quality_avg: float,
    activity_level_score: float
) -> float:
    """
    Estimate biological age based on physiological markers.
    
    WARNING: Coefficients are PROTOTYPES and must be validated
    with real cohort data before production use.
    
    Args:
        chronological_age: Actual age (years)
        resting_hr: Resting heart rate (bpm)
        rmssd_baseline: Baseline RMSSD (ms)
        sleep_quality_avg: Average sleep quality (0-100)
        activity_level_score: Activity level (0-100)
    
    Returns:
        Estimated biological age (years)
    """
    age = float(chronological_age)
    age += BIO_AGE_RHR_COEFF * (resting_hr - BIO_AGE_RHR_BASELINE)
    age += BIO_AGE_RMSSD_COEFF * (rmssd_baseline - BIO_AGE_RMSSD_BASELINE)
    age += BIO_AGE_SLEEP_COEFF * (sleep_quality_avg - BIO_AGE_SLEEP_BASELINE)
    age += BIO_AGE_ACTIVITY_COEFF * (activity_level_score - BIO_AGE_ACTIVITY_BASELINE)
    
    return float(np.round(age, 2))


def forecast_capacity_trend(
    capacity_series: Sequence[float],
    days_forward: int = FORECAST_DEFAULT_DAYS,
    fit_window: int = FORECAST_FIT_WINDOW_DAYS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast capacity trend using linear regression.
    
    Args:
        capacity_series: Historical capacity values
        days_forward: Number of days to forecast
        fit_window: Historical window for fitting (days)
    
    Returns:
        Tuple of (days_ahead, forecast_values)
    """
    arr = np.asarray(capacity_series, dtype=float)
    
    if arr.size == 0:
        return np.arange(1, days_forward + 1), np.full(days_forward, np.nan)
    
    arr = arr[-fit_window:]
    x = np.arange(arr.size)
    valid = ~np.isnan(arr)
    
    if valid.sum() < FORECAST_MIN_POINTS:
        return np.arange(1, days_forward + 1), np.full(days_forward, np.nan)
    
    a, b = np.polyfit(x[valid], arr[valid], 1)
    future_x = np.arange(arr.size, arr.size + days_forward)
    preds = a * future_x + b
    preds = np.clip(preds, 0.0, 100.0)
    
    return np.arange(1, days_forward + 1), preds
