# metabolic_metrics.py
"""
Metabolic and Longevity Metrics Computation Module

Provides working baseline implementations for:
- HRV metrics (RMSSD, SDNN, pNN20, pNN50, LF/HF)
- Sleep metrics (sleep efficiency, sleep quality index)
- Activity load (simple TRIMP-like)
- Recovery score and recovery debt
- Composite metrics: metabolic capacity, metabolic momentum,
  cardiovascular health score, biological age (simple)
- Simple linear forecast for metabolic trajectory

Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-10-17
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Sequence, Optional, Tuple, Dict, Any, Union
import json
import csv
import ast  # for safely evaluating string lists
from pathlib import Path
import logging
import sys

# Optional: frequency-domain HRV needs scipy.signal
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# Add logging configuration
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------
# --- Helper methods ---
# ----------------------
def zscore(value: float, baseline_mean: float, baseline_std: float) -> float:
    """Return z-score; if std is zero or NaN, return 0."""
    if baseline_std is None or baseline_std == 0 or np.isnan(baseline_std):
        return 0.0
    return (value - baseline_mean) / baseline_std


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def rolling_baseline(series: Sequence[float], window: int = 14) -> Tuple[float, float]:
    """Compute baseline mean and std from recent window values (ignores NaNs)."""
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


# ----------------------
# --- HRV metrics ---
# ----------------------
def rr_clean(rr_intervals_ms: Sequence[float],
             min_ms: int = 300,
             max_ms: int = 2000,
             replace_method: str = 'interpolate') -> np.ndarray:
    """
    Basic artifact removal and cleaning for R-R intervals.
    - Removes intervals outside [min_ms, max_ms] as artifacts.
    - If replace_method == 'interpolate', linear interpolate removed values.
    Returns cleaned numpy array of RR intervals (ms).
    """
    rr = np.asarray(rr_intervals_ms, dtype=float).copy()
    if rr.size == 0:
        return rr
    mask = (rr >= min_ms) & (rr <= max_ms) & (~np.isnan(rr))
    if mask.all():
        return rr
    if replace_method == 'interpolate':
        # Simple linear interpolation over invalid points
        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return np.array([])
        interp = np.interp(np.arange(rr.size), valid_idx, rr[valid_idx])
        return interp
    else:
        return rr[mask]


def compute_rmssd(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute RMSSD (ms) from R-R intervals (ms).
    Requires cleaned R-R for best accuracy.
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    diff = np.diff(rr)
    sq = diff ** 2
    return float(np.sqrt(np.mean(sq)))


def compute_sdnn(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute SDNN (ms): standard deviation of NN intervals.
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    return float(np.std(rr, ddof=1))


def compute_pnn(rr_intervals_ms: Sequence[float], threshold_ms: int = 50) -> float:
    """
    Compute pNNx: percentage of successive differences exceeding threshold_ms.
    Returns value in % (0-100).
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    diffs = np.abs(np.diff(rr))
    count = np.sum(diffs > threshold_ms)
    return float(100.0 * count / diffs.size)


def compute_pnn20(rr_intervals_ms: Sequence[float]) -> float:
    return compute_pnn(rr_intervals_ms, threshold_ms=20)


def compute_pnn50(rr_intervals_ms: Sequence[float]) -> float:
    return compute_pnn(rr_intervals_ms, threshold_ms=50)


def compute_lf_hf(rr_intervals_ms: Sequence[float], fs: float = 4.0) -> Optional[Tuple[float, float, float]]:
    """
    Approximate LF and HF band power and LF/HF ratio using Welch PSD.
    rr_intervals_ms: R-R intervals in ms.
    fs: resampling frequency for interpolation (Hz). Typical 4 Hz is fine.
    Returns (lf_power, hf_power, lf_hf_ratio) or None if scipy not available.
    Bands used: LF 0.04-0.15 Hz, HF 0.15-0.4 Hz
    """
    if not SCIPY_AVAILABLE:
        return None
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 4:
        return None
    # Convert to seconds and compute time series
    rr_s = rr / 1000.0
    # Build time axis assuming successive beats
    t = np.cumsum(rr_s)
    # Interpolate to even sampling
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    try:
        interp_hr = 60.0 / np.interp(t_uniform, t, rr_s)  # convert to instantaneous HR
    except Exception:
        return None
    f, pxx = signal.welch(interp_hr, fs=fs, nperseg=min(256, interp_hr.size))
    # Frequency bands
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f <= 0.4)
    # Use trapezoid instead of deprecated trapz
    lf_power = np.trapezoid(pxx[lf_mask], f[lf_mask]) if lf_mask.any() else 0.0
    hf_power = np.trapezoid(pxx[hf_mask], f[hf_mask]) if hf_mask.any() else 0.0
    lf_hf = float(lf_power / hf_power) if hf_power > 0 else float('inf')
    return float(lf_power), float(hf_power), float(lf_hf)


# ----------------------
# --- Sleep metrics ---
# ----------------------
def compute_sleep_efficiency(time_in_bed_min: float, total_sleep_min: float) -> float:
    """
    Sleep efficiency = total_sleep_time / time_in_bed * 100
    Inputs in minutes. Returns percent (0-100).
    """
    if time_in_bed_min <= 0 or np.isnan(time_in_bed_min):
        return float('nan')
    return float(100.0 * total_sleep_min / time_in_bed_min)


def compute_sleep_quality(sleep_stage_series: Sequence[str],
                          epoch_min: float = 0.5) -> float:
    """
    Compute a simple Sleep Quality Index based on proportion deep & REM vs fragmentation.
    - sleep_stage_series: sequence of labels per epoch: 'awake', 'light', 'deep', 'rem'
      epoch_min: length of each epoch in minutes (default 30s=0.5min)
    Returns a score 0-100 (higher is better).
    Heuristic:
      + 40% weight to %deep
      + 30% weight to %REM
      + -20% fragmentation (awakenings per night normalized)
      + 10% sleep efficiency proxy (derived from non-awake percent)
    Note: This is a heuristic baseline — tune weights with data.
    """
    stages = np.asarray(sleep_stage_series, dtype=object)
    if stages.size == 0:
        return float('nan')
    total_epochs = stages.size
    deep = np.sum(stages == 'deep')
    rem = np.sum(stages == 'rem')
    awake = np.sum(stages == 'awake')
    non_awake_epochs = total_epochs - awake
    pct_deep = deep / total_epochs
    pct_rem = rem / total_epochs
    pct_non_awake = non_awake_epochs / total_epochs
    # simple awakening count: transitions into 'awake'
    awak_transitions = np.sum((stages[:-1] != 'awake') & (stages[1:] == 'awake'))
    # normalize awakenings per 8 hours baseline (16 awakenings -> high fragmentation)
    frag = awak_transitions / max(1.0, (8.0 / (epoch_min / 60.0)))  # rough normalization
    score = (40.0 * pct_deep + 30.0 * pct_rem + 10.0 * pct_non_awake * 100.0 - 20.0 * frag)
    return clamp(score, 0.0, 100.0)


# ----------------------
# --- Activity / Strain ---
# ----------------------
def compute_activity_trimp_simple(hr_series_bpm: Sequence[float],
                                  duration_min: float,
                                  hr_rest: float,
                                  hr_max: float,
                                  multiplier_male: float = 1.92,
                                  multiplier_female: float = 1.67,
                                  sex: str = 'male') -> float:
    """
    Very simple TRIMP-like training load estimator:
      TRIMP = duration_min * avg_intensity * sex_multiplier
      where avg_intensity = (avgHR - hr_rest) / (hr_max - hr_rest)
    Returns a non-negative float. This is a baseline proxy for 'strain'.
    Note: more sophisticated TRIMP uses HR zones, lactate thresholds, or session-RPE.
    """
    hr = np.asarray(hr_series_bpm, dtype=float)
    if hr.size == 0 or duration_min <= 0:
        return 0.0
    avg_hr = float(np.nanmean(hr))
    denom = hr_max - hr_rest if hr_max > hr_rest else 1.0
    intensity = max(0.0, (avg_hr - hr_rest) / denom)
    mult = multiplier_male if sex.lower().startswith('m') else multiplier_female
    return float(duration_min * intensity * mult)


# ----------------------
# --- Recovery / composites ---
# ----------------------
def compute_recovery_score(rmssd_today: float,
                           rmssd_baseline_mean: float,
                           rmssd_baseline_std: float,
                           sleep_quality_score: float,
                           strain_today: float,
                           baseline_strain_mean: float,
                           baseline_strain_std: float) -> float:
    """
    Compute a composite recovery score (0-100).
    Combines RMSSD relative to baseline (positive), sleep quality (positive),
    and penalizes high strain relative to baseline.
    Heuristic weights: 50% RMSSD zscore, 30% sleep_quality (0-100 -> 0-1), -20% strain zscore.
    """
    z_hrv = zscore(rmssd_today, rmssd_baseline_mean, rmssd_baseline_std)
    z_strain = zscore(strain_today, baseline_strain_mean, baseline_strain_std)
    # Map z_hrv to a 0-100-ish contribution using logistic-ish or linear scaling
    # Here simple linear: multiply z by 12.5 to map typical z ranges to +-25 points
    hrv_contrib = 12.5 * z_hrv  # tuned scale
    sleep_contrib = 0.3 * (sleep_quality_score)  # sleep_quality_score in 0-100
    strain_contrib = -12.5 * z_strain
    base = 50.0
    score = base + hrv_contrib + sleep_contrib + strain_contrib
    return clamp(score, 0.0, 100.0)


def compute_recovery_debt(strain_series: Sequence[float],
                          recovery_credit_series: Sequence[float],
                          window_days: int = 7) -> float:
    """
    Compute recovery debt over a rolling window:
    debt_t = max(0, sum(strain) - sum(recovery_credit))
    Returns the current debt (higher is worse).
    Intended inputs: daily arrays of strain and recovery credit (same length).
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


def compute_metabolic_capacity(rmssd_today: float,
                               rmssd_baseline_mean: float,
                               rmssd_baseline_std: float,
                               rhr_today: float,
                               rhr_baseline_mean: float,
                               rhr_baseline_std: float,
                               sleep_quality_score: float,
                               recovery_debt: float) -> float:
    """
    Prototype composite Metabolic Capacity in 0-100.
    Inputs:
      - RMSSD today and baseline stats
      - Resting HR today and baseline stats
      - Sleep quality score (0-100)
      - recovery_debt (positive number; larger is worse)
    Heuristic weightings (tunable):
      +25 for HRV (positive)
      -20 for resting HR (negative)
      +15 for sleep quality (scaled 0-100 -> 0-15)
      -10 for recovery_debt (scaled reasonably)
    """
    z_hrv = zscore(rmssd_today, rmssd_baseline_mean, rmssd_baseline_std)
    z_rhr = zscore(rhr_today, rhr_baseline_mean, rhr_baseline_std)
    # Map zscores to points
    hrv_pts = 25.0 * z_hrv
    rhr_pts = -20.0 * z_rhr
    sleep_pts = 0.15 * sleep_quality_score  # maps 0-100 -> 0-15
    # Scale recovery debt - assume typical debts in range 0-200; scale to -10..0
    debt_scale = -10.0 * (recovery_debt / max(1.0, (200.0)))
    raw_score = 50.0 + hrv_pts + rhr_pts + sleep_pts + debt_scale
    return clamp(raw_score, 0.0, 100.0)


def compute_metabolic_momentum(capacity_series: Sequence[float],
                               window_days: int = 28) -> float:
    """
    Compute momentum as slope (capacity points per week) of last window_days.
    Returns slope in capacity-points/week (positive = improving).
    Uses simple linear regression (numpy.polyfit).
    """
    arr = np.asarray(capacity_series, dtype=float)
    if arr.size < 3:
        return 0.0
    arr = arr[-window_days:]
    x = np.arange(arr.size)
    # remove NaNs
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return 0.0
    coeffs = np.polyfit(x[valid], arr[valid], 1)
    slope_per_day = coeffs[0]
    slope_per_week = slope_per_day * 7.0
    return float(slope_per_week)


def compute_cardiovascular_health(resting_hr: float,
                                  sdnn_24h: float,
                                  spo2: Optional[float],
                                  hr_recovery_1min: Optional[float]) -> float:
    """
    Simple cardiovascular health composite (0-100).
    Components (example weights):
      - resting HR: lower is better (30%)
      - SDNN: higher is better (40%)
      - SpO2 baseline (if provided): higher is better (15%)
      - HR recovery after 1 min (if provided): higher is better (15%)
    This is a heuristic baseline; normalize components with expected ranges.
    """
    # expected ranges and normalization helpers
    def norm_inv(x, low, high):
        # maps [low,high] to [100,0]
        if np.isnan(x):
            return 50.0
        x = float(x)
        if x <= low:
            return 100.0
        if x >= high:
            return 0.0
        return 100.0 * (1.0 - (x - low) / (high - low))
    def norm(x, low, high):
        if np.isnan(x):
            return 50.0
        x = float(x)
        if x <= low:
            return 0.0
        if x >= high:
            return 100.0
        return 100.0 * ((x - low) / (high - low))

    rhr_score = norm_inv(resting_hr, 50, 90)  # resting HR: 50 best (100 pts), 90 worst (0)
    sdnn_score = norm(sdnn_24h, 20, 100)      # SDNN: 20 low, 100 high
    spo2_score = norm(spo2 if spo2 is not None else np.nan, 94, 99)
    hrrec_score = norm(hr_recovery_1min if hr_recovery_1min is not None else np.nan, 10, 40)
    score = 0.3 * rhr_score + 0.4 * sdnn_score + 0.15 * spo2_score + 0.15 * hrrec_score
    return clamp(score, 0.0, 100.0)


# ----------------------
# --- Biological age (prototype) ---
# ----------------------
def compute_biological_age(chronological_age: float,
                           resting_hr: float,
                           rmssd_baseline: float,
                           sleep_quality_avg: float,
                           activity_level_score: float) -> float:
    """
    Prototype physiological age estimate.
    This simple linear model converts biomarkers into an 'apparent age' relative to chronological_age.
    Coefficients are illustrative and MUST be retrained with real cohort data for production use.

    Inputs:
      - chronological_age: years
      - resting_hr: bpm
      - rmssd_baseline: ms
      - sleep_quality_avg: 0-100
      - activity_level_score: 0-100 (higher = more active)
    Returns estimated biological age in years.
    """
    # Coefficients (toy example): baseline = chrono age,
    # add penalty for high RHR, add benefit for higher RMSSD, sleep, activity.
    # These numbers are arbitrary starting points.
    age = float(chronological_age)
    age += 0.2 * (resting_hr - 60)           # each +5 bpm ~ +1 year
    age -= 0.05 * (rmssd_baseline - 30)     # each +20ms ~ -1 year
    age -= 0.02 * (sleep_quality_avg - 70)  # each +5 sleep -> -0.1 year
    age -= 0.03 * (activity_level_score - 50)
    return float(np.round(age, 2))


# ----------------------
# --- Forecast / Trajectory ---
# ----------------------
def forecast_capacity_trend(capacity_series: Sequence[float],
                            days_forward: int = 14,
                            fit_window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple linear trend forecast of capacity.
    Returns tuple (dates_relative, forecast_values) where dates_relative is
    array [1..days_forward] (days ahead) and forecast_values are predicted capacity.
    Uses linear regression on the last fit_window days.
    """
    arr = np.asarray(capacity_series, dtype=float)
    if arr.size == 0:
        return np.arange(1, days_forward + 1), np.full(days_forward, np.nan)
    arr = arr[-fit_window:]
    x = np.arange(arr.size)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return np.arange(1, days_forward + 1), np.full(days_forward, np.nan)
    a, b = np.polyfit(x[valid], arr[valid], 1)
    future_x = np.arange(arr.size, arr.size + days_forward)
    preds = a * future_x + b
    preds = np.clip(preds, 0.0, 100.0)
    return np.arange(1, days_forward + 1), preds


# ----------------------
# --- Data Import Functions ---
# ----------------------
def import_csv_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Import health data from CSV file.
    Expected columns: timestamp, heart_rate, rr_intervals, sleep_stage, etc.
    Returns pandas DataFrame with standardized column names.
    """
    df = pd.read_csv(filepath)
    # Validate required columns
    required_cols = ['timestamp']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def import_json_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Import health data from JSON file.
    Expected format: {
        "user_info": {...},
        "measurements": [{
            "timestamp": "ISO8601",
            "heart_rate": float,
            "rr_intervals": [float, ...],
            "sleep_stage": str,
            ...
        }]
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate basic structure
    if not isinstance(data, dict):
        raise ValueError("JSON must contain a root object")
    if "measurements" not in data:
        raise ValueError("JSON must contain 'measurements' array")
    
    return data

def preprocess_health_data(data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Standardize and clean imported health data.
    - Handles both CSV DataFrame and JSON dict inputs
    - Normalizes column names
    - Removes duplicates
    - Sorts by timestamp
    - Validates data types
    Returns cleaned DataFrame
    """
    if isinstance(data, dict):
        # Convert JSON format to DataFrame
        df = pd.DataFrame(data['measurements'])
    else:
        df = data.copy()
    
    # Standardize column names
    column_mapping = {
        'heart_rate': 'heart_rate',
        'hr': 'heart_rate',
        'rr': 'rr_intervals',
        'rr_intervals': 'rr_intervals',
        'sleep': 'sleep_stage',
        'sleep_stage': 'sleep_stage'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Sort and remove duplicates
    if 'timestamp' in df.columns:
        df.sort_values('timestamp', inplace=True)
        df.drop_duplicates(subset=['timestamp'], inplace=True)
    
    return df

def load_demo_data(filepath: Union[str, Path] = 'demo.csv') -> pd.DataFrame:
    """
    Load and preprocess the demo dataset with error handling.
    """
    try:
        script_dir = Path(__file__).parent
        demo_file = script_dir / filepath
        
        if not demo_file.exists():
            raise FileNotFoundError(f"Demo file not found at {demo_file}")
        
        df = pd.read_csv(demo_file)
        if df.empty:
            raise ValueError("Demo file is empty")
            
        # Convert timestamp to datetime first
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Now we can use datetime accessors
        logging.info(f"Raw records loaded: {len(df)}")
        logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logging.info(f"Unique dates: {df['timestamp'].dt.date.nunique()}")
        logging.info(f"Records per day: {len(df) / df['timestamp'].dt.date.nunique():.1f}")
        
        # Safely convert RR intervals from string to list
        def safe_eval(x):
            try:
                if pd.isna(x):
                    return []
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                logging.warning(f"Failed to parse RR interval: {x}")
                return []
                
        df['rr_intervals'] = df['rr_intervals'].apply(safe_eval)
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading demo data: {str(e)}")
        raise


# ----------------------
# --- Example / CLI ---
# ----------------------
if __name__ == "__main__":
    try:
        print("=== metabolic_metrics.py demo with extended dataset ===")
        
        # Load demo data
        demo_df = load_demo_data()
        logging.info(f"Loaded {len(demo_df)} records from demo dataset")
        
        if demo_df.empty:
            raise ValueError("No data available for processing")
        
        # Process one day of data as example
        day_data = demo_df[demo_df['timestamp'].dt.date == demo_df['timestamp'].dt.date.min()]
        if day_data.empty:
            raise ValueError("No data available for the first day")
            
        # Calculate daily metrics
        rr_intervals = [x for x in day_data['rr_intervals'] if x]  # Filter empty lists
        if not rr_intervals:
            raise ValueError("No valid RR intervals found")
            
        daily_rmssd = compute_rmssd(np.concatenate(rr_intervals))
        daily_sdnn = compute_sdnn(np.concatenate(rr_intervals))
        
        # Get sleep stages for the night
        night_data = day_data[day_data['sleep_stage'] != 'awake']
        if not night_data.empty:
            sleep_quality = compute_sleep_quality(night_data['sleep_stage'].tolist())
            sleep_eff = compute_sleep_efficiency(
                night_data['time_in_bed_min'].iloc[0],
                night_data['total_sleep_min'].iloc[0]
            )
        else:
            logging.warning("No sleep data available")
            sleep_quality = float('nan')
            sleep_eff = float('nan')
        
        print(f"\nDaily Metrics:")
        print(f"RMSSD: {daily_rmssd:.2f} ms")
        print(f"SDNN: {daily_sdnn:.2f} ms")
        print(f"Sleep Quality: {sleep_quality:.1f}")
        print(f"Sleep Efficiency: {sleep_eff:.1f}%")
        
        # Calculate recovery and capacity scores
        rmssd_baseline = compute_rmssd(np.concatenate(demo_df['rr_intervals'].tolist()))
        rhr_today = day_data['heart_rate'].min()
        rhr_baseline = demo_df['heart_rate'].min()
        
        capacity = compute_metabolic_capacity(
            daily_rmssd,
            rmssd_baseline,
            5.0,  # example baseline std
            rhr_today,
            rhr_baseline,
            3.0,  # example baseline std
            sleep_quality,
            0.0   # example recovery debt
        )
        
        print(f"\nMetabolic Capacity Score: {capacity:.1f}")

        # Demo with synthetic data to show outputs
        print("=== metabolic_metrics.py demo ===")
        # Synthetic RR intervals (ms) approximating a calm subject
        synthetic_rr = np.random.normal(loc=850, scale=30, size=300)  # ~70 bpm
        rmssd_val = compute_rmssd(synthetic_rr)
        sdnn_val = compute_sdnn(synthetic_rr)
        pnn20_val = compute_pnn20(synthetic_rr)
        pnn50_val = compute_pnn50(synthetic_rr)
        print(f"RMSSD: {rmssd_val:.2f} ms, SDNN: {sdnn_val:.2f} ms, pNN20: {pnn20_val:.2f}%, pNN50: {pnn50_val:.2f}%")

        if SCIPY_AVAILABLE:
            lf, hf, lf_hf_ratio = compute_lf_hf(synthetic_rr)
            print(f"LF: {lf:.4f}, HF: {hf:.4f}, LF/HF: {lf_hf_ratio:.2f}")
        else:
            print("scipy not available — skipping LF/HF")

        # Synthetic sleep: 8 hours, epochs of 30s -> 960 epochs
        epochs = 8 * 60 * 2
        stages = np.random.choice(['light', 'deep', 'rem', 'awake'], size=epochs, p=[0.55, 0.15, 0.25, 0.05])
        sleep_q = compute_sleep_quality(stages)
        eff = compute_sleep_efficiency(8 * 60, 7.5 * 60)
        print(f"Sleep quality score: {sleep_q:.1f}, Sleep efficiency: {eff:.1f}%")

        # Activity strain example
        hr_series = np.random.normal(loc=150, scale=10, size=30)  # 30-samples during activity
        trimp = compute_activity_trimp_simple(hr_series, duration_min=30, hr_rest=55, hr_max=190, sex='male')
        print(f"Example TRIMP-like strain: {trimp:.2f}")

        # Recovery & capacity demo using synthetic baselines
        rmssd_baseline_mean, rmssd_baseline_std = 35.0, 8.0
        rhr_baseline_mean, rhr_baseline_std = 58.0, 4.0
        rhr_today = 60.0
        recovery_score = compute_recovery_score(rmssd_val, rmssd_baseline_mean, rmssd_baseline_std,
                                                sleep_q, trimp, baseline_strain_mean=20.0, baseline_strain_std=5.0)
        recovery_debt = compute_recovery_debt([trimp] * 7, [recovery_score] * 7)
        capacity = compute_metabolic_capacity(rmssd_val, rmssd_baseline_mean, rmssd_baseline_std,
                                              rhr_today, rhr_baseline_mean, rhr_baseline_std,
                                              sleep_q, recovery_debt)
        print(f"Recovery score: {recovery_score:.2f}, Recovery debt: {recovery_debt:.2f}, Capacity: {capacity:.2f}")

        # Momentum & forecast
        capacity_series = np.clip(50 + np.cumsum(np.random.normal(loc=0.1, scale=0.5, size=90)), 0, 100)
        momentum = compute_metabolic_momentum(capacity_series)
        days, preds = forecast_capacity_trend(capacity_series, days_forward=14)
        print(f"Momentum (capacity points/week): {momentum:.3f}")
        print(f"Forecast next 14 days (sample): {preds[:5]}")

        # Cardiovascular & biological age demo
        cardio = compute_cardiovascular_health(resting_hr=60, sdnn_24h=70, spo2=97.0, hr_recovery_1min=25.0)
        bio_age = compute_biological_age(chronological_age=40, resting_hr=60, rmssd_baseline=rmssd_baseline_mean,
                                         sleep_quality_avg=80.0, activity_level_score=60.0)
        print(f"Cardio health score: {cardio:.2f}, Biological age estimate: {bio_age:.2f} years")

        # Data Import Example
        print("\n=== Data Import Example ===")
        # Example with synthetic data saved to CSV
        example_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'heart_rate': np.random.normal(70, 5, 100),
            'rr_intervals': [list(np.random.normal(850, 30, 5)) for _ in range(100)]
        })
        
        # Save and reload to demonstrate import
        temp_csv = 'example_health_data.csv'
        example_df.to_csv(temp_csv, index=False)
        
        try:
            loaded_df = import_csv_data(temp_csv)
            processed_df = preprocess_health_data(loaded_df)
            print(f"Successfully loaded {len(processed_df)} records")
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    except FileNotFoundError as e:
        logging.error(f"Demo file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
