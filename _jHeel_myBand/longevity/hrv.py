"""
Heart Rate Variability (HRV) metrics computation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Optional, Tuple
from .config import (
    RR_INTERVAL_MIN_MS,
    RR_INTERVAL_MAX_MS,
    LF_BAND_LOW,
    LF_BAND_HIGH,
    HF_BAND_LOW,
    HF_BAND_HIGH,
    HRV_RESAMPLE_FREQ,
    PNN20_THRESHOLD_MS,
    PNN50_THRESHOLD_MS,
)

# Optional scipy for frequency-domain analysis
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def rr_clean(
    rr_intervals_ms: Sequence[float],
    min_ms: int = RR_INTERVAL_MIN_MS,
    max_ms: int = RR_INTERVAL_MAX_MS,
    replace_method: str = 'interpolate'
) -> np.ndarray:
    """
    Clean RR intervals by removing artifacts.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
        min_ms: Minimum valid interval (default from config)
        max_ms: Maximum valid interval (default from config)
        replace_method: 'interpolate' or 'remove' artifacts
    
    Returns:
        Cleaned numpy array of RR intervals (ms)
    """
    rr = np.asarray(rr_intervals_ms, dtype=float).copy()
    if rr.size == 0:
        return rr
    
    mask = (rr >= min_ms) & (rr <= max_ms) & (~np.isnan(rr))
    
    if mask.all():
        return rr
    
    if replace_method == 'interpolate':
        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return np.array([])
        interp = np.interp(np.arange(rr.size), valid_idx, rr[valid_idx])
        return interp
    else:
        return rr[mask]


def compute_rmssd(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute RMSSD (Root Mean Square of Successive Differences).
    Primary time-domain HRV metric.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
    
    Returns:
        RMSSD in milliseconds (NaN if insufficient data)
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    
    diff = np.diff(rr)
    sq = diff ** 2
    return float(np.sqrt(np.mean(sq)))


def compute_sdnn(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute SDNN (Standard Deviation of NN intervals).
    Reflects overall HRV.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
    
    Returns:
        SDNN in milliseconds (NaN if insufficient data)
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    
    return float(np.std(rr, ddof=1))


def compute_pnn(rr_intervals_ms: Sequence[float], threshold_ms: int = 50) -> float:
    """
    Compute pNNx: percentage of successive differences exceeding threshold.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
        threshold_ms: Threshold for successive difference (default 50ms)
    
    Returns:
        Percentage (0-100) of intervals exceeding threshold (NaN if insufficient data)
    """
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 2:
        return float('nan')
    
    diffs = np.abs(np.diff(rr))
    count = np.sum(diffs > threshold_ms)
    return float(100.0 * count / diffs.size)


def compute_pnn20(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute pNN20: percentage of successive differences > 20ms.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
    
    Returns:
        pNN20 percentage (0-100)
    """
    return compute_pnn(rr_intervals_ms, threshold_ms=PNN20_THRESHOLD_MS)


def compute_pnn50(rr_intervals_ms: Sequence[float]) -> float:
    """
    Compute pNN50: percentage of successive differences > 50ms.
    Common HRV metric correlating with parasympathetic activity.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
    
    Returns:
        pNN50 percentage (0-100)
    """
    return compute_pnn(rr_intervals_ms, threshold_ms=PNN50_THRESHOLD_MS)


def compute_lf_hf(
    rr_intervals_ms: Sequence[float],
    fs: float = HRV_RESAMPLE_FREQ
) -> Optional[Tuple[float, float, float]]:
    """
    Compute LF and HF power spectral density and LF/HF ratio using Welch's method.
    
    Requires scipy. LF (0.04-0.15 Hz) reflects both sympathetic and parasympathetic.
    HF (0.15-0.4 Hz) reflects parasympathetic activity.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
        fs: Resampling frequency in Hz (default from config)
    
    Returns:
        Tuple of (lf_power, hf_power, lf_hf_ratio) or None if scipy unavailable
    """
    if not SCIPY_AVAILABLE:
        return None
    
    rr = rr_clean(rr_intervals_ms)
    if rr.size < 4:
        return None
    
    # Convert to seconds
    rr_s = rr / 1000.0
    
    # Build time axis (cumulative sum)
    t = np.cumsum(rr_s)
    
    # Interpolate to uniform sampling
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    
    try:
        # Convert to instantaneous heart rate
        interp_hr = 60.0 / np.interp(t_uniform, t, rr_s)
    except Exception:
        return None
    
    # Welch's power spectral density
    f, pxx = signal.welch(interp_hr, fs=fs, nperseg=min(256, interp_hr.size))
    
    # Extract frequency bands
    lf_mask = (f >= LF_BAND_LOW) & (f < LF_BAND_HIGH)
    hf_mask = (f >= HF_BAND_LOW) & (f <= HF_BAND_HIGH)
    
    # Integrate power using trapezoidal rule
    lf_power = np.trapezoid(pxx[lf_mask], f[lf_mask]) if lf_mask.any() else 0.0
    hf_power = np.trapezoid(pxx[hf_mask], f[hf_mask]) if hf_mask.any() else 0.0
    
    lf_hf = float(lf_power / hf_power) if hf_power > 0 else float('inf')
    
    return float(lf_power), float(hf_power), float(lf_hf)


def compute_hrv_summary(rr_intervals_ms: Sequence[float]) -> dict:
    """
    Compute all HRV metrics in one call.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
    
    Returns:
        Dictionary with all HRV metrics
    """
    result = {
        'rmssd': compute_rmssd(rr_intervals_ms),
        'sdnn': compute_sdnn(rr_intervals_ms),
        'pnn20': compute_pnn20(rr_intervals_ms),
        'pnn50': compute_pnn50(rr_intervals_ms),
    }
    
    # Add frequency domain if scipy available
    lf_hf_result = compute_lf_hf(rr_intervals_ms)
    if lf_hf_result is not None:
        result['lf_power'] = lf_hf_result[0]
        result['hf_power'] = lf_hf_result[1]
        result['lf_hf_ratio'] = lf_hf_result[2]
    
    return result
