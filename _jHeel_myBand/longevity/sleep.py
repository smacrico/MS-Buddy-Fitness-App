"""
Sleep metrics computation.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence
from .config import (
    SLEEP_WEIGHT_DEEP,
    SLEEP_WEIGHT_REM,
    SLEEP_WEIGHT_EFFICIENCY,
    SLEEP_WEIGHT_FRAGMENTATION,
    SLEEP_EPOCH_MINUTES,
    SLEEP_BASELINE_HOURS,
)
from .utils import clamp


def compute_sleep_efficiency(time_in_bed_min: float, total_sleep_min: float) -> float:
    """
    Compute sleep efficiency as percentage of time in bed actually spent sleeping.
    
    Args:
        time_in_bed_min: Total time in bed (minutes)
        total_sleep_min: Total time asleep (minutes)
    
    Returns:
        Sleep efficiency percentage (0-100), NaN if invalid input
    """
    if time_in_bed_min <= 0 or np.isnan(time_in_bed_min):
        return float('nan')
    
    return float(100.0 * total_sleep_min / time_in_bed_min)


def compute_sleep_quality(
    sleep_stage_series: Sequence[str],
    epoch_min: float = SLEEP_EPOCH_MINUTES
) -> float:
    """
    Compute Sleep Quality Index based on sleep stage distribution.
    
    Scoring heuristic:
    - 40% weight to deep sleep percentage
    - 30% weight to REM sleep percentage
    - 10% weight to overall sleep efficiency (non-awake %)
    - 20% penalty for sleep fragmentation (awakenings)
    
    Args:
        sleep_stage_series: Sequence of sleep stages ('awake', 'light', 'deep', 'rem')
        epoch_min: Length of each epoch in minutes (default 0.5 = 30 seconds)
    
    Returns:
        Sleep quality score (0-100)
    """
    stages = np.asarray(sleep_stage_series, dtype=object)
    if stages.size == 0:
        return float('nan')
    
    total_epochs = stages.size
    deep_count = np.sum(stages == 'deep')
    rem_count = np.sum(stages == 'rem')
    awake_count = np.sum(stages == 'awake')
    non_awake_epochs = total_epochs - awake_count
    
    # Calculate percentages
    pct_deep = deep_count / total_epochs
    pct_rem = rem_count / total_epochs
    pct_non_awake = non_awake_epochs / total_epochs
    
    # Count awakenings (transitions into 'awake' state)
    awak_transitions = np.sum((stages[:-1] != 'awake') & (stages[1:] == 'awake'))
    
    # Normalize awakenings per baseline hours
    baseline_epochs = SLEEP_BASELINE_HOURS / (epoch_min / 60.0)
    frag = awak_transitions / max(1.0, baseline_epochs / total_epochs)
    
    # Compute weighted score
    score = (
        SLEEP_WEIGHT_DEEP * 100.0 * pct_deep +
        SLEEP_WEIGHT_REM * 100.0 * pct_rem +
        SLEEP_WEIGHT_EFFICIENCY * 100.0 * pct_non_awake +
        SLEEP_WEIGHT_FRAGMENTATION * 100.0 * frag
    )
    
    return clamp(score, 0.0, 100.0)


def compute_sleep_architecture(sleep_stage_series: Sequence[str]) -> dict:
    """
    Analyze sleep architecture: distribution of sleep stages.
    
    Args:
        sleep_stage_series: Sequence of sleep stages
    
    Returns:
        Dictionary with stage percentages and counts
    """
    stages = np.asarray(sleep_stage_series, dtype=object)
    if stages.size == 0:
        return {
            'total_epochs': 0,
            'awake_pct': 0.0,
            'light_pct': 0.0,
            'deep_pct': 0.0,
            'rem_pct': 0.0,
        }
    
    total = stages.size
    
    return {
        'total_epochs': total,
        'awake_pct': float(100.0 * np.sum(stages == 'awake') / total),
        'light_pct': float(100.0 * np.sum(stages == 'light') / total),
        'deep_pct': float(100.0 * np.sum(stages == 'deep') / total),
        'rem_pct': float(100.0 * np.sum(stages == 'rem') / total),
        'awake_count': int(np.sum(stages == 'awake')),
        'light_count': int(np.sum(stages == 'light')),
        'deep_count': int(np.sum(stages == 'deep')),
        'rem_count': int(np.sum(stages == 'rem')),
    }


def compute_sleep_fragmentation(sleep_stage_series: Sequence[str]) -> dict:
    """
    Compute sleep fragmentation metrics.
    
    Args:
        sleep_stage_series: Sequence of sleep stages
    
    Returns:
        Dictionary with fragmentation metrics
    """
    stages = np.asarray(sleep_stage_series, dtype=object)
    if stages.size < 2:
        return {
            'total_transitions': 0,
            'awakenings': 0,
            'stage_transitions': 0,
        }
    
    # Count all stage transitions
    transitions = np.sum(stages[:-1] != stages[1:])
    
    # Count transitions into awake state
    awakenings = np.sum((stages[:-1] != 'awake') & (stages[1:] == 'awake'))
    
    # Count transitions between sleep stages (excluding awake)
    sleep_mask = (stages[:-1] != 'awake') & (stages[1:] != 'awake')
    stage_transitions = np.sum(sleep_mask & (stages[:-1] != stages[1:]))
    
    return {
        'total_transitions': int(transitions),
        'awakenings': int(awakenings),
        'stage_transitions': int(stage_transitions),
    }


def compute_sleep_summary(
    sleep_stage_series: Sequence[str],
    time_in_bed_min: float,
    total_sleep_min: float
) -> dict:
    """
    Compute comprehensive sleep metrics.
    
    Args:
        sleep_stage_series: Sequence of sleep stages
        time_in_bed_min: Total time in bed (minutes)
        total_sleep_min: Total time asleep (minutes)
    
    Returns:
        Dictionary with all sleep metrics
    """
    return {
        'efficiency': compute_sleep_efficiency(time_in_bed_min, total_sleep_min),
        'quality': compute_sleep_quality(sleep_stage_series),
        'architecture': compute_sleep_architecture(sleep_stage_series),
        'fragmentation': compute_sleep_fragmentation(sleep_stage_series),
        'time_in_bed_min': time_in_bed_min,
        'total_sleep_min': total_sleep_min,
    }
