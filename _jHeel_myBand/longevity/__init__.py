"""
Longevity Metrics - Health and wellness analytics package.

This package provides tools for computing metabolic and longevity metrics
from wearable device data including HRV, sleep, and activity measurements.
"""

__version__ = "1.0.0"
__author__ = "MS-Buddy-Fitness-App"

from . import config
from . import utils
from . import hrv
from . import sleep
from . import activity
from . import core
from . import visualizations
from . import export

# Convenience imports
from .hrv import (
    rr_clean,
    compute_rmssd,
    compute_sdnn,
    compute_pnn20,
    compute_pnn50,
    compute_lf_hf
)

from .sleep import (
    compute_sleep_efficiency,
    compute_sleep_quality
)

from .activity import (
    compute_activity_trimp_simple
)

from .core import (
    compute_recovery_score,
    compute_recovery_debt,
    compute_metabolic_capacity,
    compute_metabolic_momentum,
    compute_cardiovascular_health,
    compute_biological_age,
    forecast_capacity_trend
)

__all__ = [
    'config',
    'utils',
    'hrv',
    'sleep',
    'activity',
    'core',
    'visualizations',
    'export',
    # HRV functions
    'rr_clean',
    'compute_rmssd',
    'compute_sdnn',
    'compute_pnn20',
    'compute_pnn50',
    'compute_lf_hf',
    # Sleep functions
    'compute_sleep_efficiency',
    'compute_sleep_quality',
    # Activity functions
    'compute_activity_trimp_simple',
    # Core functions
    'compute_recovery_score',
    'compute_recovery_debt',
    'compute_metabolic_capacity',
    'compute_metabolic_momentum',
    'compute_cardiovascular_health',
    'compute_biological_age',
    'forecast_capacity_trend',
]
