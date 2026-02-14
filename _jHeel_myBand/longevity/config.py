"""
Configuration and constants for longevity metrics computation.
All magic numbers and weights are centralized here for easy tuning.
"""

# ----------------------
# RR Interval Cleaning
# ----------------------
RR_INTERVAL_MIN_MS = 300  # Minimum valid RR interval (ms)
RR_INTERVAL_MAX_MS = 2000  # Maximum valid RR interval (ms)

# ----------------------
# HRV Frequency Bands (Hz)
# ----------------------
LF_BAND_LOW = 0.04  # Low Frequency band lower bound
LF_BAND_HIGH = 0.15  # Low Frequency band upper bound
HF_BAND_LOW = 0.15  # High Frequency band lower bound
HF_BAND_HIGH = 0.4  # High Frequency band upper bound
HRV_RESAMPLE_FREQ = 4.0  # Resampling frequency for HRV analysis (Hz)

# ----------------------
# pNN Thresholds
# ----------------------
PNN20_THRESHOLD_MS = 20  # Threshold for pNN20 calculation
PNN50_THRESHOLD_MS = 50  # Threshold for pNN50 calculation

# ----------------------
# Sleep Quality Weights
# ----------------------
# Based on sleep research indicating importance of each stage
SLEEP_WEIGHT_DEEP = 0.40  # Weight for deep sleep percentage
SLEEP_WEIGHT_REM = 0.30  # Weight for REM sleep percentage
SLEEP_WEIGHT_EFFICIENCY = 0.10  # Weight for sleep efficiency
SLEEP_WEIGHT_FRAGMENTATION = -0.20  # Penalty for sleep fragmentation

# Sleep stage fragmentation normalization
SLEEP_EPOCH_MINUTES = 0.5  # Default epoch length (30 seconds)
SLEEP_BASELINE_HOURS = 8.0  # Baseline sleep duration for normalization

# ----------------------
# TRIMP Activity Load
# ----------------------
TRIMP_MULTIPLIER_MALE = 1.92  # Sex-specific multiplier for males
TRIMP_MULTIPLIER_FEMALE = 1.67  # Sex-specific multiplier for females

# ----------------------
# Recovery Score Weights
# ----------------------
# Weights for recovery score calculation (must sum to reasonable range)
RECOVERY_HRV_WEIGHT = 12.5  # Z-score to points conversion for HRV
RECOVERY_SLEEP_WEIGHT = 0.3  # Weight for sleep quality (0-100 scale)
RECOVERY_STRAIN_WEIGHT = -12.5  # Penalty for high strain (z-score)
RECOVERY_BASE_SCORE = 50.0  # Baseline recovery score

# ----------------------
# Metabolic Capacity Weights
# ----------------------
CAPACITY_HRV_WEIGHT = 25.0  # Points contribution from HRV z-score
CAPACITY_RHR_WEIGHT = -20.0  # Points contribution from resting HR z-score (negative)
CAPACITY_SLEEP_SCALE = 0.15  # Scale factor for sleep quality (0-100 -> 0-15)
CAPACITY_DEBT_SCALE = -10.0  # Scale factor for recovery debt penalty
CAPACITY_DEBT_BASELINE = 200.0  # Typical maximum recovery debt
CAPACITY_BASE_SCORE = 50.0  # Baseline capacity score

# ----------------------
# Recovery Debt
# ----------------------
RECOVERY_DEBT_WINDOW_DAYS = 7  # Rolling window for recovery debt calculation

# ----------------------
# Metabolic Momentum
# ----------------------
MOMENTUM_WINDOW_DAYS = 28  # Rolling window for momentum calculation (4 weeks)
MOMENTUM_MIN_POINTS = 3  # Minimum data points for trend calculation

# ----------------------
# Cardiovascular Health Weights
# ----------------------
CARDIO_RHR_WEIGHT = 0.30  # Weight for resting heart rate
CARDIO_SDNN_WEIGHT = 0.40  # Weight for SDNN (HRV measure)
CARDIO_SPO2_WEIGHT = 0.15  # Weight for SpO2
CARDIO_HR_RECOVERY_WEIGHT = 0.15  # Weight for HR recovery

# Cardiovascular Health - Expected Ranges
CARDIO_RHR_BEST = 50  # Best resting HR (bpm)
CARDIO_RHR_WORST = 90  # Worst resting HR (bpm)
CARDIO_SDNN_LOW = 20  # Low SDNN (ms)
CARDIO_SDNN_HIGH = 100  # High SDNN (ms)
CARDIO_SPO2_LOW = 94  # Low SpO2 (%)
CARDIO_SPO2_HIGH = 99  # High SpO2 (%)
CARDIO_HR_RECOVERY_LOW = 10  # Low HR recovery (bpm)
CARDIO_HR_RECOVERY_HIGH = 40  # High HR recovery (bpm)

# ----------------------
# Biological Age Coefficients
# ----------------------
# WARNING: These are PLACEHOLDER coefficients for demonstration only
# Must be retrained with real cohort data for production use
BIO_AGE_RHR_COEFF = 0.2  # Each +5 bpm adds ~1 year
BIO_AGE_RHR_BASELINE = 60  # Baseline resting HR (bpm)
BIO_AGE_RMSSD_COEFF = -0.05  # Each +20ms RMSSD subtracts ~1 year
BIO_AGE_RMSSD_BASELINE = 30  # Baseline RMSSD (ms)
BIO_AGE_SLEEP_COEFF = -0.02  # Sleep quality contribution
BIO_AGE_SLEEP_BASELINE = 70  # Baseline sleep quality score
BIO_AGE_ACTIVITY_COEFF = -0.03  # Activity level contribution
BIO_AGE_ACTIVITY_BASELINE = 50  # Baseline activity score

# ----------------------
# Forecasting
# ----------------------
FORECAST_DEFAULT_DAYS = 14  # Default forecast horizon (days)
FORECAST_FIT_WINDOW_DAYS = 60  # Historical data window for trend fitting
FORECAST_MIN_POINTS = 2  # Minimum points needed for forecast

# ----------------------
# Baseline Calculation
# ----------------------
BASELINE_WINDOW_DAYS = 14  # Rolling window for baseline statistics

# ----------------------
# Data Validation
# ----------------------
HEART_RATE_MIN_BPM = 30  # Minimum valid heart rate
HEART_RATE_MAX_BPM = 220  # Maximum valid heart rate
SPO2_MIN_PERCENT = 70  # Minimum valid SpO2
SPO2_MAX_PERCENT = 100  # Maximum valid SpO2

# ----------------------
# Score Ranges
# ----------------------
SCORE_MIN = 0.0  # Minimum score value
SCORE_MAX = 100.0  # Maximum score value
