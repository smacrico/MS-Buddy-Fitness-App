# Metabolic and Longevity Metrics Computation Module

A comprehensive Python module for computing metabolic, cardiovascular, and longevity metrics from health and fitness data. This module processes heart rate variability (HRV), sleep patterns, activity levels, and recovery data to provide actionable health insights.

## Overview

This module provides working baseline implementations for computing various health metrics from wearable device data, including:

- **HRV Metrics**: RMSSD, SDNN, pNN20, pNN50, LF/HF ratio
- **Sleep Metrics**: Sleep efficiency, sleep quality index
- **Activity & Strain**: TRIMP-like training load calculation
- **Recovery Metrics**: Recovery score and recovery debt tracking
- **Composite Scores**: Metabolic capacity, metabolic momentum, cardiovascular health
- **Longevity Indicators**: Biological age estimation
- **Forecasting**: Linear trend prediction for metabolic trajectory

## Features

### 1. Heart Rate Variability (HRV) Analysis

**Time-Domain Metrics:**
- `compute_rmssd()` - Root Mean Square of Successive Differences
- `compute_sdnn()` - Standard Deviation of NN intervals
- `compute_pnn20()` - Percentage of successive differences > 20ms
- `compute_pnn50()` - Percentage of successive differences > 50ms

**Frequency-Domain Metrics:**
- `compute_lf_hf()` - Low Frequency/High Frequency power ratio using Welch PSD

**Data Quality:**
- `rr_clean()` - Artifact removal and interpolation for R-R intervals

### 2. Sleep Quality Assessment

- `compute_sleep_efficiency()` - Total sleep time / time in bed ratio
- `compute_sleep_quality()` - Composite score based on:
  - Deep sleep percentage (40% weight)
  - REM sleep percentage (30% weight)
  - Sleep fragmentation (-20% weight)
  - Overall sleep efficiency (10% weight)

### 3. Activity & Training Load

- `compute_activity_trimp_simple()` - Training Impulse calculation
  - Accounts for heart rate intensity
  - Sex-specific multipliers
  - Duration-weighted scoring

### 4. Recovery & Capacity Metrics

**Recovery Assessment:**
- `compute_recovery_score()` - Composite recovery score (0-100)
  - HRV relative to baseline (50% weight)
  - Sleep quality (30% weight)
  - Strain penalty (-20% weight)

**Recovery Debt:**
- `compute_recovery_debt()` - Cumulative strain vs recovery balance over rolling window

**Metabolic Capacity:**
- `compute_metabolic_capacity()` - Overall readiness score (0-100)
  - HRV contribution (+25 points max)
  - Resting heart rate (-20 points max)
  - Sleep quality (+15 points max)
  - Recovery debt penalty (-10 points max)

**Momentum:**
- `compute_metabolic_momentum()` - Rate of capacity improvement (points/week)

### 5. Cardiovascular Health

- `compute_cardiovascular_health()` - Multi-factor cardiovascular score (0-100)
  - Resting heart rate (30% weight)
  - 24-hour SDNN (40% weight)
  - SpO2 levels (15% weight)
  - HR recovery post-exercise (15% weight)

### 6. Biological Age Estimation

- `compute_biological_age()` - Prototype physiological age estimate
  - Based on resting HR, HRV, sleep quality, and activity level
  - **Note**: Coefficients are illustrative; requires calibration with real cohort data

### 7. Forecasting

- `forecast_capacity_trend()` - Linear regression forecast
  - Predicts metabolic capacity trajectory
  - Configurable forecast window (default 14 days)
  - Based on rolling 60-day trend

### 8. Data Import & Processing

**Import Functions:**
- `import_csv_data()` - Load health data from CSV files
- `import_json_data()` - Load health data from JSON files
- `preprocess_health_data()` - Standardize and clean imported data
- `load_demo_data()` - Load and validate demo dataset

**Features:**
- Automatic timestamp parsing
- Column name standardization
- Duplicate removal
- Data validation
- Safe RR interval parsing with `ast.literal_eval`

## Installation

### Requirements

```bash
pip install numpy pandas scipy
```

**Optional Dependencies:**
- `scipy` - Required for frequency-domain HRV analysis (LF/HF ratio)

### Python Version

- Python 3.7+ (uses `from __future__ import annotations`)

## Usage

### Basic Example

```python
from Longevity import *
import numpy as np

# Calculate HRV from R-R intervals
rr_intervals = [850, 820, 840, 860, 830, 845]  # milliseconds
rmssd = compute_rmssd(rr_intervals)
sdnn = compute_sdnn(rr_intervals)
pnn50 = compute_pnn50(rr_intervals)

print(f"RMSSD: {rmssd:.2f} ms")
print(f"SDNN: {sdnn:.2f} ms")
print(f"pNN50: {pnn50:.2f}%")
```

### Sleep Quality Analysis

```python
# Sleep stages from overnight recording (30-second epochs)
sleep_stages = ['light', 'light', 'deep', 'deep', 'rem', 'rem', 'awake', 'light']

sleep_quality = compute_sleep_quality(sleep_stages, epoch_min=0.5)
sleep_efficiency = compute_sleep_efficiency(
    time_in_bed_min=480,  # 8 hours
    total_sleep_min=420    # 7 hours
)

print(f"Sleep Quality Score: {sleep_quality:.1f}/100")
print(f"Sleep Efficiency: {sleep_efficiency:.1f}%")
```

### Recovery & Capacity Assessment

```python
# Daily metrics
rmssd_today = 35.0
rmssd_baseline_mean = 30.0
rmssd_baseline_std = 8.0
sleep_quality_score = 75.0
strain_today = 25.0

# Calculate recovery score
recovery = compute_recovery_score(
    rmssd_today, rmssd_baseline_mean, rmssd_baseline_std,
    sleep_quality_score, strain_today,
    baseline_strain_mean=20.0, baseline_strain_std=5.0
)

# Calculate metabolic capacity
capacity = compute_metabolic_capacity(
    rmssd_today, rmssd_baseline_mean, rmssd_baseline_std,
    rhr_today=58.0, rhr_baseline_mean=60.0, rhr_baseline_std=4.0,
    sleep_quality_score=75.0, recovery_debt=10.0
)

print(f"Recovery Score: {recovery:.1f}/100")
print(f"Metabolic Capacity: {capacity:.1f}/100")
```

### Loading Real Data

```python
# Load data from CSV
df = import_csv_data('health_data.csv')
processed_df = preprocess_health_data(df)

# Or load demo dataset
demo_df = load_demo_data('demo.csv')

# Process daily metrics
day_data = demo_df[demo_df['timestamp'].dt.date == demo_df['timestamp'].dt.date.min()]
rr_intervals = np.concatenate(day_data['rr_intervals'].tolist())
daily_rmssd = compute_rmssd(rr_intervals)
```

### Forecasting

```python
# Historical capacity scores
capacity_history = [65, 67, 66, 70, 72, 71, 73, 75, 74, 76]

# Calculate momentum
momentum = compute_metabolic_momentum(capacity_history, window_days=7)
print(f"Momentum: {momentum:.2f} points/week")

# Forecast next 14 days
days, forecast = forecast_capacity_trend(capacity_history, days_forward=14)
print(f"14-day forecast: {forecast}")
```

## Data Format

### CSV Format

Expected columns:
- `timestamp` (required) - ISO 8601 format
- `heart_rate` - Beats per minute
- `rr_intervals` - String representation of list: "[850, 820, 840]"
- `sleep_stage` - One of: 'awake', 'light', 'deep', 'rem'
- `time_in_bed_min` - Minutes in bed
- `total_sleep_min` - Minutes of actual sleep

### JSON Format

```json
{
  "user_info": {
    "age": 40,
    "sex": "male"
  },
  "measurements": [
    {
      "timestamp": "2023-01-01T00:00:00",
      "heart_rate": 70,
      "rr_intervals": [850, 820, 840, 860],
      "sleep_stage": "light"
    }
  ]
}
```

## Key Differences: Longevity.py vs Longevity v0.1.py

### Enhanced Version (`Longevity.py`)

**Advantages:**
1. **Comprehensive Logging** - Uses Python `logging` module for debugging and monitoring
2. **Robust Error Handling** - Try-except blocks with specific error messages
3. **Demo Data Integration** - `load_demo_data()` function with validation
4. **Safe Data Parsing** - Uses `ast.literal_eval` for RR interval lists
5. **Production Ready** - System exit codes, detailed error messages
6. **Updated Dependencies** - Uses `np.trapezoid()` instead of deprecated `np.trapz()`
7. **Real Data Processing** - Works with actual demo dataset (`demo.csv`)
8. **Better Validation** - Extensive data quality checks and logging

### Original Version (`Longevity v0.1.py`)

**Characteristics:**
- Simpler, prototype-focused
- Relies on synthetic data only
- Minimal error handling
- Uses deprecated numpy functions
- Better for initial understanding and testing

**Recommendation:** Use `Longevity.py` for production applications and `Longevity v0.1.py` for educational purposes or initial prototyping.

## Performance Considerations

### Computational Complexity

- **HRV Metrics**: O(n) where n = number of R-R intervals
- **Frequency-Domain HRV**: O(n log n) due to Welch's method
- **Sleep Quality**: O(m) where m = number of sleep epochs
- **Forecasting**: O(w) where w = fitting window size

### Optimization Tips

1. **Batch Processing**: Process multiple days of data in chunks
2. **Caching**: Store baseline statistics to avoid recomputation
3. **Data Validation**: Clean data once, use multiple times
4. **Vectorization**: Uses NumPy for efficient array operations

## Limitations & Considerations

### Accuracy

1. **HRV Metrics**: Require clean R-R intervals; artifact removal is basic
2. **Sleep Quality**: Heuristic weights may need tuning for specific populations
3. **Biological Age**: Coefficients are **illustrative only** - requires calibration with cohort data
4. **TRIMP**: Simplified version; advanced versions use lactate thresholds

### Data Requirements

- **Minimum RR Intervals**: 2+ for time-domain, 4+ for frequency-domain
- **Sleep Stages**: At least one complete sleep cycle for meaningful scores
- **Baseline Calculations**: 14+ days recommended for stable baselines

### Medical Disclaimer

**This module is for research and educational purposes only.** It is not intended for medical diagnosis or treatment. Consult healthcare professionals for medical advice.

## Error Handling

The enhanced version (`Longevity.py`) includes comprehensive error handling:

```python
try:
    demo_df = load_demo_data()
except FileNotFoundError as e:
    logging.error(f"Demo file not found: {e}")
    sys.exit(1)
except ValueError as e:
    logging.error(f"Data validation error: {e}")
    sys.exit(1)
```

Common errors handled:
- Missing files
- Empty datasets
- Invalid data formats
- Missing required columns
- Parsing failures

## Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log levels:
- `INFO` - Normal operation, data loading stats
- `WARNING` - Data quality issues, missing values
- `ERROR` - Critical failures, file not found

## Future Enhancements

Potential improvements:
1. Machine learning-based biological age models
2. Advanced HRV artifact detection (Kubios-like)
3. Multi-day trend analysis
4. Personalized baseline calculations
5. Integration with wearable device APIs
6. Database storage for longitudinal tracking
7. Statistical significance testing
8. Circadian rhythm analysis

## Contributing

When modifying this module:
1. Maintain backward compatibility
2. Add comprehensive docstrings
3. Include error handling
4. Update this README
5. Add unit tests for new features
6. Validate with real health data

## License

Research and educational use. See project license for details.

## Authors

- Original Implementation: ChatGPT (GPT-5 Thinking mini)
- Date: 2025-10-17
- Enhanced Version: Includes production-ready error handling and logging

## References

### HRV Standards
- Task Force of the European Society of Cardiology (1996)
- Shaffer & Ginsberg (2017) - HRV review

### Sleep Science
- National Sleep Foundation guidelines
- AASM sleep stage scoring manual

### Training Load
- Banister TRIMP model
- Session RPE methodology

---

**Version**: 1.0 (Enhanced)  
**Last Updated**: 2025-01-XX  
**Python**: 3.7+  
**Status**: Production-ready with demo data support
