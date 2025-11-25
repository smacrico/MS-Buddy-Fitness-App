# Longevity Metrics

A comprehensive Python package for computing health and longevity metrics from wearable device data (fitness trackers, smartwatches, etc.).

## Features

### 📊 Comprehensive Metrics
- **HRV (Heart Rate Variability)**: RMSSD, SDNN, pNN20, pNN50, LF/HF ratio
- **Sleep Quality**: Sleep efficiency, quality scoring, architecture analysis
- **Activity Load**: TRIMP-based strain computation, heart rate zones
- **Recovery Metrics**: Daily recovery scores, recovery debt tracking
- **Metabolic Capacity**: Composite health score, momentum tracking
- **Cardiovascular Health**: Multi-factor cardiovascular assessment
- **Biological Age**: Physiological age estimation (prototype)

### 📈 Visualizations
- HRV trends over time
- Sleep quality and efficiency plots
- Recovery and capacity tracking
- Comprehensive health dashboard
- Forecast projections

### 💾 Data Export
- CSV, JSON, Excel, Parquet formats
- Automated report generation
- Multi-format export support

### 🖥️ Command-Line Interface
Easy-to-use CLI for batch processing and automation

## Installation

```bash
# Clone the repository
git clone https://github.com/smacrico/MS-Buddy-Fitness-App.git
cd MS-Buddy-Fitness-App/_jHeel_myBand

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Python API

```python
from longevity import (
    compute_hrv_summary,
    compute_sleep_summary,
    plot_dashboard,
    export_to_csv
)
import pandas as pd

# Load your data
df = pd.read_csv('data/demo.csv')

# Compute HRV metrics
rr_intervals = [800, 820, 810, 830, 805]  # milliseconds
hrv_metrics = compute_hrv_summary(rr_intervals)
print(f"RMSSD: {hrv_metrics['rmssd']:.2f} ms")
print(f"SDNN: {hrv_metrics['sdnn']:.2f} ms")

# Compute sleep metrics
sleep_stages = ['light', 'deep', 'rem', 'light', 'awake']
sleep_metrics = compute_sleep_summary(
    sleep_stages,
    time_in_bed_min=480,
    total_sleep_min=420
)
print(f"Sleep Quality: {sleep_metrics['quality']:.1f}")

# Export results
export_to_csv(df, 'outputs/metrics.csv')
```

### Command-Line Interface

```bash
# Analyze data and export metrics
longevity analyze -i data/demo.csv -o results/metrics.csv -f csv

# Create visualizations
longevity visualize -i data/demo.csv -o plots/ -t all

# Generate comprehensive report
longevity report -i data/demo.csv -o reports/summary.txt

# Export to multiple formats
longevity export -i data/demo.csv -o outputs/data.xlsx -f all
```

## CLI Commands

### Analyze
Compute all metrics from input data:
```bash
longevity analyze -i <input.csv> -o <output.csv> -f <format>
```

Options:
- `-i, --input`: Input CSV file (required)
- `-o, --output`: Output file path
- `-f, --format`: Output format (csv/json/excel/all)

### Visualize
Create plots and dashboards:
```bash
longevity visualize -i <input.csv> -o <output_dir> -t <plot_type>
```

Options:
- `-i, --input`: Input CSV file (required)
- `-o, --output`: Output directory for plots
- `-t, --plot-type`: Type of plot (hrv/sleep/recovery/dashboard/all)
- `--show`: Display plots interactively

### Export
Export data to various formats:
```bash
longevity export -i <input.csv> -o <output_file> -f <format>
```

Options:
- `-i, --input`: Input CSV file (required)
- `-o, --output`: Output file path (required)
- `-f, --format`: Export format (csv/json/excel/parquet/all)

### Report
Generate comprehensive text report:
```bash
longevity report -i <input.csv> -o <report.txt>
```

## Data Format

Input CSV should contain:
- `timestamp`: Date/time of measurement
- `heart_rate`: Heart rate in bpm
- `rr_intervals`: RR intervals in ms (can be string representation of list)
- `sleep_stage`: Sleep stage ('awake', 'light', 'deep', 'rem')
- `spo2`: Oxygen saturation percentage
- `activity_level`: Activity level (0-100)
- `time_in_bed_min`: Time in bed (minutes)
- `total_sleep_min`: Total sleep time (minutes)

Example:
```csv
timestamp,heart_rate,rr_intervals,sleep_stage,spo2,activity_level
2023-12-01 00:00:00,65,"[850, 840, 860]",light,98,5
2023-12-01 00:30:00,62,"[900, 910, 895]",deep,97,3
```

## Modules

### `longevity.hrv`
Heart rate variability metrics computation.

### `longevity.sleep`
Sleep quality and architecture analysis.

### `longevity.activity`
Activity load and training impulse calculation.

### `longevity.core`
Core composite metrics (recovery, capacity, biological age).

### `longevity.visualizations`
Plotting and dashboard creation.

### `longevity.export`
Data export to multiple formats.

### `longevity.config`
Configuration constants and parameters.

### `longevity.utils`
Utility functions.

## Scientific Basis

This package implements established formulas and metrics:

- **RMSSD/SDNN**: Standard HRV time-domain metrics
- **LF/HF**: Frequency-domain HRV using Welch's method
- **TRIMP**: Training Impulse methodology
- **Sleep Scoring**: Multi-factor sleep quality assessment

⚠️ **Note**: Biological age coefficients are prototypes and require validation with cohort data before clinical use.

## Contributing

Contributions welcome! Areas for improvement:
- Additional HRV metrics (DFA, entropy)
- Machine learning for biological age
- Integration with specific wearable APIs
- More visualization options

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=longevity tests/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in research, please cite:

```
Longevity Metrics Package (2025)
MS-Buddy-Fitness-App / jHeel MyBand
https://github.com/smacrico/MS-Buddy-Fitness-App
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/smacrico/MS-Buddy-Fitness-App/issues
- Documentation: See `/docs` folder

## Disclaimer

This software is for research and educational purposes. Not intended for medical diagnosis or treatment. Consult healthcare professionals for medical advice.

# MS Buddy Fitness App - Demo Data Generator

This tool generates synthetic health monitoring data for testing and development purposes.

## Installation

Install required dependencies:

```bash
pip install pandas numpy
```

## CLI Usage

### Basic Usage

Generate 30 days of demo data with default settings:

```bash
python generate_demo_data.py
```

This creates a `demo.csv` file in the same directory with readings every 30 minutes.

### Advanced Options

#### Specify number of days

Generate 60 days of data:

```bash
python generate_demo_data.py --days 60
```

#### Change reading interval

Generate data with readings every 15 minutes:

```bash
python generate_demo_data.py --interval 15
```

#### Custom output file

Save to a specific location:

```bash
python generate_demo_data.py --output path/to/custom_demo.csv
```

#### Custom start date

Start from a specific date (format: YYYY-MM-DD):

```bash
python generate_demo_data.py --start-date 2024-01-01
```

#### Combined options

```bash
python generate_demo_data.py --days 90 --interval 20 --output my_data.csv --start-date 2024-01-15
```

### CLI Help

View all available options:

```bash
python generate_demo_data.py --help
```

## Generated Data Format

The CSV file contains the following columns:

- `timestamp`: Date and time of the reading
- `heart_rate`: Heart rate in BPM
- `rr_intervals`: R-R intervals for HRV analysis (as list)
- `sleep_stage`: Sleep stage (deep, rem, light, awake)
- `spo2`: Blood oxygen saturation percentage
- `activity_level`: Activity level (0-100)
- `time_in_bed_min`: Minutes in bed (480 during sleep periods)
- `total_sleep_min`: Actual sleep minutes (450 during sleep periods)

## Data Patterns

The generator creates realistic patterns:

- **Sleep period** (22:00-06:00): Lower heart rate, various sleep stages
- **Morning exercise** (07:00-08:30, weekdays): Elevated heart rate and activity
- **Regular daytime**: Normal heart rate and moderate activity
- **Weekends**: No morning exercise pattern

## Using in Jupyter Notebooks

### Load the generated data

```python
import pandas as pd

# Load the demo data
df = pd.read_csv('demo.csv')

# Parse timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Display basic info
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Parse RR intervals

```python
import ast

# Convert RR intervals from string to list
df['rr_intervals'] = df['rr_intervals'].apply(ast.literal_eval)

# Access individual RR interval lists
print(df['rr_intervals'].iloc[0])
```

### Basic analysis examples

```python
# Sleep analysis
sleep_data = df[df['sleep_stage'] != 'awake']
print(f"Average sleep heart rate: {sleep_data['heart_rate'].mean():.1f} BPM")

# Activity patterns
print(f"Average daily activity: {df['activity_level'].mean():.1f}")

# Heart rate zones
print(df['heart_rate'].describe())
```

### Visualization examples

```python
import matplotlib.pyplot as plt

# Plot heart rate over time
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['heart_rate'])
plt.xlabel('Time')
plt.ylabel('Heart Rate (BPM)')
plt.title('Heart Rate Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sleep stages distribution
df['sleep_stage'].value_counts().plot(kind='bar')
plt.title('Sleep Stage Distribution')
plt.ylabel('Count')
plt.show()
```

## Troubleshooting

### Import errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install pandas numpy matplotlib jupyter
```

### File not found

Make sure you're running the script from the correct directory or use absolute paths for the output file.

### Invalid date format

The `--start-date` parameter must use YYYY-MM-DD format:

```bash
# Correct
python generate_demo_data.py --start-date 2024-01-15

# Incorrect
python generate_demo_data.py --start-date 01/15/2024
```

## Examples

### Generate data for a full year with hourly readings

```bash
python generate_demo_data.py --days 365 --interval 60 --output yearly_data.csv
```

### Generate high-frequency data for one week

```bash
python generate_demo_data.py --days 7 --interval 5 --output high_freq_data.csv
```

### Generate data starting from today

```bash
python generate_demo_data.py --start-date 2024-01-15 --days 30
```

## Notes

- Generated data is synthetic and for testing purposes only
- Random variations are added for realistic patterns
- RR intervals are stored as string representations of Python lists
- All timestamps are consecutive based on the specified interval
- Activity levels are capped between 0 and 100
- SpO2 values are kept within realistic ranges (94-99%)
