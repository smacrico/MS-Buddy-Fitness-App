# Garmin Stress Data Analyzer

A Python script for analyzing and visualizing stress trends from Garmin wearable devices using the Garth library.

## Overview

This tool retrieves stress data from your Garmin account and creates comprehensive visualizations to help you understand your stress patterns over time. It provides both weekly and daily stress trend analysis with statistical smoothing and decomposition.

## Features

- **Authentication Management**: Secure Garmin account login with session persistence
- **Weekly Stress Trends**: Visualize average weekly stress levels over the past 2 years
- **Daily Stress Analysis**: Detailed daily stress tracking with multiple visualization options:
  - Scatter plot of daily stress levels
  - 28-day rolling average to identify longer-term patterns
  - Seasonal decomposition to isolate underlying trends from daily fluctuations
- **Professional Visualizations**: Clean, publication-ready charts using Matplotlib and Seaborn

## Prerequisites

- Python 3.7 or higher
- Google Colab account (for the original notebook version)
- Garmin Connect account with stress data from a compatible device

## Installation

Install the required dependencies:

```bash
pip install garth pandas seaborn matplotlib statsmodels
```

For Google Colab users, the script will automatically install Garth when run.

## Setup

### Google Colab (Original Version)

1. Mount Google Drive for session persistence
2. The script will prompt for your Garmin credentials on first run
3. Subsequent runs will resume the saved session

### Local Usage

If running locally (not in Colab), modify the following:

1. Remove or comment out Google Drive mounting:
   ```python
   # from google.colab import drive
   # drive.mount("/content/drive")
   ```

2. Update the `GARTH_HOME` path to a local directory:
   ```python
   GARTH_HOME = "path/to/your/local/directory"
   ```

## Usage

Run the script:

```bash
python Colab_stress.py
```

On first run, you'll be prompted to enter:
- **Email**: Your Garmin Connect email address
- **Password**: Your Garmin Connect password (input is hidden)

The script will generate three visualizations:

1. **Average Weekly Stress**: Line chart showing weekly stress averages over 2 years
2. **Daily Stress with Rolling Average**: Scatter plot of daily stress levels with a 28-day moving average overlay
3. **Seasonal Decomposition**: Two-panel chart showing raw daily data and extracted 28-day trend

## Data Retrieval

- **Weekly Data**: Last 104 weeks (~2 years)
- **Daily Data**: Up to 3 years (1,095 days)

Note: Retrieving daily data may take some time depending on your internet connection and the amount of historical data.

## Understanding the Visualizations

### Weekly Stress Chart
Shows the overall trajectory of your stress levels on a weekly basis. Useful for identifying long-term trends.

### Daily Stress with Rolling Average
- **Blue scatter points**: Individual daily stress measurements (can be noisy)
- **Red line**: 28-day rolling average (smooths out daily variations)

### Seasonal Decomposition
- **Top panel**: Raw daily stress data
- **Bottom panel**: Extracted 28-day trend component using statistical analysis

The trend line helps identify underlying patterns by removing seasonal and random fluctuations.

## Data Privacy

- Your Garmin credentials are only used to authenticate with Garmin Connect
- Session tokens are saved locally (in Google Drive or your specified directory)
- No data is shared with third parties
- All data processing happens locally in your environment

## Troubleshooting

### Authentication Issues
If you get authentication errors:
1. Delete the saved session in your `GARTH_HOME` directory
2. Re-run the script and enter your credentials again
3. Ensure your Garmin account credentials are correct

### No Data Returned
- Verify that your Garmin device tracks stress data
- Check that you've worn your device consistently
- Some Garmin devices require specific settings to be enabled for stress tracking

### Import Errors
Ensure all required packages are installed:
```bash
pip install --upgrade garth pandas seaborn matplotlib statsmodels
```

## Technical Details

### Dependencies
- **garth**: Garmin Connect API client
- **pandas**: Data manipulation and analysis
- **seaborn**: Statistical data visualization
- **matplotlib**: Plotting library
- **statsmodels**: Time series statistical analysis

### Data Structure
The script retrieves:
- `WeeklyStress`: Aggregated weekly stress averages
- `DailyStress`: Individual daily stress measurements including overall stress level

## Contributing

Feel free to fork and modify this script for your own analysis needs. Potential enhancements:
- Add correlation analysis with other health metrics
- Include stress categorization (low/medium/high)
- Export data to CSV for external analysis
- Add year-over-year comparison views

## License

This project is provided as-is for personal use. Please respect Garmin's Terms of Service when using their API.

## Acknowledgments

- Built using [Garth](https://github.com/matin/garth) by Matin Tamizi
- Originally created as a Google Colab notebook

## Related Projects

Check out other health data analysis tools in the MS-Buddy-Fitness-App repository for comprehensive health tracking and visualization solutions.
