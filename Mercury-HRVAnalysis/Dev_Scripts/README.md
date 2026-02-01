# HRV Analytics v6.25-DEV

## Overview

A comprehensive **Heart Rate Variability (HRV) analytics tool** specifically designed for MS (Multiple Sclerosis) health monitoring. This Python-based application analyzes HRV metrics from health tracking devices, provides MS-specific health recovery scoring, and monitors trends to detect significant deviations from personal baselines.

## Purpose & Functionality

- **HRV Data Analysis**: Process and analyze heart rate variability metrics from health devices
- **MS-Specific Scoring**: Calculate recovery scores using MS-optimized metric weighting
- **Trend Detection**: Monitor longitudinal trends with correlation analysis
- **Alert System**: Detect and log significant deviations from personal baselines
- **Data Persistence**: Store measurements, baselines, trends, and alerts in SQLite database
- **Visualization Suite**: Generate comprehensive charts for health monitoring

## Key Features

### 1. Database Schema (4 Tables)

#### myHRV_data
Stores raw HRV measurements from devices
- Date, source name, 8 HRV metrics
- Timestamp tracking

#### hrv_alerts
Logs significant health deviations
- Alert date, metric, current/baseline values
- Deviation percentage and messages

#### myHRV_baselines
90-day baseline averages for personalized comparison
- Source name, analysis date
- Average values for all 8 metrics

#### myHRV_trends
Trend statistics and correlations
- Metric correlations, trend direction/strength
- Statistical summaries (mean, std, min, max)
- Latest MS recovery score

### 2. HRV Metrics Tracked (8 Metrics)

**Time-Domain Metrics:**
- `SD1` - Short-term HRV variability
- `SD2` - Long-term HRV variability
- `SDNN` - Standard deviation of NN intervals
- `RMSSD` - Root mean square of successive differences
- `pNN50` - Percentage of successive NN intervals > 50ms

**Frequency-Domain Metrics:**
- `VLF` - Very Low Frequency power
- `LF` - Low Frequency power
- `HF` - High Frequency power

### 3. MS-Specific Recovery Score Algorithm

Custom weighted algorithm prioritizing metrics most relevant to MS health monitoring:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| SDNN | 24% | Primary autonomic function indicator |
| RMSSD | 18% | Parasympathetic activity marker |
| SD1 | 15% | Short-term variability |
| pNN50 | 14% | Vagal tone indicator |
| SD2 | 13% | Long-term variability |
| LF | 8% | Sympathetic/parasympathetic balance |
| HF | 8% | Parasympathetic modulation |

**Score Range:** 0-100 (normalized)

## Installation

### Requirements
```bash
pip install sqlite3 pandas numpy matplotlib seaborn
```

### Database Setup
- Default database path: `c:/smakrykoDBs/Artemis.db`
- Schema is automatically created on first run
- Tables are created if they don't exist

## Usage

### Basic Usage
```python
from HRV_Analytics_v6 import HRVAnalyticsV33

# Initialize
hrv = HRVAnalyticsV33("path/to/database.db")

# Import data from view
hrv.import_myhrv_view(source_view="myHRV_view", device_name="MyHRV_import")

# Analyze trends
results = hrv.analyze_hrv_trends(days_back=30, source_name="MyHRV_import")

# Generate visualizations
df = results['dataframe']
baselines = hrv._get_personal_baselines("MyHRV_import")
latest = results['current_values']

hrv.plot_time_trends(df)
hrv.plot_ms_score(df)
hrv.plot_radar_chart(latest, baselines)
```

### Running the Demo
```bash
python HRV_Analytics_v6.25-DEV.py
```

## Visualization Suite

1. **Time Trends Plot** - All 8 metrics over time
2. **Histograms** - Distribution of each metric
3. **Baseline Bar Chart** - 90-day baseline values
4. **Radar Chart** - Latest vs baseline comparison
5. **MS Score Trend** - Recovery score over time
6. **Trend Summary** - Correlation coefficients with direction labels

## Core Methods

### Data Import & Retrieval
- `import_myhrv_view()` - Import data from database view
- `get_daily_hrv_dataframe()` - Retrieve HRV data for specified period

### Analysis
- `analyze_hrv_trends()` - Comprehensive trend analysis
- `_calculate_ms_recovery_score()` - Calculate MS-weighted health score
- `_calculate_trend_statistics()` - Compute correlations and trends
- `_get_personal_baselines()` - Retrieve 90-day baseline averages

### Persistence
- `save_baselines()` - Store baseline calculations
- `save_trends()` - Store trend statistics
- `report_alerts()` - Check for deviations and log alerts

### Visualization
- `plot_time_trends()` - Time series visualization
- `plot_hrv_histograms()` - Distribution plots
- `plot_baseline_bar()` - Baseline comparison
- `plot_radar_chart()` - Multi-metric radar view
- `plot_ms_score()` - Recovery score trend
- `plot_trend_summary()` - Statistical summary

## Strengths

✅ **Well-structured OOP design** - Clean class-based architecture  
✅ **Comprehensive error handling** - Logging throughout  
✅ **Database-driven persistence** - All data stored for historical analysis  
✅ **Rich visualization suite** - 6 different plot types  
✅ **Trend analysis** - Automated correlation detection  
✅ **Alert system** - Proactive health monitoring  
✅ **Sample data generation** - Built-in testing capability  
✅ **MS-specific optimization** - Tailored metric weighting  

## Known Issues & Recommendations

### Critical Issues

1. **Variable Naming Convention**
   - Uses `seLF` instead of standard `self` (intentional choice?)
   - May cause confusion for other developers

2. **Deprecated Code**
   - `report_alertsXXXX()` function (lines 399-415) should be removed

3. **SQL Injection Risk**
   - Using f-strings for table names in queries
   - Should use parameterized queries or whitelist validation

### Moderate Issues

4. **Hardcoded Configuration**
   - Database path hardcoded in constructor
   - Should use config file or environment variables

5. **Missing Input Validation**
   - No validation for `days_back`, device names, or metric values
   - Could cause unexpected behavior with invalid inputs

6. **No Documentation**
   - Missing docstrings for all methods
   - No type hints for parameters/returns

7. **Inefficient Database Access**
   - Baselines retrieved multiple times
   - Consider caching frequently accessed data

### Minor Issues

8. **Magic Numbers**
   - Threshold value `2.5` hardcoded
   - Deviation formula `/ 10` unclear
   - Should be documented or configurable

9. **Plot Blocking**
   - `plt.show()` blocks execution
   - Consider saving to files for automation/batch processing

10. **Incomplete Error Recovery**
    - Sample data generation doesn't fully replicate real data structure

## Suggested Improvements

### High Priority
1. ✏️ Fix SQL injection vulnerabilities
2. 📝 Add comprehensive docstrings and type hints
3. ⚙️ Create configuration file for paths, weights, thresholds
4. 🔒 Add data validation layer for all inputs
5. 🧹 Remove deprecated `report_alertsXXXX()` function

### Medium Priority
6. 📊 Separate visualization into dedicated module
7. 💾 Add export functionality (CSV, JSON, PDF reports)
8. 🚀 Implement caching for baselines and frequently accessed data
9. 🧪 Add comprehensive unit tests
10. 📈 Add data quality checks (outlier detection, missing data handling)

### Low Priority
11. 🎨 Make plots configurable (colors, sizes, styles)
12. 📱 Add command-line interface with argparse
13. 🔔 Implement email/SMS alerts for critical deviations
14. 📊 Add additional statistical tests (Mann-Kendall, etc.)
15. 🌐 Consider web dashboard interface

## Security Considerations

⚠️ **Important for Production Use:**

- **SQL Injection**: Parameterize all queries
- **Authentication**: Add user authentication if multi-user
- **Data Encryption**: Encrypt sensitive health data
- **Access Control**: Implement role-based permissions
- **Audit Logging**: Track all data access and modifications
- **HIPAA Compliance**: Ensure compliance if handling protected health information

## Configuration Example

```python
# Recommended config.yaml structure
database:
  path: "c:/smakrykoDBs/Artemis.db"
  
metrics:
  tracked: ['SD1', 'SD2', 'sdnn', 'rmssd', 'pNN50', 'VLF', 'LF', 'HF']
  
weights:
  rmssd: 0.18
  sdnn: 0.24
  pNN50: 0.14
  SD1: 0.15
  SD2: 0.13
  LF: 0.08
  HF: 0.08
  
alerts:
  deviation_threshold: 0.25  # 25%
  metrics_thresholds:
    SD1: 2.5
    SD2: 2.5
    # ... etc
    
analysis:
  default_days_back: 30
  baseline_days: 90
```

## Output Examples

### Console Output
```
=== HRV Analytics V3.3 Demo ===
Data points: 30
Date range: 2025-12-07 to 2026-01-06

Current HRV values:
 SD1: 32.4
 SD2: 43.1
 SDNN: 52.7
 RMSSD: 44.3
 PNN50: 14.2
 VLF: 745.3
 LF: 1089.2
 HF: 831.5

MS-Optimized Recovery Score:
 MS-Aware: 67.3/100

Trend Analysis:
 Sd1: improving (moderate)
 Sd2: stable (weak)
 Sdnn: improving (strong)
 ...
```

### Database Alerts Example
```
Alert: RMSSD deviation: current=25.3, baseline=42.1, deviation=-39.9%
Alert: SDNN deviation: current=35.2, baseline=52.7, deviation=-33.2%
```

## Contributing

When contributing to this project:
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Include unit tests for new features
4. Update this README with changes
5. Test with both real and sample data

## License

[Specify License]

## Author

[Your Name/Organization]

## Version History

- **v6.25-DEV** (Current) - Development version with alert logging
- Previous versions - [Document version history]

## Support & Contact

For issues, questions, or contributions:
- [GitHub Issues]
- [Email]
- [Documentation Website]

---

**Note:** This tool is for health monitoring purposes. Always consult with healthcare professionals for medical decisions. HRV data should be interpreted in conjunction with other health indicators and professional medical advice.

# HRV Analytics Dashboard

## Overview

The HRV Analytics Dashboard is a web-based visualization tool for Heart Rate Variability (HRV) analysis generated by the `HRV_Analytics_v6.25-DEV.py` script. It provides an interactive interface to view and analyze HRV metrics with a focus on MS-Aware health monitoring.

## Features

- 📈 **Interactive Visualization**: Display 6 types of HRV analytics charts
- 📂 **File System Integration**: Direct access to local chart files using the File System Access API
- 🔄 **Auto-Refresh**: Automatic discovery of the latest charts
- 🖼️ **Full-Screen Modal**: Click any chart to view in full-screen mode
- 📱 **Responsive Design**: Mobile-friendly layout
- 🎨 **Modern UI**: Clean, gradient-based design with smooth animations

## Chart Types

The dashboard displays the following analytics:

1. **📈 HRV Metrics Time Trends** - 30-day trends for all HRV metrics
2. **📊 HRV Metrics Distributions** - Statistical distribution of HRV values
3. **📉 90-Day Baseline Profile** - Long-term HRV baseline averages
4. **🎯 Latest vs Baseline Comparison** - Multi-dimensional HRV comparison (radar chart)
5. **💜 MS-Aware Recovery Score** - Weighted health recovery trending
6. **📋 Trend Statistics Summary** - Correlation analysis and trend directions

## Prerequisites

- Modern web browser with File System Access API support:
  - ✅ Google Chrome 86+
  - ✅ Microsoft Edge 86+
  - ✅ Opera 72+
  - ⚠️ Firefox (limited support)
  - ❌ Safari (not supported)
- HRV charts generated by `HRV_Analytics_v6.25-DEV.py`

## Setup and Usage

### Step 1: Generate HRV Charts

Run the Python analytics script to generate charts:

```bash
python HRV_Analytics_v6.25-DEV.py
```

Charts are automatically saved to: `C:\temp\logsFitnessApp\HRV_DashBoards`

### Step 2: Open the Dashboard

1. Open `HRV_Dashboard.html` in a supported browser
2. The dashboard will load with placeholder cards

### Step 3: Load Charts

1. Click the **"📂 Select Dashboard Folder"** button
2. Navigate to `C:\temp\logsFitnessApp\HRV_DashBoards`
3. Select the folder to grant access
4. Charts will automatically load and display

### Step 4: Interact with Charts

- **View Full-Screen**: Click any chart to open in modal view
- **Close Modal**: Click outside the image, press ESC, or click the X button
- **Refresh Data**: Click **"🔄 Refresh Dashboard"** after running the Python script again

## File Naming Convention

The dashboard automatically detects the latest files matching these patterns:

- `HRV_TimeTrends_YYYYMMDD_*.png`
- `HRV_Histograms_YYYYMMDD_*.png`
- `HRV_BaselineProfile_YYYYMMDD_*.png`
- `HRV_RadarChart_YYYYMMDD_*.png`
- `HRV_MSScore_YYYYMMDD_*.png`
- `HRV_TrendSummary_YYYYMMDD_*.png`

When multiple files match a pattern, the most recent file (by filename) is displayed.

## Technical Details

### File System Access API

The dashboard uses the modern File System Access API to:
- Request permission to read a local directory
- Scan for PNG files matching chart patterns
- Load images as blob URLs for display
- Automatically select the latest version of each chart

### Security Considerations

- The browser will prompt for permission before accessing the file system
- Access is limited to the selected directory only
- No data is uploaded or transmitted
- All processing happens client-side in the browser

### Browser Compatibility

If using a browser without File System Access API support, you'll see an error message. Consider using:
- Chrome-based browsers (recommended)
- Microsoft Edge (recommended)

## Troubleshooting

### Charts Not Displaying

1. **Verify folder selection**: Make sure you selected the correct folder (`C:\temp\logsFitnessApp\HRV_DashBoards`)
2. **Check file names**: Ensure PNG files match the expected patterns
3. **Run Python script**: Generate charts using `HRV_Analytics_v6.25-DEV.py`
4. **Browser compatibility**: Use Chrome or Edge for best results

### Permission Errors

- The browser may require you to re-grant folder access after closing
- Click "Select Dashboard Folder" again if charts don't load

### Refresh Not Working

- Ensure you've selected a folder first
- Click "Select Dashboard Folder" again to refresh permissions
- Clear browser cache if issues persist

## Development

### File Structure

```
Dev_Scripts/
├── HRV_Dashboard.html          # Main dashboard file
├── HRV_Analytics_v6.25-DEV.py  # Chart generation script
└── README.md                    # This file
```

### Customization

To add new chart types, modify the `chartTypes` array in the JavaScript section:

```javascript
const chartTypes = [
    { 
        pattern: 'YourPattern_', 
        title: '🎯 Your Chart Title', 
        description: 'Chart description' 
    },
    // ...existing charts
];
```

## Future Enhancements

- [ ] Add chart download functionality
- [ ] Implement date range filtering
- [ ] Add comparison between multiple time periods
- [ ] Export dashboard as PDF report
- [ ] Add data summary statistics panel
- [ ] Integration with backend API for automatic updates

## License

Part of the MS-Buddy Fitness App project.

## Support

For issues or questions, please refer to the main project documentation or contact the development team.

---

**Last Updated**: 2024
**Version**: 1.0
**Compatible with**: HRV_Analytics_v6.25-DEV.py
