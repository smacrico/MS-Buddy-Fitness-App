# MS-Buddy Fitness App - Analytics Dashboard System

## Overview

The MS-Buddy Fitness App is a comprehensive health and fitness monitoring system designed specifically for individuals with Multiple Sclerosis (MS). It consists of multiple specialized analysis modules, each with its own Python analytics script and interactive HTML dashboard for visualization.

## System Architecture

### Dashboard Components

The system includes three main dashboard modules:

1. **HRV Analytics Dashboard** - Heart Rate Variability monitoring with MS-aware scoring
2. **Running Analytics Dashboard** - Comprehensive running performance tracking
3. **Garmin Stress Analytics Dashboard** - Stress and health metrics from Garmin devices

### File Structure

```
MS-Buddy-Fitness-App/
├── Mercury-HRVAnalysis/
│   └── Dev_Scripts/
│       ├── HRV_Analytics_v6.25-DEV.py
│       └── HRV_Dashboard.html
├── APEX-RunAnalysis/
│   └── Dev_Scripts/
│       ├── RunningAnalysis_v6.26-Dev.py
│       ├── Run_Dashboard.html
│       └── README.md
└── Colab/
    ├── GarminStressDataAnalyzer.py
    └── Garmin_Dashboard.html
```

### Output Directories

All charts are saved to organized dashboard folders:
```
C:\temp\logsFitnessApp\
├── HRV_DashBoards\          # HRV analytics charts
├── Run_DashBoards\          # Running analytics charts
└── Garmin_DashBoards\       # Garmin stress analytics charts
```

---

## 1. HRV Analytics Dashboard

### Purpose
Monitor Heart Rate Variability (HRV) metrics with MS-aware weighting to track autonomic nervous system health and recovery status.

### Python Script: `HRV_Analytics_v6.25-DEV.py`

#### Features
- **Data Source**: Custom HRV data files or wearable devices
- **MS-Aware Scoring**: Weighted recovery score considering MS-specific factors
- **Time Series Analysis**: 30-day and 90-day baseline tracking
- **Statistical Analysis**: Distributions, correlations, and trend detection

#### Generated Charts

1. **HRV Time Trends** (`HRV_TimeTrends_*.png`)
   - 30-day trends for all HRV metrics
   - RMSSD, SDNN, pNN50, LF/HF ratio
   - Moving averages and trend lines

2. **HRV Histograms** (`HRV_Histograms_*.png`)
   - Statistical distribution of HRV values
   - Normal distribution overlays
   - Outlier detection

3. **90-Day Baseline Profile** (`HRV_BaselineProfile_*.png`)
   - Long-term HRV baseline averages
   - Stability indicators
   - Reference ranges

4. **Latest vs Baseline Comparison** (`HRV_RadarChart_*.png`)
   - Multi-dimensional HRV comparison
   - Radar chart visualization
   - Deviation from baseline

5. **MS-Aware Recovery Score** (`HRV_MSScore_*.png`)
   - Weighted health recovery trending
   - MS-specific metric weighting
   - Recovery trajectory

6. **Trend Statistics Summary** (`HRV_TrendSummary_*.png`)
   - Correlation analysis
   - Trend directions and significance
   - Statistical summaries

#### Usage
```bash
python HRV_Analytics_v6.25-DEV.py
```

### HTML Dashboard: `HRV_Dashboard.html`

#### Features
- Dynamic chart loading using File System Access API
- 6 interactive chart categories
- Full-screen modal view for detailed inspection
- Automatic latest file detection
- Responsive design for mobile/desktop
- Keyboard shortcuts (ESC to close modal)

#### How to Use
1. Run the Python analysis script to generate charts
2. Open `HRV_Dashboard.html` in a modern browser (Chrome/Edge recommended)
3. Click "Select Dashboard Folder"
4. Navigate to `C:\temp\logsFitnessApp\HRV_DashBoards`
5. View all generated HRV analytics charts
6. Click any chart for full-screen view
7. Use "Refresh Dashboard" after running analysis again

---

## 2. Running Analytics Dashboard

### Purpose
Comprehensive running performance analysis including speed, efficiency, HR-RS deviation, and training load management.

### Python Script: `RunningAnalysis_v6.26-Dev.py`

#### Data Source
SQLite database: `c:/smakrykoDBs/Apex.db`

#### Database Schema
```sql
CREATE TABLE running_sessions (
    date TEXT,
    running_economy REAL,
    vo2max REAL,
    distance REAL,
    time REAL,
    heart_rate REAL,
    avg_speed REAL,
    max_speed REAL,
    HR_RS_Deviation_Index REAL,
    cardiacdrift REAL,
    sport TEXT,
    resting_hr REAL,
    sleep_quality INTEGER,
    fatigue_level INTEGER
);
```

#### Core Metrics

**Basic Metrics:**
- Running Economy
- VO2Max
- Distance & Time
- Heart Rate
- Speed (Average, Max, Reserve, Consistency)

**Advanced Metrics:**
- HR-RS Deviation Index (heart rate recovery stress)
- Cardiac Drift
- Physiological Efficiency
- Fatigue Index
- TRIMP (Training Impulse)
- ACWR (Acute:Chronic Workload Ratio)
- Recovery & Readiness Scores

#### Generated Charts

1. **Training Load Analysis** (`Run_TrainingLoad_*.png`)
   - TRIMP per session
   - Weekly TRIMP load
   - Acute/chronic load trends
   - ACWR with safety thresholds (0.8-1.3)

2. **Performance Trends** (`Run_Trends_*.png`)
   - Running economy progression
   - Efficiency score trends
   - Energy cost vs distance
   - Heart rate vs running economy

3. **Recovery & Readiness** (`Run_RecoveryReadiness_*.png`)
   - Recovery score timeline
   - Readiness score timeline
   - Threshold indicators

4. **Advanced Metrics** (`Run_AdvancedMetrics_*.png`)
   - Cumulative distance
   - Running economy moving average
   - Heart rate vs pace correlation
   - Training zones distribution
   - Performance radar chart
   - Seasonal performance heatmap

5. **Training Score Impact** (`Run_ScoreImpact_*.png`)
   - Overall training score (0-100)
   - Recovery and readiness comparison
   - Multi-metric scoring

6. **Speed Metrics** (`Run_SpeedMetrics_*.png`)
   - Speed trends (avg/max)
   - Speed reserve progression
   - Speed vs heart rate
   - Speed efficiency
   - Pace progression
   - Speed zone distribution

7. **HR-RS Deviation Index** (`Run_HRRSDeviation_*.png`)
   - Deviation index trend
   - 3-session rolling average
   - Deviation vs speed performance
   - Distribution histogram
   - Deviation vs TRIMP correlation

8. **Complete Performance Dashboard** (`Run_PerformanceDashboard_*.png`)
   - All-in-one comprehensive view
   - 11 different metrics
   - Speed, efficiency, physiological metrics

#### Key Analytics

**Training Score Calculation (0-100):**
- Running Economy: 25%
- VO2Max: 20%
- Distance: 15%
- Efficiency Score: 20%
- Heart Rate: 20%

**Recovery Score Components:**
- Resting Heart Rate: 30%
- Training Load: 30%
- Sleep Quality: 20%
- Fatigue Level: 20%

**Speed Metrics:**
- Speed Reserve = Max Speed - Avg Speed
- Speed Consistency = Avg Speed / Max Speed
- Speed Efficiency = Avg Speed / Heart Rate
- Pace per km = 60 / Avg Speed

#### Database Tables Created

1. **training_logs**: Complete training data with calculated fields
2. **monthly_summaries**: Monthly aggregated statistics
3. **metrics_breakdown**: Detailed daily metrics breakdown

#### Usage
```bash
python RunningAnalysis_v6.26-Dev.py
```

#### Adding New Sessions
```python
analysis = RunningAnalysis('c:/smakrykoDBs/Apex.db')
analysis.add_session(
    date='2024-12-15',
    running_economy=73.5,
    vo2max=19.2,
    distance=5.0,
    time=27.5,
    heart_rate=150,
    sport='Running',
    cardicdrift=2.5
)
```

### HTML Dashboard: `Run_Dashboard.html`

#### Features
- 8 interactive chart categories
- Dynamic latest file loading
- Full-screen modal view
- Mobile-responsive design
- Chart filename display
- Automatic timestamp updates

#### How to Use
1. Run `RunningAnalysis_v6.26-Dev.py` to generate charts
2. Open `Run_Dashboard.html` in browser
3. Click "Select Dashboard Folder"
4. Select `C:\temp\logsFitnessApp\Run_DashBoards`
5. View all running analytics
6. Click charts for detailed view
7. Refresh after generating new analysis

---

## 3. Garmin Stress Analytics Dashboard

### Purpose
Analyze stress levels and their correlations with activity, sleep, heart rate, and other health metrics from Garmin devices.

### Python Script: `GarminStressDataAnalyzer.py`

#### Data Source
SQLite database: `C:\smakryko\myHealthData\DBs\garmin_summary.db`

#### Database Schema

**weeks_summary table** includes:
- `first_day`: Week start date
- `stress_avg`: Average weekly stress
- `hr_avg`: Average heart rate
- `rhr_avg`: Average resting heart rate
- `inactive_hr_avg`: Inactive heart rate
- `sleep_avg`: Average sleep duration
- `rem_sleep_avg`: REM sleep duration
- `steps`: Weekly step count
- `intensity_time`: Weekly intensity minutes
- `moderate_activity_time`: Moderate activity duration
- `vigorous_activity_time`: Vigorous activity duration
- `calories_avg`: Average daily calories
- `spo2_avg`: Average SpO2
- `rr_waking_avg`: Average waking respiratory rate
- `hydration_avg`: Average hydration
- `weight_avg`: Average weight

**days_summary table** includes:
- `day`: Date
- `stress_avg`: Daily stress
- Other daily metrics (subset of weekly fields)

#### Generated Charts

1. **Weekly Stress Trends** (`Garmin_WeeklyStress_*.png`)
   - Average weekly stress over 2 years
   - 4-week rolling average
   - Trend analysis
   - Statistical summary

2. **Stress Correlations** (`Garmin_StressCorrelations_*.png`)
   - Correlation matrix heatmap
   - Stress vs all health metrics
   - Color-coded correlation strength
   - Identifies strongest relationships

3. **Stress vs Activity** (`Garmin_StressVsActivity_*.png`)
   - 4 scatter plot analysis:
     - Stress vs Daily Steps
     - Stress vs Sleep Duration
     - Stress vs Average Heart Rate
     - Stress vs Intensity Time
   - Trend lines for each relationship
   - Correlation coefficients

4. **Daily Stress Analysis** (`Garmin_DailyStress_*.png`)
   - Daily stress levels (up to 3 years)
   - 28-day rolling average
   - Noise reduction through averaging
   - Long-term trend visibility

5. **Stress Decomposition** (`Garmin_StressDecomposition_*.png`)
   - Seasonal decomposition analysis
   - Original daily data
   - 28-day trend extraction
   - Identifies underlying patterns

#### Key Analytics

**Correlation Analysis:**
- Automatically calculates correlations between stress and:
  - Heart rate metrics (avg, resting, inactive)
  - Sleep metrics (total, REM)
  - Activity metrics (steps, intensity time)
  - Physiological metrics (SpO2, respiratory rate)
  - Lifestyle metrics (hydration, weight)

**Statistical Outputs:**
- Mean, min, max stress levels
- Standard deviation
- Trend analysis (recent vs early periods)
- Percentage change calculations

**Time Series Analysis:**
- Weekly aggregation for trend clarity
- Daily granularity with smoothing
- Seasonal decomposition (additive model, 28-day period)

#### Usage
```bash
python GarminStressDataAnalyzer.py
```

### HTML Dashboard: `Garmin_Dashboard.html`

#### Features
- 5 comprehensive chart categories
- Stress-health correlation visualization
- Interactive trend exploration
- Full-screen chart viewing
- Automatic latest file detection
- Responsive grid layout

#### How to Use
1. Run `GarminStressDataAnalyzer.py` to generate charts
2. Open `Garmin_Dashboard.html` in browser
3. Click "Select Dashboard Folder"
4. Navigate to `C:\temp\logsFitnessApp\Garmin_DashBoards`
5. Explore stress analytics
6. Click any chart for full-screen view
7. Use "Refresh Dashboard" for updated data

---

## Common Features Across All Dashboards

### File System Access API
All dashboards use the modern File System Access API to:
- Browse and select dashboard folders
- Read PNG files from directories
- Display the latest chart for each pattern
- Handle file errors gracefully

### Chart Naming Convention
All charts follow timestamped naming:
```
[Module]_[ChartType]_YYYYMMDD_HHMMSS.png

Examples:
HRV_TimeTrends_20241215_143022.png
Run_SpeedMetrics_20241215_143025.png
Garmin_WeeklyStress_20241215_143028.png
```

### Browser Compatibility
**Recommended Browsers:**
- Google Chrome (latest)
- Microsoft Edge (latest)
- Firefox (latest with File System Access API support)

**Required Features:**
- File System Access API
- ES6+ JavaScript
- CSS Grid support
- Blob URL support

### Keyboard Shortcuts
- `ESC`: Close full-screen modal view
- Standard browser shortcuts work as expected

### Responsive Design
All dashboards are mobile-responsive:
- Desktop: 2-column grid (600px min width per chart)
- Tablet: 2-column grid (adjusted)
- Mobile: 1-column stack layout

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Required Python packages
pip install pandas numpy matplotlib seaborn sqlite3 statsmodels
```

### Directory Setup
```bash
# Create output directories (automatically created by scripts)
mkdir C:\temp\logsFitnessApp\HRV_DashBoards
mkdir C:\temp\logsFitnessApp\Run_DashBoards
mkdir C:\temp\logsFitnessApp\Garmin_DashBoards
```

### Database Setup

**For Running Analytics:**
- Database: `c:/smakrykoDBs/Apex.db`
- Required table: `running_sessions`

**For Garmin Analytics:**
- Database: `C:\smakryko\myHealthData\DBs\garmin_summary.db`
- Required tables: `weeks_summary`, `days_summary`

---

## Workflow

### Typical Analysis Workflow

1. **Data Collection**
   - Record workouts/activities
   - Sync wearable devices
   - Update databases

2. **Run Analysis Scripts**
   ```bash
   # HRV Analysis
   cd Mercury-HRVAnalysis\Dev_Scripts
   python HRV_Analytics_v6.25-DEV.py

   # Running Analysis
   cd APEX-RunAnalysis\Dev_Scripts
   python RunningAnalysis_v6.26-Dev.py

   # Garmin Stress Analysis
   cd Colab
   python GarminStressDataAnalyzer.py
   ```

3. **View Dashboards**
   - Open corresponding HTML dashboards
   - Select dashboard folders
   - Explore visualizations
   - Export/save important insights

4. **Interpret Results**
   - Review trends and correlations
   - Identify patterns
   - Adjust training based on insights
   - Monitor recovery status

---

## Performance Insights

### HRV Insights
- **High HRV**: Good recovery, ready for training
- **Low HRV**: May need rest or recovery day
- **Declining Trend**: Potential overtraining
- **Stable Baseline**: Good training adaptation

### Running Insights
- **ACWR 0.8-1.3**: Optimal training load (injury prevention)
- **ACWR > 1.3**: Risk of overtraining
- **ACWR < 0.8**: Possible detraining
- **HR-RS Deviation**: Lower values = better recovery

### Stress Insights
- **Negative correlation with sleep**: Poor sleep increases stress
- **Positive correlation with activity**: May indicate overtraining
- **Rising trend**: Need stress management intervention
- **High variability**: Inconsistent lifestyle factors

---

## Troubleshooting

### Charts Not Displaying
1. Verify Python script ran successfully
2. Check output directory for PNG files
3. Ensure browser supports File System Access API
4. Try different browser (Chrome/Edge recommended)
5. Check browser console for errors

### Database Errors
1. Verify database file paths
2. Check database is not locked
3. Ensure tables exist with correct schema
4. Validate date formats (YYYY-MM-DD)
5. Check for null/invalid values

### Performance Issues
1. Limit data range for large datasets
2. Increase figure DPI for better quality
3. Close unused matplotlib figures
4. Clear old chart files periodically

---

## Best Practices

### Data Management
- **Consistent data entry**: Regular, accurate logging
- **Backup databases**: Regular SQLite backups
- **Clean old charts**: Periodic cleanup of old PNGs
- **Validate inputs**: Check data quality before analysis

### Analysis Frequency
- **Daily HRV**: Morning measurements for consistency
- **Weekly Running**: After 3-5 sessions minimum
- **Weekly Stress**: Sunday review of week's data
- **Monthly Review**: Comprehensive trend analysis

### Interpretation
- **Consider context**: Training cycles, life stress, illness
- **Look for patterns**: Not single data points
- **Combine metrics**: Holistic view of health
- **Track interventions**: Note when changing training/recovery

---

## Future Enhancements

### Planned Features
- Automated data import from wearables
- Predictive analytics and ML models
- Race performance predictions
- Training plan recommendations
- Multi-user support
- Cloud storage integration
- Mobile app companion
- Real-time data streaming
- Export to PDF reports
- Social sharing features

### Potential Integrations
- Strava API
- Garmin Connect API
- Polar Flow API
- Fitbit API
- Apple Health
- Google Fit

---

## Technical Notes

### Chart Generation
- **DPI**: 300 for publication quality
- **Format**: PNG for web compatibility
- **Size**: Optimized for 1400px container width
- **Colors**: Seaborn default themes
- **Fonts**: System fonts (Segoe UI, etc.)

### Data Processing
- **Pandas**: DataFrame operations
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualizations
- **Statsmodels**: Time series decomposition

### Security
- Local file access only (File System Access API)
- No cloud data transmission
- SQLite databases stored locally
- No API keys or credentials required

---

## Support & Documentation

### Additional Resources
- See individual README files for module-specific details
- Check Python script headers for version info
- Review database schemas before modifications
- Test with sample data before production use

### Version Information
- **HRV Analytics**: v6.25-DEV
- **Running Analysis**: v6.26-DEV
- **Garmin Analyzer**: Current (December 2024)

---

## License & Credits

**Developer**: smacrico  
**Date**: December 2024  
**Purpose**: MS-Aware Fitness Tracking and Analysis

This system is designed specifically for individuals with Multiple Sclerosis to monitor health metrics, training load, and recovery status with appropriate weighting for MS-specific considerations.

---

## Quick Reference

### File Locations
```
Dashboards:   C:\temp\logsFitnessApp\*_DashBoards\
Databases:    c:\smakrykoDBs\
              C:\smakryko\myHealthData\DBs\
Scripts:      MS-Buddy-Fitness-App\[Module]\Dev_Scripts\
```

### Command Quick Start
```bash
# Navigate to each module and run:
python HRV_Analytics_v6.25-DEV.py
python RunningAnalysis_v6.26-Dev.py
python GarminStressDataAnalyzer.py

# Then open corresponding HTML dashboards in browser
```

### Dashboard URLs (local files)
```
file:///c:/smakryko/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/Dev_Scripts/HRV_Dashboard.html
file:///c:/smakryko/MS-Buddy-Fitness-App/APEX-RunAnalysis/Dev_Scripts/Run_Dashboard.html
file:///c:/smakryko/MS-Buddy-Fitness-App/Colab/Garmin_Dashboard.html
```

---

**End of Documentation**
