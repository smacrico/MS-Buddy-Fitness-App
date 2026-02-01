# Running Analytics Dashboard

## Overview

The Running Analytics Dashboard is a comprehensive system for tracking, analyzing, and visualizing running performance metrics. It consists of a Python analysis script that processes running data from an SQLite database and generates detailed performance charts, along with an interactive HTML dashboard for viewing the results.

## Features

### Python Analysis Script (`RunningAnalysis_v6.26-Dev.py`)

#### Core Metrics Tracked
- **Running Economy**: Efficiency of running movement
- **VO2Max**: Maximum oxygen uptake capacity
- **Distance & Time**: Session duration and distance covered
- **Heart Rate**: Average heart rate during sessions
- **Speed Metrics**:
  - Average Speed (km/h)
  - Max Speed (km/h)
  - Speed Reserve (Max - Avg)
  - Speed Consistency (Avg/Max ratio)
  - Pace per km (min/km)
  - Speed Efficiency (Speed/HR)

#### Advanced Analytics
- **HR-RS Deviation Index**: Heart rate recovery stress analysis
- **Cardiac Drift**: Heart rate drift during sessions
- **Physiological Efficiency**: Composite efficiency score
- **Fatigue Index**: Fatigue accumulation indicator
- **TRIMP Score**: Training impulse load calculation
- **Acute/Chronic Workload Ratio (ACWR)**: Injury risk monitoring
- **Recovery Score**: Comprehensive recovery metrics
- **Readiness Score**: Training readiness indicators

#### Visualizations Generated

1. **Training Load Analysis** (`Run_TrainingLoad_*.png`)
   - TRIMP per session over time
   - Weekly TRIMP load
   - Acute and chronic load trends
   - ACWR monitoring with safety thresholds

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
   - Overall training score over time
   - Recovery and readiness comparison
   - Multi-metric scoring visualization

6. **Speed Metrics** (`Run_SpeedMetrics_*.png`)
   - Speed trends (avg and max)
   - Speed reserve progression
   - Speed vs heart rate correlation
   - Speed efficiency timeline
   - Pace progression
   - Speed zone distribution

7. **HR-RS Deviation Index** (`Run_HRRSDeviation_*.png`)
   - Deviation index trend with rolling average
   - Deviation vs speed performance
   - Distribution histogram
   - Deviation vs TRIMP correlation

8. **Complete Performance Dashboard** (`Run_PerformanceDashboard_*.png`)
   - Comprehensive all-in-one visualization
   - 11 different metrics in a single view
   - Speed, efficiency, and physiological metrics

### HTML Dashboard (`Run_Dashboard.html`)

#### Features
- **Dynamic Chart Loading**: Uses File System Access API to load charts from local directory
- **8 Chart Categories**: Organized visualization of all analysis outputs
- **Full-Screen Modal View**: Click any chart for detailed inspection
- **Responsive Design**: Mobile-friendly layout
- **Auto-Refresh**: Easy refresh button to reload latest charts
- **Timestamp Tracking**: Shows when dashboard was last updated
- **File Name Display**: Shows which specific file is being displayed

#### Supported Browsers
- Chrome/Edge (recommended)
- Firefox
- Safari (with File System Access API support)

## Installation & Setup

### Prerequisites
- Python 3.8+
- SQLite database at `c:/smakrykoDBs/Apex.db`
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib sqlite3
  ```

### Database Schema

The script expects a `running_sessions` table with the following columns:
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

### Output Directory

All charts are automatically saved to:
```
C:\temp\logsFitnessApp\Run_DashBoards
```

The directory is created automatically if it doesn't exist.

## Usage

### Running the Analysis

1. **Execute the Python script:**
   ```bash
   python RunningAnalysis_v6.26-Dev.py
   ```

2. **Script will automatically:**
   - Load data from SQLite database
   - Calculate all derived metrics
   - Generate 8 different chart types
   - Save charts with timestamps
   - Create/update database tables:
     - `training_logs`: Full training data with calculated fields
     - `monthly_summaries`: Monthly aggregated statistics
     - `metrics_breakdown`: Detailed metrics breakdown per session

3. **Charts are saved with timestamps:**
   ```
   Run_TrainingLoad_20241215_143022.png
   Run_Trends_20241215_143023.png
   Run_SpeedMetrics_20241215_143024.png
   ...etc
   ```

### Viewing the Dashboard

1. **Open the HTML dashboard:**
   - Navigate to `Run_Dashboard.html`
   - Open in a modern web browser (Chrome/Edge recommended)

2. **Load charts:**
   - Click "Select Dashboard Folder" button
   - Navigate to `C:\temp\logsFitnessApp\Run_DashBoards`
   - Select the folder

3. **View charts:**
   - Dashboard automatically displays the latest chart for each category
   - Click any chart for full-screen view
   - Press ESC or click the X to close modal

4. **Refresh charts:**
   - After running the Python script again
   - Click "Refresh Dashboard" button
   - Latest charts will be loaded automatically

## Key Metrics Explained

### Speed Reserve
Difference between maximum and average speed. Higher values indicate greater speed variability and potential for improvement.

### Speed Consistency
Ratio of average to maximum speed. Values closer to 1.0 indicate more consistent pacing.

### Speed Efficiency
Speed per heart rate unit. Higher values indicate better cardiovascular efficiency.

### HR-RS Deviation Index
Measures heart rate recovery stress. Lower values indicate better recovery capacity and adaptation.

### Physiological Efficiency
Composite score combining speed, heart rate, and HR-RS deviation. Higher values indicate better overall efficiency.

### Fatigue Index
Combines HR-RS deviation and cardiac drift relative to speed. Higher values indicate greater fatigue accumulation.

### TRIMP (Training Impulse)
Quantifies training load based on duration and heart rate intensity. Used for tracking cumulative training stress.

### ACWR (Acute:Chronic Workload Ratio)
Ratio of recent (1 week) to long-term (4 week) training load. Values between 0.8-1.3 are considered optimal for injury prevention.

## Database Integration

### Tables Created/Updated

1. **training_logs**
   - Complete training data with all calculated fields
   - Updated on each script run
   - Includes all raw and derived metrics

2. **monthly_summaries**
   - Monthly aggregated statistics
   - Mean and standard deviation for all metrics
   - Session counts per month
   - Useful for long-term trend analysis

3. **metrics_breakdown**
   - Detailed breakdown of training scores
   - Normalized and weighted metric values
   - Performance trends and correlations
   - Daily snapshots of calculated scores

### Adding New Sessions

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

## Performance Insights

### Monthly Analysis
The script generates monthly summaries showing:
- Session counts
- Average and standard deviation for all metrics
- Trend indicators
- Performance consistency

### Training Score
Comprehensive training score (0-100) based on:
- Running Economy (25%)
- VO2Max (20%)
- Distance (15%)
- Efficiency Score (20%)
- Heart Rate (20%)

### Recovery & Readiness
Calculated from:
- Resting heart rate (30%)
- Training load (30%)
- Sleep quality (20%)
- Fatigue level (20%)

## Troubleshooting

### Charts Not Displaying
1. Ensure Python script has been run recently
2. Check that charts exist in `C:\temp\logsFitnessApp\Run_DashBoards`
3. Use "Select Dashboard Folder" to manually select the folder
4. Verify browser supports File System Access API

### Database Errors
1. Verify database path: `c:/smakrykoDBs/Apex.db`
2. Check database schema matches expected structure
3. Ensure database is not locked by another process

### Missing Data
1. Check if `running_sessions` table has data
2. Verify date formats are correct (YYYY-MM-DD)
3. Ensure numeric fields are not null or invalid

### Chart Quality Issues
1. Charts are saved at 300 DPI for high quality
2. Use full-screen modal for detailed viewing
3. Generate charts on larger display if needed

## Best Practices

### Data Entry
- Enter sessions consistently after each workout
- Include all optional fields when available
- Use accurate heart rate data for best analysis

### Analysis Frequency
- Run analysis weekly for trend monitoring
- Generate monthly reports for long-term tracking
- Review ACWR regularly for injury prevention

### Performance Monitoring
- Track speed metrics for pace improvement
- Monitor HR-RS deviation for recovery status
- Use efficiency scores to optimize training intensity

### Dashboard Usage
- Keep browser tab open for quick access
- Refresh after each analysis run
- Export important charts for sharing/reports

## Future Enhancements

Potential additions:
- Race prediction models
- Training plan recommendations
- Automated anomaly detection
- Integration with fitness trackers
- Real-time data streaming
- Multi-user support
- Cloud storage integration

## License

(c) smacrico - Dec 2024

## Support

For issues or questions, refer to the main project documentation or contact the development team.

---

**Last Updated**: December 2024  
**Version**: 6.26-DEV
