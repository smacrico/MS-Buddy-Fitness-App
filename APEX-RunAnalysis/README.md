# APEX Running Analysis System

## Recent Enhancements (v6.26 - January 2026)

### 🔧 Bug Fixes & Improvements
- **Fixed script execution**: Added proper `if __name__ == "__main__"` guard to enable direct script execution
- **Fixed SQL syntax errors**: Corrected column naming in `monthly_summaries` table (replaced dots with underscores)
- **Database table creation**: Automated creation of `monthly_summaries` and `metrics_breakdown` tables

### 📊 New Features
- **Speed Analysis Module**: Comprehensive speed metrics tracking and analysis
  - Speed zones classification
  - Speed efficiency calculations
  - Pace progression monitoring
  - Speed reserve analysis

- **HR-RS Deviation Index**: Advanced fatigue detection system
  - Heart rate to running speed correlation
  - Deviation trend analysis
  - Performance impact assessment
  - Integration with training load metrics

### 💾 Automated Visualization Export
All charts now automatically save as high-resolution PNG files (300 DPI):
- 8 different visualization outputs
- Saved to: `c:/temp/logsFitnessApp/`
- Print confirmation messages with file paths
- Publication-ready image quality

### 📈 Enhanced Database Features
- Monthly summaries with 30+ aggregated metrics
- Automatic INSERT/UPDATE with conflict resolution
- Comprehensive error logging and debugging
- Support for new speed and deviation metrics

---

## Project Overview

Main Application Files:

app_v1.0.py - The main application file (version 1.0)
createRunAnalDB_dev.py - Database creation script (development version)

Documentation:
Comprehensive wiki documentation in the docs/wiki/ folder covering:
Project overview
Getting started guide
Architecture
Installation & deployment
API reference
Developer guidelines
Testing procedures
CI/CD setup
Security considerations
Performance monitoring
And more...

Visualization Components (pyVisuals/):
VisualiseBasicTrends.py - Basic trend visualization
VisualiseAdvance.py - Advanced visualization features
VisualiseRecRedy.py - Recovery readiness visualization
VisualizationTrainingLoad.py - Training load visualization

Scripts Directory:
Contains different versions of running analysis implementation:
RunningAnalysis_v50.py
RunningAnalysis_v60.py
app.py - Another application script
createRunAnalDB.py - Database creation script

## Key Features

### Analytics & Metrics
- **Training Score Calculation**: Comprehensive scoring based on multiple performance metrics including:
  - Running Economy (25% weight)
  - VO2Max (20% weight)
  - Distance (15% weight)
  - Efficiency Score (20% weight)
  - Heart Rate (20% weight)

- **Monthly Metrics Breakdown**: Automatically calculates and displays monthly averages for all key metrics:
  - Running Economy with standard deviation
  - VO2Max performance
  - Distance covered
  - Efficiency scores
  - Heart Rate averages
  - Energy Cost
  - TRIMP (Training Impulse)
  - Recovery and Readiness scores (when available)
  - Session counts per month

- **Training Load Analysis**: 
  - TRIMP calculation and visualization
  - Acute vs Chronic load monitoring
  - ACWR (Acute:Chronic Workload Ratio) tracking
  - Weekly training load trends

- **Recovery & Readiness Monitoring**:
  - Recovery score calculation based on resting HR, load, sleep quality, and fatigue
  - Readiness score for optimal training planning
  - Visual tracking over time with threshold indicators

### Visualizations
- Basic trend analysis for running economy and efficiency
- Advanced visualizations including:
  - Cumulative distance tracking
  - Moving averages
  - Training zones distribution
  - Performance radar charts
  - Seasonal performance heatmaps
- Training load and ACWR visualization
- Recovery and readiness trend charts

### Speed & Performance Analysis (NEW)
- **Speed Metrics Analysis**:
  - Average and maximum speed tracking
  - Speed reserve calculation (Max - Avg speed)
  - Speed consistency metrics (Avg/Max ratio)
  - Pace per kilometer tracking
  - Speed efficiency (Speed per heart rate unit)
  - Economy at speed calculation
  - Speed-VO2max index
  - Speed zone classification (Slow, Moderate, Fast)
  - Speed improvement trends over time

- **HR-RS Deviation Index**:
  - Heart Rate to Running Speed deviation tracking
  - Correlation analysis with performance metrics
  - Trend detection for fatigue monitoring
  - Distribution analysis and variability assessment
  - Integration with TRIMP for comprehensive load monitoring

### Enhanced Visualizations
All visualizations are now automatically saved as high-resolution PNG files (300 DPI) in `c:/temp/logsFitnessApp/`:

1. **trends.png** - Core performance trends:
   - Running economy over time
   - Efficiency score progression
   - Energy cost vs distance
   - Heart rate vs running economy

2. **advanced_metrics.png** - Comprehensive analysis:
   - Cumulative distance tracking
   - Running economy moving averages (3-session)
   - Pace vs heart rate correlation
   - Training zones pie chart distribution
   - Performance metrics radar chart
   - Seasonal performance heatmap

3. **training_load.png** - Load management:
   - TRIMP per session timeline
   - Weekly TRIMP load trends
   - Acute load (1-week average)
   - Chronic load (4-week average)
   - ACWR with threshold indicators

4. **recovery_readiness.png** - Readiness monitoring:
   - Recovery score over time
   - Readiness score tracking
   - Caution threshold visualization

5. **score_impact.png** - Training score analysis:
   - Overall training score timeline
   - Multiple scoring method comparison
   - Score trends and patterns

6. **speed_metrics.png** - Speed performance (6 panels):
   - Speed trends (Average & Max)
   - Speed reserve over time
   - Speed vs heart rate scatter (time-coded)
   - Speed efficiency progression
   - Pace progression (inverted for clarity)
   - Speed zone distribution bar chart

7. **hr_rs_deviation.png** - Deviation analysis (4 panels):
   - HR-RS deviation index timeline with rolling average
   - Deviation vs speed performance correlation
   - Deviation distribution histogram
   - Deviation vs TRIMP relationship

8. **performance_dashboard.png** - Comprehensive multi-panel overview

### Database Integration
- **SQLite database storage** for training sessions at `c:/smakrykoDBs/Apex.db`
- **Automatic metrics breakdown storage** with date-stamped records
- **Monthly summaries table** with automated aggregation:
  - Sessions count per month
  - Mean and standard deviation for all metrics
  - Running economy, VO2max, distance, efficiency
  - Heart rate, energy cost, TRIMP
  - Recovery and readiness scores
  - Speed metrics (avg, max, reserve)
  - HR-RS deviation and speed efficiency
- **Training logs persistence** with calculated fields
- **Support for multiple scoring methods** and historical tracking
- **Automatic data validation** and error handling

## Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- Required packages: pandas, numpy, matplotlib, sqlite3

### Running the Analysis

```bash
# Navigate to the scripts directory
cd c:\smakryko\MS-Buddy-Fitness-App\APEX-RunAnalysis\Scripts

# Run the main analysis script
python RunningAnalysis_v6.26.py
```

### Output Files
All visualizations are automatically saved to: `c:/temp/logsFitnessApp/`

**Generated files:**
- `trends.png` - Core performance trends
- `advanced_metrics.png` - Comprehensive multi-panel analysis
- `training_load.png` - TRIMP and load management
- `recovery_readiness.png` - Recovery monitoring
- `score_impact.png` - Training score timeline
- `speed_metrics.png` - Speed analysis (6 panels)
- `hr_rs_deviation.png` - HR-RS deviation analysis (4 panels)
- `performance_dashboard.png` - Complete overview dashboard

### Database Tables

**running_sessions**: Raw training data
- Date, distance, time, heart rate
- Running economy, VO2max
- Speed metrics (avg, max, reserve, consistency)
- HR-RS deviation, cardiac drift
- Calculated fields (TRIMP, efficiency, zones)

**monthly_summaries**: Aggregated monthly statistics
- Automatic monthly rollup with mean/std for all metrics
- Session counts per month
- Updated on each run with UPSERT logic

**metrics_breakdown**: Training score history
- Date-stamped score calculations
- Normalized and weighted component values
- Performance trend indicators

## Performance Metrics Explained

### Running Economy
Measure of running efficiency - lower oxygen consumption at a given pace indicates better economy.

### TRIMP (Training Impulse)
Quantifies training load based on duration and heart rate intensity.

### ACWR (Acute:Chronic Workload Ratio)
Ratio of recent (1 week) to longer-term (4 weeks) training load. Optimal range: 0.8-1.3

### HR-RS Deviation Index
Measures the relationship between heart rate and running speed. Increasing deviation may indicate fatigue or decreased fitness.

### Speed Zones
- **Slow**: < 6.0 km/h
- **Moderate**: 6.0 - 8.0 km/h  
- **Fast**: > 8.0 km/h

## Version History

### v6.26 (January 2026)
- Added speed metrics analysis module
- Implemented HR-RS deviation tracking
- Fixed SQL syntax errors in database operations
- Added automatic visualization file export (8 charts)
- Enhanced monthly summaries with 30+ metrics
- Improved error handling and debugging output

### v6.0
- Major refactor to use SQLite database
- Added TRIMP and ACWR calculations
- Implemented recovery and readiness scoring

### v5.0
- Added advanced visualizations
- Training zones implementation
- Performance radar charts

---

**Database Location**: `c:/smakrykoDBs/Apex.db`  
**Output Directory**: `c:/temp/logsFitnessApp/`  
**Current Version**: RunningAnalysis_v6.26.py