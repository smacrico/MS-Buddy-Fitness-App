# APEX Running Analysis - Full Pipeline

Complete end-to-end orchestration for running performance analysis: **ETL → Data Processing → Interactive Dashboard**

## Quick Start

### Windows
```bash
# Run full pipeline (ETL + Processing + Dashboard)
run_pipeline.bat

# Skip ETL step (use existing data)
run_pipeline.bat --skip-etl

# Skip dashboard launch
run_pipeline.bat --skip-dashboard --skip-etl
```

### macOS / Linux
```bash
# Make script executable (first time only)
chmod +x run_pipeline.sh

# Run full pipeline
./run_pipeline.sh

# Skip ETL
./run_pipeline.sh --skip-etl

# Skip dashboard
./run_pipeline.sh --skip-etl --skip-dashboard
```

### Python (Cross-platform)
```bash
# Run full pipeline
python run_full_pipeline.py

# Skip ETL
python run_full_pipeline.py --skip-etl

# Skip dashboard
python run_full_pipeline.py --skip-dashboard

# Verbose output (debug)
python run_full_pipeline.py --verbose
```

---

## Pipeline Architecture

### Stage 1: ETL (Extract, Transform, Load)
**Script:** `createRunAnalDB - v6.26.py`

Extracts running data from source databases and populates the analysis database:

```
Source Databases (Input)
├── artemis.db
│   └── Artemistbl_fields (running metrics)
└── garmin_activities.db
    └── activities (device data: speeds, heart rate)
          ↓
[SQL JOIN on activity_id]
          ↓
Apex.db (Output)
└── running_sessions (12 fields)
    ├── Basic: running_economy, vo2max, distance, time, date
    ├── Heart Rate: heart_rate, cardiacdrift
    ├── Speed: avg_speed, max_speed
    └── HR Analysis: HR_RS_Deviation_Index
```

**Key Fields Extracted:**
- `running_economy` - ml/kg/min
- `vo2max` - max oxygen uptake
- `distance` - km
- `heart_rate` - avg bpm
- `avg_speed`, `max_speed` - km/h
- `HR_RS_Deviation_Index` - cardiac autonomic balance
- `cardiacdrift` - HR drift during session

### Stage 2: Data Processing
**Script:** `RunningAnalysis_v6.26.py`

Loads data from Apex.db and calculates derived metrics:

```
Apex.db
└── running_sessions (raw data)
          ↓
[RunningAnalysis class]
          ↓
Calculated Metrics
├── TRIMP Score
│   ├── HR Reserve Ratio
│   └── Duration × Intensity
├── Weekly Loads
│   ├── Acute Load (1-week rolling)
│   ├── Chronic Load (4-week rolling)
│   └── ACWR (Acute:Chronic ratio)
├── Performance Scores
│   ├── Efficiency Score: running_economy / vo2max
│   ├── Energy Cost: running_economy × (distance / time)
│   └── Speed Zones: Slow/Moderate/Fast
└── Monthly Summaries
    └── Aggregated statistics per month
```

**Key Calculations:**
```python
# TRIMP (Training Impulse)
TRIMP = duration_min × (heart_rate - rest_hr) / (max_hr - rest_hr)

# Efficiency Score
Efficiency = running_economy / vo2max

# ACWR (Injury Risk Indicator)
ACWR = acute_load / chronic_load
# Optimal: 0.8-1.3 (too low = undertraining, too high = overtraining)

# Speed Zones
- Slow: < 10 km/h
- Moderate: 10-14 km/h  
- Fast: > 14 km/h
```

### Stage 3: Interactive Dashboard
**Scripts:** `app.py` & `RunAnalysis_PythonVisuals.py`

Launches Streamlit web dashboard for visualization:

```
Dashboard Features
├── Session Filtering
│   └── Date range selector
├── Performance Metrics
│   ├── Running Economy trends
│   ├── VO2Max progression
│   ├── Distance analysis
│   └── Heart rate patterns
├── Training Load
│   ├── TRIMP per session (line plot)
│   ├── Weekly TRIMP load
│   ├── Acute/Chronic load trends
│   └── ACWR risk zones
├── Performance Profile
│   └── Normalized radar chart (5 metrics)
└── Score Visualizations
    ├── Recovery score
    ├── Readiness score
    └── Overall training score
```

**Dashboard URL:** `http://localhost:8501`

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- 500MB disk space for databases

### Required Databases
```
c:/smakrykoDBs/
├── artemis.db           (Required for ETL)
├── garmin_activities.db (Required for ETL)
└── Apex.db              (Created by ETL if doesn't exist)
```

### Required Python Packages
```
pandas       # Data manipulation
numpy        # Numerical computing
plotly       # Interactive visualizations
streamlit    # Web dashboard
sqlite3      # Database access (built-in)
```

**Install dependencies:**
```bash
pip install pandas numpy plotly streamlit
```

---

## Usage Examples

### Scenario 1: First-time Setup
```bash
# Run full pipeline (creates Apex.db from source databases)
python run_full_pipeline.py

# Takes 2-5 minutes depending on data volume
# Opens dashboard at http://localhost:8501
```

### Scenario 2: Refresh Dashboard with Existing Data
```bash
# Skip ETL (database already populated)
python run_full_pipeline.py --skip-etl

# Much faster - only re-processes existing data
```

### Scenario 3: Development/Testing
```bash
# Skip dashboard to test data pipeline
python run_full_pipeline.py --skip-etl --skip-dashboard

# Check data processing without opening browser
# Review summary statistics in console
```

### Scenario 4: Troubleshooting
```bash
# Run with verbose output to see detailed logs
python run_full_pipeline.py --verbose

# Outputs debug information for each pipeline stage
```

---

## Pipeline Output

The script provides real-time status updates:

```
[2025-01-15 14:32:15] [INFO] ======= APEX Running Analysis - Full Pipeline =======
[2025-01-15 14:32:15] [INFO] Step 1: Checking prerequisites...
[2025-01-15 14:32:15] [✓ SUCCESS] All prerequisites met

[2025-01-15 14:32:15] [INFO] Step 2: Running ETL...
[2025-01-15 14:32:47] [✓ SUCCESS] ETL completed successfully

[2025-01-15 14:32:47] [INFO] Step 3: Loading and processing training data...
[2025-01-15 14:32:52] [INFO]   Loaded 287 training sessions
[2025-01-15 14:32:52] [INFO]   Calculated weekly metrics for 42 weeks
[2025-01-15 14:32:52] [✓ SUCCESS] Data processing completed

[2025-01-15 14:32:52] [INFO] Step 4: Generating summary report...
[2025-01-15 14:32:52] [INFO] DATA SUMMARY
[2025-01-15 14:32:52] [INFO] Total Sessions:        287
[2025-01-15 14:32:52] [INFO] Date Range:            2024-01-01 to 2025-01-15

[2025-01-15 14:32:52] [INFO] Step 5: Launching dashboard...
[2025-01-15 14:32:55] [INFO] Opening dashboard in browser (http://localhost:8501)
```

---

## Output Databases

### Apex.db Tables

#### 1. running_sessions (Primary)
| Column | Type | Description |
|--------|------|-------------|
| date | TEXT | Session date (YYYY-MM-DD) |
| running_economy | INT | ml/kg/min |
| vo2max | INT | ml/min/kg |
| distance | INT | km |
| time | INT | seconds |
| heart_rate | INT | avg bpm |
| sport | TEXT | Activity type |
| cardiacdrift | INT | HR drift during session |
| avg_speed | REAL | km/h |
| max_speed | REAL | km/h |
| HR_RS_Deviation_Index | INT | Cardiac autonomic balance |

#### 2. metrics_breakdown (Created by RunningAnalysis)
Stores 48 fields per session:
- Overall score
- 5 normalized metrics (economy, vo2max, distance, efficiency, HR)
- 5 weighted metrics
- Raw statistics (mean/std for each)
- Performance trends

#### 3. monthly_summaries (Created by RunningAnalysis)
Aggregated monthly statistics with mean/std for:
- All primary metrics
- Recovery & readiness scores
- TRIMP load

---

## Troubleshooting

### Problem: "Python is not installed"
**Solution:** Install Python 3.8+ from https://www.python.org/downloads/

### Problem: "missing required files"
**Solution:** Ensure you're running from the Scripts directory:
```bash
cd c:\smakrykoDBs\Odyssey\APEX-RunAnalysis\Scripts
python run_full_pipeline.py
```

### Problem: "ETL failed"
**Causes & Solutions:**
- Source databases don't exist:
  ```bash
  # Check paths are correct in createRunAnalDB - v6.26.py
  # Verify: c:/smakrykoDBs/artemis.db
  # Verify: c:/smakrykoDBs/garmin_activities.db
  ```
- Permission denied:
  ```bash
  # Run as administrator (Windows)
  # sudo python run_full_pipeline.py (Linux/Mac)
  ```
- Use existing data:
  ```bash
  python run_full_pipeline.py --skip-etl
  ```

### Problem: "No training data found"
**Solution:** Check if database was populated:
```bash
# Run test script to verify database structure
python test_metrics_db.py
```

### Problem: Dashboard won't open
**Solution:** Streamlit may already be running
```bash
# Kill existing Streamlit process
# Windows: taskkill /IM streamlit.exe
# Linux/Mac: pkill streamlit

# Then retry
python run_full_pipeline.py --skip-etl
```

### Problem: "Port 8501 already in use"
**Solution:** Streamlit uses a different port
```bash
# Streamlit will automatically use port 8502, 8503, etc.
# Or manually specify port:
streamlit run app.py --server.port=8888
```

---

## Performance Notes

### Typical Execution Times
- **Full Pipeline (first run):** 5-10 minutes
  - ETL: 2-5 min (depends on source database size)
  - Processing: 30 sec - 2 min
  - Dashboard: Instant

- **Subsequent Runs (with --skip-etl):** 30-60 seconds
  - No database extraction
  - Only re-process existing data

### Optimization Tips
- Use `--skip-etl` when data hasn't changed
- Use `--skip-dashboard` for batch processing
- Close dashboard when not using (frees memory)

---

## Configuration

### User Constants (in RunningAnalysis class)
Located in `RunningAnalysis_v6.26.py`:
```python
rest_hr = 60      # Your resting heart rate
max_hr = 190      # Your max heart rate
```

Customize these based on your fitness level.

### Database Paths
Default: `c:/smakrykoDBs/Apex.db`

To change, edit in `createRunAnalDB - v6.26.py`:
```python
conn = sqlite3.connect(r'c:/your/custom/path.db')
```

---

## Next Steps

1. **Understand the Data:**
   - Review dashboard visualizations
   - Check console output for data summary

2. **Configure User Parameters:**
   - Update rest_hr and max_hr in RunningAnalysis class
   - Adjust TRIMP thresholds if needed

3. **Explore Enhancements:**
   - See `RunningAnalysis_proposed_enhancements.py` for upcoming features
   - Speed-based metrics
   - HR-RS deviation analysis
   - Physiological efficiency scoring

4. **Automate Runs:**
   - Schedule pipeline with Windows Task Scheduler (Windows)
   - Use cron job (Linux/Mac)
   - Example: Run pipeline weekly to update dashboard

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review log output with `--verbose` flag
3. Check database integrity with `test_metrics_db.py`
4. Review source code comments in individual scripts

---

## License & Credit
(c) smacrico - 2024-2025

Part of the APEX Running Analysis system for tracking and visualizing running performance metrics.
