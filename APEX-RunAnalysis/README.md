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

### Database Integration
- SQLite database storage for training sessions
- Automatic metrics breakdown storage
- Training logs persistence
- Support for multiple scoring methods and historical tracking