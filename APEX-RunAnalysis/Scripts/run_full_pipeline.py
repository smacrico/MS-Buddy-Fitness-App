#!/usr/bin/env python3
"""
APEX Running Analysis - Full Pipeline Orchestrator
Runs the complete data pipeline: ETL → Processing → Dashboard

Pipeline Flow:
1. createRunAnalDB (ETL) - Extracts from source DBs and populates Apex.db
2. RunningAnalysis - Loads, processes, and calculates TRIMP/performance metrics
3. Dashboard - Launches interactive visualization (Streamlit)

Usage:
    python run_full_pipeline.py [--skip-etl] [--skip-dashboard] [--verbose]
    
Options:
    --skip-etl          Skip the ETL step (use existing Apex.db)
    --skip-dashboard    Skip launching the dashboard
    --verbose           Print detailed debug information
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Get the current script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Import the RunningAnalysis class
sys.path.insert(0, str(SCRIPT_DIR))

# Logging configuration
class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"
    
    def log(self, level, message):
        timestamp = datetime.now().strftime(self.timestamp_format)
        prefix = f"[{timestamp}] [{level}]"
        print(f"{prefix} {message}")
    
    def info(self, message):
        self.log("INFO", message)
    
    def success(self, message):
        self.log("✓ SUCCESS", message)
    
    def warning(self, message):
        self.log("⚠ WARNING", message)
    
    def error(self, message):
        self.log("✗ ERROR", message)
    
    def debug(self, message):
        if self.verbose:
            self.log("DEBUG", message)


class RunningAnalysisPipeline:
    """Orchestrates the full running analysis pipeline"""
    
    def __init__(self, skip_etl=False, skip_dashboard=False, verbose=False):
        self.logger = Logger(verbose=verbose)
        self.skip_etl = skip_etl
        self.skip_dashboard = skip_dashboard
        self.db_path = r'c:/smakrykoDBs/Apex.db'
        self.analysis = None
        
        self.logger.info("=" * 70)
        self.logger.info("APEX Running Analysis - Full Pipeline")
        self.logger.info("=" * 70)
    
    def check_prerequisites(self):
        """Verify all required files and dependencies exist"""
        self.logger.info("Step 1: Checking prerequisites...")
        
        required_files = {
            'data_loader': 'createRunAnalDB - v6.26.py',
            'analysis_module': 'RunningAnalysis_v6.26.py',
            'dashboard': 'app.py',
        }
        
        missing = []
        for key, filename in required_files.items():
            filepath = SCRIPT_DIR / filename
            if filepath.exists():
                self.logger.debug(f"  ✓ Found {filename}")
            else:
                missing.append(filename)
                self.logger.error(f"  ✗ Missing {filename}")
        
        if missing:
            self.logger.error(f"\nMissing required files: {', '.join(missing)}")
            return False
        
        # Check Python dependencies
        required_packages = ['pandas', 'numpy', 'sqlite3', 'streamlit']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"  ✓ Package {package} available")
            except ImportError:
                if package != 'sqlite3':  # sqlite3 is built-in
                    missing_packages.append(package)
        
        if missing_packages:
            self.logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        self.logger.success("All prerequisites met")
        return True
    
    def run_etl(self):
        """Step 1: Run the ETL script to populate Apex.db"""
        if self.skip_etl:
            self.logger.info("Step 2: Skipping ETL (--skip-etl flag)")
            self.logger.info(f"  Using existing database at {self.db_path}")
            return True
        
        self.logger.info("Step 2: Running ETL - Processing source databases...")
        self.logger.info("  Extracting from: artemis.db + garmin_activities.db")
        self.logger.info("  Populating: Apex.db (running_sessions table)")
        
        etl_script = SCRIPT_DIR / 'createRunAnalDB - v6.26.py'
        
        try:
            result = subprocess.run(
                [sys.executable, str(etl_script)],
                cwd=str(SCRIPT_DIR),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.success("ETL completed successfully")
                if result.stdout:
                    self.logger.debug(f"ETL Output:\n{result.stdout}")
                return True
            else:
                self.logger.error(f"ETL failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error output:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("ETL script timed out (exceeded 5 minutes)")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run ETL: {str(e)}")
            return False
    
    def load_and_process_data(self):
        """Step 2: Load data and process with RunningAnalysis class"""
        self.logger.info("Step 3: Loading and processing training data...")
        
        try:
            # Import RunningAnalysis
            from RunningAnalysis_v6_26 import RunningAnalysis
            
            # Initialize analysis object
            self.logger.debug("Initializing RunningAnalysis class...")
            self.analysis = RunningAnalysis(self.db_path)
            
            # Check if data was loaded
            if self.analysis.training_log.empty:
                self.logger.warning("No training data found in database")
                self.logger.info("  Hint: Run ETL with --skip-etl flag and check source databases")
                return False
            
            data_count = len(self.analysis.training_log)
            self.logger.info(f"  Loaded {data_count} training sessions")
            
            # Verify key calculations
            if hasattr(self.analysis, 'weekly_trimp'):
                weeks = len(self.analysis.weekly_trimp)
                self.logger.info(f"  Calculated weekly metrics for {weeks} weeks")
            
            self.logger.success("Data processing completed")
            return True
            
        except ImportError as e:
            self.logger.error(f"Failed to import RunningAnalysis: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def generate_summary_report(self):
        """Generate a brief summary of processed data"""
        self.logger.info("Step 4: Generating summary report...")
        
        try:
            if self.analysis is None or self.analysis.training_log.empty:
                self.logger.warning("No data available for summary")
                return
            
            df = self.analysis.training_log
            
            # Basic statistics
            self.logger.info(f"\n{'─' * 70}")
            self.logger.info("DATA SUMMARY")
            self.logger.info(f"{'─' * 70}")
            self.logger.info(f"Total Sessions:        {len(df)}")
            self.logger.info(f"Date Range:            {df['date'].min()} to {df['date'].max()}")
            
            # Metrics summary
            metrics_to_show = {
                'running_economy': 'Running Economy',
                'vo2max': 'VO2Max',
                'distance': 'Distance (km)',
                'TRIMP': 'TRIMP Score',
                'heart_rate': 'Avg Heart Rate',
                'efficiency_score': 'Efficiency Score'
            }
            
            self.logger.info(f"\n{'METRIC':<25} {'MEAN':<12} {'MIN':<12} {'MAX':<12}")
            self.logger.info(f"{'─' * 70}")
            
            for col, label in metrics_to_show.items():
                if col in df.columns:
                    mean = df[col].mean()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    self.logger.info(f"{label:<25} {mean:>10.2f}  {min_val:>10.2f}  {max_val:>10.2f}")
            
            if hasattr(self.analysis, 'weekly_trimp'):
                acwr = self.analysis.weekly_trimp['acwr'].dropna()
                if not acwr.empty:
                    self.logger.info(f"\n{'─' * 70}")
                    self.logger.info(f"Latest ACWR:          {acwr.iloc[-1]:.2f}")
                    self.logger.info(f"  (Optimal range: 0.8-1.3)")
            
            self.logger.info(f"{'─' * 70}\n")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {str(e)}")
    
    def launch_dashboard(self):
        """Step 3: Launch the Streamlit dashboard"""
        if self.skip_dashboard:
            self.logger.info("Step 5: Skipping dashboard launch (--skip-dashboard flag)")
            return True
        
        self.logger.info("Step 5: Launching dashboard...")
        self.logger.info("  Starting Streamlit application...")
        
        dashboard_script = SCRIPT_DIR / 'app.py'
        
        if not dashboard_script.exists():
            self.logger.error(f"Dashboard script not found: {dashboard_script}")
            return False
        
        try:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("Opening dashboard in browser (http://localhost:8501)")
            self.logger.info("Press Ctrl+C to stop the dashboard")
            self.logger.info("=" * 70 + "\n")
            
            # Launch streamlit
            subprocess.run(
                [sys.executable, '-m', 'streamlit', 'run', str(dashboard_script)],
                cwd=str(SCRIPT_DIR)
            )
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("\nDashboard stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Failed to launch dashboard: {str(e)}")
            return False
    
    def run(self):
        """Execute the full pipeline"""
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                self.logger.error("\nPipeline aborted: prerequisites not met")
                return False
            
            # Step 2: Run ETL
            if not self.run_etl():
                self.logger.error("\nPipeline aborted: ETL failed")
                self.logger.info("Troubleshooting:")
                self.logger.info("  - Check source databases exist (artemis.db, garmin_activities.db)")
                self.logger.info("  - Verify database paths in createRunAnalDB - v6.26.py")
                self.logger.info("  - Try: python run_full_pipeline.py --skip-etl (to use existing data)")
                return False
            
            # Step 3: Load and process
            if not self.load_and_process_data():
                self.logger.error("\nPipeline aborted: data processing failed")
                return False
            
            # Step 4: Summary report
            self.generate_summary_report()
            
            # Step 5: Launch dashboard
            self.logger.info("=" * 70)
            if self.skip_dashboard:
                self.logger.success("Pipeline completed successfully!")
                self.logger.info("To launch dashboard, run: streamlit run app.py")
            else:
                if not self.launch_dashboard():
                    self.logger.warning("Dashboard launch failed, but pipeline completed")
                    return True
            
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("\nPipeline interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return False


def print_usage():
    """Print help information"""
    print(__doc__)
    print("\nExamples:")
    print("  python run_full_pipeline.py              # Run full pipeline with dashboard")
    print("  python run_full_pipeline.py --skip-etl   # Skip ETL, use existing data")
    print("  python run_full_pipeline.py --skip-dashboard  # Run pipeline, no dashboard")
    print("  python run_full_pipeline.py --verbose    # Run with detailed output")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='APEX Running Analysis - Full Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--skip-etl',
        action='store_true',
        help='Skip ETL step and use existing Apex.db'
    )
    
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard launch'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed debug information'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = RunningAnalysisPipeline(
        skip_etl=args.skip_etl,
        skip_dashboard=args.skip_dashboard,
        verbose=args.verbose
    )
    
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
