"""
HRV Analytics Data Warehouse - Version 2.1
Refactored for consistency, bug fixes, and record insertion support

Key improvements:
- Consistent parameter naming across recovery score functions
- Fixed SQL logic (= instead of IS, aligned column names)
- Added error handling and safe division helpers
- Schema validation and auto-creation
- Improved type hints and documentation
- Added insert_hrv_record() method
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRVAnalytics:
    """
    HRV Analytics class for processing heart rate variability data
    with consistent recovery scoring and trend analysis.
    """
    
    def __init__(self, db_path: str = "c:/smakrykoDBs/Mercury_DWH_HRV.db"):
        self.db_path = db_path
        self.recovery_constants = {
            'simple': {
                'rmssd_weight': 0.4,
                'sdnn_weight': 0.3,
                'pnn50_weight': 0.3,
                'rmssd_scale': 0.8,
                'sdnn_scale': 1.0,
                'pnn50_scale': 2.0
            },
            'comprehensive': {
                'time_domain_weight': 0.4,
                'freq_domain_weight': 0.35,
                'stress_weight': 0.25,
                'rmssd_scale': 0.8,
                'sdnn_scale': 1.0,
                'pnn50_scale': 2.0,
                'lf_scale': 100.0,
                'hf_scale': 100.0,
                'stress_scale': 10.0
            }
        }
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='hrv_data'
                """)
                if not cursor.fetchone():
                    logger.info("Creating hrv_data table...")
                    cursor.execute("""
                        CREATE TABLE hrv_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date TEXT NOT NULL,
                            name TEXT NOT NULL,
                            rmssd REAL,
                            sdnn REAL,
                            pnn50 REAL,
                            lf_power REAL,
                            hf_power REAL,
                            stress_index REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("Schema created successfully")
        except sqlite3.Error as e:
            logger.error(f"Database schema error: {e}")

    def insert_hrv_record(self, date: str, name: str = "HRV", 
                          rmssd: Optional[float] = None, sdnn: Optional[float] = None,
                          pnn50: Optional[float] = None, lf_power: Optional[float] = None,
                          hf_power: Optional[float] = None, stress_index: Optional[float] = None) -> bool:
        """Insert a new HRV record into the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO hrv_data (date, name, rmssd, sdnn, pnn50, lf_power, hf_power, stress_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, name, rmssd, sdnn, pnn50, lf_power, hf_power, stress_index))
                conn.commit()
                logger.info(f"Inserted HRV record for {date} - {name}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting HRV record: {e}")
            return False

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if denominator != 0 else default

    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        return max(min_val, min(max_val, score))

    def get_daily_hrv_dataframe(self, days_back: int = 30, source_name: str = "HRV") -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT date, name, rmssd, sdnn, pnn50, lf_power, hf_power, stress_index
                    FROM hrv_data 
                    WHERE name = ? 
                    AND date >= date('now', ?)
                    ORDER BY date DESC
                """
                df = pd.read_sql_query(query, conn, params=[source_name, f'-{days_back} days'])
                if df.empty:
                    logger.warning("No data found, generating sample data")
                    return self._generate_sample_trend_data(days_back, source_name)
                df['date'] = pd.to_datetime(df['date'])
                numeric_columns = ['rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'stress_index']
                df[numeric_columns] = df[numeric_columns].fillna(0)
                return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return self._generate_sample_trend_data(days_back, source_name)

    def _generate_sample_trend_data(self, days: int = 30, source_name: str = "HRV") -> pd.DataFrame:
        logger.info(f"Generating {days} days of sample HRV data for {source_name}")
        dates = pd.date_range(start=datetime.now() - timedelta(days=days-1), end=datetime.now(), freq='D')
        base_rmssd, base_sdnn, base_pnn50 = 45, 50, 15
        np.random.seed(42)
        data = {
            'date': dates,
            'name': [source_name] * len(dates),
            'rmssd': base_rmssd + np.random.normal(0, 8, len(dates)) + np.linspace(-5, 5, len(dates)),
            'sdnn': base_sdnn + np.random.normal(0, 10, len(dates)) + np.linspace(-3, 7, len(dates)),
            'pnn50': base_pnn50 + np.random.normal(0, 5, len(dates)) + np.linspace(-2, 3, len(dates)),
            'lf_power': 800 + np.random.normal(0, 200, len(dates)),
            'hf_power': 600 + np.random.normal(0, 150, len(dates)),
            'stress_index': 5 + np.random.normal(0, 2, len(dates))
        }
        for col in ['rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power']:
            data[col] = np.maximum(data[col], 1)
        data['stress_index'] = np.maximum(data['stress_index'], 0)
        return pd.DataFrame(data)

    def _get_personal_baselines(self, source_name: str = "HRV") -> Dict[str, float]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        AVG(rmssd) as avg_rmssd,
                        AVG(sdnn) as avg_sdnn,
                        AVG(pnn50) as avg_pnn50,
                        AVG(lf_power) as avg_lf,
                        AVG(hf_power) as avg_hf,
                        AVG(stress_index) as avg_stress
                    FROM hrv_data 
                    WHERE name = ? 
                    AND date >= date('now', '-90 days')
                """
                result = pd.read_sql_query(query, conn, params=[source_name])
                if result.empty or result.iloc[0].isna().all():
                    logger.warning("No baseline data found, using default values")
                    return {
                        'avg_rmssd': 45.0, 'avg_sdnn': 50.0, 'avg_pnn50': 15.0,
                        'avg_lf': 800.0, 'avg_hf': 600.0, 'avg_stress': 5.0
                    }
                baselines = result.iloc[0].to_dict()
                for key, value in baselines.items():
                    if pd.isna(value):
                        baselines[key] = 0.0
                return baselines
        except sqlite3.Error as e:
            logger.error(f"Error getting baselines: {e}")
            return {
                'avg_rmssd': 45.0, 'avg_sdnn': 50.0, 'avg_pnn50': 15.0,
                'avg_lf': 800.0, 'avg_hf': 600.0, 'avg_stress': 5.0
            }

    # ...
    # (Keep the rest of the scoring, analysis, and plotting methods unchanged)
    # ...

def main():
    print("=== HRV Analytics Demo ===")
    hrv = HRVAnalytics("c:/smakrykoDBs/Mercury_DWH_HRV.db")

    # Example insert
    today = datetime.now().strftime('%Y-%m-%d')
    hrv.insert_hrv_record(today, "HRV", rmssd=50, sdnn=55, pnn50=20, lf_power=850, hf_power=620, stress_index=4.5)

    print("\n1. Analyzing HRV trends...")
    results = hrv.analyze_hrv_trends(days_back=30, source_name="HRV", include_stats=True)
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Data points: {results['data_points']}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print("\nCurrent HRV values:")
    for metric, value in results['current_values'].items():
        print(f"  {metric.upper()}: {value:.1f}")
    print("\nRecovery Scores:")
    for method, score in results['recovery_scores'].items():
        print(f"  {method.capitalize()}: {score:.1f}/100")

    if 'statistics' in results:
        print("\nTrend Analysis:")
        for metric, stats in results['statistics'].items():
            if 'recovery' in metric:
                print(f"  {metric.replace('_', ' ').title()}: {stats['trend_direction']} ({stats['trend_strength']})")

    print("\n2. Creating visualizations...")
    df = results['dataframe']
    hrv.plot_hrv_trend(df, "HRV Trends - Last 30 Days")
    hrv.plot_hrv_histogram(df, "HRV Metrics Distribution")
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
