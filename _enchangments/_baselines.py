CREATE TABLE IF NOT EXISTS hrv_baselines (
    metric TEXT PRIMARY KEY,
    baseline_mean REAL NOT NULL,
    baseline_std REAL NOT NULL,
    calculated_on DATE NOT NULL
)


# script for baseline_std

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
HRV_DB_PATH = "c:/smakryko/myHealthData/DataBasesDev/Mercury_DWH-HRV.db"
BASELINE_DB_PATH = "c:/smakryko/myHealthData/DataBasesDev/Mercury_DWH-Baselines.db"
HRV_METRICS = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "mean_hr", "lf", "hf", "vlf"]
DAYS_FOR_BASELINE = 90  # Adjust based on your needs (e.g., 30, 60, 180)

def create_baseline_table():
    conn = sqlite3.connect(BASELINE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hrv_baselines (
            metric TEXT PRIMARY KEY,
            baseline_mean REAL,
            baseline_std REAL,
            calculated_on DATE
        )
    ''')
    conn.commit()
    conn.close()

def calculate_baselines():
    # Ensure table exists
    create_baseline_table()

    try:
        # Load HVR data for baseline calculation
        with sqlite3.connect(HRV_DB_PATH) as conn:
            df = pd.read_sql(
                f'''
                SELECT {', '.join(HRV_METRICS)}
                FROM hrv_sessions
                WHERE timestamp >= date('now', '-{DAYS_FOR_BASELINE} days')
                ''',
                conn
            )

        # Skip if no data
        if df.empty:
            print("No sessions in the baseline period. Skipping baseline calculation.")
            return

        # Calculate means & standard deviations
        baselines = {}
        for metric in HRV_METRICS:
            if metric in df.columns and df[metric].notna().any():
                baseline = df[metric].agg(['mean', 'std'])
                baselines[metric] = {
                    'baseline_mean': baseline['mean'],
                    'baseline_std': baseline['std']
                }

        # Store/update in baselines table
        with sqlite3.connect(BASELINE_DB_PATH) as conn:
            cursor = conn.cursor()
            calc_date = datetime.now().date()

            for metric, vals in baselines.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO hrv_baselines (metric, baseline_mean, baseline_std, calculated_on)
                    VALUES (?, ?, ?, ?)
                ''', (metric, vals['baseline_mean'], vals['baseline_std'], calc_date))

            conn.commit()

        print("Baselines updated successfully.", "Metrics:", baselines.keys())

    except Exception as e:
        print(f"Error calculating/storing baselines: {str(e)}")
        # Log to a file or system log if you have a proper logging setup

if __name__ == "__main__":
    calculate_baselines()
