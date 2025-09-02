"""
HRV Analytics Data Warehouse - Version 3.0

- Loads HRV data from view 'f3bHRV_view' and copies it to 'hrv_data' table
- Maps 'armssd'->'rmssd', 'asdnn'->'sdnn', calculates pnn50 from NN50/(NN20+NN50)
- Sets 'name' field for source tracking (default 'F3b_import')
- Normal HRV analysis and trend/statistics/visuals pipeline as before
- Adds database-saving of baselines and trends
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRVAnalytics:
    def __init__(self, db_path: str = "c:/smakrykoDBs/Mercury_HRV.db"):
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
                CREATE TABLE IF NOT EXISTS hrv_data (
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
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS hrv_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT,
                    analysis_date TEXT,
                    avg_rmssd REAL,
                    avg_sdnn REAL,
                    avg_pnn50 REAL
                )
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS hrv_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT,
                    analysis_date TEXT,
                    metric TEXT,
                    correlation REAL,
                    trend_direction TEXT,
                    trend_strength TEXT,
                    mean REAL,
                    std REAL,
                    min REAL,
                    max REAL,
                    latest_recovery_score REAL
                )
                """)
                conn.commit()
                logger.info("Database schema ensured.")
        except sqlite3.Error as e:
            logger.error(f"Database schema error: {e}")

    def import_f3b_view_to_hrv_data(self, source_view: str = "f3bHRV_view", device_name: str = "F3b_import"):
        """
        Import data from f3bHRV_view into hrv_data table for processing.
        Maps columns and calculates pnn50 from NN50/(NN20+NN50) * 100.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {source_view}", conn)
                if df.empty:
                    logger.warning(f"No rows found in {source_view}.")
                    return
                df['rmssd'] = df['armssd']
                df['sdnn'] = df['asdnn']
                # pNN50 calculation: if no NN20, fallback to just NN50 counts
                if 'NN20' in df.columns:
                    total_nn = df['NN20'] + df['NN50']
                    df['pnn50'] = np.where(total_nn>0, df['NN50']/total_nn*100, 0)
                else:
                    df['pnn50'] = df['NN50']
                df['name'] = device_name
                for col in ['lf_power', 'hf_power', 'stress_index']:
                    if col not in df.columns:
                        df[col] = 0.0
                insert_cols = ['date', 'name', 'rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'stress_index']
                df_to_insert = df[insert_cols].copy()
                # Remove any duplicates in hrv_data for this name+date before inserting
                existing = pd.read_sql_query(f"SELECT date FROM hrv_data WHERE name = ?", conn, params=[device_name])
                overlapping = set(df_to_insert['date']).intersection(set(existing['date']))
                if overlapping:
                    for d in overlapping:
                        conn.execute("DELETE FROM hrv_data WHERE name = ? AND date = ?", (device_name, d))
                df_to_insert.to_sql('hrv_data', conn, if_exists='append', index=False)
                logger.info(f"Imported {len(df_to_insert)} rows from {source_view} to hrv_data as '{device_name}'.")
        except Exception as e:
            logger.error(f"Error importing data from view: {e}")

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if denominator != 0 else default

    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        return max(min_val, min(max_val, score))

    def get_daily_hrv_dataframe(self, days_back: int = 30, source_name: str = "F3b_import") -> pd.DataFrame:
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"""
                    SELECT date, rmssd, sdnn, pnn50, lf_power, hf_power, stress_index
                    FROM hrv_data
                    WHERE name = ?
                    AND date >= date('now', '-{days_back} days')
                    ORDER BY date DESC
                """
                df = pd.read_sql_query(query, conn, params=[source_name])
            if df.empty:
                logger.warning("No data found, generating sample data")
                return self._generate_sample_trend_data(days_back)
            df['date'] = pd.to_datetime(df['date'])
            for col in ['rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'stress_index']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return self._generate_sample_trend_data(days_back)

    def _generate_sample_trend_data(self, days: int = 30) -> pd.DataFrame:
        logger.info(f"Generating {days} days of sample HRV data")
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days-1),
            end=datetime.now(),
            freq='D'
        )
        base_rmssd = 45
        base_sdnn = 50
        base_pnn50 = 15
        np.random.seed(42)
        data = {
            'date': dates,
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

    def _get_personal_baselines(self, source_name: str = "F3b_import") -> dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT AVG(rmssd) AS avg_rmssd,
                           AVG(sdnn) AS avg_sdnn,
                           AVG(pnn50) AS avg_pnn50
                    FROM hrv_data
                    WHERE name = ?
                    AND date >= date('now', '-90 days')
                """
                result = pd.read_sql_query(query, conn, params=[source_name])
            if result.empty or result.iloc[0].isna().all():
                logger.warning("No baseline data found, using default values")
                return {'avg_rmssd': 45.0, 'avg_sdnn': 50.0, 'avg_pnn50': 15.0}
            baselines = dict(result.iloc[0])
            for k in baselines:
                if pd.isna(baselines[k]):
                    baselines[k] = 0.0
            return baselines
        except sqlite3.Error as e:
            logger.error(f"Error getting baselines: {e}")
            return {'avg_rmssd': 45.0, 'avg_sdnn': 50.0, 'avg_pnn50': 15.0}

    def save_baselines(self, baselines: dict, source_name: str = "F3b_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    INSERT INTO hrv_baselines (source_name, analysis_date, avg_rmssd, avg_sdnn, avg_pnn50)
                    VALUES (?, ?, ?, ?, ?)
                """, (source_name, analysis_date, baselines['avg_rmssd'], baselines['avg_sdnn'], baselines['avg_pnn50']))
                conn.commit()
                logger.info("Baselines saved successfully")
        except sqlite3.Error as e:
            logger.error(f"Error saving baselines: {e}")

    def save_trends(self, stats: dict, latest_recovery_score: float, source_name: str = "F3b_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                for metric, data in stats.items():
                    cursor.execute("""
                        INSERT INTO hrv_trends
                        (source_name, analysis_date, metric, correlation,
                         trend_direction, trend_strength, mean, std, min, max,
                         latest_recovery_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (source_name, analysis_date, metric, data.get('correlation', 0.0),
                          data.get('trend_direction', ''), data.get('trend_strength', ''),
                          data.get('mean', 0.0), data.get('std', 0.0), data.get('min',0.0),
                          data.get('max', 0.0), latest_recovery_score))
                conn.commit()
                logger.info("Trend statistics saved successfully")
        except sqlite3.Error as e:
            logger.error(f"Error saving trend statistics: {e}")

    def _calculate_simple_recovery_score(self, rmssd, sdnn, pnn50) -> float:
        c = self.recovery_constants['simple']
        rmssd_score = (rmssd / c['rmssd_scale']) * c['rmssd_weight']
        sdnn_score = (sdnn / c['sdnn_scale']) * c['sdnn_weight']
        pnn50_score = (pnn50 / c['pnn50_scale']) * c['pnn50_weight']
        return self.normalize_score(rmssd_score + sdnn_score + pnn50_score, 0, 100)

    def analyze_hrv_trends(self, days_back=30, source_name: str = "F3b_import", include_stats: bool = True):
        df = self.get_daily_hrv_dataframe(days_back, source_name)
        if df.empty:
            return {"error": "No data available for analysis"}
        df['simple_recovery'] = df.apply(
            lambda r: self._calculate_simple_recovery_score(r['rmssd'], r['sdnn'], r['pnn50']), axis=1)

        result = {
            'data_points': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'current_values': {
                'rmssd': float(df.iloc[0]['rmssd']),
                'sdnn': float(df.iloc[0]['sdnn']),
                'pnn50': float(df.iloc[0]['pnn50'])
            },
            'recovery_scores': {
                'simple': float(df.iloc[0]['simple_recovery'])
            },
            'dataframe': df
        }

        if include_stats and len(df) > 1:
            result['statistics'] = self._calculate_trend_statistics(df)

        return result

    def _calculate_trend_statistics(self, df: pd.DataFrame) -> dict:
        stats = {}
        df_copy = df.copy()
        df_copy['day_index'] = range(len(df_copy))
        metrics = ['rmssd', 'sdnn', 'pnn50', 'simple_recovery']
        for metric in metrics:
            if metric in df_copy.columns:
                correlation = df_copy['day_index'].corr(df_copy[metric])
                if abs(correlation) >= 0.7:
                    strength = "strong"
                elif abs(correlation) >= 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                direction = "improving" if correlation > 0 else "declining" if correlation < 0 else "stable"
                stats[metric] = {
                    'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                    'trend_direction': direction,
                    'trend_strength': strength,
                    'mean': float(df_copy[metric].mean()),
                    'std': float(df_copy[metric].std()),
                    'min': float(df_copy[metric].min()),
                    'max': float(df_copy[metric].max())
                }
        return stats

    def plot_hrv_trend(self, df: pd.DataFrame, title: str = "HRV Trends") -> None:
        plt.figure(figsize=(12,6))
        plt.plot(df['date'], df['rmssd'], label='RMSSD', marker='o')
        plt.plot(df['date'], df['sdnn'], label='SDNN', marker='s')
        plt.plot(df['date'], df['pnn50'], label='pNN50', marker='^')
        plt.plot(df['date'], df['simple_recovery'], label='Simple Recovery', marker='d')
        plt.title(title)
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_hrv_histogram(self, df: pd.DataFrame, title: str = "HRV Metrics Distribution") -> None:
        plt.figure(figsize=(14,6))
        metrics = ['rmssd', 'sdnn', 'pnn50', 'simple_recovery']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 4, i)
            plt.hist(df[metric], bins=15, color='steelblue', edgecolor='black')
            plt.title(metric)
            plt.axvline(df[metric].mean(), color='red', linestyle='--')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_baselines(self, baselines: dict, title: str = "HRV Baselines") -> None:
        plt.figure(figsize=(6,4))
        keys = list(baselines.keys())
        vals = list(baselines.values())
        sns.barplot(x=keys, y=vals, palette="Blues")
        plt.title(title)
        plt.ylabel("Baseline Value")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_trend_summary(self, stats: dict, title: str = "Trend Statistics Summary") -> None:
        metrics = list(stats.keys())
        corr_vals = [stats[m]['correlation'] for m in metrics]
        directions = [stats[m]['trend_direction'] for m in metrics]
        plt.figure(figsize=(8,5))
        bars = plt.bar(metrics, corr_vals, color='steelblue')
        plt.title(title)
        plt.ylabel('Correlation coefficient')
        plt.ylim(-1,1)
        plt.axhline(0, color='black', linewidth=0.8)
        for bar, direction in zip(bars, directions):
            height = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, height, direction, ha='center', va='bottom')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_latest_recovery_score(self, latest_score: float, title: str = "Latest Recovery Score") -> None:
        plt.figure(figsize=(4,4))
        plt.bar(['Latest Recovery'], [latest_score], color='orchid')
        plt.ylim(0,100)
        plt.title(title)
        plt.ylabel('Score (0-100)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

def main():
    print("=== HRV Analytics V3.0 Demo ===")
    hrv = HRVAnalytics("c:/smakrykoDBs/Mercury_HRV.db")
    hrv.import_f3b_view_to_hrv_data(source_view="f3bHRV_view", device_name="F3b_import")

    print("\n1. Analyzing HRV trends for F3b_import...")
    results = hrv.analyze_hrv_trends(days_back=30, source_name="F3b_import", include_stats=True)
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Data points: {results['data_points']}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print("\nCurrent HRV values:")
    for metric, value in results['current_values'].items():
        print(f" {metric.upper()}: {value:.1f}")

    print("\nRecovery Scores:")
    for method, score in results['recovery_scores'].items():
        print(f" {method.capitalize()}: {score:.1f}/100")

    stats = results.get('statistics', {})
    if stats:
        print("\nTrend Analysis:")
        for metric, stat in stats.items():
            print(f" {metric.title()}: {stat['trend_direction']} ({stat['trend_strength']})")

    print("\n2. Creating visualizations...")

    df = results['dataframe']
    hrv.plot_hrv_trend(df, "F3b HRV Metrics & Recovery (Last 30 Days)")
    hrv.plot_hrv_histogram(df, "F3b HRV Metrics Distribution (Last 30 Days)")

    baselines = hrv._get_personal_baselines("F3b_import")
    hrv.plot_baselines(baselines, "F3b Baseline HRV Profile (90 Days)")

    if stats:
        hrv.plot_trend_summary(stats, "F3b Trend Statistics Summary")
    latest_score = results['recovery_scores'].get('simple')
    if latest_score is not None:
        hrv.plot_latest_recovery_score(latest_score, "F3b Latest Recovery Score")

    hrv.save_baselines(baselines, "F3b_import")
    if stats:
        hrv.save_trends(stats, latest_score, "F3b_import")

    print("\nDemo completed!")

if __name__ == "__main__":
    main()
