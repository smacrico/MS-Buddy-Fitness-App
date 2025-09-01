"""
HRV Analytics Data Warehouse - Version 2.1

Updated script with saving baselines and trend statistics to database,
plus added visuals for baselines, trend summary, and latest recovery score.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any
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
                # Ensure core table exists
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
                # Baselines table for summary
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
                # Trends table for detailed stats & latest recovery score
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

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if denominator != 0 else default

    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        return max(min_val, min(max_val, score))

    def get_daily_hrv_dataframe(self, days_back: int = 30, source_name: str = "HRV") -> pd.DataFrame:
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
            numeric_columns = ['rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'stress_index']
            df[numeric_columns] = df[numeric_columns].fillna(0)
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

    def _get_personal_baselines(self, source_name: str = "HRV") -> Dict[str, float]:
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
                return {
                    'avg_rmssd': 45.0,
                    'avg_sdnn': 50.0,
                    'avg_pnn50': 15.0
                }
            baselines = result.iloc[0].to_dict()
            for k, v in baselines.items():
                if pd.isna(v):
                    baselines[k] = 0.0
            return baselines
        except sqlite3.Error as e:
            logger.error(f"Error getting baselines: {e}")
            return {
                'avg_rmssd': 45.0,
                'avg_sdnn': 50.0,
                'avg_pnn50': 15.0
            }

    def save_baselines(self, baselines: Dict[str, float], source_name: str = "HRV"):
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

    def save_trends(self, stats: Dict[str, Any], latest_recovery_score: float, source_name: str = "HRV"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                for metric, data in stats.items():
                    cursor.execute("""
                        INSERT INTO hrv_trends (source_name, analysis_date, metric, correlation,
                                                trend_direction, trend_strength, mean, std, min, max,
                                                latest_recovery_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (source_name, analysis_date, metric, data.get('correlation', 0.0),
                          data.get('trend_direction', ''), data.get('trend_strength', ''),
                          data.get('mean', 0.0), data.get('std', 0.0), data.get('min', 0.0),
                          data.get('max', 0.0), latest_recovery_score))
                conn.commit()
                logger.info("Trend statistics saved successfully")
        except sqlite3.Error as e:
            logger.error(f"Error saving trend statistics: {e}")

    def _calculate_simple_recovery_score(self, rmssd: float, sdnn: float, pnn50: float) -> float:
        c = self.recovery_constants['simple']
        rmssd = rmssd or 0.0
        sdnn = sdnn or 0.0
        pnn50 = pnn50 or 0.0
        rmssd_score = (rmssd / c['rmssd_scale']) * c['rmssd_weight']
        sdnn_score = (sdnn / c['sdnn_scale']) * c['sdnn_weight']
        pnn50_score = (pnn50 / c['pnn50_scale']) * c['pnn50_weight']
        raw_score = rmssd_score + sdnn_score + pnn50_score
        return self.normalize_score(raw_score, 0, 100)

    def _calculate_comprehensive_recovery_score(self, rmssd: float, sdnn: float, pnn50: float,
                                                lf_power: float = 0, hf_power: float = 0,
                                                stress_index: float = 0) -> float:
        c = self.recovery_constants['comprehensive']
        rmssd = rmssd or 0.0
        sdnn = sdnn or 0.0
        pnn50 = pnn50 or 0.0
        lf_power = lf_power or 0.0
        hf_power = hf_power or 0.0
        stress_index = stress_index or 0.0
        time_domain = (
            (rmssd / c['rmssd_scale']) * 0.4 +
            (sdnn / c['sdnn_scale']) * 0.35 +
            (pnn50 / c['pnn50_scale']) * 0.25
        ) * c['time_domain_weight']
        lf_hf_ratio = self.safe_divide(lf_power, hf_power, 1.0)
        freq_domain = (
            (lf_power / c['lf_scale']) * 0.4 +
            (hf_power / c['hf_scale']) * 0.4 +
            (1 / max(lf_hf_ratio, 0.1)) * 0.2
        ) * c['freq_domain_weight']
        stress_component = (
            max(0, c['stress_scale'] - stress_index) / c['stress_scale']
        ) * c['stress_weight']
        raw_score = time_domain + freq_domain + stress_component
        return self.normalize_score(raw_score * 100, 0, 100)

    def _calculate_personalized_recovery_score(self, rmssd: float, sdnn: float, pnn50: float,
                                               lf_power: float = 0, hf_power: float = 0,
                                               stress_index: float = 0,
                                               source_name: str = "HRV") -> float:
        baselines = self._get_personal_baselines(source_name)
        rmssd = rmssd or 0.0
        sdnn = sdnn or 0.0
        pnn50 = pnn50 or 0.0
        lf_power = lf_power or 0.0
        hf_power = hf_power or 0.0
        stress_index = stress_index or 0.0
        rmssd_relative = self.safe_divide(rmssd, baselines['avg_rmssd'], 1.0)
        sdnn_relative = self.safe_divide(sdnn, baselines['avg_sdnn'], 1.0)
        pnn50_relative = self.safe_divide(pnn50, baselines['avg_pnn50'], 1.0)
        # Assuming lf/hf/stress baselines zero if missing in previous data scenario:
        lf_relative = self.safe_divide(lf_power, 1, 0)
        hf_relative = self.safe_divide(hf_power, 1, 0)
        stress_relative = self.safe_divide(1, max(stress_index, 0.1), 0)
        score = (
            rmssd_relative * 0.25 +
            sdnn_relative * 0.25 +
            pnn50_relative * 0.20 +
            lf_relative * 0.10 +
            hf_relative * 0.10 +
            stress_relative * 0.10
        ) * 100
        return self.normalize_score(score, 0, 100)

    def analyze_hrv_trends(self, days_back: int = 30, source_name: str = "HRV", include_stats: bool = True) -> Dict[str, Any]:
        df = self.get_daily_hrv_dataframe(days_back, source_name)
        if df.empty:
            return {"error": "No data available for analysis"}
        df['simple_recovery'] = df.apply(lambda r: self._calculate_simple_recovery_score(r['rmssd'], r['sdnn'], r['pnn50']), axis=1)
        df['comprehensive_recovery'] = df.apply(lambda r: self._calculate_comprehensive_recovery_score(
            r['rmssd'], r['sdnn'], r['pnn50'], r['lf_power'], r['hf_power'], r['stress_index']), axis=1)
        df['personalized_recovery'] = df.apply(lambda r: self._calculate_personalized_recovery_score(
            r['rmssd'], r['sdnn'], r['pnn50'], r['lf_power'], r['hf_power'], r['stress_index'], source_name), axis=1)

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
                'simple': float(df.iloc[0]['simple_recovery']),
                'comprehensive': float(df.iloc[0]['comprehensive_recovery']),
                'personalized': float(df.iloc[0]['personalized_recovery'])
            },
            'dataframe': df
        }

        if include_stats and len(df) > 1:
            result['statistics'] = self._calculate_trend_statistics(df)

        return result

    def _calculate_trend_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats = {}
        df_copy = df.copy()
        df_copy['day_index'] = range(len(df_copy))
        metrics = ['rmssd', 'sdnn', 'pnn50', 'simple_recovery', 'comprehensive_recovery', 'personalized_recovery']
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
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15,10))
            fig.suptitle(title, fontsize=16)

            # Time domain metrics
            axes[0,0].plot(df['date'], df['rmssd'], 'b-', label='RMSSD', marker='o')
            axes[0,0].plot(df['date'], df['sdnn'], 'r-', label='SDNN', marker='s')
            axes[0,0].set_title('Time Domain Metrics')
            axes[0,0].set_ylabel('ms')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)

            # Recovery scores
            axes[0,1].plot(df['date'], df['simple_recovery'], 'g-', label='Simple', marker='o')
            axes[0,1].plot(df['date'], df['comprehensive_recovery'], 'b-', label='Comprehensive', marker='s')
            axes[0,1].plot(df['date'], df['personalized_recovery'], 'r-', label='Personalized', marker='^')
            axes[0,1].set_title('Recovery Scores')
            axes[0,1].set_ylabel('Score (0-100)')
            axes[0,1].legend()
            axes[0,1].tick_params(axis='x', rotation=45)

            # Frequency domain (if applicable)
            if 'lf_power' in df.columns and 'hf_power' in df.columns:
                axes[1,0].plot(df['date'], df['lf_power'], 'orange', label='LF Power', marker='o')
                axes[1,0].plot(df['date'], df['hf_power'], 'purple', label='HF Power', marker='s')
                axes[1,0].set_title('Frequency Domain')
                axes[1,0].set_ylabel('ms²')
                axes[1,0].legend()
                axes[1,0].tick_params(axis='x', rotation=45)

            # pNN50 and stress
            axes[1,1].plot(df['date'], df['pnn50'], 'brown', label='pNN50', marker='o')
            if 'stress_index' in df.columns:
                ax2 = axes[1,1].twinx()
                ax2.plot(df['date'], df['stress_index'], 'red', label='Stress Index', marker='s', alpha=0.7)
                ax2.set_ylabel('Stress Index', color='red')
                ax2.legend(loc='upper right')
            axes[1,1].set_title('pNN50 & Stress')
            axes[1,1].set_ylabel('pNN50 (%)', color='brown')
            axes[1,1].legend(loc='upper left')
            axes[1,1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting HRV trends: {e}")
            print(f"Could not create trend plot: {e}")

    def plot_hrv_histogram(self, df: pd.DataFrame, title: str = "HRV Distribution") -> None:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15,8))
            fig.suptitle(title, fontsize=16)
            metrics = [
                ('rmssd', 'RMSSD (ms)', 'blue'),
                ('sdnn', 'SDNN (ms)', 'red'),
                ('pnn50', 'pNN50 (%)', 'green'),
                ('simple_recovery', 'Simple Recovery', 'orange'),
                ('comprehensive_recovery', 'Comprehensive Recovery', 'purple'),
                ('personalized_recovery', 'Personalized Recovery', 'brown')
            ]
            for i, (metric, label, color) in enumerate(metrics):
                row, col = divmod(i, 3)
                if metric in df.columns:
                    axes[row, col].hist(df[metric], bins=15, alpha=0.7, color=color, edgecolor='black')
                    mean_val = df[metric].mean()
                    axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
                    axes[row, col].set_title(label)
                    axes[row, col].set_xlabel('Value')
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting HRV histogram: {e}")
            print(f"Could not create histogram: {e}")

    def plot_baselines(self, baselines: Dict[str, float], title: str = "HRV Baselines") -> None:
        labels = list(baselines.keys())
        values = [baselines[k] for k in labels]
        plt.figure(figsize=(7,5))
        sns.barplot(x=labels, y=values, palette="Set2")
        plt.title(title)
        plt.ylabel("Baseline Value")
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()

    def plot_trend_summary(self, stats: Dict[str, Any], title: str = "Trend Statistics Summary") -> None:
        # Plot correlation strengths and directions per metric
        metrics = []
        corr_vals = []
        directions = []
        for metric, stat in stats.items():
            metrics.append(metric)
            corr_vals.append(stat['correlation'])
            directions.append(stat['trend_direction'])

        plt.figure(figsize=(8,5))
        bars = plt.bar(metrics, corr_vals, color='steelblue')
        plt.title(title)
        plt.ylabel('Correlation coefficient')
        plt.ylim(-1,1)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45)

        # Add text labels for direction above bars
        for bar, direction in zip(bars, directions):
            height = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, height, direction, ha='center', va='bottom')

        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_latest_recovery_score(self, latest_score: float, title: str = "Latest Recovery Score") -> None:
        plt.figure(figsize=(5,4))
        plt.bar(['Latest Recovery Score'], [latest_score], color='orchid')
        plt.ylim(0, 100)
        plt.title(title)
        plt.ylabel('Score (0-100)')
        plt.grid(axis='y')
        plt.show()

def main():
    print("=== HRV Analytics Demo ===")
    hrv = HRVAnalytics("c:/smakrykoDBs/Mercury_HRV.db")
    print("\n1. Analyzing HRV trends...")
    results = hrv.analyze_hrv_trends(days_back=30, include_stats=True)
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

    if 'statistics' in results:
        print("\nTrend Analysis:")
        for metric, stat in results['statistics'].items():
            if 'recovery' in metric:
                print(f" {metric.replace('_', ' ').title()}: {stat['trend_direction']} ({stat['trend_strength']})")

    print("\n2. Creating visualizations...")
    df = results['dataframe']
    hrv.plot_hrv_trend(df, "HRV Trends - Last 30 Days")
    hrv.plot_hrv_histogram(df, "HRV Metrics Distribution")

    baselines = hrv._get_personal_baselines("HRV")
    hrv.plot_baselines(baselines, "Personal HRV Baselines (Last 90 Days)")
    stats = results.get('statistics', {})
    if stats:
        hrv.plot_trend_summary(stats, "Trend Statistics Summary (Correlations & Directions)")

    latest_score = results['recovery_scores'].get('personalized', None)
    if latest_score is not None:
        hrv.plot_latest_recovery_score(latest_score, "Latest Personalized Recovery Score")

    # Save baseline and trends summaries into DB
    hrv.save_baselines(baselines, "HRV")
    hrv.save_trends(stats, latest_score, "HRV")

    print("\nDemo completed!")

if __name__ == "__main__":
    main()
