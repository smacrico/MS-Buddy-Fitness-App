import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRVAnalyticsV33:
    def __init__(seLF, db_path="c:/smakrykoDBs/Artemis.db"):
        seLF.db_path = db_path
        # Metrics tracked
        seLF.metrics = ['SD1','SD2','sdnn','rmssd','pNN50','VLF','LF','HF']
        # MS-sensitive weighting for composite recovery score
        seLF.ms_weights = {
            'rmssd': 0.18,
            'sdnn': 0.24,
            'pNN50': 0.14,
            'SD1' : 0.15,
            'SD2' : 0.13,
            'LF'  : 0.08,
            'HF'  : 0.08
        }
        seLF.scales = {
            'rmssd': 0.8,
            'sdnn': 1.0,
            'pNN50': 2.0,
            'SD1': 1.0,
            'SD2': 1.0,
            'VLF': 100.0,
            'LF': 100.0,
            'HF': 100.0
        }
        seLF._ensure_schema()

    def _ensure_schema(seLF):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        name TEXT NOT NULL,
                        SD1 REAL, SD2 REAL, sdnn REAL, rmssd REAL, pNN50 REAL,
                        VLF REAL, LF REAL, HF REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_baselines (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT,
                        analysis_date TEXT,
                        avg_SD1 REAL,
                        avg_SD2 REAL,
                        avg_sdnn REAL,
                        avg_rmssd REAL,
                        avg_pNN50 REAL,
                        avg_VLF REAL,
                        avg_LF REAL,
                        avg_HF REAL
                    )
                """)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_trends (
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
                        latest_ms_score REAL
                    )
                """)
                conn.commit()
            logger.info("Database schema ensured.")
        except sqlite3.Error as e:
            logger.error(f"Database schema error: {e}")

    def import_myhrv_view(seLF, source_view="myHRV_view", device_name="MyHRV_import"):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {source_view}", conn)
                if df.empty:
                    logger.warning(f"No rows found in {source_view}.")
                    return
                df['name'] = device_name
                for m in seLF.metrics:
                    if m not in df.columns:
                        df[m] = 0.0
                insert_cols = ['date', 'name'] + seLF.metrics
                existing = pd.read_sql_query("SELECT date FROM myHRV_data WHERE name = ?", conn, params=[device_name])
                overlapping = set(df['date']).intersection(set(existing['date']))
                if overlapping:
                    for d in overlapping:
                        conn.execute("DELETE FROM myHRV_data WHERE name = ? AND date = ?", (device_name, d))
                df[insert_cols].to_sql('myHRV_data', conn, if_exists='append', index=False)
                logger.info(f"Imported {len(df)} rows from {source_view} to myHRV_data as '{device_name}'.")
        except Exception as e:
            logger.error(f"Error importing data from view: {e}")

    def get_daily_hrv_dataframe(seLF, days_back=30, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                query = f"""SELECT date, {', '.join(seLF.metrics)}
                            FROM myHRV_data
                            WHERE name = ? AND date >= date('now', ?)
                            ORDER BY date DESC"""
                df = pd.read_sql_query(query, conn, params=[source_name, f"-{days_back} days"])
                if df.empty:
                    logger.warning("No data found, generating sample data.")
                    return seLF._generate_sample_trend_data(days_back)
                df['date'] = pd.to_datetime(df['date'])
                for m in seLF.metrics:
                    df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)
                return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return seLF._generate_sample_trend_data(days_back)

    def _generate_sample_trend_data(seLF, days=30):
        logger.info(f"Generating {days} days of sample HRV data")
        dates = pd.date_range(start=datetime.now() - timedelta(days=days-1), end=datetime.now(), freq='D')
        np.random.seed(42)
        data = {
            'date': dates,
            'SD1': 30 + np.random.normal(0, 5, len(dates)),
            'SD2': 40 + np.random.normal(0, 6, len(dates)),
            'sdnn': 50 + np.random.normal(0, 8, len(dates)),
            'rmssd': 42 + np.random.normal(0, 7, len(dates)),
            'pNN50': 13 + np.random.normal(0, 4, len(dates)),
            'VLF': 700 + np.random.normal(0, 120, len(dates)),
            'LF': 1050 + np.random.normal(0, 150, len(dates)),
            'HF': 820 + np.random.normal(0, 100, len(dates))
        }
        for k in data:
            if k != 'date':
                data[k] = np.maximum(data[k], 1)
        return pd.DataFrame(data)

    def _get_personal_baselines(seLF, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                query = f"""SELECT {', '.join([f'AVG({m}) AS avg_{m}' for m in seLF.metrics])}
                            FROM myHRV_data
                            WHERE name = ? AND date >= date('now', '-90 days')"""
                result = pd.read_sql_query(query, conn, params=[source_name])
                if result.empty or result.iloc[0].isna().all():
                    logger.warning("No baseline data found, using default values.")
                    return {f"avg_{m}": 0.0 for m in seLF.metrics}
                baselines = dict(result.iloc[0])
                for k in baselines:
                    if pd.isna(baselines[k]):
                        baselines[k] = 0.0
                return baselines
        except sqlite3.Error as e:
            logger.error(f"Error getting baselines: {e}")
            return {f"avg_{m}": 0.0 for m in seLF.metrics}

    def save_baselines(seLF, baselines, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                cursor = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                cols = ','.join(['source_name', 'analysis_date'] + list(baselines.keys()))
                vals = [source_name, analysis_date] + list(baselines.values())
                placeholders = ','.join(['?']*len(vals))
                cursor.execute(f"INSERT INTO myHRV_baselines ({cols}) VALUES ({placeholders})", vals)
                conn.commit()
            logger.info("Baselines saved successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error saving baselines: {e}")

    def save_trends(seLF, stats, latest_ms_score, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(seLF.db_path) as conn:
                cursor = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                for metric, data in stats.items():
                    cursor.execute("""
                        INSERT INTO myHRV_trends
                        (source_name, analysis_date, metric, correlation, trend_direction, trend_strength, mean, std, min, max, latest_ms_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (source_name, analysis_date, metric, data.get('correlation', 0.0),
                          data.get('trend_direction', ''), data.get('trend_strength', ''),
                          data.get('mean', 0.0), data.get('std', 0.0), data.get('min', 0.0), data.get('max', 0.0), latest_ms_score))
                conn.commit()
            logger.info("Trend statistics saved successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error saving trend statistics: {e}")

    def _calculate_ms_recovery_score(seLF, row):
        s = 0
        for m in seLF.ms_weights:
            s += (row.get(m, 0.0) / seLF.scales.get(m, 1.0)) * seLF.ms_weights[m]
        return seLF.normalize_score(s, 0, 100)

    @staticmethod
    def normalize_score(score, min_val=0.0, max_val=100.0):
        return max(min_val, min(max_val, score))

    def analyze_hrv_trends(seLF, days_back=30, source_name="MyHRV_import", include_stats=True):
        df = seLF.get_daily_hrv_dataframe(days_back, source_name)
        if df.empty:
            return {"error": "No data available for analysis"}
        df['ms_recovery'] = df.apply(seLF._calculate_ms_recovery_score, axis=1)
        result = {
            'data_points': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'current_values': {m: float(df.iloc[0][m]) for m in seLF.metrics},
            'recovery_scores': {'ms': float(df.iloc[0]['ms_recovery'])},
            'dataframe': df
        }
        if include_stats and len(df) > 1:
            result['statistics'] = seLF._calculate_trend_statistics(df)
        return result

    def _calculate_trend_statistics(seLF, df):
        stats = {}
        df_copy = df.copy()
        df_copy['day_index'] = range(len(df_copy))
        for metric in seLF.metrics + ['ms_recovery']:
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

    def plot_time_trends(seLF, df, title="HRV Metrics Time Trends"):
        plt.figure(figsize=(14, 8))
        for m in seLF.metrics:
            plt.plot(df['date'], df[m], marker='o', label=m.upper())
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_hrv_histograms(seLF, df, title="HRV Metrics Distributions"):
        plt.figure(figsize=(16, 10))
        for i, metric in enumerate(seLF.metrics, 1):
            plt.subplot(2, 4, i)
            plt.hist(df[metric], bins=15, color='royalblue', edgecolor='black')
            plt.title(metric.upper())
            plt.axvline(df[metric].mean(), color='red', linestyle='--')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_baseline_bar(seLF, baselines, title="90-Day HRV Baseline Profile"):
        plt.figure(figsize=(8, 5))
        keys = [k.replace('avg_', '').upper() for k in baselines.keys()]
        vals = list(baselines.values())
        sns.barplot(x=keys, y=vals, palette='PuRd')
        plt.title(title)
        plt.ylabel('Baseline Value')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_radar_chart(seLF, latest, baselines, title="Latest vs Baseline (Radar)"):
        import math
        categories = [m.upper() for m in seLF.metrics]
        N = len(categories)
        values = [latest[m] for m in seLF.metrics]
        baseline_vals = [baselines[f"avg_{m}"] for m in seLF.metrics]
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        values += values[:1]
        baseline_vals += baseline_vals[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7))
        ax.plot(angles, values, linewidth=2, label='Latest')
        ax.fill(angles, values, alpha=0.20)
        ax.plot(angles, baseline_vals, linewidth=2, linestyle='--', label='Baseline')
        plt.title(title, size=15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

    def plot_ms_score(seLF, df, title="MS-Aware HRV Health Score (Trend)"):
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['ms_recovery'], marker='d', color='purple', label='MS-Score')
        plt.title(title)
        plt.ylabel('Score (0-100)')
        plt.xlabel('Date')
        plt.axhline(50, color='grey', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_trend_summary(seLF, stats, title="Trend Statistics Summary"):
        metrics = list(stats.keys())
        corr_vals = [stats[m]['correlation'] for m in metrics]
        directions = [stats[m]['trend_direction'] for m in metrics]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(metrics, corr_vals, color='deepskyblue')
        plt.title(title)
        plt.ylabel('Correlation coefficient')
        plt.ylim(-1, 1)
        plt.axhline(0, color='black', linewidth=0.8)
        for bar, direction in zip(bars, directions):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, direction, ha='center', va='bottom')
        plt.xticks(rotation=40)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def report_alerts(seLF, latest, baselines, thresholds=None):
        if thresholds is None:
            thresholds = {m: 2.5 for m in seLF.metrics}
        alerts = []
        for m in seLF.metrics:
            baseline = baselines.get(f"avg_{m}", 0.0)
            value = latest.get(m, 0.0)
            if baseline == 0:
                continue
            rel_dev = (value - baseline) / baseline
            if abs(rel_dev) >= thresholds.get(m, 2.5) / 10:  # ~25% deviation threshold
                alerts.append(f"{m.upper()} deviation: current={value:.1f}, baseline={baseline:.1f}, deviation={rel_dev * 100:.1f}%")
        if alerts:
            logger.warning("SIGNIFICANT HRV DEVIATION DETECTED:\n" + "\n".join(alerts))
        else:
            logger.info("No significant deviations from HRV baseline found.")
        return alerts

def main():
    print("=== HRV Analytics V3.3 Demo ===")
    hrv = HRVAnalyticsV33("c:/smakrykoDBs/Artemis.db")

    # Import from the specified view
    hrv.import_myhrv_view(source_view="myHRV_view", device_name="MyHRV_import")

    # Analyze HRV trends
    results = hrv.analyze_hrv_trends(days_back=30, source_name="MyHRV_import", include_stats=True)
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Data points: {results['data_points']}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print("\nCurrent HRV values:")
    for metric, value in results['current_values'].items():
        print(f" {metric.upper()}: {value:.1f}")

    print("\nMS-Optimized Recovery Score:")
    print(f" MS-Aware: {results['recovery_scores']['ms']:.1f}/100")

    stats = results.get('statistics', {})
    if stats:
        print("\nTrend Analysis:")
        for metric, stat in stats.items():
            print(f" {metric.title()}: {stat['trend_direction']} ({stat['trend_strength']})")

    # Visualizations
    df = results['dataframe']
    baselines = hrv._get_personal_baselines("MyHRV_import")
    latest = results['current_values']

    hrv.plot_time_trends(df, "MyHRV HRV Metrics - Last 30 Days")
    hrv.plot_hrv_histograms(df, "MyHRV HRV Metrics Distribution")
    hrv.plot_baseline_bar(baselines, "MyHRV Baseline HRV (90 Days)")
    hrv.plot_radar_chart(latest, baselines, "MyHRV Latest vs Baseline (Radar)")
    hrv.plot_ms_score(df, "MS-Aware HRV Score (Trend)")

    if stats:
        hrv.plot_trend_summary(stats, "Trend Summary")

    # Save outputs
    hrv.save_baselines(baselines, "MyHRV_import")
    if stats:
        hrv.save_trends(stats, results['recovery_scores']['ms'], "MyHRV_import")

    # Alert reporting
    alerts = hrv.report_alerts(latest, baselines)
    if alerts:
        print("\n*** ALERT: Significant deviations detected ***")
        for alert in alerts:
            print(alert)
    else:
        print("\nNo significant deviations from baseline detected.")

    print("\nDemo completed!")

if __name__ == "__main__":
    main()
