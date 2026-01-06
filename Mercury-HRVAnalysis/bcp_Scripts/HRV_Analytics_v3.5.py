import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRVAnalyticsV35:
    # Metrics for myHRV_view
    myHRV_metrics = ['sd1', 'sd2', 'sdnn', 'rmssd', 'pnn50', 'vlf', 'lf', 'hf']
    myHRV_weights = {
        'rmssd': 0.18,
        'sdnn': 0.24,
        'pnn50': 0.14,
        'sd1': 0.15,
        'sd2': 0.13,
        'lf': 0.08,
        'hf': 0.08
    }
    myHRV_scales = {
        'rmssd': 0.8,
        'sdnn': 1.0,
        'pnn50': 2.0,
        'sd1': 1.0,
        'sd2': 1.0,
        'vlf': 100.0,
        'lf': 100.0,
        'hf': 100.0
    }

    # Metrics for myHRVanalysis_view
    analysis_metrics = ['nn20', 'nn50', 'sdnn', 'rmssd', 'pnn50', 'pnn20']
    analysis_weights = {
        'rmssd': 0.25,
        'sdnn': 0.25,
        'pnn50': 0.15,
        'pnn20': 0.15,
        'nn50': 0.10,
        'nn20': 0.10
    }

    def __init__(self, db_path="c:/smakrykoDBs/Artemis.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                # myHRV tables
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        name TEXT NOT NULL,
                        sd1 REAL, sd2 REAL, sdnn REAL, rmssd REAL, pnn50 REAL,
                        vlf REAL, lf REAL, hf REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_baselines (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT,
                        analysis_date TEXT,
                        {', '.join([f"avg_{m} REAL" for m in self.myHRV_metrics])}
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS myHRV_trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT,
                        analysis_date TEXT,
                        metric TEXT,
                        correlation REAL,
                        trend_direction TEXT,
                        trend_strength TEXT,
                        mean REAL, std REAL, min REAL, max REAL,
                        latest_ms_score REAL
                    )
                """)
                # analysis tables
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS analysis_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        name TEXT NOT NULL,
                        nn20 INTEGER,
                        nn50 INTEGER,
                        sdnn REAL,
                        rmssd REAL,
                        pnn50 REAL,
                        pnn20 REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS analysis_baseline (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT,
                        analysis_date TEXT,
                        {', '.join([f"avg_{m} REAL" for m in self.analysis_metrics])}
                    )
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS analysis_trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_name TEXT,
                        analysis_date TEXT,
                        metric TEXT,
                        correlation REAL,
                        trend_direction TEXT,
                        trend_strength TEXT,
                        mean REAL, std REAL, min REAL, max REAL,
                        latest_ms_score REAL
                    )
                """)
                conn.commit()
            logger.info("Database schema ensured.")
        except sqlite3.Error as e:
            logger.error(f"Schema error: {e}")

    # Import and data handling for myHRV_view data
    def import_myHRV_view(self, source_view="myHRV_view", device_name="MyHRV_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {source_view}", conn)
                if df.empty:
                    logger.warning(f"No rows found in {source_view}")
                    return
                df['name'] = device_name
                for m in self.myHRV_metrics:
                    if m not in df.columns:
                        df[m] = 0.0
                insert_cols = ['date', 'name'] + self.myHRV_metrics
                existing = pd.read_sql_query("SELECT date FROM myHRV_data WHERE name=?", conn, params=[device_name])
                overlapping = set(df['date']).intersection(set(existing['date']))
                for d in overlapping:
                    conn.execute("DELETE FROM myHRV_data WHERE name=? AND date=?", (device_name, d))
                df[insert_cols].to_sql('myHRV_data', conn, if_exists='append', index=False)
                logger.info(f"Imported {len(df)} rows into myHRV_data as '{device_name}'")
        except Exception as e:
            logger.error(f"Import myHRV_view error: {e}")

    def get_myHRV_dataframe(self, days_back=30, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT date, {','.join(self.myHRV_metrics)} FROM myHRV_data WHERE name=? AND date >= date('now', ?) ORDER BY date DESC"
                df = pd.read_sql_query(query, conn, params=[source_name, f"-{days_back} days"])
            if df.empty:
                logger.warning("No myHRV data found; generating sample")
                return self._generate_sample_myHRV_data(days_back)
            df['date'] = pd.to_datetime(df['date'])
            for m in self.myHRV_metrics:
                df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error loading myHRV data: {e}")
            return self._generate_sample_myHRV_data(days_back)

    def _generate_sample_myHRV_data(self, days=30):
        logger.info(f"Generating {days} days of sample myHRV data")
        dates = pd.date_range(datetime.now() - timedelta(days=days-1), datetime.now(), freq='D')
        np.random.seed(42)
        data = {
            'date': dates,
            'sd1': np.abs(30 + np.random.normal(0, 5, len(dates))),
            'sd2': np.abs(40 + np.random.normal(0, 6, len(dates))),
            'sdnn': np.abs(50 + np.random.normal(0, 8, len(dates))),
            'rmssd': np.abs(40 + np.random.normal(0, 7, len(dates))),
            'pnn50': np.abs(15 + np.random.normal(0, 3, len(dates))),
            'vlf': np.abs(700 + np.random.normal(0, 120, len(dates))),
            'lf': np.abs(1050 + np.random.normal(0, 150, len(dates))),
            'hf': np.abs(820 + np.random.normal(0, 100, len(dates))),
        }
        return pd.DataFrame(data)

    def get_myHRV_baselines(self, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT {', '.join([f'AVG({m}) AS avg_{m}' for m in self.myHRV_metrics])} FROM myHRV_data WHERE name=? AND date >= date('now', '-90 days')"
                result = pd.read_sql_query(query, conn, params=[source_name])
            if result.empty or result.isna().all(axis=1).iloc[0]:
                logger.warning("No myHRV baseline data found; using zeros")
                return {f'avg_{m}': 0 for m in self.myHRV_metrics}
            baselines = dict(result.iloc[0])
            return {k: (0 if pd.isna(v) else v) for k, v in baselines.items()}
        except Exception as e:
            logger.error(f"Error fetching myHRV baselines: {e}")
            return {f'avg_{m}': 0 for m in self.myHRV_metrics}

    def save_myHRV_baselines(self, baselines, source_name="MyHRV_import"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                analysis_date = datetime.now().strftime('%Y-%m-%d')
                cols = ['source_name', 'analysis_date'] + list(baselines.keys())
                vals = [source_name, analysis_date] + list(baselines.values())
                placeholders = ','.join(['?'] * len(vals))
                cur.execute(f"INSERT INTO myHRV_baselines ({', '.join(cols)}) VALUES ({placeholders})", vals)
                conn.commit()
            logger.info("myHRV baselines saved")
        except Exception as e:
            logger.error(f"Error saving myHRV baselines: {e}")

    # Compute MS weighted recovery score for myHRV metrics
    def calculate_myHRV_ms_score(self, row):
        total_score = 0
        for m, weight in self.myHRV_weights.items():
            total_score += (row.get(m, 0) / self.myHRV_scales.get(m, 1)) * weight
        return self.normalize_score(total_score, 0, 100)

    def analyze_myHRV_trends(self, days_back=30, source_name="MyHRV_import", include_stats=True):
        df = self.get_myHRV_dataframe(days_back, source_name)
        if df.empty:
            return {"error": "No data to analyze in myHRV"}
        df['ms_score'] = df.apply(self.calculate_myHRV_ms_score, axis=1)
        results = {
            "data_points": len(df),
            "date_range": {"start": df['date'].min().strftime('%Y-%m-%d'), "end": df['date'].max().strftime('%Y-%m-%d')},
            "current_values": {m: float(df.iloc[0][m]) for m in self.myHRV_metrics},
            "recovery_scores": {"ms_score": float(df.iloc[0]['ms_score'])},
            "dataframe": df
        }
        if include_stats and len(df) > 1:
            results['statistics'] = self._calculate_trend_stats(df, metrics=self.myHRV_metrics+['ms_score'])
        return results

    def _calculate_trend_stats(self, df, metrics):
        stats = {}
        dfc = df.copy()
        dfc['day_index'] = range(len(dfc))
        for metric in metrics:
            if metric in dfc.columns:
                correlation = dfc['day_index'].corr(dfc[metric])
                if abs(correlation) >= 0.7:
                    strength = "strong"
                elif abs(correlation) >= 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                direction = "improving" if correlation > 0 else "declining" if correlation < 0 else "stable"
                stats[metric] = {
                    "correlation": float(correlation) if pd.notna(correlation) else 0.0,
                    "trend_direction": direction,
                    "trend_strength": strength,
                    "mean": float(dfc[metric].mean()),
                    "std": float(dfc[metric].std()),
                    "min": float(dfc[metric].min()),
                    "max": float(dfc[metric].max())
                }
        return stats

    # Visualization methods for myHRV data
    def plot_time_trends(self, df, title="HRV Time Trends"):
        plt.figure(figsize=(14,8))
        for m in self.myHRV_metrics:
            plt.plot(df['date'], df[m], marker='o', label=m.upper())
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_histograms(self, df, title="HRV Metrics Distribution"):
        plt.figure(figsize=(16, 10))
        for i, metric in enumerate(self.myHRV_metrics, 1):
            plt.subplot(2, 4, i)
            plt.hist(df[metric], bins=15, color='coral', edgecolor='black')
            plt.title(metric.upper())
            plt.axvline(df[metric].mean(), color='red', linestyle='--')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_baseline_bar(self, baselines, title="HRV Baseline Profile (90 Days)"):
        plt.figure(figsize=(8,5))
        keys = [k.replace('avg_', '').upper() for k in baselines.keys()]
        vals = list(baselines.values())
        sns.barplot(x=keys, y=vals, palette='Oranges')
        plt.title(title)
        plt.ylabel('Baseline Value')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_radar(self, latest, baselines, title="Latest vs Baseline (Radar)"):
        import math
        categories = [m.upper() for m in self.myHRV_metrics]
        N = len(categories)
        values = [latest[m] for m in self.myHRV_metrics]
        baseline_vals = [baselines[f'avg_{m}'] for m in self.myHRV_metrics]
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        values += values[:1]
        baseline_vals += baseline_vals[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7,7))
        ax.plot(angles, values, label='Latest', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, baseline_vals, label='Baseline', linestyle='--', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title(title)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

    def plot_recovery_score_trend(self, df, title="MS-Aware HRV Score Trend"):
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['ms_score'], marker='d', color='darkorange')
        plt.title(title)
        plt.ylabel('Score (0-100)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trend_summary(self, stats, title="Trend Statistics Summary"):
        metrics = list(stats.keys())
        corr_vals = [stats[m]['correlation'] for m in metrics]
        directions = [stats[m]['trend_direction'] for m in metrics]
        plt.figure(figsize=(10,5))
        bars = plt.bar(metrics, corr_vals, color='sandybrown')
        plt.title(title)
        plt.ylabel('Correlation coefficient')
        plt.ylim(-1,1)
        plt.axhline(0, color='black', linewidth=0.8)
        for bar, direction in zip(bars, directions):
            height = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, height, direction, ha='center', va='bottom')
        plt.xticks(rotation=40)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def report_alerts(self, latest, baselines, thresholds=None):
        if thresholds is None:
            thresholds = {m: 2.5 for m in self.myHRV_metrics}
        alerts = []
        for m in self.myHRV_metrics:
            baseline = baselines.get(f'avg_{m}', 0.0)
            val = latest.get(m, 0.0)
            if baseline == 0:
                continue
            rel_dev = (val - baseline)/baseline
            if abs(rel_dev) >= thresholds.get(m, 2.5)/10:
                alerts.append(f"{m.upper()} deviation: current={val:.1f}, baseline={baseline:.1f}, deviation={rel_dev*100:.1f}%")
        if alerts:
            logger.warning("SIGNIFICANT HRV DEVIATION DETECTED:\n" + "\n".join(alerts))
        else:
            logger.info("No significant HRV deviations.")
        return alerts

# Example main usage for myHRV_view analysis
def main():
    print("=== HRV Analytics V3.5 Extended ===")
    hrv = HRVAnalyticsV35()

    # Import and analyze myHRV_view data
    hrv.import_myHRV_view()
    results = hrv.analyze_myHRV_trends()

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Data points: {results['data_points']}")
    print(f"Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print("\nCurrent HRV values:")
    for k, v in results['current_values'].items():
        print(f" {k.upper()}: {v:.2f}")

    print("\nMS-Aware Recovery Score:")
    print(f" {results['recovery_scores']['ms_score']:.2f}/100")

    stats = results.get('statistics', {})

    if stats:
        print("\nTrend analysis:")
        for m, stat in stats.items():
            print(f" {m}: {stat['trend_direction']} ({stat['trend_strength']})")

    df = results['dataframe']
    baselines = hrv.get_myHRV_baselines()
    latest = results['current_values']

    hrv.plot_time_trends(df)
    hrv.plot_histograms(df)
    hrv.plot_baseline_bar(baselines)
    hrv.plot_radar(latest, baselines)
    hrv.plot_recovery_score_trend(df)

    if stats:
        hrv.plot_trend_summary(stats)

    hrv.save_myHRV_baselines(baselines)
    if stats:
        hrv.save_trends(stats, results['recovery_scores']['ms_score'])

    alerts = hrv.report_alerts(latest, baselines)
    if alerts:
        print("\n*** ALERTS ***")
        for alert in alerts:
            print(alert)
    else:
        print("\nNo significant alerts.")

if __name__ == "__main__":
    main()
