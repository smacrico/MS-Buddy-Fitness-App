# code refactor to use data from SQLi Database
# (c)smacrico - Dec2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
# import streamlit as st

class RunningAnalysis:

    rest_hr = 60  # Set this to the user's resting heart rate
    max_hr = 170  # Set this to the user's max HR
    
    def __init__(self, db_path):
        self.db_path = r'c:/smakrykoDBs/Apex.db'  # Use consistent path
        self.training_log = self.load_training_data()
    
    def load_training_data(self):
        """Load training data with new speed and HR-RS fields"""
        try:
            conn = sqlite3.connect(r'c:/smakrykoDBs/Apex.db')
            query = """
            SELECT 
                date,
                COALESCE(running_economy, 0) as running_economy,
                COALESCE(vo2max, 0) as vo2max,
                COALESCE(distance, 0) as distance,
                COALESCE(time, 0) as time,
                COALESCE(heart_rate, 0) as heart_rate,
                COALESCE(avg_speed, 0) as avg_speed,
                COALESCE(max_speed, 0) as max_speed,
                COALESCE(HR_RS_Deviation_Index, 0) as hr_rs_deviation,
                COALESCE(cardiacdrift, 0) as cardiac_drift,
                COALESCE(running_economy / NULLIF(vo2max, 0), 0) AS efficiency_score,
                COALESCE(running_economy * (distance / NULLIF(time, 0)), 0) AS energy_cost,
                -- NEW CALCULATED FIELDS
                COALESCE(max_speed - avg_speed, 0) as speed_reserve,
                COALESCE(avg_speed / NULLIF(max_speed, 0), 0) as speed_consistency,
                COALESCE(60.0 / NULLIF(avg_speed, 0), 0) as pace_per_km,
                COALESCE(avg_speed / NULLIF(heart_rate, 0), 0) as speed_efficiency,
                COALESCE(running_economy / NULLIF(avg_speed, 0), 0) as economy_at_speed,
                COALESCE(avg_speed * vo2max, 0) as speed_vo2max_index
            FROM running_sessions
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("WARNING: No data loaded from database!")
                return pd.DataFrame()

            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])

            # Calculate additional derived metrics
            df['duration_min'] = df['time'] / 60
            
            # TRIMP calculation
            rest_hr = 60
            max_hr = 190
            df['hr_ratio'] = (df['heart_rate'] - rest_hr) / (max_hr - rest_hr)
            df['TRIMP'] = df['duration_min'] * df['hr_ratio']

            # Physiological Efficiency Score (avoid division by zero)
            df['physio_efficiency'] = np.where(
                (df['hr_rs_deviation'] > 0) & (df['heart_rate'] > 0),
                (df['avg_speed'] / df['heart_rate']) * (1 / df['hr_rs_deviation']),
                0
            )

            # Fatigue Index
            df['fatigue_index'] = np.where(
                df['avg_speed'] > 0,
                (df['hr_rs_deviation'] * df['cardiac_drift']) / df['avg_speed'],
                0
            )

            # Speed zones (example: slow < 10, moderate 10-14, fast > 14 km/h)
            df['speed_zone'] = pd.cut(
                df['avg_speed'],
                bins=[0, 10, 14, np.inf],
                labels=['Slow', 'Moderate', 'Fast']
            )

            # Calculate weekly metrics
            df['week'] = df['date'].dt.isocalendar().week
            weekly_trimp = df.groupby('week')['TRIMP'].sum().reset_index(name='weekly_trimp')
            
            # Acute and Chronic loads
            weekly_trimp['acute_load'] = weekly_trimp['weekly_trimp'].rolling(window=1).mean()
            weekly_trimp['chronic_load'] = weekly_trimp['weekly_trimp'].rolling(window=4).mean()
            weekly_trimp['acwr'] = weekly_trimp['acute_load'] / (weekly_trimp['chronic_load'] + 1e-8)
            
            self.weekly_trimp = weekly_trimp
            
            print(f"[DEBUG] Loaded {len(df)} rows from database")
            print(f"[DEBUG] Columns available: {list(df.columns)}")

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def add_session(self, date, running_economy, vo2max, distance, time, heart_rate, sport=None, cardicdrift=None):
        """Add a new running session to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO running_sessions 
            (date, running_economy, vo2max, distance, time, heart_rate, sport, cardicdrift)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, running_economy, vo2max, distance, time, heart_rate, sport, cardicdrift))
            
            conn.commit()
            conn.close()
            
            self.training_log = self.load_training_data()
            print(self.training_log)
        except Exception as e:
            print(f"Error adding session: {e}")
        
    def save_training_log_to_db(self):
        """Save training log DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(r'c:/smakrykoDBs/Apex.db')
            
            # Create a new table for training logs if it doesn't exist
            self.training_log.to_sql('training_logs', 
                                    conn, 
                                    if_exists='replace',  # 'replace' will overwrite existing table
                                    index=False)
            
            conn.close()
            print("Training log successfully saved to database")
        except Exception as e:
            print(f"Error saving training log to database: {e}")
            
    def create_monthly_summaries_table(self):
       
        """Create monthly_summaries table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monthly_summaries (
                    year_month TEXT PRIMARY KEY,
                    sessions INTEGER,
                    running_economy_mean REAL,
                    running_economy_std REAL,
                    vo2max_mean REAL,
                    vo2max_std REAL,
                    distance_mean REAL,
                    distance_std REAL,
                    efficiency_score_mean REAL,
                    efficiency_score_std REAL,
                    heart_rate_mean REAL,
                    heart_rate_std REAL,
                    energy_cost_mean REAL,
                    energy_cost_std REAL,
                    trimp_mean REAL,
                    trimp_std REAL,
                    recovery_score_mean REAL,
                    recovery_score_std REAL,
                    readiness_score_mean REAL,
                    readiness_score_std REAL,
                    avg_speed_mean REAL,
                    avg_speed_std REAL,
                    max_speed_mean REAL,
                    max_speed_std REAL,
                    speed_reserve_mean REAL,
                    speed_reserve_std REAL,
                    hr_rs_deviation_mean REAL,
                    hr_rs_deviation_std REAL,
                    speed_efficiency_mean REAL,
                    speed_efficiency_std REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error creating monthly_summaries table: {e}")


    def save_monthly_summaries(self):
        """Save monthly averages as one record per month in monthly_summaries table"""
        monthly_avg = self.calculate_monthly_metrics_averages()
        if monthly_avg is None or monthly_avg.empty:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for year_month in monthly_avg.index:
                sessions = int(monthly_avg.loc[year_month, ('running_economy', 'count')])

                def get_val(metric, stat):
                    if metric in monthly_avg.columns.get_level_values(0):
                        return float(monthly_avg.loc[year_month, (metric, stat)])
                    return None

                cursor.execute("""
                    INSERT INTO monthly_summaries (
                        year_month,
                        sessions,
                        running_economy_mean, running_economy_std,
                        vo2max_mean, vo2max_std,
                        distance_mean, distance_std,
                        efficiency_score_mean, efficiency_score_std,
                        heart_rate_mean, heart_rate_std,
                        energy_cost_mean, energy_cost_std,
                        trimp_mean, trimp_std,
                        recovery_score_mean, recovery_score_std,
                        readiness_score_mean, readiness_score_std,
                        avg_speed_mean, avg_speed_std,
                        max_speed_mean, max_speed_std,
                        speed_reserve_mean, speed_reserve_std,
                        hr_rs_deviation_mean, hr_rs_deviation_std,
                        speed_efficiency_mean, speed_efficiency_std
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(year_month) DO UPDATE SET
                        sessions=excluded.sessions,
                        running_economy_mean=excluded.running_economy_mean,
                        running_economy_std=excluded.running_economy_std,
                        vo2max_mean=excluded.vo2max_mean,
                        vo2max_std=excluded.vo2max_std,
                        distance_mean=excluded.distance_mean,
                        distance_std=excluded.distance_std,
                        efficiency_score_mean=excluded.efficiency_score_mean,
                        efficiency_score_std=excluded.efficiency_score_std,
                        heart_rate_mean=excluded.heart_rate_mean,
                        heart_rate_std=excluded.heart_rate_std,
                        energy_cost_mean=excluded.energy_cost_mean,
                        energy_cost_std=excluded.energy_cost_std,
                        trimp_mean=excluded.trimp_mean,
                        trimp_std=excluded.trimp_std,
                        recovery_score_mean=excluded.recovery_score_mean,
                        recovery_score_std=excluded.recovery_score_std,
                        readiness_score_mean=excluded.readiness_score_mean,
                        readiness_score_std=excluded.readiness_score_std,
                        avg_speed_mean=excluded.avg_speed_mean,
                        avg_speed_std=excluded.avg_speed_std,
                        max_speed_mean=excluded.max_speed_mean,
                        max_speed_std=excluded.max_speed_std,
                        speed_reserve_mean=excluded.speed_reserve_mean,
                        speed_reserve_std=excluded.speed_reserve_std,
                        hr_rs_deviation_mean=excluded.hr_rs_deviation_mean,
                        hr_rs_deviation_std=excluded.hr_rs_deviation_std,
                        speed_efficiency_mean=excluded.speed_efficiency_mean,
                        speed_efficiency_std=excluded.speed_efficiency_std
                """, (
                    str(year_month),
                    sessions,
                    get_val('running_economy', 'mean'), get_val('running_economy', 'std'),
                    get_val('vo2max', 'mean'),          get_val('vo2max', 'std'),
                    get_val('distance', 'mean'),        get_val('distance', 'std'),
                    get_val('efficiency_score', 'mean'), get_val('efficiency_score', 'std'),
                    get_val('heart_rate', 'mean'),      get_val('heart_rate', 'std'),
                    get_val('energy_cost', 'mean'),     get_val('energy_cost', 'std'),
                    get_val('TRIMP', 'mean'),           get_val('TRIMP', 'std'),
                    get_val('recovery_score', 'mean'),  get_val('recovery_score', 'std'),
                    get_val('readiness_score', 'mean'), get_val('readiness_score', 'std'),
                    get_val('avg_speed', 'mean'),       get_val('avg_speed', 'std'),
                    get_val('max_speed', 'mean'),       get_val('max_speed', 'std'),
                    get_val('speed_reserve', 'mean'),   get_val('speed_reserve', 'std'),
                    get_val('hr_rs_deviation', 'mean'), get_val('hr_rs_deviation', 'std'),
                    get_val('speed_efficiency', 'mean'), get_val('speed_efficiency', 'std'),
                ))

            conn.commit()
            conn.close()
            print("Monthly summaries saved successfully")
        except Exception as e:
            print(f"Error saving monthly summaries: {e}")

        
    # Create a new method called create_metrics_breakdown_table to create a new table in the database to store the metrics breakdown data.        
    def create_metrics_breakdown_table(self):
        """Create metrics_breakdown table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_breakdown (
                date TEXT,
                overall_score REAL,
                running_economy_normalized REAL,
                running_economy_weighted REAL,
                running_economy_raw_mean REAL,
                running_economy_raw_std REAL,
                vo2max_normalized REAL,
                vo2max_weighted REAL,
                vo2max_raw_mean REAL,
                vo2max_raw_std REAL,
                distance_normalized REAL,
                distance_weighted REAL,
                distance_raw_mean REAL,
                distance_raw_std REAL,
                efficiency_score_normalized REAL,
                efficiency_score_weighted REAL,
                efficiency_score_raw_mean REAL,
                efficiency_score_raw_std REAL,
                heart_rate_normalized REAL,
                heart_rate_weighted REAL,
                heart_rate_raw_mean REAL,
                heart_rate_raw_std REAL,
                running_economy_trend REAL,
                distance_progression REAL,
                avg_speed_mean REAL,
                avg_speed_std REAL,
                max_speed_mean REAL,
                max_speed_std REAL,
                speed_reserve_mean REAL,
                speed_reserve_std REAL,
                speed_consistency_mean REAL,
                speed_consistency_std REAL,
                pace_per_km_mean REAL,
                pace_per_km_std REAL,
                speed_efficiency_mean REAL,
                speed_efficiency_std REAL,
                economy_at_speed_mean REAL,
                economy_at_speed_std REAL,
                speed_vo2max_index_mean REAL,
                speed_vo2max_index_std REAL,
                hr_rs_deviation_mean REAL,
                hr_rs_deviation_std REAL,
                cardiac_drift_mean REAL,
                cardiac_drift_std REAL,
                physio_efficiency_mean REAL,
                physio_efficiency_std REAL,
                fatigue_index_mean REAL,
                fatigue_index_std REAL
            )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error creating metrics_breakdown table: {e}")

    def save_metrics_breakdown(self, training_score):
        """Save metrics breakdown to database"""
        try:
            if self.training_log.empty:
                print("Cannot save metrics breakdown: No training data available")
                return
            
            # Validate training_score structure
            if not training_score or 'overall_score' not in training_score:
                print("ERROR: Invalid training_score structure")
                return
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            current_date = datetime.now().strftime('%Y-%m-%d')
            metrics = training_score.get('metric_breakdown', {})
            trends = training_score.get('performance_trends', {})
            
            # Helper function to safely get metric values
            def safe_metric_value(metric_dict, key, default=0.0):
                try:
                    if metric_dict and key in metric_dict:
                        val = metric_dict[key]
                        return float(val) if not pd.isna(val) else default
                    return default
                except:
                    return default
            
            # Helper function to safely get mean/std from DataFrame column
            def safe_stat_from_df(col_name, stat_func):
                try:
                    if col_name in self.training_log.columns:
                        series = self.training_log[col_name]
                        result = stat_func(series)
                        return float(result) if not pd.isna(result) else 0.0
                    return 0.0
                except:
                    return 0.0
            
            # Calculate means and stds for speed/HR metrics
            avg_speed_mean = safe_stat_from_df('avg_speed', pd.Series.mean)
            avg_speed_std = safe_stat_from_df('avg_speed', pd.Series.std)
            max_speed_mean = safe_stat_from_df('max_speed', pd.Series.mean)
            max_speed_std = safe_stat_from_df('max_speed', pd.Series.std)
            speed_reserve_mean = safe_stat_from_df('speed_reserve', pd.Series.mean)
            speed_reserve_std = safe_stat_from_df('speed_reserve', pd.Series.std)
            speed_consistency_mean = safe_stat_from_df('speed_consistency', pd.Series.mean)
            speed_consistency_std = safe_stat_from_df('speed_consistency', pd.Series.std)
            pace_per_km_mean = safe_stat_from_df('pace_per_km', pd.Series.mean)
            pace_per_km_std = safe_stat_from_df('pace_per_km', pd.Series.std)
            speed_efficiency_mean = safe_stat_from_df('speed_efficiency', pd.Series.mean)
            speed_efficiency_std = safe_stat_from_df('speed_efficiency', pd.Series.std)
            economy_at_speed_mean = safe_stat_from_df('economy_at_speed', pd.Series.mean)
            economy_at_speed_std = safe_stat_from_df('economy_at_speed', pd.Series.std)
            speed_vo2max_index_mean = safe_stat_from_df('speed_vo2max_index', pd.Series.mean)
            speed_vo2max_index_std = safe_stat_from_df('speed_vo2max_index', pd.Series.std)
            hr_rs_deviation_mean = safe_stat_from_df('hr_rs_deviation', pd.Series.mean)
            hr_rs_deviation_std = safe_stat_from_df('hr_rs_deviation', pd.Series.std)
            cardiac_drift_mean = safe_stat_from_df('cardiac_drift', pd.Series.mean)
            cardiac_drift_std = safe_stat_from_df('cardiac_drift', pd.Series.std)
            physio_efficiency_mean = safe_stat_from_df('physio_efficiency', pd.Series.mean)
            physio_efficiency_std = safe_stat_from_df('physio_efficiency', pd.Series.std)
            fatigue_index_mean = safe_stat_from_df('fatigue_index', pd.Series.mean)
            fatigue_index_std = safe_stat_from_df('fatigue_index', pd.Series.std)
            
            print(f"\n[DEBUG] Preparing to insert metrics breakdown...")
            print(f"[DEBUG] Current date: {current_date}")
            print(f"[DEBUG] Overall score: {training_score['overall_score']}")
            
            # Extract metric values with proper error handling
            values_to_insert = (
                current_date,
                float(training_score['overall_score']),
                safe_metric_value(metrics.get('running_economy', {}), 'normalized_value'),
                safe_metric_value(metrics.get('running_economy', {}), 'weighted_value'),
                safe_metric_value(metrics.get('running_economy', {}), 'raw_mean'),
                safe_metric_value(metrics.get('running_economy', {}), 'raw_std'),
                safe_metric_value(metrics.get('vo2max', {}), 'normalized_value'),
                safe_metric_value(metrics.get('vo2max', {}), 'weighted_value'),
                safe_metric_value(metrics.get('vo2max', {}), 'raw_mean'),
                safe_metric_value(metrics.get('vo2max', {}), 'raw_std'),
                safe_metric_value(metrics.get('distance', {}), 'normalized_value'),
                safe_metric_value(metrics.get('distance', {}), 'weighted_value'),
                safe_metric_value(metrics.get('distance', {}), 'raw_mean'),
                safe_metric_value(metrics.get('distance', {}), 'raw_std'),
                safe_metric_value(metrics.get('efficiency_score', {}), 'normalized_value'),
                safe_metric_value(metrics.get('efficiency_score', {}), 'weighted_value'),
                safe_metric_value(metrics.get('efficiency_score', {}), 'raw_mean'),
                safe_metric_value(metrics.get('efficiency_score', {}), 'raw_std'),
                safe_metric_value(metrics.get('heart_rate', {}), 'normalized_value'),
                safe_metric_value(metrics.get('heart_rate', {}), 'weighted_value'),
                safe_metric_value(metrics.get('heart_rate', {}), 'raw_mean'),
                safe_metric_value(metrics.get('heart_rate', {}), 'raw_std'),
                safe_metric_value(trends, 'running_economy_trend'),
                safe_metric_value(trends, 'distance_progression'),
                avg_speed_mean, avg_speed_std,
                max_speed_mean, max_speed_std,
                speed_reserve_mean, speed_reserve_std,
                speed_consistency_mean, speed_consistency_std,
                pace_per_km_mean, pace_per_km_std,
                speed_efficiency_mean, speed_efficiency_std,
                economy_at_speed_mean, economy_at_speed_std,
                speed_vo2max_index_mean, speed_vo2max_index_std,
                hr_rs_deviation_mean, hr_rs_deviation_std,
                cardiac_drift_mean, cardiac_drift_std,
                physio_efficiency_mean, physio_efficiency_std,
                fatigue_index_mean, fatigue_index_std
            )
            
            print(f"[DEBUG] Number of values to insert: {len(values_to_insert)}")
            
            # FIXED: Correct number of placeholders (48 total)
            cursor.execute('''
            INSERT INTO metrics_breakdown VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            ''', values_to_insert)
            
            print(f"[DEBUG] INSERT statement executed successfully")
            print(f"[DEBUG] Rows affected: {cursor.rowcount}")
            
            conn.commit()
            print(f"[DEBUG] Transaction committed")
            
            # Verify the data was saved
            cursor.execute("SELECT COUNT(*) FROM metrics_breakdown")
            count = cursor.fetchone()[0]
            print(f"[DEBUG] Total rows in metrics_breakdown table: {count}")
            
            cursor.execute("SELECT date, overall_score FROM metrics_breakdown ORDER BY date DESC LIMIT 1")
            last_row = cursor.fetchone()
            if last_row:
                print(f"[DEBUG] Last saved record: date={last_row[0]}, score={last_row[1]}")
            else:
                print("[DEBUG] WARNING: No rows found after insert!")
            
            conn.close()
            print(f"✓ Metrics breakdown saved successfully for {current_date}")
            
        except sqlite3.Error as db_err:
            print(f"DATABASE ERROR saving metrics breakdown: {db_err}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
        except Exception as e:
            print(f"ERROR saving metrics breakdown: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in locals():
                conn.rollback()
                conn.close()
        
        
        
        
    def visualize_training_load(self):
        import matplotlib.pyplot as plt

        try:
            if self.training_log.empty or self.weekly_trimp.empty:
                print("No training data available for visualization.")
                return

            # Plot TRIMP per run over time
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(self.training_log['date'], self.training_log['TRIMP'], marker='o', linestyle='-')
            plt.title('TRIMP per Session Over Time')
            plt.xlabel('Date')
            plt.ylabel('TRIMP Score')
            plt.xticks(rotation=45)

            # Plot weekly TRIMP, Acute load, Chronic load, ACWR
            plt.subplot(1, 2, 2)
            weeks = self.weekly_trimp['week']
            plt.plot(weeks, self.weekly_trimp['weekly_trimp'], label='Weekly TRIMP Load', marker='o')
            plt.plot(weeks, self.weekly_trimp['acute_load'], label='Acute Load (1 week avg)', linestyle='--')
            plt.plot(weeks, self.weekly_trimp['chronic_load'], label='Chronic Load (4 week avg)', linestyle='--')
            plt.plot(weeks, self.weekly_trimp['acwr'], label='ACWR', linestyle='-.')
            plt.axhline(1.3, color='red', linestyle=':', label='Upper ACWR Threshold (~1.3)')
            plt.axhline(0.8, color='green', linestyle=':', label='Lower ACWR Threshold (~0.8)')
            plt.title('Weekly Training Load and ACWR')
            plt.xlabel('Week Number')
            plt.ylabel('Load / Ratio')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('c:/temp/logsFitnessApp/training_load.png', dpi=300, bbox_inches='tight')
            print("Training load chart saved to: c:/temp/logsFitnessApp/training_load.png")
            plt.show()

        except Exception as e:
            print(f"Error during visualization: {e}")
    
        
    def visualize_trends(self):
        """Create visualizations of running data"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Convert date to datetime
            self.training_log['date'] = pd.to_datetime(self.training_log['date'])
            
            # Plot 1: Running Economy over time
            plt.subplot(2, 2, 1)
            plt.plot(self.training_log['date'], self.training_log['running_economy'], 'b-o')
            plt.title('Running Economy Trend')
            plt.xticks(rotation=45)
            plt.ylabel('Running Economy')
            
            # Plot 2: Efficiency Score over time
            plt.subplot(2, 2, 2)
            plt.plot(self.training_log['date'], self.training_log['efficiency_score'], 'g-o')
            plt.title('Efficiency Score Trend')
            plt.xticks(rotation=45)
            plt.ylabel('Efficiency Score')
            
            # Plot 3: Energy Cost vs Distance
            plt.subplot(2, 2, 3)
            plt.scatter(self.training_log['distance'], self.training_log['energy_cost'])
            plt.title('Energy Cost vs Distance')
            plt.xlabel('Distance (km)')
            plt.ylabel('Energy Cost')
            
            # Plot 4: Heart Rate vs Running Economy
            plt.subplot(2, 2, 4)
            plt.scatter(self.training_log['heart_rate'], self.training_log['running_economy'])
            plt.title('Heart Rate vs Running Economy')
            plt.xlabel('Heart Rate (bpm)')
            plt.ylabel('Running Economy')
            
            plt.tight_layout()
            plt.savefig('c:/temp/logsFitnessApp/trends.png', dpi=300, bbox_inches='tight')
            print("Trends chart saved to: c:/temp/logsFitnessApp/trends.png")
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def calculate_recovery_and_readiness(self):
        df = self.training_log.copy()
        # Fill missing subjective cols with reasonable defaults
        df['resting_hr'] = df.get('resting_hr', pd.Series([np.nan]*len(df)))
        df['sleep_quality'] = df.get('sleep_quality', 3)
        df['fatigue_level'] = df.get('fatigue_level', 5)

        rhr_baseline = df['resting_hr'].dropna().mean() if df['resting_hr'].notna().any() else 60
        trimp_baseline = df['TRIMP'].rolling(window=4, min_periods=1).mean() if 'TRIMP' in df.columns else pd.Series(np.repeat(50, len(df)))

        # Normalized scores (all on 0–1 scale, higher is better)
        df['rhr_score'] = 1 - ((df['resting_hr'] - rhr_baseline) / rhr_baseline)
        df['load_score'] = 1 - (df['TRIMP'] / (trimp_baseline + 1e-8))
        df['sleep_score'] = df['sleep_quality'] / 5
        df['fatigue_score'] = 1 - (df['fatigue_level'] / 10)

        # Composite Recovery Score
        df['recovery_score'] = (
            0.3 * df['rhr_score'].fillna(1) +
            0.3 * df['load_score'].fillna(1) +
            0.2 * df['sleep_score'].fillna(0.6) +
            0.2 * df['fatigue_score'].fillna(0.5)
        )

        # Readiness Score (can weight recovery more, or add freshness/load components)
        df['readiness_score'] = (
            0.5 * df['recovery_score'] +
            0.3 * df['load_score'].fillna(1) +
            0.2 * df['sleep_score'].fillna(0.6)
        )

        self.training_log = df
        return df[['date','recovery_score','readiness_score']]


    def visualize_recovery_and_readiness(self):
        import matplotlib.pyplot as plt
        self.calculate_recovery_and_readiness()
        plt.figure(figsize=(12, 5))
        plt.plot(self.training_log['date'], self.training_log['recovery_score'], label='Recovery')
        plt.plot(self.training_log['date'], self.training_log['readiness_score'], label='Readiness')
        plt.axhline(0.7, color='orange', linestyle='--', label='Caution threshold')
        plt.xlabel('Date')
        plt.ylabel('Score (0–1)')
        plt.title('Recovery and Readiness Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig('c:/temp/logsFitnessApp/recovery_readiness.png', dpi=300, bbox_inches='tight')
        print("Recovery and readiness chart saved to: c:/temp/logsFitnessApp/recovery_readiness.png")
        plt.show()


    def calculate_training_zones(self, running_economy, vo2max):
        """Calculate training zones based on running economy"""
        zones = {
            'Recovery': (0.6 * running_economy, 0.7 * running_economy),
            'Endurance': (0.7 * running_economy, 0.8 * running_economy),
            'Tempo': (0.8 * running_economy, 0.9 * running_economy),
            'Threshold': (0.9 * running_economy, running_economy),
            'VO2Max': (running_economy, 1.1 * running_economy)
        }
        return zones
    
    def print_training_zones(self, running_economy, vo2max):
        """Print training zones"""
        training_zones = self.calculate_training_zones(running_economy, vo2max)
        print("\nTraining Zones based on Running Economy:")
        for zone, (lower, upper) in training_zones.items():
            print(f"{zone}: {lower:.1f} - {upper:.1f}")
            
    def advanced_visualizations(self):
        """Create advanced performance visualizations"""
        plt.figure(figsize=(20, 15))
        
        # 1. Cumulative Distance Over Time
        plt.subplot(2, 3, 1)
        self.training_log['cumulative_distance'] = self.training_log['distance'].cumsum()
        plt.plot(self.training_log['date'], self.training_log['cumulative_distance'], 'b-o')
        plt.title('Cumulative Running Distance')
        plt.xlabel('Date')
        plt.ylabel('Total Distance (km)')
        plt.xticks(rotation=45)
        
        # 2. Moving Average of Running Economy
        plt.subplot(2, 3, 2)
        self.training_log['running_economy_ma'] = self.training_log['running_economy'].rolling(window=3).mean()
        plt.plot(self.training_log['date'], self.training_log['running_economy'], 'g-', label='Original')
        plt.plot(self.training_log['date'], self.training_log['running_economy_ma'], 'r-', label='3-Session Moving Avg')
        plt.title('Running Economy Trend')
        plt.xlabel('Date')
        plt.ylabel('Running Economy')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 3. Heart Rate vs Pace Correlation
        plt.subplot(2, 3, 3)
        pace = self.training_log['time'] / self.training_log['distance']
        plt.scatter(pace, self.training_log['heart_rate'], alpha=0.7)
        plt.title('Pace vs Heart Rate')
        plt.xlabel('Pace (min/km)')
        plt.ylabel('Heart Rate (bpm)')
        
       # In the advanced_visualizations method, modify the pie chart section:

        # 4. Training Zones Pie Chart
        plt.subplot(2, 3, 4)
        try:
            # Calculate zones only for rows with valid running_economy and vo2max
            valid_rows = self.training_log[
                (self.training_log['running_economy'].notna()) & 
                (self.training_log['vo2max'].notna())
            ]
            
            if not valid_rows.empty:
                # Use the first valid row for zone calculation
                first_valid = valid_rows.iloc[0]
                zones = self.calculate_training_zones(first_valid['running_economy'], first_valid['vo2max'])
                
                zone_durations = {}
                for zone, (lower, upper) in zones.items():
                    count = len(valid_rows[
                        (valid_rows['running_economy'] >= lower) & 
                        (valid_rows['running_economy'] < upper)
                    ])
                    if count > 0:  # Only include zones with data
                        zone_durations[zone] = count
                
                if zone_durations:  # Check if we have any data to plot
                    plt.pie(
                        list(zone_durations.values()), 
                        labels=list(zone_durations.keys()), 
                        autopct='%1.1f%%'
                    )
                    plt.title('Training Zones Distribution')
                else:
                    plt.text(0.5, 0.5, 'No valid zone data', ha='center', va='center')
            else:
                plt.text(0.5, 0.5, 'No valid training data', ha='center', va='center')
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            plt.text(0.5, 0.5, 'Error creating pie chart', ha='center', va='center')
        
        # 5. Performance Progression Radar Chart
        plt.subplot(2, 3, 5, polar=True)
        metrics = [
            'running_economy', 
            'vo2max', 
            'distance', 
            'efficiency_score', 
            'heart_rate'
        ]
        
        # Normalize metrics
        normalized_metrics = self.training_log[metrics].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # Average of normalized metrics for each session
        avg_metrics = normalized_metrics.mean()
        
        # Radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = avg_metrics.values
        values = np.concatenate((values, [values[0]]))  # Repeat first value to close the polygon
        angles = np.concatenate((angles, [angles[0]]))  # Repeat first angle
        
        plt.polar(angles, values, 'o-', linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        plt.xticks(angles[:-1], metrics)
        plt.title('Performance Metrics Radar Chart')
        
        # 6. Seasonal Performance Heatmap
        plt.subplot(2, 3, 6)
        self.training_log['month'] = self.training_log['date'].dt.month
        seasonal_performance = self.training_log.groupby('month')['running_economy'].mean()
        
        plt.imshow([seasonal_performance.values], cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Avg Running Economy')
        plt.title('Seasonal Performance Heatmap')
        plt.xlabel('Month')
        plt.xticks(range(len(seasonal_performance)), seasonal_performance.index)
        
        plt.tight_layout()
        plt.savefig('c:/temp/logsFitnessApp/advanced_metrics.png', dpi=300, bbox_inches='tight')
        print("Advanced metrics chart saved to: c:/temp/logsFitnessApp/advanced_metrics.png")
        plt.show()
        
        
    # trainning score calculation

    def calculate_training_score(self):
        """
        Calculate a comprehensive training score based on multiple performance metrics
        
        Returns a dictionary with detailed score breakdown and overall training score
        """
        # Normalize and weight different metrics
        try:
            # Normalize each metric
            normalized_data = self.training_log.copy()
            
            # Metrics to consider
            metrics = {
                'running_economy': {'weight': 0.25, 'higher_is_better': True},
                'vo2max': {'weight': 0.20, 'higher_is_better': True},
                'distance': {'weight': 0.15, 'higher_is_better': True},
                'efficiency_score': {'weight': 0.20, 'higher_is_better': True},
                'heart_rate': {'weight': 0.20, 'higher_is_better': False}
            }
            
            # Normalization function
            def normalize_metric(series, higher_is_better):
                if higher_is_better:
                    return (series - series.min()) / (series.max() - series.min())
                else:
                    return 1 - ((series - series.min()) / (series.max() - series.min()))
            
            # Calculate normalized scores
            normalized_scores = {}
            for metric, config in metrics.items():
                normalized_scores[metric] = normalize_metric(
                    normalized_data[metric], 
                    config['higher_is_better']
                )
            
            # Calculate weighted scores
            weighted_scores = {}
            for metric, config in metrics.items():
                weighted_scores[metric] = normalized_scores[metric] * config['weight']
            
            # Overall training score
            # overall_score = sum(weighted_scores.values()) * 100
            # Overall training score
            overall_score = sum(weighted_scores[metric].mean() for metric in metrics) * 100
            
            # Detailed analysis
            analysis = {
                'overall_score': overall_score,
                'metric_breakdown': {
                    metric: {
                        'normalized_value': normalized_scores[metric].mean(),
                        'weighted_value': weighted_scores[metric].mean(),
                        'raw_mean': self.training_log[metric].mean(),
                        'raw_std': self.training_log[metric].std()
                    } for metric in metrics
                },
                'performance_trends': {
                    'running_economy_trend': normalized_scores['running_economy'].corr(normalized_data['date']),
                    'distance_progression': normalized_scores['distance'].corr(normalized_data['date'])
                }
            }
            
            return analysis
        
        except Exception as e:
            print(f"Error calculating training score: {e}")
            return None 

    def visualize_score_impact_over_time(self, extra_scores=None):
        """
        Visualizes different scoring systems over time.
        extra_scores: dict, e.g. {'Recovery Score': 'recovery_score', 'Readiness': 'readiness_score'}
        """
        import matplotlib.pyplot as plt

        # Ensure dates are sorted and converted
        df = self.training_log.sort_values('date').copy()
        df['date'] = pd.to_datetime(df['date'])

        plt.figure(figsize=(14,7))

        # Plot the overall training score (current main score)
        training_score_result = self.calculate_training_score()
        # Assume you store a time series of scores; otherwise, recalculate for each row
        df['Overall Score'] = self.calculate_training_score()['overall_score']  # If per-session, else plot as flat line

        # Plot the score(s) over time
        plt.plot(df['date'], df['Overall Score'], label="Overall Training Score", linewidth=2)

        # Overlay additional scoring methods, if provided
        if extra_scores:
            for label, col in extra_scores.items():
                if col in df.columns:
                    plt.plot(df['date'], df[col], linestyle='--', label=label)

        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.title('Comparison of Scoring Calculations Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig('c:/temp/logsFitnessApp/score_impact.png', dpi=300, bbox_inches='tight')
        print("Score impact chart saved to: c:/temp/logsFitnessApp/score_impact.png")
        plt.show()

    def visualize_speed_metrics(self):
        """Create comprehensive speed-related visualizations"""
        try:
            if self.training_log.empty:
                print("No data available")
                return
            
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('Speed Metrics Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Average Speed Trend
            ax1 = axes[0, 0]
            ax1.plot(self.training_log['date'], self.training_log['avg_speed'], 
                    marker='o', color='blue', label='Avg Speed')
            ax1.plot(self.training_log['date'], self.training_log['max_speed'], 
                    marker='s', color='red', alpha=0.6, label='Max Speed')
            ax1.set_title('Speed Trends Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Speed (km/h)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Speed Reserve
            ax2 = axes[0, 1]
            ax2.plot(self.training_log['date'], self.training_log['speed_reserve'], 
                    marker='o', color='green')
            ax2.set_title('Speed Reserve (Max - Avg)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Speed Reserve (km/h)')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Speed vs Heart Rate
            ax3 = axes[1, 0]
            scatter = ax3.scatter(self.training_log['heart_rate'], 
                                self.training_log['avg_speed'],
                                c=self.training_log['date'].astype(np.int64),
                                cmap='viridis', s=100, alpha=0.6)
            ax3.set_title('Speed vs Heart Rate (colored by time)')
            ax3.set_xlabel('Heart Rate (bpm)')
            ax3.set_ylabel('Average Speed (km/h)')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Date')
            
            # Plot 4: Speed Efficiency (Speed/HR)
            ax4 = axes[1, 1]
            ax4.plot(self.training_log['date'], self.training_log['speed_efficiency'], 
                    marker='o', color='purple')
            ax4.set_title('Speed Efficiency (Speed per HR unit)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Speed/HR (km/h per bpm)')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            # Plot 5: Pace Progression
            ax5 = axes[2, 0]
            ax5.plot(self.training_log['date'], self.training_log['pace_per_km'], 
                    marker='o', color='orange')
            ax5.set_title('Pace Progression')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Pace (min/km)')
            ax5.invert_yaxis()  # Lower is better for pace
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='x', rotation=45)
            
            # Plot 6: Speed Zone Distribution
            ax6 = axes[2, 1]
            if 'speed_zone' in self.training_log.columns:
                zone_counts = self.training_log['speed_zone'].value_counts()
                ax6.bar(zone_counts.index.astype(str), zone_counts.values, 
                       color=['#3498db', '#2ecc71', '#e74c3c'])
                ax6.set_title('Training Sessions by Speed Zone')
                ax6.set_xlabel('Speed Zone')
                ax6.set_ylabel('Number of Sessions')
                ax6.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('c:/temp/logsFitnessApp/speed_metrics.png', dpi=300, bbox_inches='tight')
            print("Speed metrics chart saved to: c:/temp/logsFitnessApp/speed_metrics.png")
            plt.show()
            
        except Exception as e:
            print(f"Error in speed visualization: {e}")

    def visualize_hr_rs_deviation(self):
        """Create HR-RS Deviation Index visualizations"""
        try:
            if self.training_log.empty:
                print("No data available")
                return
            
            # Filter valid data
            valid_data = self.training_log[self.training_log['hr_rs_deviation'] > 0].copy()
            
            if valid_data.empty:
                print("No HR-RS Deviation data available")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('HR-RS Deviation Index Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: HR-RS Deviation Trend
            ax1 = axes[0, 0]
            ax1.plot(valid_data['date'], valid_data['hr_rs_deviation'], 
                    marker='o', color='red', linewidth=2)
            # Add rolling average
            rolling_avg = valid_data['hr_rs_deviation'].rolling(window=3, min_periods=1).mean()
            ax1.plot(valid_data['date'], rolling_avg, 
                    linestyle='--', color='blue', linewidth=2, label='3-session avg')
            ax1.set_title('HR-RS Deviation Index Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Deviation Index')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Deviation vs Performance (Speed)
            ax2 = axes[0, 1]
            ax2.scatter(valid_data['hr_rs_deviation'], valid_data['avg_speed'],
                       s=100, alpha=0.6, c='green')
            ax2.set_title('HR-RS Deviation vs Speed Performance')
            ax2.set_xlabel('HR-RS Deviation Index')
            ax2.set_ylabel('Average Speed (km/h)')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(valid_data) >= 3:
                z = np.polyfit(valid_data['hr_rs_deviation'], valid_data['avg_speed'], 1)
                p = np.poly1d(z)
                ax2.plot(valid_data['hr_rs_deviation'], 
                        p(valid_data['hr_rs_deviation']), 
                        "r--", alpha=0.8, linewidth=2, label='Trend')
                ax2.legend()
            
            # Plot 3: Deviation Distribution
            ax3 = axes[1, 0]
            ax3.hist(valid_data['hr_rs_deviation'], bins=15, color='purple', 
                    alpha=0.7, edgecolor='black')
            ax3.axvline(valid_data['hr_rs_deviation'].mean(), 
                       color='red', linestyle='--', linewidth=2, label='Mean')
            ax3.set_title('HR-RS Deviation Distribution')
            ax3.set_xlabel('Deviation Index')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Deviation vs TRIMP (if available)
            ax4 = axes[1, 1]
            if 'TRIMP' in valid_data.columns:
                ax4.scatter(valid_data['TRIMP'], valid_data['hr_rs_deviation'],
                           s=100, alpha=0.6, c='orange')
                ax4.set_title('HR-RS Deviation vs Training Load (TRIMP)')
                ax4.set_xlabel('TRIMP Score')
                ax4.set_ylabel('HR-RS Deviation Index')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'TRIMP data not available', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            plt.savefig('c:/temp/logsFitnessApp/hr_rs_deviation.png', dpi=300, bbox_inches='tight')
            print("HR-RS deviation chart saved to: c:/temp/logsFitnessApp/hr_rs_deviation.png")
            plt.show()
            
        except Exception as e:
            print(f"Error in HR-RS deviation visualization: {e}")

    def create_performance_dashboard(self):
        """Create comprehensive dashboard with all new metrics"""
        try:
            if self.training_log.empty:
                print("No data available")
                return
            
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            fig.suptitle('Comprehensive Running Performance Dashboard', 
                        fontsize=18, fontweight='bold')
            
            # Row 1: Speed metrics
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.training_log['date'], self.training_log['avg_speed'], 
                    marker='o', color='blue')
            ax1.set_title('Average Speed Trend')
            ax1.set_ylabel('Speed (km/h)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.training_log['date'], self.training_log['speed_reserve'], 
                    marker='o', color='green')
            ax2.set_title('Speed Reserve')
            ax2.set_ylabel('km/h')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(self.training_log['date'], self.training_log['pace_per_km'], 
                    marker='o', color='orange')
            ax3.set_title('Pace')
            ax3.set_ylabel('min/km')
            ax3.invert_yaxis()
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Row 2: HR-RS Deviation
            valid_hr_rs = self.training_log[self.training_log['hr_rs_deviation'] > 0]
            
            ax4 = fig.add_subplot(gs[1, 0])
            if not valid_hr_rs.empty:
                ax4.plot(valid_hr_rs['date'], valid_hr_rs['hr_rs_deviation'], 
                        marker='o', color='red')
                ax4.set_title('HR-RS Deviation Index')
                ax4.set_ylabel('Index')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(gs[1, 1])
            if not valid_hr_rs.empty:
                ax5.scatter(valid_hr_rs['hr_rs_deviation'], valid_hr_rs['avg_speed'],
                           s=100, alpha=0.6, c='purple')
                ax5.set_title('Deviation vs Speed')
                ax5.set_xlabel('HR-RS Deviation')
                ax5.set_ylabel('Speed (km/h)')
                ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[1, 2])
            if not valid_hr_rs.empty:
                ax6.hist(valid_hr_rs['hr_rs_deviation'], bins=15, 
                        color='purple', alpha=0.7, edgecolor='black')
                ax6.set_title('Deviation Distribution')
                ax6.set_xlabel('Index')
                ax6.grid(True, alpha=0.3, axis='y')
            
            # Row 3: Efficiency metrics
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(self.training_log['date'], self.training_log['speed_efficiency'], 
                    marker='o', color='teal')
            ax7.set_title('Speed Efficiency (Speed/HR)')
            ax7.set_ylabel('km/h per bpm')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3)
            
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(self.training_log['date'], self.training_log['economy_at_speed'], 
                    marker='o', color='brown')
            ax8.set_title('Economy at Speed')
            ax8.set_ylabel('RE / Speed')
            ax8.tick_params(axis='x', rotation=45)
            ax8.grid(True, alpha=0.3)
            
            ax9 = fig.add_subplot(gs[2, 2])
            if 'physio_efficiency' in self.training_log.columns:
                valid_physio = self.training_log[self.training_log['physio_efficiency'] > 0]
                if not valid_physio.empty:
                    ax9.plot(valid_physio['date'], valid_physio['physio_efficiency'], 
                            marker='o', color='darkgreen')
                    ax9.set_title('Physiological Efficiency')
                    ax9.set_ylabel('Composite Score')
                    ax9.tick_params(axis='x', rotation=45)
                    ax9.grid(True, alpha=0.3)
            
            # Row 4: Combined analysis
            ax10 = fig.add_subplot(gs[3, :2])
            ax10_twin = ax10.twinx()
            
            line1 = ax10.plot(self.training_log['date'], self.training_log['avg_speed'], 
                             marker='o', color='blue', label='Avg Speed')
            line2 = ax10_twin.plot(self.training_log['date'], self.training_log['heart_rate'], 
                                  marker='s', color='red', alpha=0.6, label='Heart Rate')
            
            ax10.set_title('Speed vs Heart Rate Over Time')
            ax10.set_xlabel('Date')
            ax10.set_ylabel('Speed (km/h)', color='blue')
            ax10_twin.set_ylabel('Heart Rate (bpm)', color='red')
            ax10.tick_params(axis='x', rotation=45)
            ax10.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax10.legend(lines, labels, loc='upper left')
            
            ax11 = fig.add_subplot(gs[3, 2])
            if 'speed_zone' in self.training_log.columns:
                zone_counts = self.training_log['speed_zone'].value_counts()
                colors = ['#3498db', '#2ecc71', '#e74c3c']
                ax11.pie(zone_counts.values, labels=zone_counts.index, 
                        autopct='%1.1f%%', colors=colors, startangle=90)
                ax11.set_title('Speed Zone Distribution')
            
            plt.savefig('c:/temp/logsFitnessApp/performance_dashboard.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nDashboard saved to: c:/temp/logsFitnessApp/performance_dashboard.png")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")

    def calculate_monthly_metrics_averages(self):
        """Calculate monthly averages for all metrics"""
        try:
            if self.training_log.empty:
                print("No training data available for monthly averages.")
                return None
            
            # Ensure date is datetime
            df = self.training_log.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['year_month'] = df['date'].dt.to_period('M')
            
            # Define metrics to average
            metrics = [
                'running_economy', 
                'vo2max', 
                'distance', 
                'efficiency_score', 
                'heart_rate',
                'energy_cost',
                'TRIMP'
            ]
            
            # Add optional metrics if they exist
            optional_metrics = [
                'recovery_score', 
                'readiness_score',
                'avg_speed',
                'max_speed',
                'speed_reserve',
                'hr_rs_deviation',
                'speed_efficiency',
                'pace_per_km',
                'economy_at_speed',
                'physio_efficiency'
            ]
            for metric in optional_metrics:
                if metric in df.columns:
                    metrics.append(metric)
            
            # Calculate monthly averages
            monthly_averages = df.groupby('year_month')[metrics].agg(['mean', 'std', 'count'])
            
            return monthly_averages
        
        except Exception as e:
            print(f"Error calculating monthly averages: {e}")
            return None
    
    def analyze_speed_metrics(self):
        """Analyze speed-related metrics"""
        try:
            if self.training_log.empty:
                print("No data available")
                return None
            
            print("\n" + "="*80)
            print("SPEED METRICS ANALYSIS")
            print("="*80)
            
            # Overall speed statistics
            print("\nOverall Speed Statistics:")
            print(f"Average Speed (mean):     {self.training_log['avg_speed'].mean():.2f} km/h")
            print(f"Average Speed (std):      {self.training_log['avg_speed'].std():.2f} km/h")
            print(f"Max Speed (mean):         {self.training_log['max_speed'].mean():.2f} km/h")
            print(f"Max Speed (peak):         {self.training_log['max_speed'].max():.2f} km/h")
            print(f"Speed Reserve (mean):     {self.training_log['speed_reserve'].mean():.2f} km/h")
            print(f"Speed Consistency (mean): {self.training_log['speed_consistency'].mean():.2%}")
            print(f"Average Pace:             {self.training_log['pace_per_km'].mean():.2f} min/km")
            
            # Speed efficiency (speed per heart beat)
            print(f"\nSpeed Efficiency:         {self.training_log['speed_efficiency'].mean():.4f} km/h per bpm")
            print(f"Economy at Speed:         {self.training_log['economy_at_speed'].mean():.2f}")
            
            # Speed zone distribution
            print("\nSpeed Zone Distribution:")
            zone_counts = self.training_log['speed_zone'].value_counts()
            for zone, count in zone_counts.items():
                pct = (count / len(self.training_log)) * 100
                print(f"  {zone}: {count} sessions ({pct:.1f}%)")
            
            # Trend analysis (last 5 vs first 5 sessions)
            if len(self.training_log) >= 10:
                recent_avg = self.training_log.tail(5)['avg_speed'].mean()
                early_avg = self.training_log.head(5)['avg_speed'].mean()
                improvement = ((recent_avg - early_avg) / early_avg) * 100
                print(f"\nSpeed Improvement (recent vs early): {improvement:+.2f}%")
            
            return {
                'avg_speed_mean': self.training_log['avg_speed'].mean(),
                'max_speed_peak': self.training_log['max_speed'].max(),
                'speed_reserve': self.training_log['speed_reserve'].mean(),
                'speed_consistency': self.training_log['speed_consistency'].mean(),
                'pace_per_km': self.training_log['pace_per_km'].mean()
            }
        
        except Exception as e:
            print(f"Error in speed analysis: {e}")
            return None

    def analyze_hr_rs_deviation(self):
        """Analyze HR-RS Deviation Index patterns"""
        try:
            if self.training_log.empty:
                print("No data available")
                return None
            
            print("\n" + "="*80)
            print("HR-RS DEVIATION INDEX ANALYSIS")
            print("="*80)
            
            # Filter out zero values
            valid_data = self.training_log[self.training_log['hr_rs_deviation'] > 0]
            
            if valid_data.empty:
                print("No HR-RS Deviation data available")
                return None
            
            print("\nOverall HR-RS Deviation Statistics:")
            print(f"Mean:                 {valid_data['hr_rs_deviation'].mean():.2f}")
            print(f"Std Dev:              {valid_data['hr_rs_deviation'].std():.2f}")
            print(f"Min:                  {valid_data['hr_rs_deviation'].min():.2f}")
            print(f"Max:                  {valid_data['hr_rs_deviation'].max():.2f}")
            
            # Calculate stability (coefficient of variation)
            cv = (valid_data['hr_rs_deviation'].std() / valid_data['hr_rs_deviation'].mean()) * 100
            print(f"Coefficient of Variation: {cv:.2f}% ", end="")
            if cv < 10:
                print("(Very Stable)")
            elif cv < 20:
                print("(Stable)")
            elif cv < 30:
                print("(Moderate Variability)")
            else:
                print("(High Variability)")
            
            # Trend analysis - calculate change rate
            if len(valid_data) >= 5:
                valid_data = valid_data.sort_values('date')
                recent_mean = valid_data.tail(3)['hr_rs_deviation'].mean()
                earlier_mean = valid_data.head(3)['hr_rs_deviation'].mean()
                change_rate = ((recent_mean - earlier_mean) / earlier_mean) * 100
                
                print(f"\nRecent Trend: {change_rate:+.2f}% ", end="")
                if abs(change_rate) < 5:
                    print("(Stable)")
                elif change_rate > 5:
                    print("(Increasing - may indicate fatigue)")
                else:
                    print("(Decreasing - may indicate improved adaptation)")
            
            # Correlation with performance metrics
            if len(valid_data) >= 10:
                corr_speed = valid_data['hr_rs_deviation'].corr(valid_data['avg_speed'])
                corr_hr = valid_data['hr_rs_deviation'].corr(valid_data['heart_rate'])
                corr_vo2 = valid_data['hr_rs_deviation'].corr(valid_data['vo2max'])
                
                print("\nCorrelations with Performance:")
                print(f"  vs. Average Speed:  {corr_speed:+.3f}")
                print(f"  vs. Heart Rate:     {corr_hr:+.3f}")
                print(f"  vs. VO2max:         {corr_vo2:+.3f}")
            
            return {
                'mean': valid_data['hr_rs_deviation'].mean(),
                'std': valid_data['hr_rs_deviation'].std(),
                'stability_cv': cv
            }
        
        except Exception as e:
            print(f"Error in HR-RS deviation analysis: {e}")
            return None

    def print_monthly_metrics_averages(self):
        """Print monthly averages for metrics breakdown"""
        monthly_avg = self.calculate_monthly_metrics_averages()
        
        if monthly_avg is None or monthly_avg.empty:
            return
        
        print("\n" + "="*80)
        print("MONTHLY METRICS BREAKDOWN - AVERAGES")
        print("="*80)
        
        for month in monthly_avg.index:
            print(f"\n{month} ({int(monthly_avg.loc[month, ('running_economy', 'count')])} sessions)")
            print("-" * 80)
            
            # Running Economy
            if 'running_economy' in monthly_avg.columns.get_level_values(0):
                re_mean = monthly_avg.loc[month, ('running_economy', 'mean')]
                re_std = monthly_avg.loc[month, ('running_economy', 'std')]
                print(f"Running Economy:     {re_mean:>8.2f} ± {re_std:>6.2f}")
            
            # VO2Max
            if 'vo2max' in monthly_avg.columns.get_level_values(0):
                vo2_mean = monthly_avg.loc[month, ('vo2max', 'mean')]
                vo2_std = monthly_avg.loc[month, ('vo2max', 'std')]
                print(f"VO2Max:              {vo2_mean:>8.2f} ± {vo2_std:>6.2f}")
            
            # Distance
            if 'distance' in monthly_avg.columns.get_level_values(0):
                dist_mean = monthly_avg.loc[month, ('distance', 'mean')]
                dist_std = monthly_avg.loc[month, ('distance', 'std')]
                print(f"Distance (km):       {dist_mean:>8.2f} ± {dist_std:>6.2f}")
            
            # Efficiency Score
            if 'efficiency_score' in monthly_avg.columns.get_level_values(0):
                eff_mean = monthly_avg.loc[month, ('efficiency_score', 'mean')]
                eff_std = monthly_avg.loc[month, ('efficiency_score', 'std')]
                print(f"Efficiency Score:    {eff_mean:>8.2f} ± {eff_std:>6.2f}")
            
            # Heart Rate
            if 'heart_rate' in monthly_avg.columns.get_level_values(0):
                hr_mean = monthly_avg.loc[month, ('heart_rate', 'mean')]
                hr_std = monthly_avg.loc[month, ('heart_rate', 'std')]
                print(f"Heart Rate (bpm):    {hr_mean:>8.2f} ± {hr_std:>6.2f}")
            
            # Energy Cost
            if 'energy_cost' in monthly_avg.columns.get_level_values(0):
                ec_mean = monthly_avg.loc[month, ('energy_cost', 'mean')]
                ec_std = monthly_avg.loc[month, ('energy_cost', 'std')]
                print(f"Energy Cost:         {ec_mean:>8.2f} ± {ec_std:>6.2f}")
            
            # TRIMP
            if 'TRIMP' in monthly_avg.columns.get_level_values(0):
                trimp_mean = monthly_avg.loc[month, ('TRIMP', 'mean')]
                trimp_std = monthly_avg.loc[month, ('TRIMP', 'std')]
                print(f"TRIMP:               {trimp_mean:>8.2f} ± {trimp_std:>6.2f}")
            
            # Recovery Score (if available)
            if 'recovery_score' in monthly_avg.columns.get_level_values(0):
                rec_mean = monthly_avg.loc[month, ('recovery_score', 'mean')]
                rec_std = monthly_avg.loc[month, ('recovery_score', 'std')]
                print(f"Recovery Score:      {rec_mean:>8.2f} ± {rec_std:>6.2f}")
            
            # Readiness Score (if available)
            if 'readiness_score' in monthly_avg.columns.get_level_values(0):
                ready_mean = monthly_avg.loc[month, ('readiness_score', 'mean')]
                ready_std = monthly_avg.loc[month, ('readiness_score', 'std')]
                print(f"Readiness Score:     {ready_mean:>8.2f} ± {ready_std:>6.2f}")
        
        print("\n" + "="*80)
    
def main():
    # Database path
    db_path = 'c:/smakrykoDBs/Apex.db'

    # Create analysis object
    analysis = RunningAnalysis('c:/smakrykoDBs/Apex.db')
    
    # DEBUG: Check if data was loaded
    print(f"\n[DEBUG] Training log loaded: {len(analysis.training_log)} rows")
    print(f"[DEBUG] Training log empty: {analysis.training_log.empty}")
    
    # Add sample session if database is empty
    if analysis.training_log.empty:
        print("[DEBUG] Database is empty, adding sample session...")
        analysis.add_session(
            date=datetime.now().strftime('%Y-%m-%d'),
            running_economy=73,
            vo2max=19.0,
            distance=5,
            time=27,
            heart_rate=150
        )
        print(f"[DEBUG] After adding session: {len(analysis.training_log)} rows")
    
    # IMPORTANT: Reload data to ensure we have all calculated fields
    print("[DEBUG] Reloading training data to ensure all fields are present...")
    analysis.training_log = analysis.load_training_data()
    print(f"[DEBUG] After reload: {len(analysis.training_log)} rows")
    
    # Create metrics_breakdown table
    analysis.create_metrics_breakdown_table()
    
    # Save training log to database.
    analysis.save_training_log_to_db()
    
    # Print training log
    print("\n[DEBUG] Training Log Preview:")
    print(analysis.training_log.head())
    print(f"\n[DEBUG] Training log shape: {analysis.training_log.shape}")
    
    # Skip visualizations if no data
    if not analysis.training_log.empty:
        # Visualize trends
        analysis.visualize_trends()
        
        # Visualize advanced metrics
        analysis.advanced_visualizations()
    
    # Calculate and print training score
    training_score = analysis.calculate_training_score()
    
    if not analysis.training_log.empty:
        # Visualize TRIMP
        analysis.visualize_training_load()

        # Calculate and Visualize Recovery and Readiness Scores
        analysis.calculate_recovery_and_readiness()
        analysis.visualize_recovery_and_readiness()

    # Print monthly metrics averages
    analysis.print_monthly_metrics_averages()
    
    # Create monthly summaries table
    analysis.create_monthly_summaries_table()
    
    # Save monthly summaries to database
    print("\n[DEBUG] Saving monthly summaries...")
    analysis.save_monthly_summaries()

    print("\n[DEBUG] Checking training_score value...")
    if training_score:
        print(f"[DEBUG] training_score is valid: {type(training_score)}")
        print("\nTraining Score Analysis:")
        print(f"Overall Training Score: {float(training_score['overall_score']):.2f}")
        
        # Save metrics breakdown to database
        print("\n[DEBUG] About to call save_metrics_breakdown()...")
        analysis.save_metrics_breakdown(training_score)
        print("[DEBUG] Returned from save_metrics_breakdown()")
        
        print("\nMetric Breakdown:")
        for metric, details in training_score['metric_breakdown'].items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Normalized Value: {details['normalized_value']:.4f}")
            print(f"  Weighted Value: {details['weighted_value']:.4f}")
            print(f"  Raw Mean: {details['raw_mean']:.2f}")
            print(f"  Raw Std Dev: {details['raw_std']:.2f}")
        
        print("\nPerformance Trends:")
        for trend, value in training_score['performance_trends'].items():
            print(f"{trend.replace('_', ' ').title()}: {value:.4f}")
    else:
        print("[DEBUG] WARNING: training_score is None!")

    if not analysis.training_log.empty:
        # Make sure your DataFrame has columns: 'recovery_score', 'readiness_score', etc.
        analysis.visualize_score_impact_over_time(
            extra_scores={
                'Recovery Score': 'recovery_score',
                'Readiness Score': 'readiness_score'
            }
        )

        # NEW ENHANCEMENTS: Speed and HR-RS Deviation Analysis
        print("\n" + "="*80)
        print("ENHANCED PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Analyze speed metrics
        speed_analysis = analysis.analyze_speed_metrics()
        
        # Analyze HR-RS deviation
        hr_rs_analysis = analysis.analyze_hr_rs_deviation()
        
        # Visualize speed metrics
        analysis.visualize_speed_metrics()
        
        # Visualize HR-RS deviation
        analysis.visualize_hr_rs_deviation()
        
        # Create comprehensive performance dashboard
        analysis.create_performance_dashboard()
    
    print("\n[DEBUG] Main execution completed")

if __name__ == "__main__":
    main()