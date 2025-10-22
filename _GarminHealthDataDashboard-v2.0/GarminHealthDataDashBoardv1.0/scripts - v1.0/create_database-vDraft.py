import sqlite3
import os
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
import json

class GarminHealthMetricsDB:
    def __init__(self, db_path='c:/smakrykoDBs/health_hydra.db'):
        """Initialize database connection following Garmin data structure"""
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
        # Enable foreign keys
        self.cursor.execute("PRAGMA foreign_keys = ON")
        
    def create_garmin_health_tables(self):
        """Create tables matching Garmin Health API structure"""
        
        # Daily Summary - corresponds to Garmin's "My Day" section
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            steps INTEGER,
            total_distance_in_meters INTEGER,
            duration_in_seconds INTEGER,
            active_time_in_seconds INTEGER,
            active_kilocalories INTEGER,
            bmr_kilocalories INTEGER,
            wellness_kilocalories INTEGER,
            wellness_active_kilocalories INTEGER,
            wellness_distance_in_meters INTEGER,
            wellness_active_time_in_seconds INTEGER,
            high_stress_duration_in_seconds INTEGER,
            low_stress_duration_in_seconds INTEGER,
            overall_stress_level INTEGER,
            stress_qualifier TEXT,
            measured_heart_rate INTEGER,
            resting_heart_rate INTEGER,
            last_seven_days_avg_resting_heart_rate REAL,
            source TEXT,
            time_offset_heart_rate_samples INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )''')
        
        # Epoch Summary - 15-minute interval wellness data
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS epoch_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            summary_start_time_in_seconds INTEGER NOT NULL,
            summary_end_time_in_seconds INTEGER NOT NULL,
            activity_type TEXT,
            duration_in_seconds INTEGER,
            active_time_in_seconds INTEGER,
            steps INTEGER,
            distance_in_meters INTEGER,
            active_kilocalories INTEGER,
            met_1_minute_average REAL,
            intensity_time_goal INTEGER,
            floors_climbed INTEGER,
            met_max REAL,
            mean_motion_intensity REAL,
            max_motion_intensity REAL,
            start_time_offset INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Sleep Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sleep_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            sleep_start_timestamp_gmt INTEGER,
            sleep_end_timestamp_gmt INTEGER,
            sleep_start_timestamp_local INTEGER,
            sleep_end_timestamp_local INTEGER,
            duration_in_seconds INTEGER,
            unmeasurable_sleep_in_seconds INTEGER,
            deep_sleep_duration_in_seconds INTEGER,
            light_sleep_duration_in_seconds INTEGER,
            rem_sleep_duration_in_seconds INTEGER,
            awake_duration_in_seconds INTEGER,
            device_id TEXT,
            sleep_levels_map TEXT, -- JSON string
            validation TEXT,
            auto_sleep_start_timestamp_gmt INTEGER,
            auto_sleep_end_timestamp_gmt INTEGER,
            sleep_need_in_seconds INTEGER,
            overall_sleep_score INTEGER,
            sleep_score_personalization TEXT,
            sleep_score_insight TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )''')
        
        # HRV Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS hrv_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            weekly_avg REAL,
            last_night_avg REAL,
            last_night_5_min_high REAL,
            baseline_low_upper REAL,
            baseline_balanced_low REAL,
            baseline_balanced_upper REAL,
            status TEXT,
            feedback_phrase TEXT,
            measure_timestamp_gmt INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )''')
        
        # Stress Detail Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS stress_detail_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            start_time_in_seconds INTEGER NOT NULL,
            duration_in_seconds INTEGER,
            stress_avg_3_min INTEGER,
            stress_max_3_min INTEGER,
            stress_avg_rest INTEGER,
            stress_avg_activity INTEGER,
            stress_avg_uncategorized INTEGER,
            rest_stress_avg_3_min INTEGER,
            activity_stress_avg_3_min INTEGER,
            uncategorized_stress_avg_3_min INTEGER,
            start_time_offset INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Body Composition Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS body_composition_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            measurement_timestamp_gmt INTEGER NOT NULL,
            weight REAL,
            bmi REAL,
            body_fat_percentage REAL,
            body_water_percentage REAL,
            bone_mass REAL,
            muscle_mass REAL,
            visceral_fat_rating INTEGER,
            metabolic_age INTEGER,
            physique_rating INTEGER,
            source_type TEXT,
            timestamp_offset INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, measurement_timestamp_gmt)
        )''')
        
        # Pulse Ox Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS pulse_ox_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            single_reading_type TEXT,
            reading_timestamp_gmt INTEGER,
            reading_timestamp_local INTEGER,
            spo2_value INTEGER,
            spo2_readings TEXT, -- JSON array of readings
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )''')
        
        # Respiration Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS respiration_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            latest_reading_timestamp_gmt INTEGER,
            latest_reading_value REAL,
            highest_reading_value REAL,
            lowest_reading_value REAL,
            avg_waking_reading_value REAL,
            avg_sleep_reading_value REAL,
            avg_reading_value REAL,
            reading_count INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )''')
        
        # Blood Pressure Summary
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS blood_pressure_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            measurement_timestamp_gmt INTEGER NOT NULL,
            measurement_timestamp_local INTEGER,
            systolic_pressure INTEGER,
            diastolic_pressure INTEGER,
            pulse INTEGER,
            mean_arterial_pressure INTEGER,
            pulse_pressure INTEGER,
            source_type TEXT,
            timestamp_offset INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, measurement_timestamp_gmt)
        )''')
        
        print("✅ Garmin Health API tables created successfully!")

    def create_garmin_activity_tables(self):
        """Create tables matching Garmin Activity API structure"""
        
        # Activity Summary - high-level activity information
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_summary (
            activity_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            activity_name TEXT,
            activity_description TEXT,
            activity_type TEXT,
            activity_type_key TEXT,
            activity_subtype TEXT,
            activity_subtype_key TEXT,
            parent_activity_id TEXT,
            activity_level INTEGER,
            start_time_in_seconds INTEGER,
            start_time_offset_in_seconds INTEGER,
            duration_in_seconds INTEGER,
            elapsed_duration_in_seconds INTEGER,
            moving_duration_in_seconds INTEGER,
            distance_in_meters REAL,
            max_speed_in_meters_per_second REAL,
            avg_speed_in_meters_per_second REAL,
            steps INTEGER,
            floors_climbed REAL,
            min_elevation_in_meters REAL,
            max_elevation_in_meters REAL,
            elevation_gain_in_meters REAL,
            elevation_loss_in_meters REAL,
            avg_vertical_oscillation_in_centimeters REAL,
            avg_ground_contact_time_in_milliseconds REAL,
            avg_stride_length_in_centimeters REAL,
            avg_fractional_cadence REAL,
            max_fractional_cadence REAL,
            training_effect_aerobic REAL,
            training_effect_anaerobic REAL,
            avg_heart_rate INTEGER,
            max_heart_rate INTEGER,
            calories INTEGER,
            bmr_calories INTEGER,
            avg_bike_cadence REAL,
            max_bike_cadence REAL,
            avg_bike_power INTEGER,
            max_bike_power INTEGER,
            total_work_in_joules INTEGER,
            avg_power_position TEXT, -- JSON
            max_power_position TEXT, -- JSON
            avg_left_torque_effectiveness REAL,
            avg_right_torque_effectiveness REAL,
            avg_left_pedal_smoothness REAL,
            avg_right_pedal_smoothness REAL,
            avg_combined_pedal_smoothness REAL,
            vo2_max_value REAL,
            device_id TEXT,
            activity_training_load REAL,
            finish_date INTEGER,
            start_latitude REAL,
            start_longitude REAL,
            has_start_finish_waypoints BOOLEAN,
            is_favorite BOOLEAN,
            is_parent BOOLEAN,
            location_name TEXT,
            lap_count INTEGER,
            end_latitude REAL,
            end_longitude REAL,
            min_activity_lap_duration REAL,
            has_splits BOOLEAN,
            has_heart_rate BOOLEAN,
            has_speed BOOLEAN,
            has_cadence BOOLEAN,
            has_power BOOLEAN,
            has_elevation BOOLEAN,
            workout_id TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Activity Details Summary - detailed metrics and splits
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_details_summary (
            detail_id TEXT PRIMARY KEY,
            activity_id TEXT NOT NULL,
            user_access_token TEXT NOT NULL,
            measurement_count INTEGER,
            metered_metrics TEXT, -- JSON array
            metric_descriptors TEXT, -- JSON array
            activity_detail_metrics TEXT, -- JSON array of detailed metrics
            geo_polyline_segments TEXT, -- JSON array for GPS data
            manual_activity_segments TEXT, -- JSON array
            segment_splits TEXT, -- JSON array of lap/split data
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (activity_id) REFERENCES activity_summary (activity_id)
        )''')
        
        # MoveIQ Activity Summary - automatically detected activities
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS moveiq_activity_summary (
            activity_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            activity_type TEXT,
            activity_type_key TEXT,
            start_time_in_seconds INTEGER,
            start_time_offset_in_seconds INTEGER,
            duration_in_seconds INTEGER,
            distance_in_meters REAL,
            steps INTEGER,
            calories INTEGER,
            activity_level INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Manually Updated Activity Summary - user-created activities
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS manual_activity_summary (
            activity_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            activity_name TEXT,
            activity_type TEXT,
            start_time_in_seconds INTEGER,
            duration_in_seconds INTEGER,
            distance_in_meters REAL,
            calories INTEGER,
            avg_heart_rate INTEGER,
            max_heart_rate INTEGER,
            notes TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        print("✅ Garmin Activity API tables created successfully!")

    def create_user_management_tables(self):
        """Create user and device management tables"""
        
        # User Access Tokens (simulate OAuth tokens)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_access_tokens (
            user_access_token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token_secret TEXT,
            token_expiry INTEGER,
            scope TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Device Information
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            device_name TEXT,
            device_type TEXT,
            unit_id TEXT,
            software_version TEXT,
            max_heart_rate INTEGER,
            rest_heart_rate INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_access_token) REFERENCES user_access_tokens (user_access_token)
        )''')
        
        print("✅ User management tables created successfully!")

    def create_indexes(self):
        """Create indexes for optimal query performance"""
        
        indexes = [
            # Health data indexes
            "CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_daily_summary_user ON daily_summary(user_access_token)",
            "CREATE INDEX IF NOT EXISTS idx_epoch_summary_time ON epoch_summary(summary_start_time_in_seconds)",
            "CREATE INDEX IF NOT EXISTS idx_sleep_summary_date ON sleep_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_hrv_summary_date ON hrv_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_stress_detail_time ON stress_detail_summary(start_time_in_seconds)",
            "CREATE INDEX IF NOT EXISTS idx_pulse_ox_date ON pulse_ox_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_respiration_date ON respiration_summary(calendar_date)",
            
            # Activity data indexes
            "CREATE INDEX IF NOT EXISTS idx_activity_summary_start ON activity_summary(start_time_in_seconds)",
            "CREATE INDEX IF NOT EXISTS idx_activity_summary_user ON activity_summary(user_access_token)",
            "CREATE INDEX IF NOT EXISTS idx_activity_summary_type ON activity_summary(activity_type)",
            "CREATE INDEX IF NOT EXISTS idx_moveiq_start_time ON moveiq_activity_summary(start_time_in_seconds)",
            "CREATE INDEX IF NOT EXISTS idx_manual_activity_start ON manual_activity_summary(start_time_in_seconds)",
            
            # User management indexes
            "CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_access_token)"
        ]
        
        for index_sql in indexes:
            self.cursor.execute(index_sql)
        
        print("✅ Database indexes created successfully!")

    def create_default_user(self):
        """Create default test user with access token"""
        
        try:
            # Create user access token
            user_token = "GARMIN_USER_TOKEN_12345"
            self.cursor.execute('''
            INSERT OR REPLACE INTO user_access_tokens 
            (user_access_token, user_id, token_secret, token_expiry, scope)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_token, 'test_user_001', 'secret_key_12345', 
                  int((datetime.now() + timedelta(days=365)).timestamp()), 
                  'wellness:read,activities:read'))
            
            # Create default device
            device_id = "GARMIN_DEVICE_12345"
            self.cursor.execute('''
            INSERT OR REPLACE INTO devices 
            (device_id, user_access_token, device_name, device_type, unit_id, 
             software_version, max_heart_rate, rest_heart_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (device_id, user_token, 'vívoactive 4', 'fitness_tracker', 
                  '3851981234', '8.20', 190, 65))
            
            print(f"✅ Default user created with token: {user_token}")
            return user_token
            
        except sqlite3.IntegrityError:
            print("ℹ️ Default user already exists")
            self.cursor.execute("SELECT user_access_token FROM user_access_tokens WHERE user_id = 'test_user_001'")
            return self.cursor.fetchone()[0]

    def generate_garmin_sample_data(self, days=30, user_token="GARMIN_USER_TOKEN_12345"):
        """Generate realistic Garmin data for testing"""
        
        print(f"🔄 Generating {days} days of Garmin sample data...")
        
        device_id = "GARMIN_DEVICE_12345"
        
        for day in range(days):
            current_date = datetime.now() - timedelta(days=days-day-1)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Generate Daily Summary
            steps = random.randint(5000, 15000)
            distance = steps * random.uniform(0.65, 0.85)  # meters per step
            active_time = random.randint(1800, 7200)  # 30min to 2hr
            calories = random.randint(1800, 2800)
            resting_hr = random.randint(58, 75)
            stress_level = random.randint(15, 45)
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO daily_summary
            (summary_id, user_access_token, calendar_date, steps, total_distance_in_meters,
             duration_in_seconds, active_time_in_seconds, active_kilocalories, 
             bmr_kilocalories, resting_heart_rate, overall_stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (f"daily_{user_token}_{date_str}", user_token, date_str, steps, 
                  int(distance), 86400, active_time, calories-400, 400, resting_hr, stress_level))
            
            # Generate Sleep Summary
            sleep_start = int((current_date.replace(hour=22, minute=30) + 
                             timedelta(minutes=random.randint(-60, 60))).timestamp())
            sleep_duration = random.randint(6*3600, 9*3600)  # 6-9 hours
            deep_sleep = int(sleep_duration * random.uniform(0.15, 0.25))
            light_sleep = int(sleep_duration * random.uniform(0.45, 0.55))
            rem_sleep = int(sleep_duration * random.uniform(0.20, 0.30))
            awake_time = sleep_duration - deep_sleep - light_sleep - rem_sleep
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO sleep_summary
            (summary_id, user_access_token, calendar_date, sleep_start_timestamp_gmt,
             sleep_end_timestamp_gmt, duration_in_seconds, deep_sleep_duration_in_seconds,
             light_sleep_duration_in_seconds, rem_sleep_duration_in_seconds,
             awake_duration_in_seconds, overall_sleep_score, device_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (f"sleep_{user_token}_{date_str}", user_token, date_str, sleep_start,
                  sleep_start + sleep_duration, sleep_duration, deep_sleep, light_sleep,
                  rem_sleep, awake_time, random.randint(70, 95), device_id))
            
            # Generate HRV Summary
            hrv_avg = random.uniform(25, 55)
            self.cursor.execute('''
            INSERT OR REPLACE INTO hrv_summary
            (summary_id, user_access_token, calendar_date, weekly_avg, last_night_avg,
             last_night_5_min_high, status, feedback_phrase)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (f"hrv_{user_token}_{date_str}", user_token, date_str, hrv_avg,
                  hrv_avg + random.uniform(-5, 5), hrv_avg + random.uniform(5, 15),
                  random.choice(['BALANCED', 'UNBALANCED', 'LOW', 'HIGH']),
                  'Your HRV is within normal range'))
            
            # Generate Epoch Summary (24 records per day - hourly)
            for hour in range(24):
                epoch_start = int(current_date.replace(hour=hour, minute=0).timestamp())
                epoch_steps = max(0, int(steps/24 + random.randint(-100, 200)))
                
                self.cursor.execute('''
                INSERT OR REPLACE INTO epoch_summary
                (summary_id, user_access_token, summary_start_time_in_seconds,
                 summary_end_time_in_seconds, activity_type, duration_in_seconds,
                 steps, distance_in_meters, active_kilocalories)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (f"epoch_{user_token}_{date_str}_{hour:02d}", user_token, epoch_start,
                      epoch_start + 3600, random.choice(['WALKING', 'SEDENTARY', 'LIGHT_ACTIVE']),
                      3600, epoch_steps, epoch_steps * 0.75, random.randint(20, 120)))
            
            # Generate some activities (2-3 per week)
            if random.random() < 0.35:  # ~35% chance of activity per day
                activity_types = [
                    ('RUNNING', 'running'), ('CYCLING', 'cycling'), 
                    ('WALKING', 'casual_walking'), ('SWIMMING', 'pool_swim')
                ]
                activity_type, activity_key = random.choice(activity_types)
                
                activity_start = int(current_date.replace(hour=random.randint(6, 20)).timestamp())
                duration = random.randint(1200, 5400)  # 20min to 1.5hr
                distance = random.randint(2000, 15000)  # 2-15km
                avg_hr = random.randint(140, 180)
                activity_calories = random.randint(200, 800)
                
                activity_id = f"activity_{user_token}_{date_str}_{random.randint(1000, 9999)}"
                
                self.cursor.execute('''
                INSERT INTO activity_summary
                (activity_id, user_access_token, activity_name, activity_type, activity_type_key,
                 start_time_in_seconds, duration_in_seconds, distance_in_meters,
                 avg_heart_rate, max_heart_rate, calories, device_id, steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (activity_id, user_token, f"Morning {activity_type.title()}", 
                      activity_type, activity_key, activity_start, duration, distance,
                      avg_hr, avg_hr + random.randint(10, 30), activity_calories, 
                      device_id, int(distance/0.75) if activity_type == 'RUNNING' else 0))
        
        print("✅ Garmin sample data generation completed!")

    def get_database_stats(self):
        """Print comprehensive database statistics"""
        tables = [
            'daily_summary', 'epoch_summary', 'sleep_summary', 'hrv_summary',
            'stress_detail_summary', 'body_composition_summary', 'pulse_ox_summary',
            'respiration_summary', 'blood_pressure_summary', 'activity_summary',
            'activity_details_summary', 'moveiq_activity_summary', 'manual_activity_summary',
            'user_access_tokens', 'devices'
        ]
        
        print("\n📊 Garmin Database Statistics:")
        print("=" * 60)
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                print(f"{table:<30}: {count:>8} records")
            except Exception as e:
                print(f"{table:<30}: Error - {e}")
        
        # Additional stats
        try:
            self.cursor.execute("SELECT COUNT(DISTINCT calendar_date) FROM daily_summary")
            days_count = self.cursor.fetchone()[0]
            print(f"\n📅 Total days of health data: {days_count}")
            
            self.cursor.execute("SELECT COUNT(*) FROM activity_summary")
            activities_count = self.cursor.fetchone()[0]
            print(f"🏃 Total recorded activities: {activities_count}")
            
        except Exception as e:
            print(f"Error getting additional stats: {e}")

    def commit_and_close(self):
        """Commit changes and close database connection"""
        self.connection.commit()
        self.connection.close()
        print("✅ Database connection closed successfully!")

def main():
    """Main function to create Garmin-compatible health metrics database"""
    print("🏥 Creating Garmin Health & Activity Metrics Database...")
    print("=" * 70)
    
    # Initialize database
    db = GarminHealthMetricsDB()
    
    try:
        # Create all Garmin-compatible tables
        db.create_garmin_health_tables()
        db.create_garmin_activity_tables()
        db.create_user_management_tables()
        
        # Create indexes for performance
        db.create_indexes()
        
        # Create default user and device
        user_token = db.create_default_user()
        
        # Generate sample Garmin data
        generate_sample = input("\n🤔 Generate sample Garmin data for testing? (y/n): ").lower().strip()
        if generate_sample in ['y', 'yes']:
            days = int(input("How many days of sample data? (default: 30): ") or 30)
            db.generate_garmin_sample_data(days=days, user_token=user_token)
        
        # Show database statistics
        db.get_database_stats()
        
        print(f"\n🎉 Garmin-compatible database setup completed!")
        print(f"📁 Database file: {db.db_path}")
        print(f"🔑 Test user token: {user_token}")
        print(f"🔗 Connection string: sqlite:///{db.db_path}")
        print(f"\n💡 This database now mirrors actual Garmin Connect data structure!")
        
    except Exception as e:
        print(f"❌ Error during database setup: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.commit_and_close()

if __name__ == "__main__":
    main()
