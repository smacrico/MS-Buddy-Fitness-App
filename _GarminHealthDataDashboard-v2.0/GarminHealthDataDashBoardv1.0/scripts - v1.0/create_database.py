import sqlite3
import os
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

class HealthMetricsDB:
    def __init__(self, db_path='C:/smakrykoDBs/health_metrics.db'):
        """Initialize database connection and create directory if needed"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
        # Enable foreign keys
        self.cursor.execute("PRAGMA foreign_keys = ON")
        
    def create_tables(self):
        """Create all necessary tables for health metrics storage"""
        
        # Main health data table - normalized structure
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            device_id TEXT,
            sync_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, metric_name)
        )''')
        
        # User profiles and personal baselines
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            height_cm REAL,
            weight_kg REAL,
            activity_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Personal baselines for each metric
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS personal_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            metric_name TEXT NOT NULL,
            target_value REAL NOT NULL,
            min_threshold REAL NOT NULL,
            max_threshold REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id),
            UNIQUE(user_id, metric_name)
        )''')
        
        # Activity sessions (for workout/exercise data)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL,
            activity_type TEXT NOT NULL,
            duration_minutes INTEGER,
            distance_km REAL,
            calories_burned INTEGER,
            avg_heart_rate INTEGER,
            max_heart_rate INTEGER,
            avg_pace_min_km REAL,
            elevation_gain_m REAL,
            device_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
        )''')
        
        # Sleep sessions (detailed sleep tracking)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sleep_sessions (
            sleep_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sleep_date DATE NOT NULL,
            bedtime DATETIME,
            sleep_start DATETIME,
            sleep_end DATETIME,
            wake_time DATETIME,
            total_sleep_minutes INTEGER,
            deep_sleep_minutes INTEGER,
            light_sleep_minutes INTEGER,
            rem_sleep_minutes INTEGER,
            awake_minutes INTEGER,
            sleep_efficiency_percent REAL,
            sleep_score INTEGER,
            device_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id),
            UNIQUE(user_id, sleep_date)
        )''')
        
        # Alert/notification logs
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            metric_name TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            metric_value REAL NOT NULL,
            threshold_value REAL,
            alert_message TEXT NOT NULL,
            acknowledged BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at DATETIME,
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
        )''')
        
        # Metric definitions and metadata
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS metric_definitions (
            metric_name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            unit TEXT,
            category TEXT,
            data_type TEXT,
            normal_range_min REAL,
            normal_range_max REAL,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        print("✅ All tables created successfully!")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_health_data_timestamp ON health_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_health_data_metric ON health_data(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_health_data_timestamp_metric ON health_data(timestamp, metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_activity_sessions_start_time ON activity_sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_activity_sessions_user ON activity_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sleep_sessions_date ON sleep_sessions(sleep_date)",
            "CREATE INDEX IF NOT EXISTS idx_sleep_sessions_user ON sleep_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_health_alerts_created ON health_alerts(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_health_alerts_user ON health_alerts(user_id)"
        ]
        
        for index_sql in indexes:
            self.cursor.execute(index_sql)
        
        print("✅ Database indexes created successfully!")
    
    def insert_metric_definitions(self):
        """Insert standard Garmin health metric definitions"""
        
        metrics = [
            ('heart_rate', 'Heart Rate', 'bpm', 'cardiovascular', 'integer', 60, 100, 'Resting heart rate measurement'),
            ('steps', 'Daily Steps', 'steps', 'activity', 'integer', 8000, 12000, 'Daily step count'),
            ('sleep_efficiency', 'Sleep Efficiency', '%', 'sleep', 'float', 80, 95, 'Percentage of time spent sleeping while in bed'),
            ('hrv_score', 'HRV Score', 'ms', 'recovery', 'float', 20, 60, 'Heart rate variability score'),
            ('stress_level', 'Stress Level', 'score', 'wellness', 'integer', 0, 40, 'Stress level on 0-100 scale'),
            ('spo2', 'Blood Oxygen', '%', 'respiratory', 'float', 95, 100, 'Blood oxygen saturation percentage'),
            ('calories', 'Calories Burned', 'kcal', 'activity', 'integer', 1500, 3000, 'Total daily calories burned'),
            ('active_minutes', 'Active Minutes', 'minutes', 'activity', 'integer', 30, 90, 'Minutes of moderate to vigorous activity'),
            ('body_battery', 'Body Battery', 'score', 'energy', 'integer', 50, 100, 'Energy reserves on 0-100 scale'),
            ('respiration_rate', 'Respiration Rate', 'breaths/min', 'respiratory', 'integer', 12, 20, 'Breathing rate per minute'),
            ('vo2_max', 'VO2 Max', 'ml/kg/min', 'fitness', 'float', 25, 60, 'Maximum oxygen uptake'),
            ('body_weight', 'Body Weight', 'kg', 'biometric', 'float', 40, 150, 'Body weight in kilograms'),
            ('body_fat_percent', 'Body Fat %', '%', 'biometric', 'float', 5, 40, 'Body fat percentage'),
            ('hydration_level', 'Hydration', 'score', 'wellness', 'integer', 0, 100, 'Hydration level score')
        ]
        
        self.cursor.executemany('''
        INSERT OR REPLACE INTO metric_definitions 
        (metric_name, display_name, unit, category, data_type, normal_range_min, normal_range_max, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', metrics)
        
        print("✅ Metric definitions inserted successfully!")
    
    def create_default_user(self):
        """Create a default user profile for testing"""
        
        try:
            self.cursor.execute('''
            INSERT INTO user_profiles (username, age, gender, height_cm, weight_kg, activity_level)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', ('default_user', 30, 'unspecified', 170.0, 70.0, 'moderate'))
            
            user_id = self.cursor.lastrowid
            
            # Insert default baselines for the user
            default_baselines = [
                ('heart_rate', 65, 50, 80),
                ('steps', 10000, 8000, 12000),
                ('sleep_efficiency', 85, 75, 95),
                ('hrv_score', 40, 25, 60),
                ('stress_level', 25, 10, 40),
                ('spo2', 98, 95, 100),
                ('calories', 2200, 1500, 3000),
                ('active_minutes', 60, 30, 90),
                ('body_battery', 75, 50, 100),
                ('respiration_rate', 16, 12, 20)
            ]
            
            for metric, target, min_val, max_val in default_baselines:
                self.cursor.execute('''
                INSERT INTO personal_baselines (user_id, metric_name, target_value, min_threshold, max_threshold)
                VALUES (?, ?, ?, ?, ?)
                ''', (user_id, metric, target, min_val, max_val))
            
            print(f"✅ Default user created with ID: {user_id}")
            return user_id
            
        except sqlite3.IntegrityError:
            print("ℹ️ Default user already exists")
            self.cursor.execute("SELECT user_id FROM user_profiles WHERE username = 'default_user'")
            return self.cursor.fetchone()[0]
    
    def generate_sample_data(self, days=30, user_id=1):
        """Generate sample health data for testing"""
        
        print(f"🔄 Generating {days} days of sample data...")
        
        # Get metric definitions
        self.cursor.execute("SELECT metric_name, normal_range_min, normal_range_max FROM metric_definitions")
        metrics = self.cursor.fetchall()
        
        # Generate data for each day
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate hourly data points (24 per day)
            for hour in range(24):
                timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                for metric_name, min_val, max_val in metrics:
                    # Generate realistic values within normal ranges
                    if metric_name == 'steps':
                        # Steps accumulate throughout the day
                        base_steps = hour * 400 + random.randint(0, 200)
                        value = min(base_steps, 15000)
                    elif metric_name == 'body_battery':
                        # Body battery pattern: high in morning, depletes during day
                        base_battery = max(20, 100 - (hour * 3) + random.randint(-10, 10))
                        value = min(100, max(0, base_battery))
                    elif metric_name == 'sleep_efficiency' and 6 <= hour <= 22:
                        # Only generate sleep efficiency during typical sleep hours
                        continue
                    else:
                        # Normal distribution around the middle of normal range
                        mid_point = (min_val + max_val) / 2
                        std_dev = (max_val - min_val) / 6  # 99.7% within range
                        value = max(min_val, min(max_val, np.random.normal(mid_point, std_dev)))
                    
                    # Insert the data point
                    try:
                        self.cursor.execute('''
                        INSERT OR REPLACE INTO health_data (timestamp, metric_name, metric_value, device_id)
                        VALUES (?, ?, ?, ?)
                        ''', (timestamp, metric_name, round(value, 2), 'simulator'))
                    except Exception as e:
                        print(f"Error inserting {metric_name} at {timestamp}: {e}")
        
        # Generate some sample sleep sessions
        for day in range(days):
            sleep_date = (start_date + timedelta(days=day)).date()
            bedtime = datetime.combine(sleep_date, datetime.min.time().replace(hour=22, minute=30))
            sleep_start = bedtime + timedelta(minutes=random.randint(10, 60))
            sleep_end = sleep_start + timedelta(hours=random.randint(6, 9), minutes=random.randint(0, 59))
            wake_time = sleep_end + timedelta(minutes=random.randint(5, 30))
            
            total_sleep = int((sleep_end - sleep_start).total_seconds() / 60)
            deep_sleep = int(total_sleep * random.uniform(0.15, 0.25))
            rem_sleep = int(total_sleep * random.uniform(0.20, 0.30))
            light_sleep = total_sleep - deep_sleep - rem_sleep
            awake_minutes = int((wake_time - bedtime).total_seconds() / 60) - total_sleep
            
            efficiency = (total_sleep / ((wake_time - bedtime).total_seconds() / 60)) * 100
            
            try:
                self.cursor.execute('''
                INSERT OR REPLACE INTO sleep_sessions 
                (user_id, sleep_date, bedtime, sleep_start, sleep_end, wake_time, 
                 total_sleep_minutes, deep_sleep_minutes, light_sleep_minutes, 
                 rem_sleep_minutes, awake_minutes, sleep_efficiency_percent, 
                 sleep_score, device_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, sleep_date, bedtime, sleep_start, sleep_end, wake_time,
                      total_sleep, deep_sleep, light_sleep, rem_sleep, awake_minutes,
                      round(efficiency, 1), random.randint(60, 95), 'simulator'))
            except Exception as e:
                print(f"Error inserting sleep data for {sleep_date}: {e}")
        
        print("✅ Sample data generation completed!")
    
    def commit_and_close(self):
        """Commit changes and close database connection"""
        self.connection.commit()
        self.connection.close()
        print("✅ Database connection closed successfully!")
    
    def get_database_stats(self):
        """Print database statistics"""
        tables = ['health_data', 'user_profiles', 'personal_baselines', 
                 'activity_sessions', 'sleep_sessions', 'health_alerts', 'metric_definitions']
        
        print("\n📊 Database Statistics:")
        print("-" * 50)
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                print(f"{table:<20}: {count:>8} records")
            except Exception as e:
                print(f"{table:<20}: Error - {e}")

def main():
    """Main function to create and populate the database"""
    print("🏥 Creating Garmin Health Metrics Database...")
    print("=" * 60)
    
    # Initialize database
    db = HealthMetricsDB()
    
    try:
        # Create all tables
        db.create_tables()
        
        # Create indexes for performance
        db.create_indexes()
        
        # Insert metric definitions
        db.insert_metric_definitions()
        
        # Create default user
        user_id = db.create_default_user()
        
        # Generate sample data
        generate_sample = input("\n🤔 Generate sample data for testing? (y/n): ").lower().strip()
        if generate_sample in ['y', 'yes']:
            days = int(input("How many days of sample data? (default: 30): ") or 30)
            db.generate_sample_data(days=days, user_id=user_id)
        
        # Show database statistics
        db.get_database_stats()
        
        print(f"\n🎉 Database setup completed successfully!")
        print(f"📁 Database file: {db.db_path}")
        print(f"🔗 Connection string: sqlite:///{db.db_path}")
        
    except Exception as e:
        print(f"❌ Error during database setup: {e}")
    
    finally:
        db.commit_and_close()

if __name__ == "__main__":
    main()
