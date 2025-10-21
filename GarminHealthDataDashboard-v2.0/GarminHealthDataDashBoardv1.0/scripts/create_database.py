import sqlite3
import os
from datetime import datetime, timedelta
import random

class GarminHealthMetricsDB:
    def __init__(self, db_path='c:/smakrykoDBs/GarminHealth_Hydra.db'):
        """Initialize DB connection"""
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")

    # ------------------------
    # CREATE TABLES
    # ------------------------
    def create_garmin_health_tables(self):
        """Create Garmin Health API tables"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            steps INTEGER,
            total_distance_in_meters INTEGER,
            active_time_in_seconds INTEGER,
            active_kilocalories INTEGER,
            bmr_kilocalories INTEGER,
            resting_heart_rate INTEGER,
            overall_stress_level INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )""")

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            sleep_start_timestamp_gmt INTEGER,
            sleep_end_timestamp_gmt INTEGER,
            duration_in_seconds INTEGER,
            deep_sleep_duration_in_seconds INTEGER,
            light_sleep_duration_in_seconds INTEGER,
            rem_sleep_duration_in_seconds INTEGER,
            awake_duration_in_seconds INTEGER,
            overall_sleep_score INTEGER,
            device_id TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )""")

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS hrv_summary (
            summary_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            calendar_date DATE NOT NULL,
            weekly_avg REAL,
            last_night_avg REAL,
            last_night_5_min_high REAL,
            status TEXT,
            feedback_phrase TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_access_token, calendar_date)
        )""")
        print("✅ Health tables created.")

    def create_garmin_activity_tables(self):
        """Create Garmin Activity API tables"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_summary (
            activity_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            activity_name TEXT,
            activity_type TEXT,
            start_time_in_seconds INTEGER,
            duration_in_seconds INTEGER,
            distance_in_meters REAL,
            avg_heart_rate INTEGER,
            max_heart_rate INTEGER,
            calories INTEGER,
            device_id TEXT,
            steps INTEGER,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        print("✅ Activity tables created.")

    def create_user_management_tables(self):
        """User and device tables"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_access_tokens (
            user_access_token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            user_access_token TEXT NOT NULL,
            device_name TEXT,
            device_type TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        print("✅ User/device tables created.")

    def create_indexes(self):
        """Indexes for faster read access"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_summary(calendar_date)",
            "CREATE INDEX IF NOT EXISTS idx_activity_date ON activity_summary(start_time_in_seconds)"
        ]
        for idx in indexes:
            self.cursor.execute(idx)
        print("✅ Indexes created.")

    # ------------------------
    # DATA CREATION
    # ------------------------
    def create_default_user(self):
        """Create a default user + device"""
        user_token = "GARMIN_USER_TOKEN_12345"
        self.cursor.execute("""
        INSERT OR REPLACE INTO user_access_tokens (user_access_token, user_id)
        VALUES (?, ?)""", (user_token, 'test_user_001'))
        self.cursor.execute("""
        INSERT OR REPLACE INTO devices (device_id, user_access_token, device_name, device_type)
        VALUES (?, ?, ?, ?)""", ('GARMIN_DEVICE_12345', user_token, 'vívoactive 4', 'fitness_tracker'))
        print("✅ Default user/device created.")
        return user_token

    def generate_garmin_sample_data(self, days=30, user_token="GARMIN_USER_TOKEN_12345"):
        """Generate simulated Garmin-like data"""
        device_id = "GARMIN_DEVICE_12345"
        for day in range(days):
            date_val = (datetime.now() - timedelta(days=days - day - 1)).strftime('%Y-%m-%d')
            # Daily Summary
            steps = random.randint(5000, 15000)
            dist = int(steps * 0.75)
            self.cursor.execute("""
            INSERT OR REPLACE INTO daily_summary 
            (summary_id, user_access_token, calendar_date, steps, total_distance_in_meters,
             active_time_in_seconds, active_kilocalories, bmr_kilocalories, 
             resting_heart_rate, overall_stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"daily_{user_token}_{date_val}", user_token, date_val, steps, dist,
             random.randint(1800, 7200), random.randint(1400, 2400), 400,
             random.randint(55, 75), random.randint(15, 45)))
            # Sleep Summary
            duration = random.randint(6*3600, 9*3600)
            self.cursor.execute("""
            INSERT OR REPLACE INTO sleep_summary 
            (summary_id, user_access_token, calendar_date, duration_in_seconds,
             deep_sleep_duration_in_seconds, light_sleep_duration_in_seconds,
             rem_sleep_duration_in_seconds, awake_duration_in_seconds, 
             overall_sleep_score, device_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"sleep_{user_token}_{date_val}", user_token, date_val, duration,
             int(duration*0.2), int(duration*0.5), int(duration*0.25),
             duration - int(duration*0.95), random.randint(65, 95), device_id))
            # HRV Summary
            hrv = random.uniform(25, 55)
            self.cursor.execute("""
            INSERT OR REPLACE INTO hrv_summary
            (summary_id, user_access_token, calendar_date, weekly_avg, 
             last_night_avg, last_night_5_min_high, status, feedback_phrase)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"hrv_{user_token}_{date_val}", user_token, date_val, hrv,
             hrv + random.uniform(-5, 5), hrv + random.uniform(5, 15),
             "BALANCED", "Your HRV is within normal range"))
            # Activity Summary sometimes
            if random.random() < 0.3:
                act_id = f"activity_{user_token}_{date_val}_{random.randint(1000,9999)}"
                self.cursor.execute("""
                INSERT OR REPLACE INTO activity_summary
                (activity_id, user_access_token, activity_name, activity_type,
                 start_time_in_seconds, duration_in_seconds, distance_in_meters,
                 avg_heart_rate, max_heart_rate, calories, device_id, steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (act_id, user_token, "Morning Run", "RUNNING",
                 int(datetime.now().timestamp()), random.randint(1200, 5400),
                 random.randint(2000, 10000), random.randint(140, 175),
                 random.randint(160, 190), random.randint(200, 800),
                 device_id, steps))
        print(f"✅ Generated {days} days of Garmin sample data.")

    # ------------------------
    # IMPORT FROM LOCAL GARMIN DB
    # ------------------------
    def import_data_from_local_garmin_dbs(self, paths):
        """Import from existing Garmin SQLite DB(s) stored locally"""
        for file_path in paths:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            print(f"📥 Importing from Garmin DB: {file_path}")
            local_conn = sqlite3.connect(file_path)
            try:
                # Example: adapt table/column names based on your local DB schema
                for table in ['daily_summary', 'sleep_summary', 'hrv_summary', 'activity_summary']:
                    try:
                        df = self._read_table(local_conn, table)
                        if not df.empty:
                            df.to_sql(table, self.connection, if_exists='append', index=False)
                            print(f"   ➕ Imported {len(df)} rows into {table}")
                    except Exception as e:
                        print(f"   ⚠️ Skipping {table} - {e}")
                self.connection.commit()
            finally:
                local_conn.close()
        print("✅ Local Garmin DB import complete.")

    def _read_table(self, conn, table):
        """Read table from SQLite DB into DataFrame"""
        import pandas as pd
        try:
            return pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()

    # ------------------------
    # STATS & CLOSE
    # ------------------------
    def get_database_stats(self):
        for t in ['daily_summary', 'sleep_summary', 'hrv_summary', 'activity_summary']:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {t}")
                print(f"{t:<20}: {self.cursor.fetchone()[0]} rows")
            except:
                print(f"{t:<20}: missing")

    def commit_and_close(self):
        self.connection.commit()
        self.connection.close()


# ------------------------
# MAIN SCRIPT FLOW
# ------------------------
def main():
    print("🏥 Setting up Garmin Health & Activity Database...")
    db = GarminHealthMetricsDB()
    db.create_garmin_health_tables()
    db.create_garmin_activity_tables()
    db.create_user_management_tables()
    db.create_indexes()
    user_token = db.create_default_user()

    choice = input(
        "\n📊 Choose how to populate data:\n"
        "1 - Generate sample Garmin data\n"
        "2 - Import from local Garmin SQLite DB(s)\n"
        "Select 1 or 2: ").strip()

    if choice == "1":
        days = int(input("How many days of sample data? (30 default): ") or 30)
        db.generate_garmin_sample_data(days=days, user_token=user_token)
    elif choice == "2":
        paths_str = input("Enter full path(s) to Garmin DB files, separated by commas: ")
        paths = [p.strip() for p in paths_str.split(",") if p.strip()]
        db.import_data_from_local_garmin_dbs(paths)

    db.get_database_stats()
    db.commit_and_close()
    print(f"✅ Setup complete. Database stored at: {db.db_path}")

if __name__ == "__main__":
    main()
