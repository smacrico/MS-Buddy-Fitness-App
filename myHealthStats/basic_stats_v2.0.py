import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def connect_db(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def query_to_df(conn, query: str) -> pd.DataFrame:
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Query failed: {e}")
        return pd.DataFrame()

def plot_bar(df, x_col, y_col, title, xlabel, ylabel, rotate_xticks=45):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks, ha='right')
    plt.tight_layout()
    plt.show()

def plot_line(df, x_col, y_col, title, xlabel, ylabel, marker='o', rotate_xticks=45):
    plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(rotation=rotate_xticks, ha='right')
    plt.tight_layout()
    plt.show()

def get_avg_heart_rate_last_40_days(conn):
    query = """
    SELECT a.sport, AVG(r.hr) AS avg_heart_rate
    FROM activities a
    JOIN activity_records r ON a.activity_id = r.activity_id
    WHERE a.start_time >= DATE('now', '-40 days')
    GROUP BY a.sport;
    """
    return query_to_df(conn, query)

def get_avg_sleep_duration_last_40_days(conn):
    query = """
    SELECT DATE(day) AS sleep_date, AVG(total_sleep / 3600.0) AS avg_sleep_hours
    FROM sleep
    WHERE day >= DATE('now', '-40 days')
    GROUP BY sleep_date;
    """
    return query_to_df(conn, query)

def get_daily_steps_last_40_days(conn):
    query = """
    SELECT day, steps
    FROM days_summary
    WHERE day >= DATE('now', '-40 days');
    """
    df = query_to_df(conn, query)
    if not df.empty:
        df['day'] = pd.to_datetime(df['day'])
    return df

def get_resting_heart_rate(conn):
    """Extract resting heart rate per day."""
    query = """
    SELECT DATE(day) AS date, resting_heart_rate
    FROM resting_hr
    WHERE resting_heart_rate IS NOT NULL
    ORDER BY date;
    """
    return query_to_df(conn, query)

def get_resting_heart_rate_last_40_days(conn):
    """Extract resting heart rate per day for the last 40 days."""
    query = """
    SELECT DATE(day) AS date, resting_heart_rate
    FROM resting_hr
    WHERE resting_heart_rate IS NOT NULL
      AND day >= DATE('now', '-40 days')
    ORDER BY date;
    """
    return query_to_df(conn, query)



def plot_resting_hr(df):
    if df.empty:
        print("No resting heart rate data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['resting_heart_rate'], marker='o', color='royalblue')
    plt.title('Resting Heart Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Resting Heart Rate (bpm)')
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    activities_db = r'c:/users/jheel/jheelHealthData/DBs/garmin_activities.db'
    sleep_db = r'c:/users/jheel/jheelHealthData/DBs/garmin.db'
    summary_db = r'c:/users/jheel/jheelHealthData/DBs/garmin_summary.db'
    garmin_db = r'c:/users/jheel/jheelHealthData/DBs/garmin.db'

    conn_activities = connect_db(activities_db)
    if conn_activities:
        hr_df = get_avg_heart_rate_last_40_days(conn_activities)
        print("\nAverage Heart Rate per Sport - Last 40 Days:")
        print(hr_df)
        if not hr_df.empty:
            plot_bar(hr_df, 'sport', 'avg_heart_rate', 'Avg Heart Rate per Sport (Last 40 Days)', 'Sport', 'Avg Heart Rate')
        conn_activities.close()

    conn_sleep = connect_db(sleep_db)
    if conn_sleep:
        sleep_df = get_avg_sleep_duration_last_40_days(conn_sleep)
        print("\nAverage Sleep Duration per Day - Last 40 Days:")
        print(sleep_df.head())
        if not sleep_df.empty:
            plot_line(sleep_df, 'sleep_date', 'avg_sleep_hours', 'Avg Sleep Duration per Day (Last 40 Days)', 'Date', 'Sleep Hours')
        conn_sleep.close()

    conn_summary = connect_db(summary_db)
    if conn_summary:
        steps_df = get_daily_steps_last_40_days(conn_summary)
        print("\nDaily Steps - Last 40 Days:")
        print(steps_df.head())
        if not steps_df.empty:
            plot_line(steps_df, 'day', 'steps', 'Daily Steps (Last 40 Days)', 'Date', 'Steps', marker='.')
        conn_summary.close()

    conn_health = connect_db(garmin_db)  # or appropriate DB path with resting HR data
    if conn_health:
        rhr_df = get_resting_heart_rate_last_40_days(conn_health)
        print("\nResting Heart Rate Over Last 40 Days:")
        print(rhr_df.head())
        plot_resting_hr(rhr_df)
        conn_health.close()

if __name__ == "__main__":
    main()
