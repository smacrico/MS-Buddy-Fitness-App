import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def connect_db(db_path: str) -> sqlite3.Connection:
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def query_to_df(conn, query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
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


def get_avg_heart_rate(conn):
    """Calculate average heart rate per sport and extended stats."""
    query = """
    SELECT a.sport, 
           AVG(r.hr) AS avg_heart_rate,
           MIN(r.hr) AS min_hr,
           MAX(r.hr) AS max_hr,
           STDDEV_POP(r.hr) AS std_hr
    FROM activities a
    JOIN activity_records r ON a.activity_id = r.activity_id
    GROUP BY a.sport;
    """
    # Note: SQLite does not support STDDEV_POP by default; fallback:
    # We'll calculate std dev using pandas after querying avg HR per sport
    query_simple = """
    SELECT a.sport, r.hr
    FROM activities a
    JOIN activity_records r ON a.activity_id = r.activity_id;
    """
    df = query_to_df(conn, query_simple)
    if df.empty:
        return df
    
    stats_df = df.groupby('sport')['hr'].agg(['mean', 'min', 'max', 'std']).reset_index()
    stats_df.rename(columns={'mean': 'avg_heart_rate', 'min': 'min_hr', 'max': 'max_hr', 'std': 'std_hr'}, inplace=True)
    return stats_df


def get_avg_sleep_duration(conn):
    """Calculate average sleep duration per day in hours."""
    query = """
    SELECT DATE(day) AS sleep_date, AVG(total_sleep / 3600.0) AS avg_sleep_hours
    FROM sleep
    GROUP BY sleep_date;
    """
    return query_to_df(conn, query)


def get_daily_steps(conn):
    """Extract daily steps from summary."""
    query = """
    SELECT day, steps
    FROM days_summary;
    """
    df = query_to_df(conn, query)
    if not df.empty:
        df['day'] = pd.to_datetime(df['day'])
    return df


def correlate_sleep_steps(sleep_df, steps_df):
    """Merge sleep and steps data on date and calculate correlation."""
    if sleep_df.empty or steps_df.empty:
        print("One of the dataframes is empty, cannot correlate.")
        return
    
    merged = pd.merge(sleep_df, steps_df, left_on='sleep_date', right_on='day')
    corr = merged['avg_sleep_hours'].corr(merged['steps'])
    print(f"Correlation between average sleep hours and steps: {corr:.3f}")

    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='steps', y='avg_sleep_hours', data=merged)
    plt.title('Correlation Between Daily Steps and Sleep Duration')
    plt.xlabel('Daily Steps')
    plt.ylabel('Average Sleep Hours')
    plt.tight_layout()
    plt.show()


def main():
    # Paths to your databases (update as necessary)
    activities_db = r'c:/users/jheel/jheelHealthData/DBs/garmin_activities.db'
    sleep_db = r'c:/users/jheel/jheelHealthData/DBs/garmin.db'
    summary_db = r'c:/users/jheel/jheelHealthData/DBs/garmin_summary.db'

    # Connect and analyze Activities heart rate
    conn_activities = connect_db(activities_db)
    if conn_activities:
        hr_stats = get_avg_heart_rate(conn_activities)
        print("\nHeart Rate Stats per Sport:")
        print(hr_stats)
        if not hr_stats.empty:
            plot_bar(hr_stats, 'sport', 'avg_heart_rate', 'Average Heart Rate per Sport', 'Sport', 'Avg Heart Rate')
        conn_activities.close()

    # Connect and analyze Sleep data
    conn_sleep = connect_db(sleep_db)
    if conn_sleep:
        sleep_df = get_avg_sleep_duration(conn_sleep)
        print("\nAverage Sleep Duration per Day:")
        print(sleep_df.head())
        if not sleep_df.empty:
            plot_line(sleep_df, 'sleep_date', 'avg_sleep_hours', 'Average Sleep Duration per Day', 'Date', 'Sleep Hours')
        conn_sleep.close()

    # Connect and analyze Daily Steps
    conn_summary = connect_db(summary_db)
    if conn_summary:
        steps_df = get_daily_steps(conn_summary)
        print("\nDaily Steps:")
        print(steps_df.head())
        if not steps_df.empty:
            plot_line(steps_df, 'day', 'steps', 'Daily Steps', 'Date', 'Steps', marker='.')
        conn_summary.close()

    # Correlate Sleep and Steps if data available
    if 'sleep_df' in locals() and 'steps_df' in locals() and not sleep_df.empty and not steps_df.empty:
        correlate_sleep_steps(sleep_df, steps_df)


if __name__ == "__main__":
    main()
