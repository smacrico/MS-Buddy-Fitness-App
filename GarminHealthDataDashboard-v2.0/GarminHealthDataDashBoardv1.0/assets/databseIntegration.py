def load_data_from_db(self, metric):
    """Load data from database"""
    import sqlite3
    
    conn = sqlite3.connect('data/health_metrics.db')
    query = f"""
    SELECT timestamp, {metric} 
    FROM health_data 
    WHERE timestamp >= datetime('now', '-{self.get_hours_for_timerange()} hours')
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['timestamp'], index_col='timestamp')
    conn.close()
    
    return df[metric]
