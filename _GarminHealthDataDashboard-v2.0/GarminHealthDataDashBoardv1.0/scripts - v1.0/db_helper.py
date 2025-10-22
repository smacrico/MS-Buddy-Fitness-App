import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class HealthDataManager:
    def __init__(self, db_path='C:/smakrykoDBs/health_metrics.db'):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_metric_data(self, metric_name: str, hours: int = 24, user_id: int = 1) -> pd.Series:
        """Get time series data for a specific metric"""
        
        with self.get_connection() as conn:
            query = """
            SELECT timestamp, metric_value 
            FROM health_data 
            WHERE metric_name = ? 
            AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp
            """.format(hours)
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=[metric_name],
                parse_dates=['timestamp'], 
                index_col='timestamp'
            )
            
            return df['metric_value'] if not df.empty else pd.Series()
    
    def get_user_baselines(self, user_id: int = 1) -> Dict:
        """Get personal baselines for a user"""
        
        with self.get_connection() as conn:
            query = """
            SELECT metric_name, target_value, min_threshold, max_threshold
            FROM personal_baselines
            WHERE user_id = ?
            """
            
            cursor = conn.cursor()
            cursor.execute(query, [user_id])
            
            baselines = {}
            for row in cursor.fetchall():
                metric_name, target, min_val, max_val = row
                baselines[metric_name] = {
                    'target': target,
                    'min': min_val,
                    'max': max_val
                }
            
            return baselines
    
    def insert_health_data(self, metric_name: str, value: float, 
                          timestamp: Optional[datetime] = None, device_id: str = 'manual'):
        """Insert a single health data point"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO health_data (timestamp, metric_name, metric_value, device_id)
            VALUES (?, ?, ?, ?)
            """, (timestamp, metric_name, value, device_id))
            
            conn.commit()
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics in the database"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT metric_name FROM health_data ORDER BY metric_name")
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_latest_values(self, user_id: int = 1) -> Dict:
        """Get the latest value for each metric"""
        
        with self.get_connection() as conn:
            query = """
            SELECT metric_name, metric_value, timestamp
            FROM health_data hd1
            WHERE timestamp = (
                SELECT MAX(timestamp) 
                FROM health_data hd2 
                WHERE hd2.metric_name = hd1.metric_name
            )
            ORDER BY metric_name
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            
            latest_values = {}
            for metric_name, value, timestamp in cursor.fetchall():
                latest_values[metric_name] = {
                    'value': value,
                    'timestamp': timestamp
                }
            
            return latest_values
