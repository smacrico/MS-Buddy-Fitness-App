import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_demo_data():
    # Get script directory
    script_dir = Path(__file__).parent
    output_file = script_dir / 'demo.csv'
    
    # Generate 30 days of data with 48 readings per day (every 30 minutes)
    start_date = datetime(2023, 12, 1)
    dates = [start_date + timedelta(minutes=30*i) for i in range(30*48)]
    
    data = []
    for date in dates:
        hour = date.hour
        is_weekend = date.weekday() >= 5
        
        # Sleep period (22:00-06:00)
        if 22 <= hour or hour < 6:
            hr = np.random.normal(60, 5)
            # Generate RR intervals as regular Python list of floats
            rr = [float(x) for x in np.random.normal(1000, 50, 5)]  # Higher RR intervals
            if 0 <= hour < 2:
                stage = 'deep'
            elif 2 <= hour < 4:
                stage = 'rem'
            else:
                stage = 'light'
            activity = np.random.normal(5, 2)
            time_in_bed = 480
            total_sleep = 450
        
        # Morning exercise (07:00-08:30) on weekdays
        elif not is_weekend and 7 <= hour < 9:
            hr = np.random.normal(140, 10)
            rr = [float(x) for x in np.random.normal(430, 30, 5)]
            stage = 'awake'
            activity = np.random.normal(85, 5)
            time_in_bed = 0
            total_sleep = 0
            
        # Regular daytime
        else:
            hr = np.random.normal(75, 8)
            rr = [float(x) for x in np.random.normal(800, 40, 5)]
            stage = 'awake'
            activity = np.random.normal(20, 10)
            time_in_bed = 0
            total_sleep = 0
            
        # Add some random variation for realism
        hr += np.random.normal(0, 2)
        activity = max(0, min(100, activity))
        
        # SPO2 varies less but drops slightly during sleep
        spo2 = 98 if hour >= 6 and hour < 22 else 96
        spo2 += np.random.normal(0, 0.5)
        spo2 = max(94, min(99, spo2))
        
        data.append({
            'timestamp': date,
            'heart_rate': round(float(hr), 1),
            'rr_intervals': str(rr),  # Convert list of floats to string
            'sleep_stage': stage,
            'spo2': round(float(spo2), 1),
            'activity_level': round(float(activity), 1),
            'time_in_bed_min': time_in_bed,
            'total_sleep_min': total_sleep
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} records of demo data")
    print(f"File saved at: {output_file}")

if __name__ == "__main__":
    generate_demo_data()
