import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Format dataset with your exact renaming pattern
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance',
    'TRIMP': 'TRIMP',
    'resting_hr': 'rest_hr',
    'Time': 'time',  # Critical for TRIMP calculation
    'VO2Max': 'vo2max'
})

# Ensure date is datetime upfront
dataset = dataset.copy()
dataset['date'] = pd.to_datetime(dataset['date'])

def visualize_training_load(dataset):
    """
    Power BI compatible visualization for training load
    dataset: Power BI DataFrame input (after column renaming)
    """
    try:
        df = dataset.copy()
        
        # Calculate TRIMP if not already present
        if 'TRIMP' not in df.columns and 'time' in df.columns and 'heart_rate' in df.columns:
            rest_hr = 60
            max_hr = 190
            df['duration_min'] = df['time'] / 60
            df['hr_ratio'] = (df['heart_rate'] - rest_hr) / (max_hr - rest_hr)
            df['TRIMP'] = df['duration_min'] * df['hr_ratio']
        elif 'TRIMP' not in df.columns:
            # Fallback if time/heart_rate missing
            df['TRIMP'] = 50  # default value
        
        # Calculate weekly metrics (safe handling)
        df['week'] = df['date'].dt.isocalendar().week
        weekly_trimp = df.groupby('week')['TRIMP'].sum().reset_index(name='TRIMP')
        
        if len(weekly_trimp) > 0:
            weekly_trimp['acute_load'] = weekly_trimp['TRIMP'].rolling(window=1, min_periods=1).mean()
            weekly_trimp['chronic_load'] = weekly_trimp['TRIMP'].rolling(window=4, min_periods=1).mean()
            weekly_trimp['acwr'] = weekly_trimp['acute_load'] / (weekly_trimp['chronic_load'] + 1e-8)

        plt.figure(figsize=(14, 6))
        
        # Plot training load metrics
        plt.subplot(1, 2, 1)
        plt.plot(df['date'], df['TRIMP'], 'o-', label='TRIMP per Session', markersize=6, linewidth=2)
        plt.title('TRIMP per Session')
        plt.xlabel('Date')
        plt.ylabel('TRIMP Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Weekly analysis
        plt.subplot(1, 2, 2)
        plt.plot(weekly_trimp['week'], weekly_trimp['TRIMP'], 'o-', label='Weekly TRIMP', markersize=6)
        plt.plot(weekly_trimp['week'], weekly_trimp['acute_load'], '--', label='Acute Load (1w)', linewidth=2)
        plt.plot(weekly_trimp['week'], weekly_trimp['chronic_load'], '--', label='Chronic Load (4w)', linewidth=2)
        plt.plot(weekly_trimp['week'], weekly_trimp['acwr'], '.-', label='ACWR', markersize=8)
        
        # Add threshold lines
        plt.axhline(1.3, color='red', linestyle=':', label='ACWR High Risk (1.3)', linewidth=2)
        plt.axhline(0.8, color='green', linestyle=':', label='ACWR Low Risk (0.8)', linewidth=2)
        
        plt.title('Weekly Training Load & ACWR')
        plt.xlabel('Week Number')
        plt.ylabel('Load / Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}\n\nRequired columns:\ndate, time, heart_rate\n(for TRIMP calculation)", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.show()

# Call the function
visualize_training_load(dataset)
