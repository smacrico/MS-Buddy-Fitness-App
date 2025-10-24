import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_training_load(dataset):
    """
    Power BI compatible visualization for training load
    dataset: Power BI DataFrame input
    """
    try:
        # Calculate TRIMP
        rest_hr = 60
        max_hr = 190
        dataset['duration_min'] = dataset['time'] / 60
        dataset['hr_ratio'] = (dataset['heart_rate'] - rest_hr) / (max_hr - rest_hr)
        dataset['TRIMP'] = dataset['duration_min'] * dataset['hr_ratio']

        # Calculate weekly metrics
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['week'] = dataset['date'].dt.isocalendar().week
        weekly_trimp = dataset.groupby('week')['TRIMP'].sum().reset_index()
        
        # Calculate loads
        weekly_trimp['acute_load'] = weekly_trimp['TRIMP'].rolling(window=1).mean()
        weekly_trimp['chronic_load'] = weekly_trimp['TRIMP'].rolling(window=4).mean()
        weekly_trimp['acwr'] = weekly_trimp['acute_load'] / (weekly_trimp['chronic_load'] + 1e-8)

        plt.figure(figsize=(14, 6))
        
        # Plot training load metrics
        plt.plot(weekly_trimp['week'], weekly_trimp['TRIMP'], label='Weekly TRIMP', marker='o')
        plt.plot(weekly_trimp['week'], weekly_trimp['acute_load'], label='Acute Load', linestyle='--')
        plt.plot(weekly_trimp['week'], weekly_trimp['chronic_load'], label='Chronic Load', linestyle='--')
        plt.plot(weekly_trimp['week'], weekly_trimp['acwr'], label='ACWR', linestyle='-.')
        
        # Add threshold lines
        plt.axhline(1.3, color='red', linestyle=':', label='Upper ACWR Threshold')
        plt.axhline(0.8, color='green', linestyle=':', label='Lower ACWR Threshold')
        
        plt.title('Training Load Analysis')
        plt.xlabel('Week Number')
        plt.ylabel('Load / Ratio')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    except Exception as e:
        plt.figure()
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        
    return plt