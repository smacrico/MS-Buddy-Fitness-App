import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_advanced_metrics(dataset):
    """
    Power BI compatible visualization for advanced performance metrics
    dataset: Power BI DataFrame input
    """
    try:
        plt.figure(figsize=(20, 15))
        
        # 1. Cumulative Distance
        plt.subplot(2, 3, 1)
        dataset['cumulative_distance'] = dataset['distance'].cumsum()
        plt.plot(dataset['date'], dataset['cumulative_distance'], 'b-o')
        plt.title('Cumulative Distance')
        plt.xlabel('Date')
        plt.ylabel('Total Distance (km)')
        plt.xticks(rotation=45)
        
        # 2. Running Economy Trend
        plt.subplot(2, 3, 2)
        dataset['running_economy_ma'] = dataset['running_economy'].rolling(window=3).mean()
        plt.plot(dataset['date'], dataset['running_economy'], 'g-', label='Actual')
        plt.plot(dataset['date'], dataset['running_economy_ma'], 'r-', label='3-Session Avg')
        plt.title('Running Economy Trend')
        plt.xlabel('Date')
        plt.ylabel('Running Economy')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 3. Performance Radar
        plt.subplot(2, 3, 3, polar=True)
        metrics = ['running_economy', 'vo2max', 'distance', 'efficiency_score', 'heart_rate']
        
        # Normalize metrics
        normalized_metrics = dataset[metrics].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        avg_metrics = normalized_metrics.mean()
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = avg_metrics.values
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.polar(angles, values, 'o-', linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        plt.xticks(angles[:-1], metrics)
        plt.title('Performance Metrics Radar')
        
        plt.tight_layout()
        
    except Exception as e:
        plt.figure()
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        
    return plt