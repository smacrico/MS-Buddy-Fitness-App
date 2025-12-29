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
    'VO2Max': 'vo2max',  # CRITICAL for training score
    'TRIMP': 'TRIMP',
    'resting_hr': 'rest_hr'
})

# Ensure date is datetime upfront
dataset = dataset.copy()
dataset['date'] = pd.to_datetime(dataset['date'])

def visualize_training_score_trends(dataset):
    """Overall training score components over time"""
    try:
        df = dataset.copy()
        
        # Metrics configuration (weights sum to 1.0)
        metrics = {
            'running_economy': {'weight': 0.25, 'higher_better': True},
            'vo2max': {'weight': 0.20, 'higher_better': True},
            'distance': {'weight': 0.15, 'higher_better': True},
            'efficiency_score': {'weight': 0.20, 'higher_better': True},
            'heart_rate': {'weight': 0.20, 'higher_better': False}
        }
        
        normalized_scores = {}
        
        # Normalize each metric (0-1 scale)
        for metric, config in metrics.items():
            if metric in df.columns and df[metric].notna().any():
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:  # Avoid divide by zero
                    if config['higher_better']:
                        norm = (df[metric] - min_val) / (max_val - min_val)
                    else:
                        norm = 1 - ((df[metric] - min_val) / (max_val - min_val))
                    normalized_scores[metric] = norm * config['weight']
                else:
                    normalized_scores[metric] = pd.Series(0.5, index=df.index) * config['weight']
            else:
                # Missing metric gets neutral score
                normalized_scores[metric] = pd.Series(0.5, index=df.index) * config['weight']
        
        # FIXED: Sum Series properly (no axis parameter needed)
        df['training_score'] = sum(normalized_scores.values()) * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['training_score'], 'purple', linewidth=3, label='Training Score')
        plt.fill_between(df['date'], df['training_score'], alpha=0.3, color='purple')
        plt.title('Overall Training Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score (0-100)')
        plt.axhline(70, color='green', linestyle=':', label='Good (70+)', alpha=0.8, linewidth=2)
        plt.axhline(50, color='orange', linestyle=':', label='Caution (50)', alpha=0.8, linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}\n\nRequired columns:\ndate, running_economy, vo2max,\ndistance, efficiency_score, heart_rate", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.show()

# Call the function
visualize_training_score_trends(dataset)
