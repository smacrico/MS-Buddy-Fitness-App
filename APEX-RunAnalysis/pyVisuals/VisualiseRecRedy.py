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
    'resting_hr': 'rest_hr',  # or 'Resting HR': 'rest_hr'
    'sleep_quality': 'sleep_quality',
    'fatigue_level': 'fatigue_level'
})

# Ensure date is datetime
dataset = dataset.copy()
dataset['date'] = pd.to_datetime(dataset['date'])

def visualize_recovery_readiness(dataset):
    """
    Power BI compatible visualization for recovery and readiness scores
    dataset: Power BI DataFrame input (after column renaming)
    """
    try:
        df = dataset.copy()
        
        # Calculate baselines with safe defaults
        rhr_baseline = df['rest_hr'].dropna().mean() if 'rest_hr' in df.columns else 60
        trimp_baseline = df['TRIMP'].rolling(window=4, min_periods=1).mean() if 'TRIMP' in df.columns else pd.Series(50, index=df.index)
        
        # Calculate normalized component scores (0-1 scale, higher=better)
        df['rhr_score'] = 1 - ((df['rest_hr'].fillna(rhr_baseline) - rhr_baseline) / rhr_baseline)
        df['load_score'] = 1 - (df['TRIMP'].fillna(50) / (trimp_baseline + 1e-8))
        df['sleep_score'] = df.get('sleep_quality', pd.Series(3, index=df.index)).fillna(3) / 5
        df['fatigue_score'] = 1 - (df.get('fatigue_level', pd.Series(5, index=df.index)).fillna(5) / 10)
        
        # Composite scores
        df['recovery_score'] = (
            0.3 * df['rhr_score'].fillna(1) +
            0.3 * df['load_score'].fillna(1) +
            0.2 * df['sleep_score'].fillna(0.6) +
            0.2 * df['fatigue_score'].fillna(0.5)
        )
        
        df['readiness_score'] = (
            0.5 * df['recovery_score'] +
            0.3 * df['load_score'].fillna(1) +
            0.2 * df['sleep_score'].fillna(0.6)
        )

        # Create visualization
        plt.figure(figsize=(12, 5))
        plt.plot(df['date'], df['recovery_score'], 'b-o', label='Recovery', linewidth=2, markersize=5)
        plt.plot(df['date'], df['readiness_score'], 'r-s', label='Readiness', linewidth=2, markersize=5)
        plt.axhline(0.7, color='orange', linestyle='--', label='Caution (0.7)', linewidth=2)
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='High Risk (0.5)')
        plt.xlabel('Date')
        plt.ylabel('Score (0–1)')
        plt.title('Recovery and Readiness Tracking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}\n\nRequired columns:\ndate, rest_hr, TRIMP", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.show()

# Call the function
visualize_recovery_readiness(dataset)
