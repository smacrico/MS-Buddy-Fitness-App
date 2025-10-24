import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_recovery_readiness(dataset):
    """
    Power BI compatible visualization for recovery and readiness scores
    dataset: Power BI DataFrame input
    """
    try:
        df = dataset.copy()
        
        # Calculate recovery and readiness scores
        rhr_baseline = df['resting_hr'].dropna().mean() if 'resting_hr' in df.columns else 60
        trimp_baseline = df['TRIMP'].rolling(window=4).mean() if 'TRIMP' in df.columns else pd.Series(50, index=df.index)
        
        # Calculate normalized scores
        df['rhr_score'] = 1 - ((df['resting_hr'] - rhr_baseline) / rhr_baseline)
        df['load_score'] = 1 - (df['TRIMP'] / (trimp_baseline + 1e-8))
        df['sleep_score'] = df.get('sleep_quality', pd.Series(3, index=df.index)) / 5
        df['fatigue_score'] = 1 - (df.get('fatigue_level', pd.Series(5, index=df.index)) / 10)
        
        # Calculate composite scores
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
        plt.plot(df['date'], df['recovery_score'], label='Recovery')
        plt.plot(df['date'], df['readiness_score'], label='Readiness')
        plt.axhline(0.7, color='orange', linestyle='--', label='Caution threshold')
        plt.xlabel('Date')
        plt.ylabel('Score (0–1)')
        plt.title('Recovery and Readiness Tracking')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    except Exception as e:
        plt.figure()
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        
    return plt