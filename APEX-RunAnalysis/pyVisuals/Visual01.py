# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(undefined)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:

# PowerBI_Visuals.py
"""
Power BI compatible visualizations for RunningAnalysis pipeline
Connect this to your training_log DataFrame from Apex.db
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_trends_basic(dataset):
    """Basic 2x2 trends - matches original VisualiseBasicTrends.py"""
    plt.figure(figsize=(15, 10))
    
    # Ensure date is datetime
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Plot 1: Running Economy over time
    plt.subplot(2, 2, 1)
    plt.plot(dataset['date'], dataset['running_economy'], 'b-o', markersize=4)
    plt.title('Running Economy Trend')
    plt.xticks(rotation=45)
    plt.ylabel('Running Economy')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency Score over time
    plt.subplot(2, 2, 2)
    plt.plot(dataset['date'], dataset['efficiency_score'], 'g-o', markersize=4)
    plt.title('Efficiency Score Trend')
    plt.xticks(rotation=45)
    plt.ylabel('Efficiency Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Energy Cost vs Distance
    plt.subplot(2, 2, 3)
    plt.scatter(dataset['distance'], dataset['energy_cost'], alpha=0.7, s=60)
    plt.title('Energy Cost vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Energy Cost')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Heart Rate vs Running Economy
    plt.subplot(2, 2, 4)
    plt.scatter(dataset['heart_rate'], dataset['running_economy'], alpha=0.7, s=60)
    plt.title('Heart Rate vs Running Economy')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Running Economy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_training_load(dataset):
    """TRIMP and ACWR - requires pre-computed weekly_trimp column"""
    # Calculate TRIMP if not present
    if 'TRIMP' not in dataset.columns:
        rest_hr, max_hr = 60, 190
        dataset['duration_min'] = dataset['time'] / 60
        dataset['hr_ratio'] = (dataset['heart_rate'] - rest_hr) / (max_hr - rest_hr)
        dataset['TRIMP'] = dataset['duration_min'] * dataset['hr_ratio']
    
    # Weekly aggregation (Power BI will handle filters, so compute on current dataset)
    dataset['week'] = dataset['date'].dt.isocalendar().week
    weekly_trimp = dataset.groupby('week')['TRIMP'].sum().reset_index(name='weekly_trimp')
    weekly_trimp['acute_load'] = weekly_trimp['weekly_trimp'].rolling(window=1).mean()
    weekly_trimp['chronic_load'] = weekly_trimp['weekly_trimp'].rolling(window=4).mean()
    weekly_trimp['acwr'] = weekly_trimp['acute_load'] / (weekly_trimp['chronic_load'] + 1e-8)
    
    plt.figure(figsize=(14, 6))
    
    # TRIMP per session
    plt.subplot(1, 2, 1)
    plt.plot(dataset['date'], dataset['TRIMP'], 'o-', markersize=6, linewidth=2)
    plt.title('TRIMP per Session Over Time')
    plt.xlabel('Date')
    plt.ylabel('TRIMP Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Weekly loads + ACWR
    plt.subplot(1, 2, 2)
    weeks = weekly_trimp['week']
    plt.plot(weeks, weekly_trimp['weekly_trimp'], 'o-', label='Weekly TRIMP', markersize=6)
    plt.plot(weeks, weekly_trimp['acute_load'], '--', label='Acute Load (1w)', linewidth=2)
    plt.plot(weeks, weekly_trimp['chronic_load'], '--', label='Chronic Load (4w)', linewidth=2)
    plt.plot(weeks, weekly_trimp['acwr'], '.-', label='ACWR', markersize=8)
    plt.axhline(1.3, color='red', linestyle=':', label='ACWR High Risk', linewidth=2)
    plt.axhline(0.8, color='green', linestyle=':', label='ACWR Low Risk', linewidth=2)
    plt.title('Weekly Training Load & ACWR')
    plt.xlabel('Week Number')
    plt.ylabel('Load / Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_recovery_readiness(dataset):
    """Recovery and Readiness scores over time"""
    df = dataset.copy()
    
    # Calculate if missing (with defaults)
    if 'recovery_score' not in df.columns or df['recovery_score'].isna().all():
        # Simplified calculation matching main script
        df['resting_hr'] = df.get('resting_hr', 60)
        df['sleep_quality'] = df.get('sleep_quality', 3)
        df['fatigue_level'] = df.get('fatigue_level', 5)
        
        rhr_baseline = df['resting_hr'].mean()
        trimp_baseline = df['TRIMP'].rolling(4, min_periods=1).mean()
        
        df['rhr_score'] = 1 - ((df['resting_hr'] - rhr_baseline) / rhr_baseline)
        df['load_score'] = 1 - (df['TRIMP'] / (trimp_baseline + 1e-8))
        df['sleep_score'] = df['sleep_quality'] / 5
        df['fatigue_score'] = 1 - (df['fatigue_level'] / 10)
        
        df['recovery_score'] = (0.3*df['rhr_score'].fillna(1) + 
                               0.3*df['load_score'].fillna(1) + 
                               0.2*df['sleep_score'].fillna(0.6) + 
                               0.2*df['fatigue_score'].fillna(0.5))
        df['readiness_score'] = 0.5*df['recovery_score'] + 0.3*df['load_score'] + 0.2*df['sleep_score']
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['recovery_score'], 'b-o', label='Recovery', linewidth=2, markersize=5)
    plt.plot(df['date'], df['readiness_score'], 'r-s', label='Readiness', linewidth=2, markersize=5)
    plt.axhline(0.7, color='orange', linestyle='--', label='Caution (0.7)', linewidth=2)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='High Risk (0.5)')
    plt.title('Recovery & Readiness Over Time')
    plt.xlabel('Date')
    plt.ylabel('Score (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_training_score_trends(dataset):
    """Overall training score components over time"""
    # Calculate training score per session (simplified)
    metrics = {
        'running_economy': {'weight': 0.25, 'higher_better': True},
        'vo2max': {'weight': 0.20, 'higher_better': True},
        'distance': {'weight': 0.15, 'higher_better': True},
        'efficiency_score': {'weight': 0.20, 'higher_better': True},
        'heart_rate': {'weight': 0.20, 'higher_better': False}
    }
    
    df = dataset.copy()
    normalized_scores = {}
    
    for metric, config in metrics.items():
        if metric in df.columns:
            if config['higher_better']:
                norm = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            else:
                norm = 1 - ((df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min()))
            normalized_scores[metric] = norm * config['weight']
    
    df['training_score'] = sum(normalized_scores.values(), axis=1) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['training_score'], 'purple', linewidth=3, label='Training Score')
    plt.fill_between(df['date'], df['training_score'], alpha=0.3, color='purple')
    plt.title('Overall Training Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Score (0-100)')
    plt.axhline(70, color='green', linestyle=':', label='Good (70+)', alpha=0.8)
    plt.axhline(50, color='orange', linestyle=':', label='Caution (50)', alpha=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Usage in Power BI: Call ONE function per Python visual
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance'
})
visualize_trends_basic(dataset)
# visualize_training_load(dataset) 
# visualize_recovery_readiness(dataset)
# visualize_training_score_trends(dataset)

