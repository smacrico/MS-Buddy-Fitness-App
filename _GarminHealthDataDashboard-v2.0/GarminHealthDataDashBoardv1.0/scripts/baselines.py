import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Generate sample data for demonstration
def generate_sample_data(days=90):
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Health metrics data
    health_data = {
        'date': dates,
        'steps': np.random.normal(8500, 2000, days).clip(2000, 15000),
        'active_minutes': np.random.normal(45, 15, days).clip(10, 120),
        'calories': np.random.normal(2200, 300, days).clip(1500, 3500),
        'sleep_hours': np.random.normal(7.5, 1, days).clip(4, 10),
        'sleep_efficiency': np.random.normal(85, 8, days).clip(60, 98),
        'deep_sleep_pct': np.random.normal(20, 5, days).clip(10, 35),
        'resting_hr': np.random.normal(65, 8, days).clip(45, 85),
        'hrv_score': np.random.normal(35, 8, days).clip(15, 60),
        'stress_avg': np.random.normal(35, 15, days).clip(5, 80),
        'spo2_avg': np.random.normal(97, 2, days).clip(92, 100),
    }
    
    # Activity metrics data (3x per week)
    activity_dates = dates[::2]  # Every other day
    activity_data = {
        'date': activity_dates,
        'distance_km': np.random.normal(5.5, 2, len(activity_dates)).clip(1, 15),
        'duration_min': np.random.normal(35, 10, len(activity_dates)).clip(15, 80),
        'avg_pace_min_km': np.random.normal(5.2, 0.8, len(activity_dates)).clip(3.5, 7.5),
        'avg_hr': np.random.normal(155, 15, len(activity_dates)).clip(120, 190),
        'calories_burned': np.random.normal(350, 100, len(activity_dates)).clip(150, 700),
        'elevation_gain': np.random.normal(120, 50, len(activity_dates)).clip(0, 300),
    }
    
    return pd.DataFrame(health_data), pd.DataFrame(activity_data)

# Calculate derived metrics
def calculate_derived_metrics(health_df, activity_df):
    # Health-derived metrics
    health_df['activity_score'] = (health_df['steps'] / 10000) * 0.4 + (health_df['active_minutes'] / 60) * 0.6
    health_df['movement_ratio'] = health_df['active_minutes'] / (24 * 60) * 100
    health_df['recovery_score'] = (health_df['hrv_score'] * 0.4 + health_df['sleep_efficiency'] * 0.3 + 
                                  (100 - health_df['resting_hr']) * 0.3)
    health_df['wellness_score'] = (health_df['sleep_efficiency'] * 0.3 + health_df['hrv_score'] * 0.2 + 
                                  health_df['activity_score'] * 30 + (100 - health_df['stress_avg']) * 0.2)
    
    # Activity-derived metrics
    activity_df['pace_consistency'] = 1 - (np.random.normal(0.15, 0.05, len(activity_df))).clip(0.05, 0.4)
    activity_df['intensity_factor'] = activity_df['avg_hr'] / 170  # Assuming threshold HR of 170
    activity_df['caloric_efficiency'] = activity_df['distance_km'] / activity_df['calories_burned'] * 1000
    activity_df['elevation_efficiency'] = activity_df['elevation_gain'] / activity_df['distance_km']
    
    return health_df, activity_df

# Establish baselines
def establish_baselines(health_df, activity_df):
    # Health baselines (using percentiles and targets)
    health_baselines = {
        'steps': {'target': 10000, 'good': health_df['steps'].quantile(0.75), 'poor': health_df['steps'].quantile(0.25)},
        'sleep_efficiency': {'target': 85, 'good': 80, 'poor': 70},
        'hrv_score': {'target': health_df['hrv_score'].quantile(0.75), 'good': health_df['hrv_score'].median(), 'poor': health_df['hrv_score'].quantile(0.25)},
        'stress_avg': {'target': 25, 'good': 35, 'poor': 50},
        'wellness_score': {'target': health_df['wellness_score'].quantile(0.9), 'good': health_df['wellness_score'].quantile(0.75), 'poor': health_df['wellness_score'].quantile(0.25)},
        'recovery_score': {'target': health_df['recovery_score'].quantile(0.8), 'good': health_df['recovery_score'].median(), 'poor': health_df['recovery_score'].quantile(0.3)}
    }
    
    # Activity baselines
    activity_baselines = {
        'avg_pace_min_km': {'target': activity_df['avg_pace_min_km'].quantile(0.25), 'good': activity_df['avg_pace_min_km'].median(), 'poor': activity_df['avg_pace_min_km'].quantile(0.75)},
        'intensity_factor': {'target': 0.9, 'good': 0.8, 'poor': 0.7},
        'pace_consistency': {'target': 0.9, 'good': 0.8, 'poor': 0.7},
        'caloric_efficiency': {'target': activity_df['caloric_efficiency'].quantile(0.75), 'good': activity_df['caloric_efficiency'].median(), 'poor': activity_df['caloric_efficiency'].quantile(0.25)}
    }
    
    return health_baselines, activity_baselines

# Generate data
health_df, activity_df = generate_sample_data(90)
health_df, activity_df = calculate_derived_metrics(health_df, activity_df)
health_baselines, activity_baselines = establish_baselines(health_df, activity_df)

# Create comprehensive visualization dashboard
def create_health_dashboard(health_df, health_baselines):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Garmin Health Metrics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Daily Steps with baseline
    ax = axes[0, 0]
    ax.plot(health_df['date'], health_df['steps'], linewidth=2, label='Daily Steps')
    ax.axhline(y=health_baselines['steps']['target'], color='green', linestyle='--', alpha=0.7, label='Target (10K)')
    ax.axhline(y=health_baselines['steps']['good'], color='orange', linestyle='--', alpha=0.7, label='Good')
    ax.axhline(y=health_baselines['steps']['poor'], color='red', linestyle='--', alpha=0.7, label='Poor')
    ax.fill_between(health_df['date'], health_baselines['steps']['poor'], health_baselines['steps']['good'], alpha=0.2, color='yellow')
    ax.set_title('Daily Steps')
    ax.set_ylabel('Steps')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Sleep Efficiency
    ax = axes[0, 1]
    ax.plot(health_df['date'], health_df['sleep_efficiency'], linewidth=2, color='purple', label='Sleep Efficiency')
    ax.axhline(y=health_baselines['sleep_efficiency']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['sleep_efficiency']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['sleep_efficiency']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(health_df['date'], health_baselines['sleep_efficiency']['poor'], 
                    health_baselines['sleep_efficiency']['good'], alpha=0.2, color='purple')
    ax.set_title('Sleep Efficiency (%)')
    ax.set_ylabel('Efficiency %')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. HRV Score
    ax = axes[0, 2]
    ax.plot(health_df['date'], health_df['hrv_score'], linewidth=2, color='blue', label='HRV Score')
    ax.axhline(y=health_baselines['hrv_score']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['hrv_score']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['hrv_score']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(health_df['date'], health_baselines['hrv_score']['poor'], 
                    health_baselines['hrv_score']['good'], alpha=0.2, color='blue')
    ax.set_title('Heart Rate Variability')
    ax.set_ylabel('HRV Score')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Stress Levels
    ax = axes[1, 0]
    ax.plot(health_df['date'], health_df['stress_avg'], linewidth=2, color='red', label='Avg Stress')
    ax.axhline(y=health_baselines['stress_avg']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['stress_avg']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['stress_avg']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(health_df['date'], health_baselines['stress_avg']['target'], 
                    health_baselines['stress_avg']['good'], alpha=0.2, color='green')
    ax.set_title('Average Daily Stress')
    ax.set_ylabel('Stress Level')
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Recovery Score
    ax = axes[1, 1]
    ax.plot(health_df['date'], health_df['recovery_score'], linewidth=2, color='cyan', label='Recovery Score')
    ax.axhline(y=health_baselines['recovery_score']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['recovery_score']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['recovery_score']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(health_df['date'], health_baselines['recovery_score']['poor'], 
                    health_baselines['recovery_score']['good'], alpha=0.2, color='cyan')
    ax.set_title('Recovery Score')
    ax.set_ylabel('Recovery Score')
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Overall Wellness Score
    ax = axes[1, 2]
    ax.plot(health_df['date'], health_df['wellness_score'], linewidth=2, color='magenta', label='Wellness Score')
    ax.axhline(y=health_baselines['wellness_score']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['wellness_score']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=health_baselines['wellness_score']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(health_df['date'], health_baselines['wellness_score']['poor'], 
                    health_baselines['wellness_score']['good'], alpha=0.2, color='magenta')
    ax.set_title('Overall Wellness Score')
    ax.set_ylabel('Wellness Score')
    ax.tick_params(axis='x', rotation=45)
    
    # 7. Sleep vs Recovery correlation
    ax = axes[2, 0]
    scatter = ax.scatter(health_df['sleep_efficiency'], health_df['recovery_score'], 
                        c=health_df['hrv_score'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('Sleep Efficiency (%)')
    ax.set_ylabel('Recovery Score')
    ax.set_title('Sleep vs Recovery (colored by HRV)')
    plt.colorbar(scatter, ax=ax, label='HRV Score')
    
    # 8. Weekly averages heatmap
    ax = axes[2, 1]
    health_df['week'] = health_df['date'].dt.isocalendar().week
    weekly_data = health_df.groupby('week')[['steps', 'sleep_efficiency', 'hrv_score', 'stress_avg']].mean()
    sns.heatmap(weekly_data.T, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax)
    ax.set_title('Weekly Averages Heatmap')
    ax.set_xlabel('Week Number')
    
    # 9. Trend analysis
    ax = axes[2, 2]
    # Calculate 7-day rolling averages
    health_df['wellness_7d'] = health_df['wellness_score'].rolling(7).mean()
    health_df['recovery_7d'] = health_df['recovery_score'].rolling(7).mean()
    
    ax.plot(health_df['date'], health_df['wellness_7d'], label='Wellness (7d avg)', linewidth=2)
    ax.plot(health_df['date'], health_df['recovery_7d'], label='Recovery (7d avg)', linewidth=2)
    ax.set_title('Wellness & Recovery Trends')
    ax.set_ylabel('Score')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_activity_dashboard(activity_df, activity_baselines):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Garmin Activity Metrics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Pace over time
    ax = axes[0, 0]
    ax.plot(activity_df['date'], activity_df['avg_pace_min_km'], linewidth=2, marker='o', markersize=4)
    ax.axhline(y=activity_baselines['avg_pace_min_km']['target'], color='green', linestyle='--', alpha=0.7, label='Target')
    ax.axhline(y=activity_baselines['avg_pace_min_km']['good'], color='orange', linestyle='--', alpha=0.7, label='Good')
    ax.axhline(y=activity_baselines['avg_pace_min_km']['poor'], color='red', linestyle='--', alpha=0.7, label='Poor')
    ax.fill_between(activity_df['date'], activity_baselines['avg_pace_min_km']['good'], 
                    activity_baselines['avg_pace_min_km']['poor'], alpha=0.2, color='yellow')
    ax.set_title('Average Pace Progression')
    ax.set_ylabel('Pace (min/km)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Distance vs Calories
    ax = axes[0, 1]
    scatter = ax.scatter(activity_df['distance_km'], activity_df['calories_burned'], 
                        c=activity_df['avg_hr'], cmap='plasma', alpha=0.7, s=60)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Calories Burned')
    ax.set_title('Distance vs Calories (colored by HR)')
    plt.colorbar(scatter, ax=ax, label='Avg Heart Rate')
    
    # 3. Intensity Factor
    ax = axes[0, 2]
    ax.plot(activity_df['date'], activity_df['intensity_factor'], linewidth=2, color='red', marker='s', markersize=4)
    ax.axhline(y=activity_baselines['intensity_factor']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=activity_baselines['intensity_factor']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=activity_baselines['intensity_factor']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(activity_df['date'], activity_baselines['intensity_factor']['poor'], 
                    activity_baselines['intensity_factor']['good'], alpha=0.2, color='red')
    ax.set_title('Training Intensity Factor')
    ax.set_ylabel('Intensity Factor')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Caloric Efficiency
    ax = axes[1, 0]
    ax.plot(activity_df['date'], activity_df['caloric_efficiency'], linewidth=2, color='green', marker='^', markersize=4)
    ax.axhline(y=activity_baselines['caloric_efficiency']['target'], color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=activity_baselines['caloric_efficiency']['good'], color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=activity_baselines['caloric_efficiency']['poor'], color='red', linestyle='--', alpha=0.7)
    ax.fill_between(activity_df['date'], activity_baselines['caloric_efficiency']['poor'], 
                    activity_baselines['caloric_efficiency']['good'], alpha=0.2, color='green')
    ax.set_title('Caloric Efficiency')
    ax.set_ylabel('Distance/Calories (m/cal)')
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Performance metrics correlation matrix
    ax = axes[1, 1]
    corr_data = activity_df[['distance_km', 'avg_pace_min_km', 'avg_hr', 'intensity_factor', 'caloric_efficiency']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Performance Metrics Correlation')
    
    # 6. Weekly volume and intensity
    ax = axes[1, 2]
    activity_df['week'] = activity_df['date'].dt.isocalendar().week
    weekly_volume = activity_df.groupby('week')['distance_km'].sum()
    weekly_intensity = activity_df.groupby('week')['intensity_factor'].mean()
    
    ax2 = ax.twinx()
    line1 = ax.bar(weekly_volume.index, weekly_volume.values, alpha=0.6, label='Weekly Distance', color='blue')
    line2 = ax2.plot(weekly_intensity.index, weekly_intensity.values, color='red', marker='o', linewidth=2, label='Avg Intensity')
    
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Total Distance (km)', color='blue')
    ax2.set_ylabel('Average Intensity Factor', color='red')
    ax.set_title('Weekly Volume vs Intensity')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig

# Generate the dashboards
health_fig = create_health_dashboard(health_df, health_baselines)
activity_fig = create_activity_dashboard(activity_df, activity_baselines)

# Display baseline summary
def print_baseline_summary(health_baselines, activity_baselines):
    print("="*60)
    print("GARMIN METRICS BASELINES SUMMARY")
    print("="*60)
    
    print("\n📊 HEALTH METRICS BASELINES:")
    print("-" * 40)
    for metric, values in health_baselines.items():
        print(f"{metric.replace('_', ' ').title():<20} | Target: {values['target']:.1f} | Good: {values['good']:.1f} | Poor: {values['poor']:.1f}")
    
    print("\n🏃 ACTIVITY METRICS BASELINES:")
    print("-" * 40)
    for metric, values in activity_baselines.items():
        print(f"{metric.replace('_', ' ').title():<20} | Target: {values['target']:.2f} | Good: {values['good']:.2f} | Poor: {values['poor']:.2f}")
    
    print("\n📈 BASELINE INTERPRETATION:")
    print("-" * 40)
    print("• Target: Optimal performance/health zone")
    print("• Good: Above average, positive trend")
    print("• Poor: Below average, needs attention")
    print("• Shaded areas: Normal variation range")

print_baseline_summary(health_baselines, activity_baselines)

plt.show()
