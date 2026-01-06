# -*- coding: utf-8 -*-
"""Garmin Stress Data Analyzer - Local SQLite Version

Reads stress data from local garmin_summary SQLite database
and creates visualizations using Matplotlib and Seaborn.
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Database configuration
DB_PATH = r"C:\smakryko\myHealthData\DBs\garmin_summary.db"

# Output directory for charts
CHART_OUTPUT_DIR = r'C:\temp\logsFitnessApp\Garmin_DashBoards'

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    print(f"Charts will be saved to: {CHART_OUTPUT_DIR}")

def get_db_connection():
    """Establish connection to the Garmin SQLite database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")
    return sqlite3.connect(DB_PATH)

def convert_to_numeric(df, exclude_columns=None):
    """Convert appropriate columns to numeric types, handling errors gracefully."""
    if exclude_columns is None:
        exclude_columns = []
    
    for col in df.columns:
        if col not in exclude_columns:
            # Try to convert to numeric, keeping non-convertible as-is
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df

def load_weekly_stress(weeks=104):
    """Load weekly stress data from weeks_summary table."""
    conn = get_db_connection()
    query = f"""
        SELECT 
            first_day,
            stress_avg,
            hr_avg,
            rhr_avg,
            inactive_hr_avg,
            sleep_avg,
            rem_sleep_avg,
            steps,
            intensity_time,
            moderate_activity_time,
            vigorous_activity_time,
            calories_avg,
            spo2_avg,
            rr_waking_avg,
            hydration_avg,
            weight_avg
        FROM weeks_summary 
        ORDER BY first_day DESC 
        LIMIT {weeks}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert first_day to datetime
    df['first_day'] = pd.to_datetime(df['first_day'])
    
    # Convert numeric columns
    df = convert_to_numeric(df, exclude_columns=['first_day'])
    
    return df

def load_daily_stress():
    """Load daily stress data from days_summary table."""
    conn = get_db_connection()
    # Try to get daily data - adjust table name if needed
    try:
        query = """
            SELECT 
                day,
                stress_avg,
                hr_avg,
                rhr_avg,
                sleep_avg,
                steps,
                intensity_time,
                spo2_avg,
                calories_avg
            FROM days_summary 
            WHERE stress_avg IS NOT NULL
            ORDER BY day DESC 
            LIMIT 1095
        """
        df = pd.read_sql_query(query, conn)
    except:
        # Fallback if days_summary doesn't exist
        print("Note: Could not load daily data. Using weekly data for all analysis.")
        df = pd.DataFrame()
    
    conn.close()
    
    if not df.empty:
        df['day'] = pd.to_datetime(df['day'])
        # Convert numeric columns
        df = convert_to_numeric(df, exclude_columns=['day'])
    
    return df

def analyze_weekly_stress():
    """Analyze and visualize weekly stress trends."""
    print("\n" + "="*80)
    print("WEEKLY STRESS ANALYSIS")
    print("="*80)
    
    df = load_weekly_stress(weeks=104)
    df = df.sort_values("first_day")
    
    # Calculate statistics
    print(f"\nWeekly Stress Statistics (last 2 years):")
    print(f"Average Stress:        {df['stress_avg'].mean():.1f}")
    print(f"Min Stress:            {df['stress_avg'].min():.1f}")
    print(f"Max Stress:            {df['stress_avg'].max():.1f}")
    print(f"Std Dev:               {df['stress_avg'].std():.1f}")
    
    # Calculate trend
    if len(df) >= 10:
        recent_avg = df.tail(8)['stress_avg'].mean()
        early_avg = df.head(8)['stress_avg'].mean()
        change = ((recent_avg - early_avg) / early_avg) * 100
        print(f"Trend (recent vs early): {change:+.1f}%")
    
    # Create visualization
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot stress line
    sns.lineplot(x=df["first_day"], y=df["stress_avg"], 
                marker='o', linewidth=2, markersize=6, ax=ax)
    
    # Add rolling average
    df['rolling_avg'] = df['stress_avg'].rolling(window=4).mean()
    sns.lineplot(x=df["first_day"], y=df["rolling_avg"],
                linewidth=2, linestyle='--', color='red', 
                label='4-week Rolling Average', ax=ax)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_xlabel(None)
    ax.set_ylabel("Stress Level")
    ax.set_title("Average Weekly Stress Over Time", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(CHART_OUTPUT_DIR, 
                              f'Garmin_WeeklyStress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Weekly stress chart saved to: {output_path}")
    plt.show()
    
    return df

def analyze_stress_correlations(df):
    """Analyze correlations between stress and other metrics."""
    print("\n" + "="*80)
    print("STRESS CORRELATION ANALYSIS")
    print("="*80)
    
    # Select metrics for correlation
    correlation_metrics = [
        'stress_avg', 'hr_avg', 'rhr_avg', 'sleep_avg', 
        'steps', 'spo2_avg', 'hydration_avg'
    ]
    
    # Filter to available columns that exist in the dataframe
    available_metrics = [col for col in correlation_metrics if col in df.columns]
    
    # Further filter to only numeric columns
    numeric_metrics = []
    for col in available_metrics:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_metrics.append(col)
        else:
            print(f"  Skipping non-numeric column: {col} (type: {df[col].dtype})")
    
    if len(numeric_metrics) < 2:
        print("\n⚠ Warning: Not enough numeric metrics available for correlation analysis.")
        print(f"Available numeric columns: {numeric_metrics}")
        return
    
    print(f"\nAnalyzing correlations for: {', '.join(numeric_metrics)}")
    
    # Calculate correlations
    corr_matrix = df[numeric_metrics].corr()
    
    print("\nCorrelations with Stress:")
    if 'stress_avg' in corr_matrix.columns:
        stress_corr = corr_matrix['stress_avg'].sort_values(ascending=False)
        for metric, corr_value in stress_corr.items():
            if metric != 'stress_avg':
                print(f"  {metric:20s}: {corr_value:+.3f}")
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title("Stress and Health Metrics Correlation Matrix", 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(CHART_OUTPUT_DIR, 
                              f'Garmin_StressCorrelations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Correlation chart saved to: {output_path}")
    plt.show()

def analyze_stress_vs_activity(df):
    """Analyze relationship between stress and activity levels."""
    print("\n" + "="*80)
    print("STRESS vs ACTIVITY ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Stress vs Steps
    if 'steps' in df.columns and pd.api.types.is_numeric_dtype(df['steps']):
        ax1 = axes[0, 0]
        valid_data = df[['steps', 'stress_avg']].dropna()
        if len(valid_data) >= 3:
            sns.scatterplot(x=valid_data['steps'], y=valid_data['stress_avg'], 
                           alpha=0.6, s=100, ax=ax1)
            z = np.polyfit(valid_data['steps'], valid_data['stress_avg'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data['steps'], p(valid_data['steps']), "r--", alpha=0.8, linewidth=2)
        ax1.set_title('Stress vs Daily Steps')
        ax1.set_xlabel('Average Daily Steps')
        ax1.set_ylabel('Stress Level')
        ax1.grid(True, alpha=0.3)
    
    # Stress vs Sleep
    if 'sleep_avg' in df.columns and pd.api.types.is_numeric_dtype(df['sleep_avg']):
        ax2 = axes[0, 1]
        valid_data = df[['sleep_avg', 'stress_avg']].dropna()
        if len(valid_data) >= 3:
            sns.scatterplot(x=valid_data['sleep_avg'], y=valid_data['stress_avg'],
                           alpha=0.6, s=100, ax=ax2, color='purple')
            z = np.polyfit(valid_data['sleep_avg'], valid_data['stress_avg'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data['sleep_avg'], p(valid_data['sleep_avg']), "r--", alpha=0.8, linewidth=2)
        ax2.set_title('Stress vs Sleep Duration')
        ax2.set_xlabel('Average Sleep (hours)')
        ax2.set_ylabel('Stress Level')
        ax2.grid(True, alpha=0.3)
    
    # Stress vs Heart Rate
    if 'hr_avg' in df.columns and pd.api.types.is_numeric_dtype(df['hr_avg']):
        ax3 = axes[1, 0]
        valid_data = df[['hr_avg', 'stress_avg']].dropna()
        if len(valid_data) >= 3:
            sns.scatterplot(x=valid_data['hr_avg'], y=valid_data['stress_avg'],
                           alpha=0.6, s=100, ax=ax3, color='red')
            z = np.polyfit(valid_data['hr_avg'], valid_data['stress_avg'], 1)
            p = np.poly1d(z)
            ax3.plot(valid_data['hr_avg'], p(valid_data['hr_avg']), "r--", alpha=0.8, linewidth=2)
        ax3.set_title('Stress vs Average Heart Rate')
        ax3.set_xlabel('Average Heart Rate (bpm)')
        ax3.set_ylabel('Stress Level')
        ax3.grid(True, alpha=0.3)
    
    # Stress vs Activity Time
    if 'intensity_time' in df.columns and pd.api.types.is_numeric_dtype(df['intensity_time']):
        ax4 = axes[1, 1]
        valid_data = df[['intensity_time', 'stress_avg']].dropna()
        if len(valid_data) >= 3:
            sns.scatterplot(x=valid_data['intensity_time'], y=valid_data['stress_avg'],
                           alpha=0.6, s=100, ax=ax4, color='green')
            z = np.polyfit(valid_data['intensity_time'], valid_data['stress_avg'], 1)
            p = np.poly1d(z)
            ax4.plot(valid_data['intensity_time'], p(valid_data['intensity_time']), "r--", alpha=0.8, linewidth=2)
        ax4.set_title('Stress vs Intensity Time')
        ax4.set_xlabel('Weekly Intensity Time (min)')
        ax4.set_ylabel('Stress Level')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Stress vs Activity & Health Metrics', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(CHART_OUTPUT_DIR, 
                              f'Garmin_StressVsActivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Activity analysis chart saved to: {output_path}")
    plt.show()

def analyze_daily_stress_trends():
    """Analyze daily stress trends with seasonal decomposition."""
    print("\n" + "="*80)
    print("DAILY STRESS TREND ANALYSIS")
    print("="*80)
    
    df = load_daily_stress()
    
    if df.empty:
        print("Daily data not available. Using weekly data instead.")
        return
    
    df = df.sort_values('day')
    df.set_index('day', inplace=True)
    
    # Calculate rolling average
    df['rolling_avg_28'] = df['stress_avg'].rolling(window=28).mean()
    
    print(f"\nDaily Stress Statistics:")
    print(f"Average Stress:        {df['stress_avg'].mean():.1f}")
    print(f"Min Stress:            {df['stress_avg'].min():.1f}")
    print(f"Max Stress:            {df['stress_avg'].max():.1f}")
    print(f"Std Dev:               {df['stress_avg'].std():.1f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.scatterplot(x=df.index, y=df["stress_avg"],
                   color="skyblue", label="Daily Stress Level",
                   alpha=0.5, s=50, ax=ax)
    
    sns.lineplot(x=df.index, y=df["rolling_avg_28"],
                color="r", label="28-day Rolling Average",
                linewidth=2, ax=ax)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_xlabel(None)
    ax.set_ylabel("Stress Level")
    ax.set_title("Daily Stress Level Over Time", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(CHART_OUTPUT_DIR, 
                              f'Garmin_DailyStress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Daily stress chart saved to: {output_path}")
    plt.show()
    
    # Seasonal decomposition
    if len(df) >= 56:  # Need at least 2 periods
        try:
            result = seasonal_decompose(
                df["stress_avg"].dropna(), 
                model="additive", 
                period=28
            )
            trend = result.trend.dropna()
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # Original data
            axes[0].scatter(df.index, df["stress_avg"], 
                          color='skyblue', alpha=0.5, s=30)
            axes[0].set_title('Daily Stress Level', fontweight='bold')
            axes[0].set_xlim(df.index.min(), df.index.max())
            axes[0].set_ylabel('Stress Level')
            axes[0].grid(True, alpha=0.3)
            axes[0].xaxis.set_major_locator(mdates.MonthLocator())
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[0].tick_params(axis="x", rotation=45)
            
            # Trend
            axes[1].plot(trend.index, trend, color='r', linewidth=2)
            axes[1].set_title('28-Day Trend', fontweight='bold')
            axes[1].set_xlim(df.index.min(), df.index.max())
            axes[1].set_ylabel('Stress Level')
            axes[1].grid(True, alpha=0.3)
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[1].tick_params(axis="x", rotation=45)
            
            plt.suptitle('Stress Decomposition Analysis', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            output_path = os.path.join(CHART_OUTPUT_DIR, 
                                      f'Garmin_StressDecomposition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Decomposition chart saved to: {output_path}")
            plt.show()
            
        except Exception as e:
            print(f"Could not perform seasonal decomposition: {e}")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GARMIN STRESS DATA ANALYZER")
    print("="*80)
    
    # Setup output directory
    setup_output_directory()
    
    try:
        # Analyze weekly stress
        weekly_df = analyze_weekly_stress()
        
        # Analyze correlations
        analyze_stress_correlations(weekly_df)
        
        # Analyze stress vs activity
        analyze_stress_vs_activity(weekly_df)
        
        # Analyze daily trends
        analyze_daily_stress_trends()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll charts saved to: {CHART_OUTPUT_DIR}")
        print("\nOpen Garmin_Dashboard.html to view all visualizations.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()