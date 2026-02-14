#!/usr/bin/env python3
"""
Command-line interface for longevity metrics computation.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Import longevity modules
from longevity import (
    compute_hrv_summary,
    compute_sleep_summary,
    compute_activity_summary,
    compute_recovery_score,
    compute_metabolic_capacity,
    compute_cardiovascular_health,
    compute_biological_age,
    forecast_capacity_trend,
    plot_hrv_metrics,
    plot_sleep_metrics,
    plot_recovery_and_capacity,
    plot_dashboard,
    plot_forecast,
    export_to_csv,
    export_to_json,
    export_to_excel,
    export_metrics_summary,
    create_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def parse_rr_intervals(rr_string):
    """Parse RR intervals from string representation."""
    try:
        import ast
        if pd.isna(rr_string):
            return []
        return ast.literal_eval(rr_string)
    except:
        return []


def analyze_command(args):
    """Analyze health data and compute metrics."""
    print(f"\n{'='*60}")
    print(f"  Longevity Metrics Analysis")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data(args.input)
    
    # Parse RR intervals if needed
    if 'rr_intervals' in df.columns:
        df['rr_intervals'] = df['rr_intervals'].apply(parse_rr_intervals)
    
    # Group by date
    df['date'] = df['timestamp'].dt.date
    daily_metrics = []
    
    for date, day_data in df.groupby('date'):
        metrics = {'date': date}
        
        # HRV metrics
        if 'rr_intervals' in day_data.columns:
            rr_all = [x for sublist in day_data['rr_intervals'] for x in sublist if x]
            if rr_all:
                hrv_summary = compute_hrv_summary(rr_all)
                metrics.update(hrv_summary)
        
        # Sleep metrics
        sleep_data = day_data[day_data['sleep_stage'] != 'awake']
        if not sleep_data.empty and 'time_in_bed_min' in sleep_data.columns:
            sleep_summary = compute_sleep_summary(
                sleep_data['sleep_stage'].tolist(),
                sleep_data['time_in_bed_min'].iloc[0],
                sleep_data['total_sleep_min'].iloc[0]
            )
            metrics['sleep_quality'] = sleep_summary['quality']
            metrics['sleep_efficiency'] = sleep_summary['efficiency']
        
        # Resting HR
        if 'heart_rate' in day_data.columns:
            metrics['rhr'] = float(day_data['heart_rate'].min())
            metrics['avg_hr'] = float(day_data['heart_rate'].mean())
        
        daily_metrics.append(metrics)
    
    results_df = pd.DataFrame(daily_metrics)
    
    print(f"\n✓ Analyzed {len(results_df)} days of data")
    print(f"\nMetrics computed:")
    print(f"  - HRV: {sum('rmssd' in m for m in daily_metrics)} days")
    print(f"  - Sleep: {sum('sleep_quality' in m for m in daily_metrics)} days")
    print(f"  - Heart Rate: {sum('rhr' in m for m in daily_metrics)} days")
    
    # Export results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'csv':
            export_to_csv(results_df, output_path)
        elif args.format == 'json':
            export_to_json(results_df, output_path)
        elif args.format == 'excel':
            export_to_excel(results_df, output_path)
        elif args.format == 'all':
            export_metrics_summary(results_df, output_path.parent, 
                                   formats=['csv', 'json', 'excel'])
    
    return results_df


def visualize_command(args):
    """Create visualizations."""
    print(f"\n{'='*60}")
    print(f"  Creating Visualizations")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data(args.input)
    
    # Parse dates
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    dates = pd.to_datetime(df['date'].unique())
    
    # Prepare metrics dict
    metrics = {}
    for col in df.columns:
        if col not in ['timestamp', 'date']:
            # Group by date and take mean
            grouped = df.groupby('date')[col].mean()
            metrics[col] = grouped.values
    
    output_dir = Path(args.output) if args.output else Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    if args.plot_type == 'hrv' or args.plot_type == 'all':
        if 'rmssd' in metrics and 'sdnn' in metrics:
            plot_hrv_metrics(dates, metrics['rmssd'], metrics['sdnn'],
                           output_path=output_dir / 'hrv_metrics.png',
                           show=args.show)
            print("✓ Created HRV metrics plot")
    
    if args.plot_type == 'sleep' or args.plot_type == 'all':
        if 'sleep_quality' in metrics and 'sleep_efficiency' in metrics:
            plot_sleep_metrics(dates, metrics['sleep_quality'], 
                             metrics['sleep_efficiency'],
                             output_path=output_dir / 'sleep_metrics.png',
                             show=args.show)
            print("✓ Created sleep metrics plot")
    
    if args.plot_type == 'recovery' or args.plot_type == 'all':
        if 'recovery_score' in metrics and 'capacity_score' in metrics:
            plot_recovery_and_capacity(dates, metrics['recovery_score'],
                                     metrics['capacity_score'],
                                     output_path=output_dir / 'recovery_capacity.png',
                                     show=args.show)
            print("✓ Created recovery/capacity plot")
    
    if args.plot_type == 'dashboard' or args.plot_type == 'all':
        plot_dashboard(dates, metrics,
                      output_path=output_dir / 'dashboard.png',
                      show=args.show)
        print("✓ Created dashboard")
    
    print(f"\n✓ Plots saved to: {output_dir}")


def export_command(args):
    """Export data to specified format."""
    print(f"\n{'='*60}")
    print(f"  Exporting Data")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data(args.input)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'csv':
        export_to_csv(df, output_path)
    elif args.format == 'json':
        export_to_json(df, output_path)
    elif args.format == 'excel':
        export_to_excel(df, output_path)
    elif args.format == 'parquet':
        from longevity.export import export_to_parquet
        export_to_parquet(df, output_path)
    elif args.format == 'all':
        export_metrics_summary(df, output_path.parent,
                              formats=['csv', 'json', 'excel'])
    
    print(f"\n✓ Export complete")


def report_command(args):
    """Generate comprehensive report."""
    print(f"\n{'='*60}")
    print(f"  Generating Report")
    print(f"{'='*60}\n")
    
    # Run analysis
    args.output = None  # Don't export during analysis
    results_df = analyze_command(args)
    
    # Compute summary statistics
    summary = {
        'total_days': len(results_df),
        'date_range': f"{results_df['date'].min()} to {results_df['date'].max()}",
    }
    
    if 'rmssd' in results_df.columns:
        summary['avg_rmssd'] = f"{results_df['rmssd'].mean():.2f} ms"
        summary['avg_sdnn'] = f"{results_df['sdnn'].mean():.2f} ms" if 'sdnn' in results_df.columns else 'N/A'
    
    if 'sleep_quality' in results_df.columns:
        summary['avg_sleep_quality'] = f"{results_df['sleep_quality'].mean():.1f}"
    
    if 'rhr' in results_df.columns:
        summary['avg_rhr'] = f"{results_df['rhr'].mean():.1f} bpm"
    
    # Create report
    output_path = Path(args.output) if args.output else Path('outputs/report.txt')
    create_report(summary, output_path, title="Longevity Metrics Report")
    
    print(f"\n✓ Report saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Longevity Metrics - Health Analytics CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze data and export to CSV
  %(prog)s analyze -i data/demo.csv -o results/metrics.csv

  # Create all visualizations
  %(prog)s visualize -i data/demo.csv -t all -o plots/

  # Generate comprehensive report
  %(prog)s report -i data/demo.csv -o reports/summary.txt

  # Export data to multiple formats
  %(prog)s export -i data/demo.csv -o outputs/data.xlsx -f all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze health data')
    analyze_parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    analyze_parser.add_argument('-o', '--output', help='Output file path')
    analyze_parser.add_argument('-f', '--format', choices=['csv', 'json', 'excel', 'all'],
                               default='csv', help='Output format')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    viz_parser.add_argument('-o', '--output', help='Output directory')
    viz_parser.add_argument('-t', '--plot-type', 
                           choices=['hrv', 'sleep', 'recovery', 'dashboard', 'all'],
                           default='all', help='Type of plot to create')
    viz_parser.add_argument('--show', action='store_true', help='Display plots')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    export_parser.add_argument('-o', '--output', required=True, help='Output file path')
    export_parser.add_argument('-f', '--format', 
                              choices=['csv', 'json', 'excel', 'parquet', 'all'],
                              default='csv', help='Export format')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report')
    report_parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    report_parser.add_argument('-o', '--output', help='Output report path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'visualize':
            visualize_command(args)
        elif args.command == 'export':
            export_command(args)
        elif args.command == 'report':
            report_command(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
