"""
Visualization module for longevity metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Union, Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from .config import (
    DEFAULT_FIGURE_SIZE,
    DEFAULT_DPI,
    COLORS,
)


def setup_plot_style():
    """Configure matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = DEFAULT_FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DEFAULT_DPI
    plt.rcParams['font.size'] = 10


def plot_hrv_metrics(
    dates: Sequence,
    rmssd: Sequence[float],
    sdnn: Sequence[float],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot HRV metrics over time.
    
    Args:
        dates: Date/timestamp sequence
        rmssd: RMSSD values (ms)
        sdnn: SDNN values (ms)
        output_path: Path to save figure (optional)
        show: Whether to display plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)
    
    # RMSSD plot
    ax1.plot(dates, rmssd, color=COLORS['hrv'], linewidth=2, marker='o', markersize=4)
    ax1.set_ylabel('RMSSD (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Heart Rate Variability Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(dates, rmssd, alpha=0.2, color=COLORS['hrv'])
    
    # SDNN plot
    ax2.plot(dates, sdnn, color=COLORS['capacity'], linewidth=2, marker='s', markersize=4)
    ax2.set_ylabel('SDNN (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(dates, sdnn, alpha=0.2, color=COLORS['capacity'])
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved HRV metrics plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_sleep_metrics(
    dates: Sequence,
    sleep_quality: Sequence[float],
    sleep_efficiency: Sequence[float],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot sleep metrics over time.
    
    Args:
        dates: Date/timestamp sequence
        sleep_quality: Sleep quality scores (0-100)
        sleep_efficiency: Sleep efficiency percentages (0-100)
        output_path: Path to save figure (optional)
        show: Whether to display plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)
    
    # Sleep quality
    ax1.plot(dates, sleep_quality, color=COLORS['sleep'], linewidth=2, marker='o', markersize=4)
    ax1.set_ylabel('Sleep Quality', fontsize=12, fontweight='bold')
    ax1.set_title('Sleep Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Fair')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sleep efficiency
    ax2.plot(dates, sleep_efficiency, color=COLORS['activity'], linewidth=2, marker='s', markersize=4)
    ax2.set_ylabel('Sleep Efficiency (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Good')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved sleep metrics plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_recovery_and_capacity(
    dates: Sequence,
    recovery_scores: Sequence[float],
    capacity_scores: Sequence[float],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot recovery and metabolic capacity scores.
    
    Args:
        dates: Date/timestamp sequence
        recovery_scores: Recovery scores (0-100)
        capacity_scores: Metabolic capacity scores (0-100)
        output_path: Path to save figure (optional)
        show: Whether to display plot
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)
    
    # Recovery score
    ax1.plot(dates, recovery_scores, color=COLORS['recovery'], linewidth=2, marker='o', markersize=4, label='Recovery')
    ax1.set_ylabel('Recovery Score', fontsize=12, fontweight='bold')
    ax1.set_title('Recovery and Metabolic Capacity', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(dates, recovery_scores, alpha=0.2, color=COLORS['recovery'])
    
    # Capacity score
    ax2.plot(dates, capacity_scores, color=COLORS['capacity'], linewidth=2, marker='s', markersize=4, label='Capacity')
    ax2.set_ylabel('Metabolic Capacity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(dates, capacity_scores, alpha=0.2, color=COLORS['capacity'])
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved recovery/capacity plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dashboard(
    dates: Sequence,
    metrics_dict: dict,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Create comprehensive dashboard with all metrics.
    
    Args:
        dates: Date/timestamp sequence
        metrics_dict: Dictionary with keys: rmssd, sdnn, sleep_quality, 
                     recovery_score, capacity_score, rhr
        output_path: Path to save figure (optional)
        show: Whether to display plot
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # HRV metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, metrics_dict.get('rmssd', []), color=COLORS['hrv'], linewidth=2, marker='o', markersize=3)
    ax1.set_ylabel('RMSSD (ms)', fontweight='bold')
    ax1.set_title('HRV (RMSSD)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(dates, metrics_dict.get('sdnn', []), color=COLORS['capacity'], linewidth=2, marker='o', markersize=3)
    ax2.set_ylabel('SDNN (ms)', fontweight='bold')
    ax2.set_title('HRV (SDNN)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Sleep and RHR
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(dates, metrics_dict.get('sleep_quality', []), color=COLORS['sleep'], linewidth=2, marker='s', markersize=3)
    ax3.set_ylabel('Sleep Quality', fontweight='bold')
    ax3.set_title('Sleep Quality Score', fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(dates, metrics_dict.get('rhr', []), color=COLORS['rhr'], linewidth=2, marker='o', markersize=3)
    ax4.set_ylabel('RHR (bpm)', fontweight='bold')
    ax4.set_title('Resting Heart Rate', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Recovery and Capacity
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(dates, metrics_dict.get('recovery_score', []), color=COLORS['recovery'], linewidth=2, marker='o', markersize=3)
    ax5.set_ylabel('Recovery Score', fontweight='bold')
    ax5.set_title('Daily Recovery', fontweight='bold')
    ax5.set_ylim([0, 105])
    ax5.set_xlabel('Date', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(dates, metrics_dict.get('capacity_score', []), color=COLORS['capacity'], linewidth=2, marker='s', markersize=3)
    ax6.set_ylabel('Capacity Score', fontweight='bold')
    ax6.set_title('Metabolic Capacity', fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.set_xlabel('Date', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    fig.suptitle('Longevity Metrics Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved dashboard to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_forecast(
    historical_dates: Sequence,
    historical_values: Sequence[float],
    forecast_dates: Sequence,
    forecast_values: Sequence[float],
    metric_name: str = "Capacity",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot historical data with forecast.
    
    Args:
        historical_dates: Historical date sequence
        historical_values: Historical values
        forecast_dates: Forecast date sequence
        forecast_values: Forecast values
        metric_name: Name of metric being forecast
        output_path: Path to save figure (optional)
        show: Whether to display plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # Historical
    ax.plot(historical_dates, historical_values, color=COLORS['capacity'], 
            linewidth=2, marker='o', markersize=4, label='Historical')
    
    # Forecast
    ax.plot(forecast_dates, forecast_values, color='red', 
            linewidth=2, linestyle='--', marker='s', markersize=4, label='Forecast')
    
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Forecast', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved forecast plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
