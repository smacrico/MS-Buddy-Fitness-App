# -*- coding: utf-8 -*-
"""Garmin Stress Data Analyzer - Local SQLite Version

Reads stress data from local garmin_summary SQLite database
and creates visualizations using Matplotlib and Seaborn.
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

# Database configuration
DB_PATH = "C:\smakryko\myHealthData\DBs\garmin_summary.db"  # Update this path to your database location

def get_db_connection():
    """Establish connection to the Garmin SQLite database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")
    return sqlite3.connect(DB_PATH)

def load_weekly_stress(weeks=104):
    """Load weekly stress data from weeks_summary table."""
    conn = get_db_connection()
    query = f"""
        SELECT * FROM weeks_summary 
        ORDER BY calendar_date DESC 
        LIMIT {weeks}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_daily_stress(days=365*3):
    """Load daily stress data from months_summary or daily table."""
    conn = get_db_connection()
    # Adjust the table and column names based on your actual schema
    query = f"""
        SELECT * FROM months_summary 
        ORDER BY calendar_date DESC 
        LIMIT {days}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

"""## Weekly stress

Load the last two years of weekly data from SQLite database.
"""

print("Loading weekly stress data from database...")
weekly_stress_df = load_weekly_stress(weeks=104)
weekly_stress = weekly_stress_df.to_dict('records')

"""Use Pandas and Matplotlib with Seaborn styling to graph"""

import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

df = pd.DataFrame(weekly_stress).sort_values("calendar_date")
df['calendar_date'] = pd.to_datetime(df['calendar_date'])

sns.set_theme()

plt.figure(figsize=(10, 6))

sns.lineplot(x=df["calendar_date"], y=df["value"])

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)

plt.xlabel(None)
plt.ylabel(None)
plt.title("Average Weekly Stress")

plt.show()

"""## Daily stress trends

Load daily stress data from SQLite database.
"""

from datetime import date, timedelta

print("Loading daily stress data from database...")
daily_stress_df = load_daily_stress(days=365 * 3)
daily_stress = daily_stress_df.to_dict('records')

"""Daily stats are going to have a *lot* of noise, so let's also graph the 28-day rolling average."""

sns.set_theme()

df = pd.DataFrame(daily_stress)
df['calendar_date'] = pd.to_datetime(df['calendar_date'])
df.set_index("calendar_date", inplace=True)
df = df.sort_index()

df["rolling_avg"] = df["overall_stress_level"].rolling(window=28).mean()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=df.index,
    y=df["overall_stress_level"],
    color="skyblue",
    label="Daily Stress Level"
)

sns.lineplot(
    x=df.index,
    y=df["rolling_avg"],
    color="r",
    label="28-day Rolling Average of Stress"
)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)
plt.xlim(df.index.min(), df.index.max())
plt.xlabel(None)
plt.ylabel(None)
plt.title("Overall Stress Level Over Time")
plt.legend()

plt.show()

"""We can also use `seasonal_decompose` to look at the 28-day trend."""

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(
    df["overall_stress_level"], model="additive", period=28
)
trend = result.trend.dropna()

min_date = df.index.min()
max_date = df.index.max()

def plot_subplot(ax, x, y, title, color, plot_type="line"):
    if plot_type == "line":
        sns.lineplot(ax=ax, x=x, y=y, color=color)
    elif plot_type == 'scatter':
        sns.scatterplot(ax=ax, x=x, y=y, color=color)

    ax.set_title(title)
    ax.set_xlim(min_date, max_date)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=45)

fig, axes = plt.subplots(2, 1, figsize=(15, 6))

plot_subplot(
    axes[0], df.index, df["overall_stress_level"],
    "Daily Stress Level", "skyblue", plot_type='scatter'
)
plot_subplot(axes[1], trend.index, trend, "28-Day Trend", "r")

plt.tight_layout()
plt.show()