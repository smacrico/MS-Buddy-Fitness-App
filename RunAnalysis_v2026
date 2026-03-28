# Refactor to use data from SQLite database
# (c) smacrico - Dec2024

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class RunningAnalysis:
    def __init__(
        self,
        db_path: str,
        output_dir: str = "c:/temp/logsFitnessApp",
        rest_hr: int = 60,
        max_hr: int = 190,
    ) -> None:
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rest_hr = rest_hr
        self.max_hr = max_hr

        self.training_log = pd.DataFrame()
        self.weekly_trimp = pd.DataFrame()

        self.training_log = self.load_training_data()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _save_plot(self, filename: str) -> Path:
        path = self.output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logging.info("Chart saved to: %s", path)
        return path

    @staticmethod
    def _safe_numeric_corr(series_a: pd.Series, series_b: pd.Series) -> float:
        try:
            valid = pd.concat([series_a, series_b], axis=1).dropna()
            if len(valid) < 2:
                return 0.0
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            return 0.0 if pd.isna(corr) else float(corr)
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_metric(series: pd.Series, higher_is_better: bool) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val

        if pd.isna(range_val) or range_val == 0:
            return pd.Series(0.5, index=series.index, dtype=float)

        normalized = (series - min_val) / range_val
        return normalized if higher_is_better else 1 - normalized

    def load_training_data(self) -> pd.DataFrame:
        """Load training data and calculate derived metrics.

        Assumption:
        - DB column `time` is stored in MINUTES.
        - avg_speed and max_speed are km/h.
        - distance is km.
        """
        query = """
        SELECT
            date,
            COALESCE(running_economy, 0) AS running_economy,
            COALESCE(vo2max, 0) AS vo2max,
            COALESCE(distance, 0) AS distance,
            COALESCE(time, 0) AS time,
            COALESCE(heart_rate, 0) AS heart_rate,
            COALESCE(avg_speed, 0) AS avg_speed,
            COALESCE(max_speed, 0) AS max_speed,
            COALESCE(HR_RS_Deviation_Index, 0) AS hr_rs_deviation,
            COALESCE(cardiacdrift, 0) AS cardiac_drift,
            COALESCE(running_economy / NULLIF(vo2max, 0), 0) AS efficiency_score,
            COALESCE(running_economy * (distance / NULLIF(time, 0)), 0) AS energy_cost,
            COALESCE(max_speed - avg_speed, 0) AS speed_reserve,
            COALESCE(avg_speed / NULLIF(max_speed, 0), 0) AS speed_consistency,
            COALESCE(60.0 / NULLIF(avg_speed, 0), 0) AS pace_per_km,
            COALESCE(avg_speed / NULLIF(heart_rate, 0), 0) AS speed_efficiency,
            COALESCE(running_economy / NULLIF(avg_speed, 0), 0) AS economy_at_speed,
            COALESCE(avg_speed * vo2max, 0) AS speed_vo2max_index
        FROM running_sessions
        """

        try:
            with self._connect() as conn:
                df = pd.read_sql_query(query, conn)

            if df.empty:
                logging.warning("No data loaded from database.")
                self.weekly_trimp = pd.DataFrame()
                return pd.DataFrame()

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            # Time is assumed to be in minutes
            df["duration_min"] = pd.to_numeric(df["time"], errors="coerce").fillna(0)
            df["duration_hr"] = df["duration_min"] / 60.0

            # TRIMP proxy
            hr_ratio = (df["heart_rate"] - self.rest_hr) / (self.max_hr - self.rest_hr)
            df["hr_ratio"] = hr_ratio.clip(lower=0, upper=1)
            df["TRIMP"] = df["duration_min"] * df["hr_ratio"]

            # Physiological Efficiency Score
            df["physio_efficiency"] = np.where(
                (df["hr_rs_deviation"] > 0) & (df["heart_rate"] > 0),
                (df["avg_speed"] / df["heart_rate"]) * (1 / df["hr_rs_deviation"]),
                0,
            )

            # Fatigue Index
            df["fatigue_index"] = np.where(
                df["avg_speed"] > 0,
                (df["hr_rs_deviation"] * df["cardiac_drift"]) / df["avg_speed"],
                0,
            )

            # Speed zones
            df["speed_zone"] = pd.cut(
                df["avg_speed"],
                bins=[0, 10, 14, np.inf],
                labels=["Slow", "Moderate", "Fast"],
                include_lowest=True,
            )

            # ISO year/week grouping for weekly load
            iso = df["date"].dt.isocalendar()
            df["iso_year"] = iso.year.astype(int)
            df["iso_week"] = iso.week.astype(int)
            df["week_label"] = (
                df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
            )

            weekly_trimp = (
                df.groupby(["iso_year", "iso_week", "week_label"], as_index=False)["TRIMP"]
                .sum()
                .rename(columns={"TRIMP": "weekly_trimp"})
                .sort_values(["iso_year", "iso_week"])
                .reset_index(drop=True)
            )

            weekly_trimp["acute_load"] = weekly_trimp["weekly_trimp"]
            weekly_trimp["chronic_load"] = (
                weekly_trimp["weekly_trimp"].rolling(window=4, min_periods=1).mean()
            )
            weekly_trimp["acwr"] = np.where(
                weekly_trimp["chronic_load"] > 0,
                weekly_trimp["acute_load"] / weekly_trimp["chronic_load"],
                0,
            )

            self.weekly_trimp = weekly_trimp

            logging.info("Loaded %s rows from database", len(df))
            logging.info("Columns available: %s", list(df.columns))
            return df

        except Exception as e:
            logging.exception("Error loading data: %s", e)
            self.weekly_trimp = pd.DataFrame()
            return pd.DataFrame()

    def add_session(
        self,
        date: str,
        running_economy: float,
        vo2max: float,
        distance: float,
        time: float,
        heart_rate: float,
        sport: str | None = None,
        cardiacdrift: float | None = None,
    ) -> None:
        """Add a new running session to the database.

        Assumption:
        - time is in minutes
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO running_sessions
                    (date, running_economy, vo2max, distance, time, heart_rate, sport, cardiacdrift)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date,
                        running_economy,
                        vo2max,
                        distance,
                        time,
                        heart_rate,
                        sport,
                        cardiacdrift,
                    ),
                )
                conn.commit()

            self.training_log = self.load_training_data()
            logging.info("Session added successfully.")

        except Exception as e:
            logging.exception("Error adding session: %s", e)

    def save_training_log_to_db(self) -> None:
        """Save derived training log DataFrame to SQLite database."""
        try:
            with self._connect() as conn:
                self.training_log.to_sql(
                    "training_logs",
                    conn,
                    if_exists="replace",
                    index=False,
                )
            logging.info("Training log successfully saved to database")
        except Exception as e:
            logging.exception("Error saving training log to database: %s", e)

    def create_monthly_summaries_table(self) -> None:
        """Create monthly_summaries table if it doesn't exist."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monthly_summaries (
                        year_month TEXT PRIMARY KEY,
                        sessions INTEGER,
                        running_economy_mean REAL,
                        running_economy_std REAL,
                        vo2max_mean REAL,
                        vo2max_std REAL,
                        distance_mean REAL,
                        distance_std REAL,
                        efficiency_score_mean REAL,
                        efficiency_score_std REAL,
                        heart_rate_mean REAL,
                        heart_rate_std REAL,
                        energy_cost_mean REAL,
                        energy_cost_std REAL,
                        trimp_mean REAL,
                        trimp_std REAL,
                        recovery_score_mean REAL,
                        recovery_score_std REAL,
                        readiness_score_mean REAL,
                        readiness_score_std REAL,
                        avg_speed_mean REAL,
                        avg_speed_std REAL,
                        max_speed_mean REAL,
                        max_speed_std REAL,
                        speed_reserve_mean REAL,
                        speed_reserve_std REAL,
                        hr_rs_deviation_mean REAL,
                        hr_rs_deviation_std REAL,
                        speed_efficiency_mean REAL,
                        speed_efficiency_std REAL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logging.exception("Error creating monthly_summaries table: %s", e)

    def calculate_monthly_metrics_averages(self) -> pd.DataFrame | None:
        """Calculate monthly averages for all metrics."""
        try:
            if self.training_log.empty:
                logging.warning("No training data available for monthly averages.")
                return None

            df = self.training_log.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["year_month"] = df["date"].dt.to_period("M")

            metrics = [
                "running_economy",
                "vo2max",
                "distance",
                "efficiency_score",
                "heart_rate",
                "energy_cost",
                "TRIMP",
            ]

            optional_metrics = [
                "recovery_score",
                "readiness_score",
                "avg_speed",
                "max_speed",
                "speed_reserve",
                "hr_rs_deviation",
                "speed_efficiency",
                "pace_per_km",
                "economy_at_speed",
                "physio_efficiency",
            ]

            for metric in optional_metrics:
                if metric in df.columns and metric not in metrics:
                    metrics.append(metric)

            monthly_averages = df.groupby("year_month")[metrics].agg(["mean", "std", "count"])
            return monthly_averages

        except Exception as e:
            logging.exception("Error calculating monthly averages: %s", e)
            return None

    def save_monthly_summaries(self) -> None:
        """Save monthly averages as one record per month in monthly_summaries table."""
        monthly_avg = self.calculate_monthly_metrics_averages()
        if monthly_avg is None or monthly_avg.empty:
            return

        try:
            df = self.training_log.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            monthly_sessions = (
                df.groupby(df["date"].dt.to_period("M")).size().to_dict()
            )

            def get_val(month_key, metric: str, stat: str) -> float | None:
                if metric in monthly_avg.columns.get_level_values(0):
                    val = monthly_avg.loc[month_key, (metric, stat)]
                    return None if pd.isna(val) else float(val)
                return None

            with self._connect() as conn:
                cursor = conn.cursor()

                for year_month in monthly_avg.index:
                    sessions = int(monthly_sessions.get(year_month, 0))

                    cursor.execute(
                        """
                        INSERT INTO monthly_summaries (
                            year_month,
                            sessions,
                            running_economy_mean, running_economy_std,
                            vo2max_mean, vo2max_std,
                            distance_mean, distance_std,
                            efficiency_score_mean, efficiency_score_std,
                            heart_rate_mean, heart_rate_std,
                            energy_cost_mean, energy_cost_std,
                            trimp_mean, trimp_std,
                            recovery_score_mean, recovery_score_std,
                            readiness_score_mean, readiness_score_std,
                            avg_speed_mean, avg_speed_std,
                            max_speed_mean, max_speed_std,
                            speed_reserve_mean, speed_reserve_std,
                            hr_rs_deviation_mean, hr_rs_deviation_std,
                            speed_efficiency_mean, speed_efficiency_std
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(year_month) DO UPDATE SET
                            sessions=excluded.sessions,
                            running_economy_mean=excluded.running_economy_mean,
                            running_economy_std=excluded.running_economy_std,
                            vo2max_mean=excluded.vo2max_mean,
                            vo2max_std=excluded.vo2max_std,
                            distance_mean=excluded.distance_mean,
                            distance_std=excluded.distance_std,
                            efficiency_score_mean=excluded.efficiency_score_mean,
                            efficiency_score_std=excluded.efficiency_score_std,
                            heart_rate_mean=excluded.heart_rate_mean,
                            heart_rate_std=excluded.heart_rate_std,
                            energy_cost_mean=excluded.energy_cost_mean,
                            energy_cost_std=excluded.energy_cost_std,
                            trimp_mean=excluded.trimp_mean,
                            trimp_std=excluded.trimp_std,
                            recovery_score_mean=excluded.recovery_score_mean,
                            recovery_score_std=excluded.recovery_score_std,
                            readiness_score_mean=excluded.readiness_score_mean,
                            readiness_score_std=excluded.readiness_score_std,
                            avg_speed_mean=excluded.avg_speed_mean,
                            avg_speed_std=excluded.avg_speed_std,
                            max_speed_mean=excluded.max_speed_mean,
                            max_speed_std=excluded.max_speed_std,
                            speed_reserve_mean=excluded.speed_reserve_mean,
                            speed_reserve_std=excluded.speed_reserve_std,
                            hr_rs_deviation_mean=excluded.hr_rs_deviation_mean,
                            hr_rs_deviation_std=excluded.hr_rs_deviation_std,
                            speed_efficiency_mean=excluded.speed_efficiency_mean,
                            speed_efficiency_std=excluded.speed_efficiency_std
                        """,
                        (
                            str(year_month),
                            sessions,
                            get_val(year_month, "running_economy", "mean"),
                            get_val(year_month, "running_economy", "std"),
                            get_val(year_month, "vo2max", "mean"),
                            get_val(year_month, "vo2max", "std"),
                            get_val(year_month, "distance", "mean"),
                            get_val(year_month, "distance", "std"),
                            get_val(year_month, "efficiency_score", "mean"),
                            get_val(year_month, "efficiency_score", "std"),
                            get_val(year_month, "heart_rate", "mean"),
                            get_val(year_month, "heart_rate", "std"),
                            get_val(year_month, "energy_cost", "mean"),
                            get_val(year_month, "energy_cost", "std"),
                            get_val(year_month, "TRIMP", "mean"),
                            get_val(year_month, "TRIMP", "std"),
                            get_val(year_month, "recovery_score", "mean"),
                            get_val(year_month, "recovery_score", "std"),
                            get_val(year_month, "readiness_score", "mean"),
                            get_val(year_month, "readiness_score", "std"),
                            get_val(year_month, "avg_speed", "mean"),
                            get_val(year_month, "avg_speed", "std"),
                            get_val(year_month, "max_speed", "mean"),
                            get_val(year_month, "max_speed", "std"),
                            get_val(year_month, "speed_reserve", "mean"),
                            get_val(year_month, "speed_reserve", "std"),
                            get_val(year_month, "hr_rs_deviation", "mean"),
                            get_val(year_month, "hr_rs_deviation", "std"),
                            get_val(year_month, "speed_efficiency", "mean"),
                            get_val(year_month, "speed_efficiency", "std"),
                        ),
                    )

                conn.commit()

            logging.info("Monthly summaries saved successfully")

        except Exception as e:
            logging.exception("Error saving monthly summaries: %s", e)

    def create_metrics_breakdown_table(self) -> None:
        """Create metrics_breakdown table if it doesn't exist."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics_breakdown (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        overall_score REAL,
                        running_economy_normalized REAL,
                        running_economy_weighted REAL,
                        running_economy_raw_mean REAL,
                        running_economy_raw_std REAL,
                        vo2max_normalized REAL,
                        vo2max_weighted REAL,
                        vo2max_raw_mean REAL,
                        vo2max_raw_std REAL,
                        distance_normalized REAL,
                        distance_weighted REAL,
                        distance_raw_mean REAL,
                        distance_raw_std REAL,
                        efficiency_score_normalized REAL,
                        efficiency_score_weighted REAL,
                        efficiency_score_raw_mean REAL,
                        efficiency_score_raw_std REAL,
                        heart_rate_normalized REAL,
                        heart_rate_weighted REAL,
                        heart_rate_raw_mean REAL,
                        heart_rate_raw_std REAL,
                        running_economy_trend REAL,
                        distance_progression REAL,
                        avg_speed_mean REAL,
                        avg_speed_std REAL,
                        max_speed_mean REAL,
                        max_speed_std REAL,
                        speed_reserve_mean REAL,
                        speed_reserve_std REAL,
                        speed_consistency_mean REAL,
                        speed_consistency_std REAL,
                        pace_per_km_mean REAL,
                        pace_per_km_std REAL,
                        speed_efficiency_mean REAL,
                        speed_efficiency_std REAL,
                        economy_at_speed_mean REAL,
                        economy_at_speed_std REAL,
                        speed_vo2max_index_mean REAL,
                        speed_vo2max_index_std REAL,
                        hr_rs_deviation_mean REAL,
                        hr_rs_deviation_std REAL,
                        cardiac_drift_mean REAL,
                        cardiac_drift_std REAL,
                        physio_efficiency_mean REAL,
                        physio_efficiency_std REAL,
                        fatigue_index_mean REAL,
                        fatigue_index_std REAL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logging.exception("Error creating metrics_breakdown table: %s", e)

    def save_metrics_breakdown(self, training_score: dict) -> None:
        """Save metrics breakdown to database."""
        try:
            if self.training_log.empty:
                logging.warning("Cannot save metrics breakdown: no training data available")
                return

            if not training_score or "overall_score" not in training_score:
                logging.error("Invalid training_score structure")
                return

            current_date = datetime.now().strftime("%Y-%m-%d")
            metrics = training_score.get("metric_breakdown", {})
            trends = training_score.get("performance_trends", {})

            def safe_metric_value(metric_dict: dict, key: str, default: float = 0.0) -> float:
                try:
                    if metric_dict and key in metric_dict:
                        val = metric_dict[key]
                        return default if pd.isna(val) else float(val)
                    return default
                except (TypeError, ValueError, KeyError):
                    return default

            def safe_stat_from_df(col_name: str, stat_func) -> float:
                try:
                    if col_name in self.training_log.columns:
                        result = stat_func(self.training_log[col_name])
                        return 0.0 if pd.isna(result) else float(result)
                    return 0.0
                except Exception:
                    return 0.0

            values_to_insert = (
                current_date,
                float(training_score["overall_score"]),
                safe_metric_value(metrics.get("running_economy", {}), "normalized_value"),
                safe_metric_value(metrics.get("running_economy", {}), "weighted_value"),
                safe_metric_value(metrics.get("running_economy", {}), "raw_mean"),
                safe_metric_value(metrics.get("running_economy", {}), "raw_std"),
                safe_metric_value(metrics.get("vo2max", {}), "normalized_value"),
                safe_metric_value(metrics.get("vo2max", {}), "weighted_value"),
                safe_metric_value(metrics.get("vo2max", {}), "raw_mean"),
                safe_metric_value(metrics.get("vo2max", {}), "raw_std"),
                safe_metric_value(metrics.get("distance", {}), "normalized_value"),
                safe_metric_value(metrics.get("distance", {}), "weighted_value"),
                safe_metric_value(metrics.get("distance", {}), "raw_mean"),
                safe_metric_value(metrics.get("distance", {}), "raw_std"),
                safe_metric_value(metrics.get("efficiency_score", {}), "normalized_value"),
                safe_metric_value(metrics.get("efficiency_score", {}), "weighted_value"),
                safe_metric_value(metrics.get("efficiency_score", {}), "raw_mean"),
                safe_metric_value(metrics.get("efficiency_score", {}), "raw_std"),
                safe_metric_value(metrics.get("heart_rate", {}), "normalized_value"),
                safe_metric_value(metrics.get("heart_rate", {}), "weighted_value"),
                safe_metric_value(metrics.get("heart_rate", {}), "raw_mean"),
                safe_metric_value(metrics.get("heart_rate", {}), "raw_std"),
                safe_metric_value(trends, "running_economy_trend"),
                safe_metric_value(trends, "distance_progression"),
                safe_stat_from_df("avg_speed", pd.Series.mean),
                safe_stat_from_df("avg_speed", pd.Series.std),
                safe_stat_from_df("max_speed", pd.Series.mean),
                safe_stat_from_df("max_speed", pd.Series.std),
                safe_stat_from_df("speed_reserve", pd.Series.mean),
                safe_stat_from_df("speed_reserve", pd.Series.std),
                safe_stat_from_df("speed_consistency", pd.Series.mean),
                safe_stat_from_df("speed_consistency", pd.Series.std),
                safe_stat_from_df("pace_per_km", pd.Series.mean),
                safe_stat_from_df("pace_per_km", pd.Series.std),
                safe_stat_from_df("speed_efficiency", pd.Series.mean),
                safe_stat_from_df("speed_efficiency", pd.Series.std),
                safe_stat_from_df("economy_at_speed", pd.Series.mean),
                safe_stat_from_df("economy_at_speed", pd.Series.std),
                safe_stat_from_df("speed_vo2max_index", pd.Series.mean),
                safe_stat_from_df("speed_vo2max_index", pd.Series.std),
                safe_stat_from_df("hr_rs_deviation", pd.Series.mean),
                safe_stat_from_df("hr_rs_deviation", pd.Series.std),
                safe_stat_from_df("cardiac_drift", pd.Series.mean),
                safe_stat_from_df("cardiac_drift", pd.Series.std),
                safe_stat_from_df("physio_efficiency", pd.Series.mean),
                safe_stat_from_df("physio_efficiency", pd.Series.std),
                safe_stat_from_df("fatigue_index", pd.Series.mean),
                safe_stat_from_df("fatigue_index", pd.Series.std),
            )

            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO metrics_breakdown (
                        date,
                        overall_score,
                        running_economy_normalized,
                        running_economy_weighted,
                        running_economy_raw_mean,
                        running_economy_raw_std,
                        vo2max_normalized,
                        vo2max_weighted,
                        vo2max_raw_mean,
                        vo2max_raw_std,
                        distance_normalized,
                        distance_weighted,
                        distance_raw_mean,
                        distance_raw_std,
                        efficiency_score_normalized,
                        efficiency_score_weighted,
                        efficiency_score_raw_mean,
                        efficiency_score_raw_std,
                        heart_rate_normalized,
                        heart_rate_weighted,
                        heart_rate_raw_mean,
                        heart_rate_raw_std,
                        running_economy_trend,
                        distance_progression,
                        avg_speed_mean,
                        avg_speed_std,
                        max_speed_mean,
                        max_speed_std,
                        speed_reserve_mean,
                        speed_reserve_std,
                        speed_consistency_mean,
                        speed_consistency_std,
                        pace_per_km_mean,
                        pace_per_km_std,
                        speed_efficiency_mean,
                        speed_efficiency_std,
                        economy_at_speed_mean,
                        economy_at_speed_std,
                        speed_vo2max_index_mean,
                        speed_vo2max_index_std,
                        hr_rs_deviation_mean,
                        hr_rs_deviation_std,
                        cardiac_drift_mean,
                        cardiac_drift_std,
                        physio_efficiency_mean,
                        physio_efficiency_std,
                        fatigue_index_mean,
                        fatigue_index_std
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values_to_insert,
                )
                conn.commit()

            logging.info("Metrics breakdown saved successfully for %s", current_date)

        except sqlite3.Error as db_err:
            logging.exception("Database error saving metrics breakdown: %s", db_err)
        except Exception as e:
            logging.exception("Error saving metrics breakdown: %s", e)

    def calculate_recovery_and_readiness(self) -> pd.DataFrame:
        df = self.training_log.copy()

        if df.empty:
            return pd.DataFrame(columns=["date", "recovery_score", "readiness_score"])

        if "resting_hr" not in df.columns:
            df["resting_hr"] = pd.Series(np.nan, index=df.index)
        if "sleep_quality" not in df.columns:
            df["sleep_quality"] = 3
        if "fatigue_level" not in df.columns:
            df["fatigue_level"] = 5

        rhr_baseline = df["resting_hr"].dropna().mean() if df["resting_hr"].notna().any() else 60
        trimp_baseline = (
            df["TRIMP"].rolling(window=4, min_periods=1).mean()
            if "TRIMP" in df.columns
            else pd.Series(np.repeat(50, len(df)), index=df.index)
        )

        df["rhr_score"] = 1 - ((df["resting_hr"] - rhr_baseline) / rhr_baseline)
        df["load_score"] = 1 - (df["TRIMP"] / (trimp_baseline + 1e-8))
        df["sleep_score"] = df["sleep_quality"] / 5
        df["fatigue_score"] = 1 - (df["fatigue_level"] / 10)

        for col in ["rhr_score", "load_score", "sleep_score", "fatigue_score"]:
            df[col] = df[col].clip(0, 1)

        df["recovery_score"] = (
            0.3 * df["rhr_score"].fillna(1)
            + 0.3 * df["load_score"].fillna(1)
            + 0.2 * df["sleep_score"].fillna(0.6)
            + 0.2 * df["fatigue_score"].fillna(0.5)
        )

        df["readiness_score"] = (
            0.5 * df["recovery_score"]
            + 0.3 * df["load_score"].fillna(1)
            + 0.2 * df["sleep_score"].fillna(0.6)
        )

        df["recovery_score"] = df["recovery_score"].clip(0, 1)
        df["readiness_score"] = df["readiness_score"].clip(0, 1)

        self.training_log = df
        return df[["date", "recovery_score", "readiness_score"]]

    def calculate_training_zones(self, running_economy: float) -> dict[str, tuple[float, float]]:
        """Calculate training zones based on running economy."""
        return {
            "Recovery": (0.6 * running_economy, 0.7 * running_economy),
            "Endurance": (0.7 * running_economy, 0.8 * running_economy),
            "Tempo": (0.8 * running_economy, 0.9 * running_economy),
            "Threshold": (0.9 * running_economy, running_economy),
            "VO2Max": (running_economy, 1.1 * running_economy),
        }

    def print_training_zones(self, running_economy: float) -> None:
        training_zones = self.calculate_training_zones(running_economy)
        print("\nTraining Zones based on Running Economy:")
        for zone, (lower, upper) in training_zones.items():
            print(f"{zone}: {lower:.1f} - {upper:.1f}")

    def calculate_training_score(self) -> dict | None:
        """
        Calculate a comprehensive training score based on multiple performance metrics.
        Returns a dictionary with detailed score breakdown and overall training score.
        """
        try:
            if self.training_log.empty:
                return None

            normalized_data = self.training_log.copy()
            normalized_data["date"] = pd.to_datetime(normalized_data["date"], errors="coerce")
            normalized_data = normalized_data.dropna(subset=["date"]).sort_values("date")

            metrics = {
                "running_economy": {"weight": 0.25, "higher_is_better": True},
                "vo2max": {"weight": 0.20, "higher_is_better": True},
                "distance": {"weight": 0.15, "higher_is_better": True},
                "efficiency_score": {"weight": 0.20, "higher_is_better": True},
                "heart_rate": {"weight": 0.20, "higher_is_better": False},
            }

            normalized_scores = {}
            for metric, config in metrics.items():
                normalized_scores[metric] = self._normalize_metric(
                    normalized_data[metric], config["higher_is_better"]
                )

            weighted_scores = {}
            for metric, config in metrics.items():
                weighted_scores[metric] = normalized_scores[metric] * config["weight"]

            overall_score = sum(weighted_scores[metric].mean() for metric in metrics) * 100

            date_num = normalized_data["date"].map(pd.Timestamp.toordinal)

            analysis = {
                "overall_score": float(overall_score),
                "metric_breakdown": {
                    metric: {
                        "normalized_value": float(normalized_scores[metric].mean()),
                        "weighted_value": float(weighted_scores[metric].mean()),
                        "raw_mean": float(normalized_data[metric].mean()),
                        "raw_std": float(normalized_data[metric].std())
                        if not pd.isna(normalized_data[metric].std())
                        else 0.0,
                    }
                    for metric in metrics
                },
                "performance_trends": {
                    "running_economy_trend": self._safe_numeric_corr(
                        normalized_scores["running_economy"], date_num
                    ),
                    "distance_progression": self._safe_numeric_corr(
                        normalized_scores["distance"], date_num
                    ),
                },
            }

            return analysis

        except Exception as e:
            logging.exception("Error calculating training score: %s", e)
            return None

    def calculate_session_scores(self) -> pd.Series:
        """Create a per-session score for time-series visualization."""
        if self.training_log.empty:
            return pd.Series(dtype=float)

        df = self.training_log.copy().sort_values("date")
        metrics = {
            "running_economy": {"weight": 0.25, "higher_is_better": True},
            "vo2max": {"weight": 0.20, "higher_is_better": True},
            "distance": {"weight": 0.15, "higher_is_better": True},
            "efficiency_score": {"weight": 0.20, "higher_is_better": True},
            "heart_rate": {"weight": 0.20, "higher_is_better": False},
        }

        total = pd.Series(0.0, index=df.index)
        for metric, config in metrics.items():
            norm = self._normalize_metric(df[metric], config["higher_is_better"])
            total += norm * config["weight"]

        return total * 100

    def visualize_training_load(self) -> None:
        try:
            if self.training_log.empty or self.weekly_trimp.empty:
                logging.warning("No training data available for visualization.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            axes[0].plot(self.training_log["date"], self.training_log["TRIMP"], marker="o")
            axes[0].set_title("TRIMP per Session Over Time")
            axes[0].set_xlabel("Date")
            axes[0].set_ylabel("TRIMP Score")
            axes[0].tick_params(axis="x", rotation=45)

            weeks = self.weekly_trimp["week_label"]
            axes[1].plot(weeks, self.weekly_trimp["weekly_trimp"], label="Weekly TRIMP Load", marker="o")
            axes[1].plot(weeks, self.weekly_trimp["acute_load"], label="Acute Load (1 week)", linestyle="--")
            axes[1].plot(weeks, self.weekly_trimp["chronic_load"], label="Chronic Load (4 week avg)", linestyle="--")
            axes[1].plot(weeks, self.weekly_trimp["acwr"], label="ACWR", linestyle="-.")
            axes[1].axhline(1.3, color="red", linestyle=":", label="Upper ACWR Threshold (~1.3)")
            axes[1].axhline(0.8, color="green", linestyle=":", label="Lower ACWR Threshold (~0.8)")
            axes[1].set_title("Weekly Training Load and ACWR")
            axes[1].set_xlabel("ISO Week")
            axes[1].set_ylabel("Load / Ratio")
            axes[1].legend()
            axes[1].grid(True)
            axes[1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            self._save_plot("training_load.png")
            plt.show()

        except Exception as e:
            logging.exception("Error during visualization: %s", e)

    def visualize_trends(self) -> None:
        """Create visualizations of running data."""
        try:
            if self.training_log.empty:
                logging.warning("No training data available for visualization.")
                return

            df = self.training_log.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.plot(df["date"], df["running_economy"], "b-o")
            plt.title("Running Economy Trend")
            plt.xticks(rotation=45)
            plt.ylabel("Running Economy")

            plt.subplot(2, 2, 2)
            plt.plot(df["date"], df["efficiency_score"], "g-o")
            plt.title("Efficiency Score Trend")
            plt.xticks(rotation=45)
            plt.ylabel("Efficiency Score")

            plt.subplot(2, 2, 3)
            plt.scatter(df["distance"], df["energy_cost"])
            plt.title("Energy Cost vs Distance")
            plt.xlabel("Distance (km)")
            plt.ylabel("Energy Cost")

            plt.subplot(2, 2, 4)
            plt.scatter(df["heart_rate"], df["running_economy"])
            plt.title("Heart Rate vs Running Economy")
            plt.xlabel("Heart Rate (bpm)")
            plt.ylabel("Running Economy")

            plt.tight_layout()
            self._save_plot("trends.png")
            plt.show()

        except Exception as e:
            logging.exception("Visualization error: %s", e)

    def visualize_recovery_and_readiness(self) -> None:
        self.calculate_recovery_and_readiness()
        if self.training_log.empty:
            return

        plt.figure(figsize=(12, 5))
        plt.plot(self.training_log["date"], self.training_log["recovery_score"], label="Recovery")
        plt.plot(self.training_log["date"], self.training_log["readiness_score"], label="Readiness")
        plt.axhline(0.7, color="orange", linestyle="--", label="Caution threshold")
        plt.xlabel("Date")
        plt.ylabel("Score (0–1)")
        plt.title("Recovery and Readiness Over Time")
        plt.legend()
        plt.tight_layout()
        self._save_plot("recovery_readiness.png")
        plt.show()

    def advanced_visualizations(self) -> None:
        """Create advanced performance visualizations."""
        try:
            if self.training_log.empty:
                logging.warning("No data available for advanced visualizations.")
                return

            df = self.training_log.copy().sort_values("date")
            df["cumulative_distance"] = df["distance"].cumsum()
            df["running_economy_ma"] = df["running_economy"].rolling(window=3, min_periods=1).mean()
            df["month"] = df["date"].dt.month

            plt.figure(figsize=(20, 15))

            plt.subplot(2, 3, 1)
            plt.plot(df["date"], df["cumulative_distance"], "b-o")
            plt.title("Cumulative Running Distance")
            plt.xlabel("Date")
            plt.ylabel("Total Distance (km)")
            plt.xticks(rotation=45)

            plt.subplot(2, 3, 2)
            plt.plot(df["date"], df["running_economy"], "g-", label="Original")
            plt.plot(df["date"], df["running_economy_ma"], "r-", label="3-Session Moving Avg")
            plt.title("Running Economy Trend")
            plt.xlabel("Date")
            plt.ylabel("Running Economy")
            plt.legend()
            plt.xticks(rotation=45)

            plt.subplot(2, 3, 3)
            pace = np.where(df["distance"] > 0, df["time"] / df["distance"], np.nan)
            plt.scatter(pace, df["heart_rate"], alpha=0.7)
            plt.title("Pace vs Heart Rate")
            plt.xlabel("Pace (min/km)")
            plt.ylabel("Heart Rate (bpm)")

            plt.subplot(2, 3, 4)
            try:
                valid_rows = df[(df["running_economy"].notna()) & (df["vo2max"].notna())]
                if not valid_rows.empty:
                    reference_re = valid_rows["running_economy"].median()
                    zones = self.calculate_training_zones(reference_re)

                    zone_durations = {}
                    for zone, (lower, upper) in zones.items():
                        count = len(valid_rows[
                            (valid_rows["running_economy"] >= lower)
                            & (valid_rows["running_economy"] < upper)
                        ])
                        if count > 0:
                            zone_durations[zone] = count

                    if zone_durations:
                        plt.pie(
                            list(zone_durations.values()),
                            labels=list(zone_durations.keys()),
                            autopct="%1.1f%%",
                        )
                        plt.title("Training Zones Distribution")
                    else:
                        plt.text(0.5, 0.5, "No valid zone data", ha="center", va="center")
                else:
                    plt.text(0.5, 0.5, "No valid training data", ha="center", va="center")
            except Exception as e:
                logging.warning("Error creating pie chart: %s", e)
                plt.text(0.5, 0.5, "Error creating pie chart", ha="center", va="center")

            plt.subplot(2, 3, 5, polar=True)
            metrics = ["running_economy", "vo2max", "distance", "efficiency_score", "heart_rate"]
            normalized_metrics = pd.DataFrame(
                {
                    metric: self._normalize_metric(df[metric], metric != "heart_rate")
                    for metric in metrics
                }
            )
            avg_metrics = normalized_metrics.mean()

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            values = avg_metrics.values
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            plt.polar(angles, values, "o-", linewidth=2)
            plt.fill(angles, values, alpha=0.25)
            plt.xticks(angles[:-1], metrics)
            plt.title("Performance Metrics Radar Chart")

            plt.subplot(2, 3, 6)
            seasonal_performance = df.groupby("month")["running_economy"].mean()
            plt.imshow([seasonal_performance.values], cmap="YlOrRd", aspect="auto")
            plt.colorbar(label="Avg Running Economy")
            plt.title("Seasonal Performance Heatmap")
            plt.xlabel("Month")
            plt.xticks(range(len(seasonal_performance)), seasonal_performance.index)

            plt.tight_layout()
            self._save_plot("advanced_metrics.png")
            plt.show()

        except Exception as e:
            logging.exception("Error in advanced visualizations: %s", e)

    def visualize_score_impact_over_time(self, extra_scores: dict | None = None) -> None:
        """
        Visualize per-session score over time with optional extra score overlays.
        """
        try:
            if self.training_log.empty:
                logging.warning("No data available for score visualization.")
                return

            df = self.training_log.sort_values("date").copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["Session Score"] = self.calculate_session_scores()

            plt.figure(figsize=(14, 7))
            plt.plot(df["date"], df["Session Score"], label="Session Training Score", linewidth=2)

            if extra_scores:
                for label, col in extra_scores.items():
                    if col in df.columns:
                        plt.plot(df["date"], df[col], linestyle="--", label=label)

            plt.xlabel("Date")
            plt.ylabel("Score")
            plt.title("Comparison of Scoring Calculations Over Time")
            plt.legend()
            plt.tight_layout()
            self._save_plot("score_impact.png")
            plt.show()

        except Exception as e:
            logging.exception("Error visualizing score impact over time: %s", e)

    def visualize_speed_metrics(self) -> None:
        """Create comprehensive speed-related visualizations."""
        try:
            if self.training_log.empty:
                logging.warning("No data available")
                return

            df = self.training_log.copy().sort_values("date")

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle("Speed Metrics Analysis", fontsize=16, fontweight="bold")

            ax1 = axes[0, 0]
            ax1.plot(df["date"], df["avg_speed"], marker="o", color="blue", label="Avg Speed")
            ax1.plot(df["date"], df["max_speed"], marker="s", color="red", alpha=0.6, label="Max Speed")
            ax1.set_title("Speed Trends Over Time")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Speed (km/h)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

            ax2 = axes[0, 1]
            ax2.plot(df["date"], df["speed_reserve"], marker="o", color="green")
            ax2.set_title("Speed Reserve (Max - Avg)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Speed Reserve (km/h)")
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

            ax3 = axes[1, 0]
            date_color = df["date"].map(pd.Timestamp.toordinal)
            scatter = ax3.scatter(
                df["heart_rate"],
                df["avg_speed"],
                c=date_color,
                cmap="viridis",
                s=100,
                alpha=0.6,
            )
            ax3.set_title("Speed vs Heart Rate (colored by time)")
            ax3.set_xlabel("Heart Rate (bpm)")
            ax3.set_ylabel("Average Speed (km/h)")
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label="Date progression")

            ax4 = axes[1, 1]
            ax4.plot(df["date"], df["speed_efficiency"], marker="o", color="purple")
            ax4.set_title("Speed Efficiency (Speed per HR unit)")
            ax4.set_xlabel("Date")
            ax4.set_ylabel("Speed/HR (km/h per bpm)")
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

            ax5 = axes[2, 0]
            ax5.plot(df["date"], df["pace_per_km"], marker="o", color="orange")
            ax5.set_title("Pace Progression")
            ax5.set_xlabel("Date")
            ax5.set_ylabel("Pace (min/km)")
            ax5.invert_yaxis()
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis="x", rotation=45)

            ax6 = axes[2, 1]
            if "speed_zone" in df.columns:
                zone_counts = df["speed_zone"].value_counts()
                ax6.bar(zone_counts.index.astype(str), zone_counts.values, color=["#3498db", "#2ecc71", "#e74c3c"])
                ax6.set_title("Training Sessions by Speed Zone")
                ax6.set_xlabel("Speed Zone")
                ax6.set_ylabel("Number of Sessions")
                ax6.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            self._save_plot("speed_metrics.png")
            plt.show()

        except Exception as e:
            logging.exception("Error in speed visualization: %s", e)

    def visualize_hr_rs_deviation(self) -> None:
        """Create HR-RS Deviation Index visualizations."""
        try:
            if self.training_log.empty:
                logging.warning("No data available")
                return

            valid_data = self.training_log[self.training_log["hr_rs_deviation"] > 0].copy()
            if valid_data.empty:
                logging.warning("No HR-RS Deviation data available")
                return

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle("HR-RS Deviation Index Analysis", fontsize=16, fontweight="bold")

            ax1 = axes[0, 0]
            ax1.plot(valid_data["date"], valid_data["hr_rs_deviation"], marker="o", color="red", linewidth=2)
            rolling_avg = valid_data["hr_rs_deviation"].rolling(window=3, min_periods=1).mean()
            ax1.plot(valid_data["date"], rolling_avg, linestyle="--", color="blue", linewidth=2, label="3-session avg")
            ax1.set_title("HR-RS Deviation Index Over Time")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Deviation Index")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

            ax2 = axes[0, 1]
            ax2.scatter(valid_data["hr_rs_deviation"], valid_data["avg_speed"], s=100, alpha=0.6, c="green")
            ax2.set_title("HR-RS Deviation vs Speed Performance")
            ax2.set_xlabel("HR-RS Deviation Index")
            ax2.set_ylabel("Average Speed (km/h)")
            ax2.grid(True, alpha=0.3)

            if len(valid_data) >= 3 and valid_data["hr_rs_deviation"].nunique() > 1:
                z = np.polyfit(valid_data["hr_rs_deviation"], valid_data["avg_speed"], 1)
                p = np.poly1d(z)
                xvals = np.sort(valid_data["hr_rs_deviation"].values)
                ax2.plot(xvals, p(xvals), "r--", alpha=0.8, linewidth=2, label="Trend")
                ax2.legend()

            ax3 = axes[1, 0]
            ax3.hist(valid_data["hr_rs_deviation"], bins=15, color="purple", alpha=0.7, edgecolor="black")
            ax3.axvline(valid_data["hr_rs_deviation"].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
            ax3.set_title("HR-RS Deviation Distribution")
            ax3.set_xlabel("Deviation Index")
            ax3.set_ylabel("Frequency")
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis="y")

            ax4 = axes[1, 1]
            if "TRIMP" in valid_data.columns:
                ax4.scatter(valid_data["TRIMP"], valid_data["hr_rs_deviation"], s=100, alpha=0.6, c="orange")
                ax4.set_title("HR-RS Deviation vs Training Load (TRIMP)")
                ax4.set_xlabel("TRIMP Score")
                ax4.set_ylabel("HR-RS Deviation Index")
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, "TRIMP data not available", ha="center", va="center", transform=ax4.transAxes)

            plt.tight_layout()
            self._save_plot("hr_rs_deviation.png")
            plt.show()

        except Exception as e:
            logging.exception("Error in HR-RS deviation visualization: %s", e)

    def create_performance_dashboard(self) -> None:
        """Create comprehensive dashboard with all new metrics."""
        try:
            if self.training_log.empty:
                logging.warning("No data available")
                return

            df = self.training_log.copy().sort_values("date")

            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            fig.suptitle("Comprehensive Running Performance Dashboard", fontsize=18, fontweight="bold")

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(df["date"], df["avg_speed"], marker="o", color="blue")
            ax1.set_title("Average Speed Trend")
            ax1.set_ylabel("Speed (km/h)")
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(df["date"], df["speed_reserve"], marker="o", color="green")
            ax2.set_title("Speed Reserve")
            ax2.set_ylabel("km/h")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(df["date"], df["pace_per_km"], marker="o", color="orange")
            ax3.set_title("Pace")
            ax3.set_ylabel("min/km")
            ax3.invert_yaxis()
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3)

            valid_hr_rs = df[df["hr_rs_deviation"] > 0]

            ax4 = fig.add_subplot(gs[1, 0])
            if not valid_hr_rs.empty:
                ax4.plot(valid_hr_rs["date"], valid_hr_rs["hr_rs_deviation"], marker="o", color="red")
                ax4.set_title("HR-RS Deviation Index")
                ax4.set_ylabel("Index")
                ax4.tick_params(axis="x", rotation=45)
                ax4.grid(True, alpha=0.3)

            ax5 = fig.add_subplot(gs[1, 1])
            if not valid_hr_rs.empty:
                ax5.scatter(valid_hr_rs["hr_rs_deviation"], valid_hr_rs["avg_speed"], s=100, alpha=0.6, c="purple")
                ax5.set_title("Deviation vs Speed")
                ax5.set_xlabel("HR-RS Deviation")
                ax5.set_ylabel("Speed (km/h)")
                ax5.grid(True, alpha=0.3)

            ax6 = fig.add_subplot(gs[1, 2])
            if not valid_hr_rs.empty:
                ax6.hist(valid_hr_rs["hr_rs_deviation"], bins=15, color="purple", alpha=0.7, edgecolor="black")
                ax6.set_title("Deviation Distribution")
                ax6.set_xlabel("Index")
                ax6.grid(True, alpha=0.3, axis="y")

            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(df["date"], df["speed_efficiency"], marker="o", color="teal")
            ax7.set_title("Speed Efficiency (Speed/HR)")
            ax7.set_ylabel("km/h per bpm")
            ax7.tick_params(axis="x", rotation=45)
            ax7.grid(True, alpha=0.3)

            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(df["date"], df["economy_at_speed"], marker="o", color="brown")
            ax8.set_title("Economy at Speed")
            ax8.set_ylabel("RE / Speed")
            ax8.tick_params(axis="x", rotation=45)
            ax8.grid(True, alpha=0.3)

            ax9 = fig.add_subplot(gs[2, 2])
            if "physio_efficiency" in df.columns:
                valid_physio = df[df["physio_efficiency"] > 0]
                if not valid_physio.empty:
                    ax9.plot(valid_physio["date"], valid_physio["physio_efficiency"], marker="o", color="darkgreen")
                    ax9.set_title("Physiological Efficiency")
                    ax9.set_ylabel("Composite Score")
                    ax9.tick_params(axis="x", rotation=45)
                    ax9.grid(True, alpha=0.3)

            ax10 = fig.add_subplot(gs[3, :2])
            ax10_twin = ax10.twinx()

            line1 = ax10.plot(df["date"], df["avg_speed"], marker="o", color="blue", label="Avg Speed")
            line2 = ax10_twin.plot(df["date"], df["heart_rate"], marker="s", color="red", alpha=0.6, label="Heart Rate")

            ax10.set_title("Speed vs Heart Rate Over Time")
            ax10.set_xlabel("Date")
            ax10.set_ylabel("Speed (km/h)", color="blue")
            ax10_twin.set_ylabel("Heart Rate (bpm)", color="red")
            ax10.tick_params(axis="x", rotation=45)
            ax10.grid(True, alpha=0.3)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax10.legend(lines, labels, loc="upper left")

            ax11 = fig.add_subplot(gs[3, 2])
            if "speed_zone" in df.columns:
                zone_counts = df["speed_zone"].value_counts()
                colors = ["#3498db", "#2ecc71", "#e74c3c"]
                ax11.pie(zone_counts.values, labels=zone_counts.index.astype(str), autopct="%1.1f%%", colors=colors, startangle=90)
                ax11.set_title("Speed Zone Distribution")

            self._save_plot("performance_dashboard.png")
            plt.show()

            logging.info("Dashboard saved successfully")

        except Exception as e:
            logging.exception("Error creating dashboard: %s", e)

    def analyze_speed_metrics(self) -> dict | None:
        """Analyze speed-related metrics."""
        try:
            if self.training_log.empty:
                logging.warning("No data available")
                return None

            df = self.training_log.copy().sort_values("date")

            print("\n" + "=" * 80)
            print("SPEED METRICS ANALYSIS")
            print("=" * 80)

            print("\nOverall Speed Statistics:")
            print(f"Average Speed (mean):     {df['avg_speed'].mean():.2f} km/h")
            print(f"Average Speed (std):      {df['avg_speed'].std():.2f} km/h")
            print(f"Max Speed (mean):         {df['max_speed'].mean():.2f} km/h")
            print(f"Max Speed (peak):         {df['max_speed'].max():.2f} km/h")
            print(f"Speed Reserve (mean):     {df['speed_reserve'].mean():.2f} km/h")
            print(f"Speed Consistency (mean): {df['speed_consistency'].mean():.2%}")
            print(f"Average Pace:             {df['pace_per_km'].mean():.2f} min/km")

            print(f"\nSpeed Efficiency:         {df['speed_efficiency'].mean():.4f} km/h per bpm")
            print(f"Economy at Speed:         {df['economy_at_speed'].mean():.2f}")

            print("\nSpeed Zone Distribution:")
            zone_counts = df["speed_zone"].value_counts()
            for zone, count in zone_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {zone}: {count} sessions ({pct:.1f}%)")

            if len(df) >= 10:
                recent_avg = df.tail(5)["avg_speed"].mean()
                early_avg = df.head(5)["avg_speed"].mean()
                improvement = ((recent_avg - early_avg) / early_avg) * 100 if early_avg else 0.0
                print(f"\nSpeed Improvement (recent vs early): {improvement:+.2f}%")

            return {
                "avg_speed_mean": float(df["avg_speed"].mean()),
                "max_speed_peak": float(df["max_speed"].max()),
                "speed_reserve": float(df["speed_reserve"].mean()),
                "speed_consistency": float(df["speed_consistency"].mean()),
                "pace_per_km": float(df["pace_per_km"].mean()),
            }

        except Exception as e:
            logging.exception("Error in speed analysis: %s", e)
            return None

    def analyze_hr_rs_deviation(self) -> dict | None:
        """Analyze HR-RS Deviation Index patterns."""
        try:
            if self.training_log.empty:
                logging.warning("No data available")
                return None

            print("\n" + "=" * 80)
            print("HR-RS DEVIATION INDEX ANALYSIS")
            print("=" * 80)

            valid_data = self.training_log[self.training_log["hr_rs_deviation"] > 0].copy()
            if valid_data.empty:
                print("No HR-RS Deviation data available")
                return None

            print("\nOverall HR-RS Deviation Statistics:")
            print(f"Mean:                 {valid_data['hr_rs_deviation'].mean():.2f}")
            print(f"Std Dev:              {valid_data['hr_rs_deviation'].std():.2f}")
            print(f"Min:                  {valid_data['hr_rs_deviation'].min():.2f}")
            print(f"Max:                  {valid_data['hr_rs_deviation'].max():.2f}")

            mean_val = valid_data["hr_rs_deviation"].mean()
            cv = ((valid_data["hr_rs_deviation"].std() / mean_val) * 100) if mean_val else 0.0
            print(f"Coefficient of Variation: {cv:.2f}% ", end="")
            if cv < 10:
                print("(Very Stable)")
            elif cv < 20:
                print("(Stable)")
            elif cv < 30:
                print("(Moderate Variability)")
            else:
                print("(High Variability)")

            if len(valid_data) >= 5:
                valid_data = valid_data.sort_values("date")
                recent_mean = valid_data.tail(3)["hr_rs_deviation"].mean()
                earlier_mean = valid_data.head(3)["hr_rs_deviation"].mean()
                change_rate = ((recent_mean - earlier_mean) / earlier_mean) * 100 if earlier_mean else 0.0

                print(f"\nRecent Trend: {change_rate:+.2f}% ", end="")
                if abs(change_rate) < 5:
                    print("(Stable)")
                elif change_rate > 5:
                    print("(Increasing - may indicate fatigue)")
                else:
                    print("(Decreasing - may indicate improved adaptation)")

            if len(valid_data) >= 10:
                corr_speed = valid_data["hr_rs_deviation"].corr(valid_data["avg_speed"])
                corr_hr = valid_data["hr_rs_deviation"].corr(valid_data["heart_rate"])
                corr_vo2 = valid_data["hr_rs_deviation"].corr(valid_data["vo2max"])

                print("\nCorrelations with Performance:")
                print(f"  vs. Average Speed:  {corr_speed:+.3f}")
                print(f"  vs. Heart Rate:     {corr_hr:+.3f}")
                print(f"  vs. VO2max:         {corr_vo2:+.3f}")

            return {
                "mean": float(valid_data["hr_rs_deviation"].mean()),
                "std": float(valid_data["hr_rs_deviation"].std()),
                "stability_cv": float(cv),
            }

        except Exception as e:
            logging.exception("Error in HR-RS deviation analysis: %s", e)
            return None

    def print_monthly_metrics_averages(self) -> None:
        """Print monthly averages for metrics breakdown."""
        monthly_avg = self.calculate_monthly_metrics_averages()
        if monthly_avg is None or monthly_avg.empty:
            return

        df = self.training_log.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        session_counts = df.groupby(df["date"].dt.to_period("M")).size().to_dict()

        print("\n" + "=" * 80)
        print("MONTHLY METRICS BREAKDOWN - AVERAGES")
        print("=" * 80)

        for month in monthly_avg.index:
            print(f"\n{month} ({session_counts.get(month, 0)} sessions)")
            print("-" * 80)

            if "running_economy" in monthly_avg.columns.get_level_values(0):
                re_mean = monthly_avg.loc[month, ("running_economy", "mean")]
                re_std = monthly_avg.loc[month, ("running_economy", "std")]
                print(f"Running Economy:     {re_mean:>8.2f} ± {re_std:>6.2f}")

            if "vo2max" in monthly_avg.columns.get_level_values(0):
                vo2_mean = monthly_avg.loc[month, ("vo2max", "mean")]
                vo2_std = monthly_avg.loc[month, ("vo2max", "std")]
                print(f"VO2Max:              {vo2_mean:>8.2f} ± {vo2_std:>6.2f}")

            if "distance" in monthly_avg.columns.get_level_values(0):
                dist_mean = monthly_avg.loc[month, ("distance", "mean")]
                dist_std = monthly_avg.loc[month, ("distance", "std")]
                print(f"Distance (km):       {dist_mean:>8.2f} ± {dist_std:>6.2f}")

            if "efficiency_score" in monthly_avg.columns.get_level_values(0):
                eff_mean = monthly_avg.loc[month, ("efficiency_score", "mean")]
                eff_std = monthly_avg.loc[month, ("efficiency_score", "std")]
                print(f"Efficiency Score:    {eff_mean:>8.2f} ± {eff_std:>6.2f}")

            if "heart_rate" in monthly_avg.columns.get_level_values(0):
                hr_mean = monthly_avg.loc[month, ("heart_rate", "mean")]
                hr_std = monthly_avg.loc[month, ("heart_rate", "std")]
                print(f"Heart Rate (bpm):    {hr_mean:>8.2f} ± {hr_std:>6.2f}")

            if "energy_cost" in monthly_avg.columns.get_level_values(0):
                ec_mean = monthly_avg.loc[month, ("energy_cost", "mean")]
                ec_std = monthly_avg.loc[month, ("energy_cost", "std")]
                print(f"Energy Cost:         {ec_mean:>8.2f} ± {ec_std:>6.2f}")

            if "TRIMP" in monthly_avg.columns.get_level_values(0):
                trimp_mean = monthly_avg.loc[month, ("TRIMP", "mean")]
                trimp_std = monthly_avg.loc[month, ("TRIMP", "std")]
                print(f"TRIMP:               {trimp_mean:>8.2f} ± {trimp_std:>6.2f}")

            if "recovery_score" in monthly_avg.columns.get_level_values(0):
                rec_mean = monthly_avg.loc[month, ("recovery_score", "mean")]
                rec_std = monthly_avg.loc[month, ("recovery_score", "std")]
                print(f"Recovery Score:      {rec_mean:>8.2f} ± {rec_std:>6.2f}")

            if "readiness_score" in monthly_avg.columns.get_level_values(0):
                ready_mean = monthly_avg.loc[month, ("readiness_score", "mean")]
                ready_std = monthly_avg.loc[month, ("readiness_score", "std")]
                print(f"Readiness Score:     {ready_mean:>8.2f} ± {ready_std:>6.2f}")

        print("\n" + "=" * 80)


def main() -> None:
    db_path = "c:/smakrykoDBs/Apex.db"

    analysis = RunningAnalysis(
        db_path=db_path,
        output_dir="c:/temp/logsFitnessApp",
        rest_hr=60,
        max_hr=190,
    )

    logging.info("Training log loaded: %s rows", len(analysis.training_log))
    logging.info("Training log empty: %s", analysis.training_log.empty)

    if analysis.training_log.empty:
        logging.warning("Database is empty. Adding sample session.")
        analysis.add_session(
            date=datetime.now().strftime("%Y-%m-%d"),
            running_economy=73,
            vo2max=19.0,
            distance=5,
            time=27,  # minutes
            heart_rate=150,
            cardiacdrift=0.0,
        )
        logging.info("After adding session: %s rows", len(analysis.training_log))

    logging.info("Reloading training data to ensure all fields are present...")
    analysis.training_log = analysis.load_training_data()
    logging.info("After reload: %s rows", len(analysis.training_log))

    analysis.create_metrics_breakdown_table()
    analysis.save_training_log_to_db()

    print("\n[DEBUG] Training Log Preview:")
    print(analysis.training_log.head())
    print(f"\n[DEBUG] Training log shape: {analysis.training_log.shape}")

    if not analysis.training_log.empty:
        analysis.visualize_trends()
        analysis.advanced_visualizations()

    training_score = analysis.calculate_training_score()

    if not analysis.training_log.empty:
        analysis.visualize_training_load()
        analysis.calculate_recovery_and_readiness()
        analysis.visualize_recovery_and_readiness()

    analysis.print_monthly_metrics_averages()
    analysis.create_monthly_summaries_table()

    logging.info("Saving monthly summaries...")
    analysis.save_monthly_summaries()

    if training_score:
        print("\nTraining Score Analysis:")
        print(f"Overall Training Score: {float(training_score['overall_score']):.2f}")

        analysis.save_metrics_breakdown(training_score)

        print("\nMetric Breakdown:")
        for metric, details in training_score["metric_breakdown"].items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Normalized Value: {details['normalized_value']:.4f}")
            print(f"  Weighted Value: {details['weighted_value']:.4f}")
            print(f"  Raw Mean: {details['raw_mean']:.2f}")
            print(f"  Raw Std Dev: {details['raw_std']:.2f}")

        print("\nPerformance Trends:")
        for trend, value in training_score["performance_trends"].items():
            print(f"{trend.replace('_', ' ').title()}: {value:.4f}")
    else:
        logging.warning("training_score is None")

    if not analysis.training_log.empty:
        analysis.visualize_score_impact_over_time(
            extra_scores={
                "Recovery Score": "recovery_score",
                "Readiness Score": "readiness_score",
            }
        )

        print("\n" + "=" * 80)
        print("ENHANCED PERFORMANCE ANALYSIS")
        print("=" * 80)

        analysis.analyze_speed_metrics()
        analysis.analyze_hr_rs_deviation()
        analysis.visualize_speed_metrics()
        analysis.visualize_hr_rs_deviation()
        analysis.create_performance_dashboard()

    logging.info("Main execution completed")


if __name__ == "__main__":
    main()