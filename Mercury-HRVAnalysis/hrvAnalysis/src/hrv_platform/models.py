from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class HRVMeasurement(Base):
    __tablename__ = "hrv_measurements"
    __table_args__ = (
        UniqueConstraint("measurement_date", "source_name", name="uq_measurement_date_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    measurement_date: Mapped[date] = mapped_column(Date, index=True)
    source_name: Mapped[str] = mapped_column(String(100), index=True)
    SD1: Mapped[float] = mapped_column(Float, default=0.0)
    SD2: Mapped[float] = mapped_column(Float, default=0.0)
    sdnn: Mapped[float] = mapped_column(Float, default=0.0)
    rmssd: Mapped[float] = mapped_column(Float, default=0.0)
    pNN50: Mapped[float] = mapped_column(Float, default=0.0)
    VLF: Mapped[float] = mapped_column(Float, default=0.0)
    LF: Mapped[float] = mapped_column(Float, default=0.0)
    HF: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HRVBaseline(Base):
    __tablename__ = "hrv_baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    analysis_date: Mapped[date] = mapped_column(Date, nullable=False)

    avg_SD1: Mapped[float] = mapped_column(Float, default=0.0)
    avg_SD2: Mapped[float] = mapped_column(Float, default=0.0)
    avg_sdnn: Mapped[float] = mapped_column(Float, default=0.0)
    avg_rmssd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_pNN50: Mapped[float] = mapped_column(Float, default=0.0)
    avg_VLF: Mapped[float] = mapped_column(Float, default=0.0)
    avg_LF: Mapped[float] = mapped_column(Float, default=0.0)
    avg_HF: Mapped[float] = mapped_column(Float, default=0.0)

    std_SD1: Mapped[float] = mapped_column(Float, default=0.0)
    std_SD2: Mapped[float] = mapped_column(Float, default=0.0)
    std_sdnn: Mapped[float] = mapped_column(Float, default=0.0)
    std_rmssd: Mapped[float] = mapped_column(Float, default=0.0)
    std_pNN50: Mapped[float] = mapped_column(Float, default=0.0)
    std_VLF: Mapped[float] = mapped_column(Float, default=0.0)
    std_LF: Mapped[float] = mapped_column(Float, default=0.0)
    std_HF: Mapped[float] = mapped_column(Float, default=0.0)


class HRVTrend(Base):
    __tablename__ = "hrv_trends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_date: Mapped[date] = mapped_column(Date, index=True)
    source_name: Mapped[str] = mapped_column(String(100), index=True)
    metric: Mapped[str] = mapped_column(String(50), index=True)
    slope: Mapped[float] = mapped_column(Float, default=0.0)
    r_value: Mapped[float] = mapped_column(Float, default=0.0)
    p_value: Mapped[float] = mapped_column(Float, default=1.0)
    trend_direction: Mapped[str] = mapped_column(String(20), default="stable")
    trend_strength: Mapped[str] = mapped_column(String(20), default="weak")
    mean: Mapped[float] = mapped_column(Float, default=0.0)
    std: Mapped[float] = mapped_column(Float, default=0.0)
    min: Mapped[float] = mapped_column(Float, default=0.0)
    max: Mapped[float] = mapped_column(Float, default=0.0)
    latest_ms_score: Mapped[float] = mapped_column(Float, default=0.0)


class HRVAlert(Base):
    __tablename__ = "hrv_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_date: Mapped[date] = mapped_column(Date, index=True)
    source_name: Mapped[str] = mapped_column(String(100), index=True)
    metric: Mapped[str] = mapped_column(String(50), index=True)
    current_value: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_value: Mapped[float] = mapped_column(Float, default=0.0)
    deviation_pct: Mapped[float] = mapped_column(Float, default=0.0)
    alert_type: Mapped[str] = mapped_column(String(50), default="threshold")
    alert_message: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HRVAnomaly(Base):
    __tablename__ = "hrv_anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    measurement_date: Mapped[date] = mapped_column(Date, index=True)
    source_name: Mapped[str] = mapped_column(String(100), index=True)
    metric: Mapped[str] = mapped_column(String(50), index=True)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_mean: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_std: Mapped[float] = mapped_column(Float, default=0.0)
    z_score: Mapped[float] = mapped_column(Float, default=0.0)
    detector: Mapped[str] = mapped_column(String(50), default="z_score")
    message: Mapped[str] = mapped_column(String(500), default="")