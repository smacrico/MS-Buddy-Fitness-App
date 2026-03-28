from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_METRICS = {"SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"}


class HRVMeasurementIn(BaseModel):
    measurement_date: date
    source_name: str = Field(min_length=1, max_length=100)
    SD1: float = Field(ge=0)
    SD2: float = Field(ge=0)
    sdnn: float = Field(ge=0)
    rmssd: float = Field(ge=0)
    pNN50: float = Field(ge=0)
    VLF: float = Field(ge=0)
    LF: float = Field(ge=0)
    HF: float = Field(ge=0)

    @field_validator("source_name")
    @classmethod
    def clean_source_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_name cannot be blank")
        return value


class HRVMeasurementOut(HRVMeasurementIn):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class BaselineOut(BaseModel):
    source_name: str
    analysis_date: date
    values: dict[str, float]


class TrendOut(BaseModel):
    metric: str
    slope: float
    r_value: float
    p_value: float
    trend_direction: str
    trend_strength: str
    mean: float
    std: float
    min: float
    max: float


class AlertOut(BaseModel):
    metric: str
    current_value: float
    baseline_value: float
    deviation_pct: float
    alert_type: str
    alert_message: str


class AnomalyOut(BaseModel):
    measurement_date: date
    source_name: str
    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    detector: str
    message: str


class SummaryOut(BaseModel):
    data_points: int
    date_range: dict[str, str | None]
    current_values: dict[str, float]
    recovery_scores: dict[str, float]
    baselines: dict[str, float]
    alerts: list[AlertOut]
    anomalies: list[AnomalyOut]


class MetricSeriesPoint(BaseModel):
    measurement_date: date
    value: float


class MetricSeriesOut(BaseModel):
    metric: str
    points: list[MetricSeriesPoint]


class ArtemisPreviewOut(BaseModel):
    source_view: str
    row_count: int
    columns: list[str]
    sample_rows: list[dict[str, Any]]


class ArtemisSyncOut(BaseModel):
    source_view: str
    imported_count: int
    source_name_used: str
    db_path: str
    analysis_triggered: bool


class EventMessage(BaseModel):
    type: str
    payload: dict[str, Any]
