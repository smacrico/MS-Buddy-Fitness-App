from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HRV_", extra="ignore")

    db_url: str = Field(default="sqlite:///./hrv_platform.db")
    alert_deviation_threshold: float = Field(default=0.25, ge=0.01, le=5.0)
    ingestion_anomaly_zscore: float = Field(default=2.5, ge=1.0, le=10.0)
    baseline_window_days: int = Field(default=90, ge=7, le=3650)
    analysis_window_days: int = Field(default=30, ge=2, le=3650)
    tracked_metrics: list[str] = [
        "SD1", "SD2", "sdnn", "rmssd", "pNN50", "VLF", "LF", "HF"
    ]
    allowed_source_views: list[str] = ["myHRV_view"]
    source_name_default: str = "MyHRV_import"

    # Artemis integration
    artemis_db_path: str = Field(default="c:/smakrykoDBs/Artemis.db")
    artemis_source_view: str = Field(default="myHRV_view")
    artemis_source_name_column: str = Field(default="name")
    artemis_date_column: str = Field(default="date")
    artemis_poll_interval_seconds: int = Field(default=60, ge=5, le=86400)

    ms_weights: dict[str, float] = {
        "rmssd": 0.18,
        "sdnn": 0.24,
        "pNN50": 0.14,
        "SD1": 0.15,
        "SD2": 0.13,
        "LF": 0.08,
        "HF": 0.08,
    }

    metric_reference_values: dict[str, float] = {
        "rmssd": 42.0,
        "sdnn": 50.0,
        "pNN50": 13.0,
        "SD1": 30.0,
        "SD2": 40.0,
        "VLF": 700.0,
        "LF": 1050.0,
        "HF": 820.0,
    }


settings = Settings()
