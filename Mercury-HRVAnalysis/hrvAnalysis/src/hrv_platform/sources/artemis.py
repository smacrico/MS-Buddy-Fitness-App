from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import settings
from ..repository import METRICS

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ArtemisSourceError(RuntimeError):
    pass


@dataclass(slots=True)
class ArtemisReadResult:
    source_view: str
    row_count: int
    dataframe: pd.DataFrame


class ArtemisSource:
    def __init__(self, db_path: str | None = None, source_view: str | None = None) -> None:
        self.db_path = db_path or settings.artemis_db_path
        self.source_view = source_view or settings.artemis_source_view

    def _validate_source_view(self, source_view: str) -> str:
        if source_view not in settings.allowed_source_views:
            raise ArtemisSourceError(
                f"View '{source_view}' is not allowed. Allowed views: {settings.allowed_source_views}"
            )
        if not _SAFE_IDENTIFIER.match(source_view):
            raise ArtemisSourceError(f"Unsafe view name: {source_view}")
        return source_view

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        if not path.exists():
            raise ArtemisSourceError(f"Artemis database not found: {path}")
        return sqlite3.connect(str(path))

    def preview(self, limit: int = 5) -> ArtemisReadResult:
        source_view = self._validate_source_view(self.source_view)
        with self._connect() as conn:
            query = f"SELECT * FROM {source_view} LIMIT {int(limit)}"
            df = pd.read_sql_query(query, conn)
        return ArtemisReadResult(source_view=source_view, row_count=len(df), dataframe=df)

    def read(self) -> ArtemisReadResult:
        source_view = self._validate_source_view(self.source_view)
        with self._connect() as conn:
            query = f"SELECT * FROM {source_view}"
            df = pd.read_sql_query(query, conn)
        if df.empty:
            return ArtemisReadResult(source_view=source_view, row_count=0, dataframe=df)
        return ArtemisReadResult(source_view=source_view, row_count=len(df), dataframe=df)

    def normalize(self, df: pd.DataFrame, default_source_name: str | None = None) -> list[dict]:
        if df.empty:
            return []

        date_column = settings.artemis_date_column
        source_name_column = settings.artemis_source_name_column

        if date_column not in df.columns:
            raise ArtemisSourceError(
                f"Artemis view must contain date column '{date_column}'. Columns={list(df.columns)}"
            )

        missing_metrics = [metric for metric in METRICS if metric not in df.columns]
        if missing_metrics:
            raise ArtemisSourceError(
                f"Artemis view is missing required HRV metric columns: {missing_metrics}"
            )

        normalized: list[dict] = []
        working = df.copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.date

        for metric in METRICS:
            working[metric] = pd.to_numeric(working[metric], errors="coerce")

        working = working.dropna(subset=[date_column, *METRICS])

        if working.empty:
            return []

        for _, row in working.iterrows():
            source_name = default_source_name or settings.source_name_default

            payload = {
                "measurement_date": row[date_column],
                "source_name": source_name,
            }
            for metric in METRICS:
                value = float(row[metric])
                if value < 0:
                    raise ArtemisSourceError(f"Negative metric value for {metric} on {row[date_column]}: {value}")
                payload[metric] = value
            normalized.append(payload)

        return normalized
