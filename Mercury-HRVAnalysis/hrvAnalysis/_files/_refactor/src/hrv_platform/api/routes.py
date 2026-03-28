from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..artemis_sync import ArtemisSyncService
from ..config import settings
from ..db import session_scope
from ..ingest import IngestionService
from ..live import event_bus
from ..repository import HRVRepository, METRICS
from ..schemas import (
    AlertOut,
    AnomalyOut,
    ArtemisPreviewOut,
    ArtemisSyncOut,
    HRVMeasurementIn,
    HRVMeasurementOut,
    MetricSeriesOut,
    SummaryOut,
    TrendOut,
)
from ..services import AnalyticsService

router = APIRouter()


def build_summary(source_name: str, days_back: int = 30) -> SummaryOut:
    with session_scope() as session:
        repo = HRVRepository(session)
        service = AnalyticsService(repo)
        result = service.analyze(source_name=source_name, days_back=days_back)
        return SummaryOut(
            data_points=result.data_points,
            date_range=result.date_range,
            current_values=result.current_values,
            recovery_scores=result.recovery_scores,
            baselines=result.baselines,
            alerts=[AlertOut(**item) for item in result.alerts],
            anomalies=[AnomalyOut(**item) for item in result.anomalies],
        )


@router.get("/api/summary", response_model=SummaryOut)
def get_summary(source_name: str = settings.source_name_default, days_back: int = 30):
    return build_summary(source_name=source_name, days_back=days_back)


@router.get("/api/series", response_model=list[MetricSeriesOut])
def get_series(source_name: str = settings.source_name_default, days_back: int = 30):
    with session_scope() as session:
        repo = HRVRepository(session)
        df = repo.get_measurements_df(source_name=source_name, days_back=days_back)
        payload = []
        if df.empty:
            return payload
        for metric in [*METRICS]:
            payload.append(
                MetricSeriesOut(
                    metric=metric,
                    points=[
                        {"measurement_date": row["measurement_date"], "value": float(row[metric])}
                        for _, row in df.iterrows()
                    ],
                )
            )
        return payload


@router.get("/api/trends", response_model=list[TrendOut])
def get_trends(source_name: str = settings.source_name_default, days_back: int = 30):
    with session_scope() as session:
        repo = HRVRepository(session)
        service = AnalyticsService(repo)
        result = service.analyze(source_name=source_name, days_back=days_back)
        return [TrendOut(**trend) for trend in result.trends.values()]


@router.get("/api/anomalies", response_model=list[AnomalyOut])
def get_anomalies(source_name: str = settings.source_name_default, limit: int = 20):
    with session_scope() as session:
        repo = HRVRepository(session)
        rows = repo.get_recent_anomalies(source_name=source_name, limit=limit)
        return [
            AnomalyOut(
                measurement_date=item.measurement_date,
                source_name=item.source_name,
                metric=item.metric,
                value=item.value,
                baseline_mean=item.baseline_mean,
                baseline_std=item.baseline_std,
                z_score=item.z_score,
                detector=item.detector,
                message=item.message,
            )
            for item in rows
        ]


@router.get("/api/import/artemis/preview", response_model=ArtemisPreviewOut)
def preview_artemis(limit: int = 5):
    service = ArtemisSyncService()
    source_view, row_count, columns, sample_rows = service.preview(limit=limit)
    return ArtemisPreviewOut(
        source_view=source_view,
        row_count=row_count,
        columns=columns,
        sample_rows=sample_rows,
    )


@router.post("/api/import/artemis", response_model=ArtemisSyncOut)
async def import_artemis(source_name: str = settings.source_name_default):
    service = ArtemisSyncService()
    result = await service.sync_and_publish(source_name=source_name)
    return ArtemisSyncOut(
        source_view=result.source_view,
        imported_count=result.imported_count,
        source_name_used=result.source_name_used,
        db_path=result.db_path,
        analysis_triggered=result.analysis_triggered,
    )


@router.post("/api/ingest", response_model=HRVMeasurementOut)
async def ingest(payload: HRVMeasurementIn):
    with session_scope() as session:
        repo = HRVRepository(session)
        service = IngestionService(repo)
        item = service.ingest_one(payload)

    summary = build_summary(payload.source_name)
    await event_bus.publish({"type": "measurement_ingested", "payload": payload.model_dump(mode="json")})
    await event_bus.publish({"type": "summary_updated", "payload": summary.model_dump(mode="json")})
    for anomaly in summary.anomalies:
        await event_bus.publish({"type": "anomaly_detected", "payload": anomaly.model_dump(mode="json")})
    return HRVMeasurementOut.model_validate(item)


@router.post("/api/ingest/batch", response_model=list[HRVMeasurementOut])
async def ingest_batch(payloads: list[HRVMeasurementIn]):
    if not payloads:
        return []
    with session_scope() as session:
        repo = HRVRepository(session)
        service = IngestionService(repo)
        items = service.ingest_batch(payloads)

    source_name = payloads[-1].source_name
    summary = build_summary(source_name)
    await event_bus.publish({"type": "batch_ingested", "payload": {"count": len(payloads), "source_name": source_name}})
    await event_bus.publish({"type": "summary_updated", "payload": summary.model_dump(mode="json")})
    return [HRVMeasurementOut.model_validate(item) for item in items]


@router.websocket("/ws/live")
async def live_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        async for event in event_bus.subscribe():
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        return
