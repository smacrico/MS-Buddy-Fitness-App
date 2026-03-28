from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import text

from ..artemis_sync import ArtemisSyncService
from ..db import session_scope
from ..live import event_bus
from ..sources.artemis import ArtemisSourceError

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/api/debug/source-names")
def debug_source_names() -> dict:
    try:
        with session_scope() as session:
            rows = session.execute(
                text(
                    """
                    SELECT source_name, COUNT(*) AS cnt
                    FROM hrv_measurements
                    GROUP BY source_name
                    ORDER BY cnt DESC, source_name ASC
                    """
                )
            ).mappings().all()

        return {"sources": [dict(r) for r in rows]}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"/api/debug/source-names failed: {type(exc).__name__}: {exc}",
        )


@router.get("/api/debug/measurements")
def debug_measurements(
    source_name: str = Query(default="MyHRV_import"),
    limit: int = Query(default=10, ge=1, le=500),
) -> dict:
    try:
        with session_scope() as session:
            total = session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM hrv_measurements
                    WHERE source_name = :source_name
                    """
                ),
                {"source_name": source_name},
            ).scalar_one()

            sample = session.execute(
                text(
                    """
                    SELECT
                        measurement_date,
                        source_name,
                        SD1,
                        SD2,
                        sdnn,
                        rmssd,
                        pNN50,
                        VLF,
                        LF,
                        HF
                    FROM hrv_measurements
                    WHERE source_name = :source_name
                    ORDER BY measurement_date DESC
                    LIMIT :limit
                    """
                ),
                {"source_name": source_name, "limit": limit},
            ).mappings().all()

        return {
            "source_name": source_name,
            "count": total,
            "sample": [dict(r) for r in sample],
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"/api/debug/measurements failed: {type(exc).__name__}: {exc}",
        )


@router.get("/api/import/artemis/preview")
def preview_artemis(limit: int = Query(default=5, ge=1, le=100)) -> dict:
    try:
        service = ArtemisSyncService()
        source_view, row_count, columns, rows = service.preview(limit=limit)

        return {
            "status": "ok",
            "source_view": source_view,
            "row_count": row_count,
            "columns": columns,
            "rows": rows,
        }
    except ArtemisSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"/api/import/artemis/preview failed: {type(exc).__name__}: {exc}",
        )


@router.post("/api/import/artemis")
async def import_artemis(
    source_name: str = Query(default="MyHRV_import"),
) -> dict:
    try:
        service = ArtemisSyncService()
        result = await service.sync_and_publish(source_name=source_name)

        return {
            "status": "ok",
            "rows_synced": result.imported_count,
            "source_name": result.source_name_used,
            "source_view": result.source_view,
            "db_path": result.db_path,
            "analysis_triggered": result.analysis_triggered,
        }
    except ArtemisSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"/api/import/artemis failed: {type(exc).__name__}: {exc}",
        )


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    await event_bus.connect(websocket)

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "payload": {"message": "WebSocket connected"},
            }
        )

        while True:
            # keep connection alive; client does not need to send useful data
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)