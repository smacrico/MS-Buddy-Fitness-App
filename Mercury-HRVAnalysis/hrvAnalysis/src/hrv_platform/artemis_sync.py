from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .config import settings
from .db import session_scope
from .live import event_bus
from .repository import HRVRepository
from .sources.artemis import ArtemisSource
from .summary_service import build_summary
from .recalc import RecalculationService


@dataclass(slots=True)
class ArtemisSyncResult:
    source_view: str
    imported_count: int
    source_name_used: str
    db_path: str
    analysis_triggered: bool = True


class ArtemisSyncService:
    def __init__(self, source: ArtemisSource | None = None) -> None:
        self.source = source or ArtemisSource()

    def preview(self, limit: int = 5) -> tuple[str, int, list[str], list[dict]]:
        result = self.source.preview(limit=limit)
        sample_rows = result.dataframe.fillna("").to_dict(orient="records")
        return result.source_view, result.row_count, list(result.dataframe.columns), sample_rows

    def sync(self, source_name: str | None = None) -> ArtemisSyncResult:
        read_result = self.source.read()
        rows = self.source.normalize(
            read_result.dataframe,
            default_source_name=source_name or settings.source_name_default,
        )

        with session_scope() as session:
            repo = HRVRepository(session)
            for row in rows:
                repo.upsert_measurement(row)

        return ArtemisSyncResult(
            source_view=read_result.source_view,
            imported_count=len(rows),
            source_name_used=source_name or settings.source_name_default,
            db_path=self.source.db_path,
            analysis_triggered=True,
        )
    def sync(self, source_name: str | None = None) -> ArtemisSyncResult:
        source_name_used = source_name or settings.source_name_default
        read_result = self.source.read()
        rows = self.source.normalize(
            read_result.dataframe,
            default_source_name=source_name_used,
        )

        with session_scope() as session:
            repo = HRVRepository(session)
            for row in rows:
                repo.upsert_measurement(row)

            recalc = RecalculationService(session)
            recalc.recompute_all(source_name_used)

        return ArtemisSyncResult(
            source_view=read_result.source_view,
            imported_count=len(rows),
            source_name_used=source_name_used,
            db_path=self.source.db_path,
            analysis_triggered=True,
        )

    async def sync_and_publish(self, source_name: str | None = None) -> ArtemisSyncResult:
        result = self.sync(source_name=source_name)
        source_name_used = source_name or settings.source_name_default

        with session_scope() as session:
            summary = build_summary(session=session, source_name=source_name_used)

        await event_bus.broadcast(
    {
        "type": "artemis_synced",
        "payload": {
            "source_view": result.source_view,
            "imported_count": result.imported_count,
            "source_name": result.source_name_used,
            "db_path": result.db_path,
        },
    }
)

        await event_bus.broadcast(
            {
                "type": "summary_updated",
                "payload": summary,
            }
        )

        await event_bus.broadcast(
            {
                "type": "summary_updated",
                "payload": summary,
            }
        )

        return result

    async def watch_forever(self, interval_seconds: int | None = None, source_name: str | None = None) -> None:
        interval = interval_seconds or settings.artemis_poll_interval_seconds
        while True:
            try:
                await self.sync_and_publish(source_name=source_name)
            except Exception as exc:
                await event_bus.broadcast(
                    {
                        "type": "artemis_sync_error",
                        "payload": {"message": str(exc)},
                    }
                )
            await asyncio.sleep(interval)