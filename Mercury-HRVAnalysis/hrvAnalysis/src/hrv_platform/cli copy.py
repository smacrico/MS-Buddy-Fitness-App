from __future__ import annotations


import typer

import argparse
import asyncio
from datetime import date, timedelta

import uvicorn

from .artemis_sync import ArtemisSyncService
from .config import settings
from .db import Base, engine, session_scope
from .repository import HRVRepository
app = typer.Typer(help="HRV Platform CLI")


@app.command("init-db")
def init_db() -> None:
    from .db import Base, engine
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")


@app.command("preview-artemis")

def preview_artemis() -> None:
    from .artemis_sync import ArtemisSyncService
    service = ArtemisSyncService()
    rows = service.preview()
    print(rows)


@app.command("sync-artemis")
def sync_artemis() -> None:
    from .artemis_sync import ArtemisSyncService
    service = ArtemisSyncService()
    count = service.sync()
    print(f"Synced {count} Artemis rows.")


@app.command("watch-artemis")
def watch_artemis(interval_seconds: int = 60) -> None:
    import time
    from .artemis_sync import ArtemisSyncService

    service = ArtemisSyncService()
    print(f"Watching Artemis every {interval_seconds} seconds. Press Ctrl+C to stop.")

    while True:
        try:
            count = service.sync()
            print(f"Synced {count} Artemis rows.")
        except Exception as exc:
            print(f"Watch error: {exc}")
        time.sleep(interval_seconds)


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("hrv_platform.api.app:app", host=host, port=port, reload=False)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {settings.db_url}")


def seed_demo() -> None:
    import random

    Base.metadata.create_all(bind=engine)
    start = date.today() - timedelta(days=29)
    with session_scope() as session:
        repo = HRVRepository(session)
        for i in range(30):
            d = start + timedelta(days=i)
            repo.upsert_measurement(
                {
                    "measurement_date": d,
                    "source_name": settings.source_name_default,
                    "SD1": 30 + random.uniform(-5, 5),
                    "SD2": 40 + random.uniform(-6, 6),
                    "sdnn": 50 + random.uniform(-8, 8),
                    "rmssd": 42 + random.uniform(-7, 7),
                    "pNN50": 13 + random.uniform(-4, 4),
                    "VLF": 700 + random.uniform(-120, 120),
                    "LF": 1050 + random.uniform(-150, 150),
                    "HF": 820 + random.uniform(-100, 100),
                }
            )
    print("Demo data seeded.")


def sync_artemis(source_name: str | None = None) -> None:
    Base.metadata.create_all(bind=engine)
    service = ArtemisSyncService()
    result = service.sync(source_name=source_name)
    print(
        f"Imported {result.imported_count} rows from Artemis view "
        f"'{result.source_view}' at '{result.db_path}' into app DB."
    )


def preview_artemis(limit: int = 5) -> None:
    service = ArtemisSyncService()
    source_view, row_count, columns, sample_rows = service.preview(limit=limit)
    print(f"Source view: {source_view}")
    print(f"Sample row count: {row_count}")
    print(f"Columns: {columns}")
    print("Sample rows:")
    for row in sample_rows:
        print(row)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("hrv_platform.api.app:app", host=host, port=port, reload=False)


def watch_artemis(interval_seconds: int | None = None, source_name: str | None = None) -> None:
    service = ArtemisSyncService()
    interval = interval_seconds or settings.artemis_poll_interval_seconds
    print(
        f"Watching Artemis DB '{settings.artemis_db_path}' view '{settings.artemis_source_view}' "
        f"every {interval} seconds."
    )
    asyncio.run(service.watch_forever(interval_seconds=interval, source_name=source_name))


def main() -> None:
    parser = argparse.ArgumentParser(prog="hrv-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    sub.add_parser("seed-demo")

    preview_parser = sub.add_parser("preview-artemis")
    preview_parser.add_argument("--limit", type=int, default=5)

    sync_parser = sub.add_parser("sync-artemis")
    sync_parser.add_argument("--source-name", default=settings.source_name_default)

    watch_parser = sub.add_parser("watch-artemis")
    watch_parser.add_argument("--interval", type=int, default=settings.artemis_poll_interval_seconds)
    watch_parser.add_argument("--source-name", default=settings.source_name_default)

    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
    elif args.command == "seed-demo":
        seed_demo()
    elif args.command == "preview-artemis":
        preview_artemis(args.limit)
    elif args.command == "sync-artemis":
        sync_artemis(args.source_name)
    elif args.command == "watch-artemis":
        watch_artemis(args.interval, args.source_name)
    elif args.command == "serve":
        serve(args.host, args.port)

if __name__ == "__main__":
    app()