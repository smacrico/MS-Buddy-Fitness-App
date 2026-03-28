from __future__ import annotations

import time

import typer

app = typer.Typer(help="HRV Platform CLI")


@app.command("init-db")
def init_db() -> None:
    from .db import Base, engine
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

@app.command("export-plots")
def export_plots(
    source_name: str = "MyHRV_import",
    days_back: int = 90,
    output_dir: str = r"C:\temp\logsFitnessApp\HRV_DashBoards",
) -> None:
    from .db import session_scope
    from .plots import PlotService

    with session_scope() as session:
        service = PlotService(session=session, output_dir=output_dir)
        result = service.export_all(source_name=source_name, days_back=days_back)

    print(f"Plots exported to: {result.output_dir}")
    for file in result.files:
        print(file)

@app.command("preview-artemis")
def preview_artemis() -> None:
    from .artemis_sync import ArtemisSyncService
    service = ArtemisSyncService()
    source_view, row_count, columns, rows = service.preview()
    print(
        {
            "source_view": source_view,
            "row_count": row_count,
            "columns": columns,
            "rows": rows,
        }
    )


@app.command("sync-artemis")
def sync_artemis(source_name: str = "MyHRV_import") -> None:
    from .artemis_sync import ArtemisSyncService
    service = ArtemisSyncService()
    result = service.sync(source_name=source_name)
    print(f"Synced {result.imported_count} Artemis rows for source '{result.source_name_used}'.")


@app.command("watch-artemis")
def watch_artemis(
    interval_seconds: int = 60,
    source_name: str = "MyHRV_import",
) -> None:
    from .artemis_sync import ArtemisSyncService

    service = ArtemisSyncService()
    print(f"Watching Artemis every {interval_seconds} seconds. Press Ctrl+C to stop.")

    while True:
        try:
            result = service.sync(source_name=source_name)
            print(f"Synced {result.imported_count} Artemis rows for source '{result.source_name_used}'.")
        except Exception as exc:
            print(f"Watch error: {exc}")
        time.sleep(interval_seconds)


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("hrv_platform.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()