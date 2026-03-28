from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..db import Base, engine
from ..live import event_bus
from .anomalies import router as anomalies_router
from .routes import router
from .summary import router as summary_router
from .trends import router as trends_router


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DASHBOARD = BASE_DIR / "templates" / "dashboard.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    app.state.event_bus = event_bus
    yield


app = FastAPI(title="HRV Platform", version="0.2.0", lifespan=lifespan)

app.include_router(router)
app.include_router(summary_router)
app.include_router(trends_router)
app.include_router(anomalies_router)


@app.get("/")
def dashboard():
    return FileResponse(str(TEMPLATE_DASHBOARD))


@app.get("/dashboard")
def dashboard_alias():
    return FileResponse(str(TEMPLATE_DASHBOARD))