from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..db import Base, engine
from ..live import event_bus
from .routes import router


BASE_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    app.state.event_bus = event_bus
    yield


app = FastAPI(title="HRV Platform", version="0.2.0", lifespan=lifespan)
app.include_router(router)

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
