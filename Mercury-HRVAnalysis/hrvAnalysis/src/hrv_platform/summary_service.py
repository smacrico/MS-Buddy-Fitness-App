from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from .repository import HRVRepository
from .services import AnalyticsService


def build_summary(session: Session, source_name: str = "MyHRV_import") -> dict[str, Any]:
    repo = HRVRepository(session)
    service = AnalyticsService(repository=repo)
    return service.build_summary(source_name=source_name, limit=90)