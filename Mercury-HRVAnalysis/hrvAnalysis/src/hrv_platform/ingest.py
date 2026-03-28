from __future__ import annotations

from .repository import HRVRepository
from .schemas import HRVMeasurementIn


class IngestionService:
    def __init__(self, repository: HRVRepository) -> None:
        self.repository = repository

    def ingest_one(self, payload: HRVMeasurementIn):
        return self.repository.upsert_measurement(payload.model_dump())

    def ingest_batch(self, payloads: list[HRVMeasurementIn]):
        return [self.repository.upsert_measurement(item.model_dump()) for item in payloads]
