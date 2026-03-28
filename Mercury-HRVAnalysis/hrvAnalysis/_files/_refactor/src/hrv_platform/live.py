from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class LiveEventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    async def publish(self, event: dict) -> None:
        for queue in list(self._subscribers):
            await queue.put(event)

    async def subscribe(self) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


event_bus = LiveEventBus()
