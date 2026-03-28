from __future__ import annotations

from fastapi import WebSocket


class EventBus:
    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        if websocket not in self.connections:
            self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []

        for connection in list(self.connections):
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)

        for connection in dead:
            self.disconnect(connection)


event_bus = EventBus()