# websocket connection manager

import asyncio
from fastapi import WebSocket


class ConnectionManager:
    # manages websocket connections and broadcasts
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        # accept and store new connection
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        # remove connection
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, websocket: WebSocket, message: str):
        # send message to specific websocket
        try:
            await websocket.send_text(message)
        except Exception:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        # broadcast message to all clients
        await asyncio.gather(
            *[self.send_message(conn, message) for conn in self.active_connections],
            return_exceptions=True
        )
