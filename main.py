from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from database import init_db
from data import router as data_router
from ws_manager import manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Road Mapper", lifespan=lifespan)

app.include_router(data_router, prefix="/api")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; clients don't need to send anything
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_map():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/collect")
async def serve_collector():
    return FileResponse(os.path.join(frontend_path, "collector.html"))
