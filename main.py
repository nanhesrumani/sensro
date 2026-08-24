from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.database import init_db
from app.data import router as data_router
from app.predict_route import router as predict_router
from app.ws_manager import manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Road Mapper", lifespan=lifespan)

# Allow the GitHub Pages frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nanhesrumani.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api")
app.include_router(predict_router, prefix="/api")

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/segments.json")
async def serve_segments_snapshot():
    path = os.path.join(frontend_path, "segments.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="segments.json not found")
    return FileResponse(path)

@app.get("/manifest.json")
async def serve_manifest():
    path = os.path.join(frontend_path, "manifest.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="manifest.json not found")
    return FileResponse(path)

@app.get("/")
async def serve_map():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/collect")
async def serve_collector():
    return FileResponse(os.path.join(frontend_path, "collector.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
