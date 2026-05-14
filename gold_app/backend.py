from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime
from typing import Annotated
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .config import PUSH_INTERVAL_SECONDS
from .snapshot_builder import build_snapshot
from .storage import init_db, load_latest_snapshot, load_snapshot_history


app = FastAPI(title="Gold Forecast Backend", version="1.0.0")
publisher_task: asyncio.Task | None = None
background_tasks: set[asyncio.Task] = set()


class SnapshotConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)

        if not connections:
            return

        failed: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                failed.append(websocket)

        if failed:
            async with self._lock:
                for websocket in failed:
                    self._connections.discard(websocket)


manager = SnapshotConnectionManager()


class RefreshRequest(BaseModel):
    force_retrain: bool = False


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value).isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _snapshot_event(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "snapshot",
        "created_at": snapshot.get("created_at"),
        "summary": snapshot.get("summary", {}),
        "market_features": snapshot.get("market_features", {}),
        "model_metrics": snapshot.get("model_metrics", {}),
        "news_status": snapshot.get("news_status", "unknown"),
        "market_status": snapshot.get("market_status", "unknown"),
    }


async def _publish_snapshot(snapshot: dict[str, Any]) -> None:
    await manager.broadcast(_sanitize(_snapshot_event(snapshot)))


async def _snapshot_publisher_loop(interval_seconds: int) -> None:
    while True:
        try:
            snapshot = build_snapshot(force_retrain=False)
            await _publish_snapshot(snapshot)
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


def _create_background_task(coro: Any) -> None:
    task = asyncio.create_task(coro)
    background_tasks.add(task)

    def _cleanup(completed: asyncio.Task) -> None:
        background_tasks.discard(completed)

    task.add_done_callback(_cleanup)


@app.on_event("startup")
def on_startup() -> None:
    global publisher_task
    init_db()
    interval = int(os.getenv("GOLD_PUSH_INTERVAL_SECONDS", str(PUSH_INTERVAL_SECONDS)))
    interval = max(15, interval)
    publisher_task = asyncio.create_task(_snapshot_publisher_loop(interval))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global publisher_task
    if publisher_task is not None:
        publisher_task.cancel()
        await asyncio.gather(publisher_task, return_exceptions=True)
        publisher_task = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/snapshot/latest")
def get_latest_snapshot() -> dict[str, Any]:
    snapshot = load_latest_snapshot()
    if snapshot is None:
        snapshot = build_snapshot(force_retrain=False)
    return _sanitize(snapshot)


@app.get("/api/v1/snapshot/live")
def get_live_snapshot(
    force_retrain: Annotated[bool, Query()] = False
) -> dict[str, Any]:
    snapshot = build_snapshot(force_retrain=force_retrain)
    # Don't try to create background task here - just return the snapshot
    # The WebSocket publisher will handle periodic updates
    return _sanitize(snapshot)


@app.post("/api/v1/snapshot/refresh")
def refresh_snapshot(payload: RefreshRequest) -> dict[str, Any]:
    snapshot = build_snapshot(force_retrain=payload.force_retrain)
    return _sanitize(snapshot)


@app.get("/api/v1/snapshot/history")
def get_snapshot_history(
    limit: Annotated[int, Query(ge=1, le=1000)] = 30
) -> dict[str, Any]:
    history_frame = load_snapshot_history(limit=limit)
    if history_frame.empty:
        return {"items": []}

    parsed = history_frame.copy()
    parsed["created_at"] = pd.to_datetime(parsed["created_at"], errors="coerce")
    parsed["predicted_price"] = parsed["payload"].map(
        lambda item: item.get("summary", {}).get("predicted_price", 0.0)
    )
    parsed = parsed.sort_values("created_at", ascending=False)

    items = [
        {
            "created_at": row["created_at"].isoformat() if pd.notna(row["created_at"]) else None,
            "predicted_price": row["predicted_price"],
        }
        for _, row in parsed.iterrows()
    ]
    return {"items": _sanitize(items)}


@app.websocket("/ws/snapshots")
async def snapshot_websocket(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        latest = load_latest_snapshot()
        if latest is None:
            latest = build_snapshot(force_retrain=False)
        await websocket.send_json(_sanitize(_snapshot_event(latest)))

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)
