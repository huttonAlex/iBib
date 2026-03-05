"""MJPEG video stream and WebSocket event push."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------

async def _mjpeg_generator(request: Request):
    """Yield MJPEG frames from the shared FrameBuffer."""
    buf = request.app.state.manager.frame_buffer
    boundary = b"--frame\r\n"
    while True:
        jpeg = await asyncio.get_event_loop().run_in_executor(None, buf.get, 1.0)
        if jpeg is None:
            # No frame yet — send a keep-alive empty boundary
            await asyncio.sleep(0.1)
            continue
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            + jpeg
            + b"\r\n"
        )


@router.get("/mjpeg")
async def mjpeg_stream(request: Request):
    return StreamingResponse(
        _mjpeg_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket — real-time crossing + progress events
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    mgr = websocket.app.state.manager
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def _on_crossing(event):
        """Called from pipeline thread — push to async queue."""
        data = {
            "type": "crossing",
            "data": {
                "sequence": event.sequence,
                "bib_number": event.bib_number,
                "confidence": round(event.confidence, 3),
                "timestamp_sec": round(event.timestamp_sec, 3),
                "source": event.source,
                "frame_idx": event.frame_idx,
            },
        }
        try:
            loop.call_soon_threadsafe(queue.put_nowait, data)
        except asyncio.QueueFull:
            pass

    def _on_progress(info):
        """Called from pipeline thread — push to async queue."""
        data = {
            "type": "progress",
            "data": {
                "frame_idx": info.frame_idx,
                "elapsed_sec": round(info.elapsed_sec, 1),
                "total_crossings": info.total_crossings,
                "unknown_crossings": info.unknown_crossings,
                "fps": round(info.fps, 1),
                "total_detections": info.total_detections,
            },
        }
        try:
            loop.call_soon_threadsafe(queue.put_nowait, data)
        except asyncio.QueueFull:
            pass

    mgr.add_crossing_callback(_on_crossing)
    mgr.add_progress_callback(_on_progress)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        mgr.remove_crossing_callback(_on_crossing)
        mgr.remove_progress_callback(_on_progress)
