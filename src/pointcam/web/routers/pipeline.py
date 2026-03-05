"""Pipeline control endpoints: start, stop, status, manual crossing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pointcam.web.manager import PipelineState

router = APIRouter()


class ManualCrossingRequest(BaseModel):
    bib_number: str


@router.post("/start")
async def start_pipeline(request: Request):
    mgr = request.app.state.manager
    try:
        mgr.start()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "state": mgr.state.value}


@router.post("/stop")
async def stop_pipeline(request: Request):
    mgr = request.app.state.manager
    mgr.stop()
    return {"ok": True, "state": mgr.state.value}


@router.get("/status")
async def pipeline_status(request: Request):
    mgr = request.app.state.manager
    return mgr.status_dict()


@router.post("/reset")
async def reset_pipeline(request: Request):
    mgr = request.app.state.manager
    mgr.reset()
    return {"ok": True, "state": mgr.state.value}


@router.post("/manual-crossing")
async def manual_crossing(body: ManualCrossingRequest, request: Request):
    """Inject a manual bib entry while pipeline is running."""
    mgr = request.app.state.manager
    if mgr.state != PipelineState.RUNNING:
        raise HTTPException(status_code=409, detail="Pipeline not running")
    # Fire crossing callbacks with a synthetic event
    for cb in mgr._crossing_callbacks:
        try:
            from pointcam.crossing import CrossingEvent

            event = CrossingEvent(
                sequence=-1,
                frame_idx=mgr.stats.frame_idx,
                timestamp_sec=mgr.stats.elapsed_sec,
                person_track_id=-1,
                bib_number=body.bib_number,
                confidence=1.0,
                person_bbox=(0, 0, 0, 0),
                source="manual",
            )
            cb(event)
        except Exception:
            pass
    return {"ok": True, "bib_number": body.bib_number}
