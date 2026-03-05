"""Setup endpoints: config CRUD, model listing, bib list upload."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from pointcam.web.manager import PipelineState

router = APIRouter()


@router.get("/config")
async def get_config(request: Request):
    mgr = request.app.state.manager
    return mgr.config.to_dict()


@router.put("/config")
async def update_config(request: Request):
    mgr = request.app.state.manager
    body = await request.json()
    try:
        mgr.configure(body)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return mgr.config.to_dict()


@router.get("/models")
async def list_models(request: Request):
    """List available model files in models/ directory."""
    root = request.app.state.manager._project_root
    models_dir = root / "models"
    result = {"detectors": [], "ocr": [], "pose": []}
    if models_dir.is_dir():
        for p in sorted(models_dir.iterdir()):
            name = p.name
            if name.startswith("bib_detector"):
                result["detectors"].append(name)
            elif "ocr" in name or "parseq" in name or "crnn" in name:
                result["ocr"].append(name)
            elif "pose" in name:
                result["pose"].append(name)
    return result


@router.post("/bib-list")
async def upload_bib_list(request: Request, file: UploadFile = File(...)):
    """Upload a bib list text file (one bib per line)."""
    mgr = request.app.state.manager
    if mgr.state not in (PipelineState.IDLE, PipelineState.CONFIGURED):
        raise HTTPException(status_code=409, detail="Cannot upload bib list while pipeline is running")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    bibs = [line.strip() for line in text.splitlines() if line.strip()]

    # Save to project root
    root = mgr._project_root
    dest = root / "bib_list_uploaded.txt"
    dest.write_text("\n".join(bibs) + "\n")

    # Update config to use the uploaded file
    cfg = mgr.config.to_dict()
    cfg["bib_set_path"] = str(dest)
    mgr.configure(cfg)

    return {"ok": True, "count": len(bibs), "path": str(dest)}
