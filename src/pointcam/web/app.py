"""FastAPI application factory for the PointCam web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pointcam.web.manager import PipelineManager
from pointcam.web.routers import pipeline, stream, setup, review, network

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(project_root: Path | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    import os

    if project_root is None:
        env_root = os.environ.get("POINTCAM_PROJECT_ROOT")
        if env_root:
            project_root = Path(env_root)

    app = FastAPI(title="PointCam", version="0.1.0")

    # Shared singleton
    mgr = PipelineManager(project_root=project_root)
    app.state.manager = mgr

    # API routers
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(stream.router, prefix="/api/stream", tags=["stream"])
    app.include_router(setup.router, prefix="/api", tags=["setup"])
    app.include_router(review.router, prefix="/api/review", tags=["review"])
    app.include_router(network.router, prefix="/api/network", tags=["network"])

    # Serve pre-built Svelte SPA (index.html + assets)
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app
