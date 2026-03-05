"""Entry point: ``python -m pointcam.web`` or ``pointcam-web``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PointCam Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument("--dev", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args(argv)

    # Resolve project root (walk up from this file to find pyproject.toml)
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        p = Path(__file__).resolve().parent
        while p != p.parent:
            if (p / "pyproject.toml").exists():
                project_root = p
                break
            p = p.parent
        else:
            project_root = Path.cwd()

    import uvicorn

    print(f"PointCam Web UI starting on http://{args.host}:{args.port}")
    print(f"Project root: {project_root}")

    if args.dev:
        # Reload mode requires import string, not app object.
        # Store project_root in env so the factory can read it.
        import os

        os.environ["POINTCAM_PROJECT_ROOT"] = str(project_root)
        uvicorn.run(
            "pointcam.web.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
        )
    else:
        from pointcam.web.app import create_app

        app = create_app(project_root=project_root)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
