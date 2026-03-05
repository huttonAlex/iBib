"""Rich terminal dashboard for the live pipeline.

Displays a live crossing feed, running totals, and system stats
(FPS, frame count, elapsed time) using ``rich.live.Live``.

Requires the ``tui`` extra: ``pip install pointcam[tui]``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pointcam.crossing import CrossingEvent
from pointcam.pipeline import ProgressInfo

# Maximum number of recent crossings shown in the feed
_FEED_SIZE = 50


@dataclass
class DashboardState:
    """Mutable state backing the dashboard display."""

    # Config metadata (set once at init)
    camera_name: str = ""
    ocr_model: str = ""
    crossing_mode: str = ""
    placement: str = ""

    # Crossing feed (most recent first)
    crossings: deque = field(default_factory=lambda: deque(maxlen=_FEED_SIZE))
    total_crossings: int = 0
    unknown_crossings: int = 0
    unique_bibs: set = field(default_factory=set)

    # Progress stats
    frame_idx: int = 0
    elapsed_sec: float = 0.0
    fps: float = 0.0
    total_detections: int = 0
    avg_det_ms: float = 0.0
    avg_ocr_ms: float = 0.0
    avg_pose_ms: float = 0.0


class LiveDashboard:
    """Context-manager that drives a ``rich.live.Live`` display.

    Usage::

        dash = LiveDashboard(camera_name="CSI 0", ocr_model="parseq", ...)
        with dash:
            process_frames(..., on_crossing=dash.on_crossing, on_progress=dash.on_progress)
    """

    def __init__(
        self,
        camera_name: str = "",
        ocr_model: str = "",
        crossing_mode: str = "zone",
        placement: str = "center",
    ):
        self._state = DashboardState(
            camera_name=camera_name,
            ocr_model=ocr_model,
            crossing_mode=crossing_mode,
            placement=placement,
        )
        self._console = Console()
        self._live: Optional[Live] = None

    # -- Callbacks -----------------------------------------------------------

    def on_crossing(self, event: CrossingEvent) -> None:
        """Called by the pipeline for each crossing event."""
        s = self._state
        s.total_crossings += 1
        s.crossings.appendleft(event)
        if event.bib_number == "UNKNOWN":
            s.unknown_crossings += 1
        else:
            s.unique_bibs.add(event.bib_number)
        self._refresh()

    def on_progress(self, info: ProgressInfo) -> None:
        """Called by the pipeline periodically with stats."""
        s = self._state
        s.frame_idx = info.frame_idx
        s.elapsed_sec = info.elapsed_sec
        s.fps = info.fps
        s.total_detections = info.total_detections
        s.avg_det_ms = info.avg_det_ms
        s.avg_ocr_ms = info.avg_ocr_ms
        s.avg_pose_ms = info.avg_pose_ms
        # Keep crossing counts in sync with pipeline totals
        s.total_crossings = info.total_crossings
        s.unknown_crossings = info.unknown_crossings
        self._refresh()

    # -- Context manager -----------------------------------------------------

    def __enter__(self):
        self._live = Live(
            self._render(),
            console=self._console,
            screen=False,
            transient=True,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # -- Rendering -----------------------------------------------------------

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        s = self._state

        # -- Header bar ------------------------------------------------------
        header_parts = []
        if s.camera_name:
            header_parts.append(f"Camera: {s.camera_name}")
        if s.ocr_model:
            header_parts.append(f"OCR: {s.ocr_model}")
        mode_str = s.crossing_mode
        if s.placement:
            mode_str += f"/{s.placement}"
        header_parts.append(f"Mode: {mode_str}")
        header = Text("  |  ".join(header_parts), style="dim")

        # -- Crossing feed table ---------------------------------------------
        feed = Table(
            show_header=True,
            header_style="bold cyan",
            expand=True,
            show_edge=False,
            pad_edge=False,
        )
        feed.add_column("#", justify="right", width=5, no_wrap=True)
        feed.add_column("Bib", justify="left", width=7, no_wrap=True)
        feed.add_column("Conf", justify="right", width=5, no_wrap=True)
        feed.add_column("Source", justify="left", width=8, no_wrap=True)
        feed.add_column("Frame", justify="right", width=8, no_wrap=True)

        for evt in list(s.crossings)[:20]:
            bib_style = "red" if evt.bib_number == "UNKNOWN" else "green"
            feed.add_row(
                str(evt.sequence),
                Text(evt.bib_number, style=bib_style),
                f"{evt.confidence:.2f}",
                evt.source,
                str(evt.frame_idx),
            )

        feed_panel = Panel(feed, title="Crossing Feed", border_style="cyan")

        # -- Stats panel -----------------------------------------------------
        elapsed_m = int(s.elapsed_sec) // 60
        elapsed_s = int(s.elapsed_sec) % 60
        known = s.total_crossings - s.unknown_crossings

        stats_lines = [
            f"  Frames:  {s.frame_idx:>8,}",
            f"  Elapsed: {elapsed_m:>4d}:{elapsed_s:02d}",
            f"  FPS:     {s.fps:>8.1f}",
            "",
            f"  Det:     {s.avg_det_ms:>5.1f} ms",
            f"  OCR:     {s.avg_ocr_ms:>5.1f} ms",
            f"  Pose:    {s.avg_pose_ms:>5.1f} ms",
            "",
            f"  Crossings: {s.total_crossings:>5}",
            f"    Known:   {known:>5}",
            f"    Unknown: {s.unknown_crossings:>5}",
            f"    Unique:  {len(s.unique_bibs):>5}",
        ]
        stats_text = Text("\n".join(stats_lines))
        stats_panel = Panel(stats_text, title="Stats", border_style="green", width=24)

        # -- Compose two-column layout using a table -------------------------
        layout = Table.grid(expand=True)
        layout.add_column(ratio=3)
        layout.add_column(width=24)
        layout.add_row(feed_panel, stats_panel)

        # -- Outer panel wrapping everything ---------------------------------
        footer = Text("  Ctrl+C to stop", style="dim italic")
        outer = Panel(
            Group(header, "", layout, footer),
            title="[bold]PointCam Live[/bold]",
            border_style="blue",
        )
        return outer
