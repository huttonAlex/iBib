"""Thread-safe single-slot JPEG frame buffer for MJPEG streaming.

The pipeline thread calls ``update(frame_bgr)`` with each annotated frame.
The MJPEG endpoint calls ``get(timeout)`` to block until the next frame is
available.  This decouples pipeline FPS from stream FPS.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

# JPEG encode params: quality 70 → ~60-80 KB/frame at 1080p
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]


class FrameBuffer:
    """Single-slot buffer holding the most recent JPEG-encoded frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._jpeg: Optional[bytes] = None

    def update(self, frame_bgr: np.ndarray) -> None:
        """Encode *frame_bgr* to JPEG and store it (called from pipeline thread)."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, _JPEG_PARAMS)
        if not ok:
            return
        jpeg_bytes = buf.tobytes()
        with self._lock:
            self._jpeg = jpeg_bytes
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        """Block until a frame is available, then return the JPEG bytes.

        Returns ``None`` on timeout.
        """
        if not self._event.wait(timeout=timeout):
            return None
        self._event.clear()
        with self._lock:
            return self._jpeg

    def clear(self) -> None:
        """Discard the current frame."""
        with self._lock:
            self._jpeg = None
        self._event.clear()
