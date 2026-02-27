"""Camera frame sources for live pipeline operation.

Provides ``CameraSource`` with factory classmethods for different capture
backends (Jetson CSI via GStreamer, USB webcam via V4L2, video file).

Each source implements the iterator protocol and context manager so it
can be plugged directly into ``process_frames()``.

Usage::

    with CameraSource.csi() as cam:
        for frame in cam:
            ...

    with CameraSource.usb(0) as cam:
        for frame in cam:
            ...

    with CameraSource.file("video.mp4") as cam:
        for frame in cam:
            ...
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraSource:
    """Iterable frame source backed by OpenCV VideoCapture.

    Use the factory classmethods rather than constructing directly.
    """

    def __init__(self, cap: cv2.VideoCapture, name: str) -> None:
        self._cap = cap
        self.name = name
        self._stopped = False
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {name}")

    # -- factory classmethods ------------------------------------------------

    @classmethod
    def csi(
        cls,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        sensor_id: int = 0,
    ) -> "CameraSource":
        """Jetson CSI camera via GStreamer ``nvarguscamerasrc``."""
        gst = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"framerate=(fraction){fps}/1, format=(string)NV12 ! "
            f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink drop=1 max-buffers=2"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return cls(cap, f"CSI sensor {sensor_id} ({width}x{height}@{fps})")

    @classmethod
    def usb(
        cls,
        device: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ) -> "CameraSource":
        """USB webcam via V4L2."""
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cls(cap, f"USB /dev/video{device} ({width}x{height}@{fps})")

    @classmethod
    def file(cls, path: str) -> "CameraSource":
        """Video file (for testing without hardware)."""
        cap = cv2.VideoCapture(path)
        return cls(cap, f"file {path}")

    # -- properties ----------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int | None:
        """Total frame count (only meaningful for file sources)."""
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None

    # -- iterator protocol ---------------------------------------------------

    def __iter__(self) -> "CameraSource":
        return self

    def __next__(self) -> np.ndarray:
        if self._stopped:
            raise StopIteration
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration
        return frame

    # -- control -------------------------------------------------------------

    def stop(self) -> None:
        """Signal the iterator to stop (e.g. from a signal handler)."""
        self._stopped = True

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        self._cap.release()

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
