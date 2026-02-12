"""Person detection and timing-line crossing detection.

Software-only approach using OpenCV background subtraction + blob tracking.
Zero additional GPU cost (~3-5ms CPU overhead per frame).

Classes:
    TimingLine           - Virtual line for crossing math
    CrossingDetector     - Fires when a tracked centroid crosses the timing line
    BackgroundSubtractorManager - MOG2 + morphological cleanup, person-sized blobs
    PersonBibAssociator  - Matches person blobs to bib detections by spatial proximity
    MergedBlobEstimator  - Estimates person count in merged blobs via area heuristic
    CrossingEvent        - Dataclass for a single crossing event
    CrossingEventLog     - Writes crossing events to CSV
    CentroidTracker      - Simple centroid-based object tracker
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CentroidTracker (moved from test_video_pipeline.py)
# ---------------------------------------------------------------------------


class CentroidTracker:
    """Simple centroid-based tracker for maintaining track IDs across frames.

    Args:
        max_disappeared: Frames before a track is dropped.
        max_distance: Maximum pixel distance for matching.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.next_id = 0
        self.objects: Dict[int, Tuple[float, float]] = OrderedDict()
        self.bboxes: Dict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int]):
        """Register a new object."""
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]]:
        """Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            Dict of {track_id: (centroid, bbox)}.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}

        # Calculate centroids
        input_centroids = []
        for x1, y1, x2, y2 in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, detections[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

            # Match using greedy approach
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = detections[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            for row in set(range(len(object_centroids))) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col], detections[col])

        return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}


# ---------------------------------------------------------------------------
# TimingLine
# ---------------------------------------------------------------------------


class TimingLine:
    """Virtual timing line defined by two endpoints in normalized [0,1] coordinates.

    The line divides the frame into two sides.  ``side_of_point`` returns +1 or -1
    depending on which side of the line a point falls.

    Args:
        x1, y1, x2, y2: Normalized coordinates of the two endpoints (0.0-1.0).
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def side_of_point(self, px: float, py: float) -> int:
        """Return +1 or -1 indicating which side of the line *px, py* falls on.

        Uses the cross-product sign of (line_vector) x (point - line_start).
        Returns 0 exactly on the line (rare with floats).
        """
        cross = (self.x2 - self.x1) * (py - self.y1) - (self.y2 - self.y1) * (px - self.x1)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0

    def to_pixel_coords(
        self, frame_width: int, frame_height: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert normalized coords to pixel coordinates."""
        pt1 = (int(self.x1 * frame_width), int(self.y1 * frame_height))
        pt2 = (int(self.x2 * frame_width), int(self.y2 * frame_height))
        return pt1, pt2


# ---------------------------------------------------------------------------
# CrossingDetector
# ---------------------------------------------------------------------------


class CrossingDetector:
    """Detects when a tracked object's centroid crosses the timing line.

    Tracks the previous side-of-line for each track ID and fires when
    the side changes (with debounce to prevent double-counting).

    Args:
        timing_line: The :class:`TimingLine` to detect crossings against.
        direction: ``"left_to_right"`` (+1 to -1), ``"right_to_left"``
            (-1 to +1), or ``"any"``.
        debounce_frames: Minimum frames between crossings for the same track.
    """

    def __init__(
        self,
        timing_line: TimingLine,
        direction: str = "any",
        debounce_frames: int = 60,
    ):
        self.timing_line = timing_line
        self.direction = direction
        self.debounce_frames = debounce_frames
        # track_id -> last known side (+1 / -1)
        self._prev_side: Dict[int, int] = {}
        # track_id -> frame number of last crossing
        self._last_crossing_frame: Dict[int, int] = {}

    def check(
        self,
        track_id: int,
        cx_norm: float,
        cy_norm: float,
        frame_idx: int,
    ) -> bool:
        """Check whether *track_id* has just crossed the timing line.

        Args:
            track_id: Unique ID from the centroid tracker.
            cx_norm: Centroid x in normalized [0,1] coords.
            cy_norm: Centroid y in normalized [0,1] coords.
            frame_idx: Current frame number (for debounce).

        Returns:
            ``True`` if a crossing was detected this frame.
        """
        side = self.timing_line.side_of_point(cx_norm, cy_norm)
        if side == 0:
            return False

        prev = self._prev_side.get(track_id)
        self._prev_side[track_id] = side

        if prev is None or prev == side:
            return False

        # Debounce
        last_frame = self._last_crossing_frame.get(track_id, -self.debounce_frames - 1)
        if frame_idx - last_frame < self.debounce_frames:
            return False

        # Direction filter
        if self.direction == "left_to_right" and not (prev == 1 and side == -1):
            return False
        if self.direction == "right_to_left" and not (prev == -1 and side == 1):
            return False

        self._last_crossing_frame[track_id] = frame_idx
        return True

    def cleanup(self, active_track_ids: set):
        """Remove state for tracks that no longer exist."""
        for tid in list(self._prev_side.keys()):
            if tid not in active_track_ids:
                self._prev_side.pop(tid, None)
                self._last_crossing_frame.pop(tid, None)


# ---------------------------------------------------------------------------
# BackgroundSubtractorManager
# ---------------------------------------------------------------------------


class BackgroundSubtractorManager:
    """Wraps MOG2 background subtraction + morphological cleanup.

    Extracts person-sized blobs filtered by area and aspect ratio.

    Args:
        history: Number of frames for background model.
        var_threshold: Variance threshold for MOG2.
        detect_shadows: Whether MOG2 should detect shadows.
        min_area: Minimum contour area in pixels to be considered a person.
        max_area: Maximum contour area in pixels.
        min_aspect_ratio: Minimum height/width ratio (persons are tall).
        max_aspect_ratio: Maximum height/width ratio.
        morph_kernel_size: Kernel size for morphological operations.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 50.0,
        detect_shadows: bool = True,
        min_area: int = 2000,
        max_area: int = 200000,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 5.0,
        morph_kernel_size: int = 5,
    ):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def process(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Apply background subtraction and return person-sized bounding boxes.

        Args:
            frame: BGR frame from video capture.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes for detected blobs.
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold: remove shadows (value 127 in MOG2), keep foreground (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup: close gaps, then open to remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0:
                continue
            aspect = h / w

            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue

            bboxes.append((x, y, x + w, y + h))

        return bboxes


# ---------------------------------------------------------------------------
# PersonBibAssociator
# ---------------------------------------------------------------------------


class PersonBibAssociator:
    """Matches person blobs to bib detections by spatial proximity.

    A bib is associated with a person if the bib center falls inside the
    person bounding box, or if it is the closest person within a distance
    threshold.

    Args:
        max_distance: Maximum pixel distance for fallback association when
            the bib center is not inside any person bbox.
    """

    def __init__(self, max_distance: float = 150.0):
        self.max_distance = max_distance

    def associate(
        self,
        person_bboxes: Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]],
        bib_bboxes: Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]],
    ) -> Dict[int, Optional[int]]:
        """Associate person track IDs with bib track IDs.

        Args:
            person_bboxes: {person_track_id: (centroid, (x1,y1,x2,y2))}.
            bib_bboxes: {bib_track_id: (centroid, (x1,y1,x2,y2))}.

        Returns:
            Dict mapping person_track_id -> bib_track_id (or None if unmatched).
        """
        result: Dict[int, Optional[int]] = {}
        used_bibs: set = set()

        for pid, (p_centroid, p_bbox) in person_bboxes.items():
            px1, py1, px2, py2 = p_bbox
            best_bib_id: Optional[int] = None
            best_dist = float("inf")

            for bid, (b_centroid, b_bbox) in bib_bboxes.items():
                if bid in used_bibs:
                    continue
                bcx, bcy = b_centroid

                # Prefer: bib center inside person bbox
                inside = px1 <= bcx <= px2 and py1 <= bcy <= py2
                dist = np.sqrt(
                    (p_centroid[0] - bcx) ** 2 + (p_centroid[1] - bcy) ** 2
                )

                if inside:
                    # Among bibs inside, pick closest
                    if dist < best_dist:
                        best_dist = dist
                        best_bib_id = bid
                elif best_bib_id is None and dist < self.max_distance and dist < best_dist:
                    # Fallback: closest bib within distance
                    best_dist = dist
                    best_bib_id = bid

            result[pid] = best_bib_id
            if best_bib_id is not None:
                used_bibs.add(best_bib_id)

        return result


# ---------------------------------------------------------------------------
# MergedBlobEstimator
# ---------------------------------------------------------------------------


class MergedBlobEstimator:
    """Estimates the number of persons in a merged (large) blob.

    Uses a simple area heuristic: estimated_count = blob_area / typical_person_area.

    Args:
        typical_person_area: Expected area of a single person in pixels.
            Depends on camera distance/resolution.  A reasonable default
            for 1080p finish-line footage is ~8000-15000 px.
    """

    def __init__(self, typical_person_area: float = 10000.0):
        self.typical_person_area = typical_person_area

    def estimate(self, bbox: Tuple[int, int, int, int]) -> int:
        """Estimate the number of persons in a bounding box.

        Returns at least 1.
        """
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        count = max(1, round(area / self.typical_person_area))
        return count


# ---------------------------------------------------------------------------
# CrossingEvent
# ---------------------------------------------------------------------------


@dataclass
class CrossingEvent:
    """A single timing-line crossing event."""

    sequence: int
    frame_idx: int
    timestamp_sec: float
    person_track_id: int
    bib_number: str  # Bib number or "UNKNOWN"
    confidence: float  # 0.0-1.0 (0 for UNKNOWN)
    estimated_count: int = 1  # >1 if merged blob
    person_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# CrossingEventLog
# ---------------------------------------------------------------------------


class CrossingEventLog:
    """Writes crossing events to a CSV file.

    Args:
        path: Output CSV file path.
    """

    HEADER = [
        "sequence",
        "frame",
        "time_sec",
        "person_track_id",
        "bib_number",
        "confidence",
        "estimated_count",
        "person_x1",
        "person_y1",
        "person_x2",
        "person_y2",
    ]

    def __init__(self, path: Path):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def write(self, event: CrossingEvent):
        """Write a single crossing event row."""
        self._writer.writerow(
            [
                event.sequence,
                event.frame_idx,
                f"{event.timestamp_sec:.3f}",
                event.person_track_id,
                event.bib_number,
                f"{event.confidence:.3f}",
                event.estimated_count,
                event.person_bbox[0],
                event.person_bbox[1],
                event.person_bbox[2],
                event.person_bbox[3],
            ]
        )

    def close(self):
        """Flush and close the file."""
        self._file.close()
