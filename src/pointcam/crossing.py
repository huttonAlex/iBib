"""Person detection and timing-line crossing detection.

YOLOv8n-pose for person detection (chest keypoint crossing) with persistent
bib association and bib-level deduplication.  Legacy MOG2 blob approach kept
but deprecated.

Classes:
    CentroidTracker              - Simple centroid-based object tracker
    PersonDetection              - Dataclass for a single pose detection
    PoseDetector                 - YOLOv8n-pose wrapper, extracts chest point
    TimingLine                   - Virtual line for crossing math
    CrossingDetector             - Fires when a tracked centroid crosses the timing line
    PersistentPersonBibAssociator - Accumulates person→bib mapping over time via voting
    BibCrossingDeduplicator      - Prevents same bib from firing twice
    CrossingEvent                - Dataclass for a single crossing event
    CrossingEventLog             - Writes crossing events to CSV
    BackgroundSubtractorManager  - (deprecated) MOG2 + morphological cleanup
    PersonBibAssociator          - (deprecated) Stateless spatial matching
    MergedBlobEstimator          - (deprecated) Area-based person count
"""

from __future__ import annotations

import csv
import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


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
        self,
        detections: List[Tuple[int, int, int, int]],
        centroids: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]]:
        """Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes.
            centroids: Optional pre-computed centroids (e.g. chest points from pose).
                If provided, must be same length as *detections*.  When ``None``,
                centroids are computed as bbox centers.

        Returns:
            Dict of {track_id: (centroid, bbox)}.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}

        # Calculate centroids (or use provided ones)
        if centroids is not None:
            input_centroids = list(centroids)
        else:
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
# PersonDetection (dataclass)
# ---------------------------------------------------------------------------


# COCO keypoint indices for torso
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12
_TORSO_INDICES = [_KP_LEFT_SHOULDER, _KP_RIGHT_SHOULDER, _KP_LEFT_HIP, _KP_RIGHT_HIP]

# Minimum keypoint confidence to consider a keypoint "visible"
_KP_CONF_THRESHOLD = 0.3


@dataclass
class PersonDetection:
    """A single person detection from YOLOv8n-pose."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    chest_point: Tuple[float, float]  # (cx, cy) in pixel coords
    keypoints: Optional[np.ndarray] = None  # (17, 3) array or None


# ---------------------------------------------------------------------------
# PoseDetector
# ---------------------------------------------------------------------------


class PoseDetector:
    """Wraps YOLOv8n-pose to detect persons and extract chest keypoints.

    Chest point = midpoint of visible torso keypoints (COCO indices:
    left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12).

    Fallback chain:
        1. Average of all visible torso keypoints (if >=2 visible)
        2. Bbox centroid (if <2 keypoints visible)

    Args:
        model_path: Path to YOLOv8n-pose weights (e.g. ``yolov8n-pose.pt``).
        conf: Minimum detection confidence.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf: float = 0.5,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """Run pose detection on a single frame.

        Returns:
            List of :class:`PersonDetection` with chest points.
        """
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        detections: List[PersonDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            kps_all = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i, (box, conf_val) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = box
                kps = kps_all[i] if kps_all is not None else None
                chest = self._chest_point(kps, (x1, y1, x2, y2))
                detections.append(
                    PersonDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf_val),
                        chest_point=chest,
                        keypoints=kps,
                    )
                )
        return detections

    @staticmethod
    def _chest_point(
        keypoints: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Compute chest point from keypoints with fallback to bbox center.

        Args:
            keypoints: (17, 3) array where columns are (x, y, conf), or None.
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            (cx, cy) chest point in pixel coordinates.
        """
        if keypoints is not None and keypoints.shape[0] >= max(_TORSO_INDICES) + 1:
            visible = []
            for idx in _TORSO_INDICES:
                if keypoints[idx, 2] >= _KP_CONF_THRESHOLD:
                    visible.append(keypoints[idx, :2])
            if len(visible) >= 2:
                pts = np.array(visible)
                return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        # Fallback: bbox centroid
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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
    the side changes.  A crossing is only confirmed after the object
    stays on the new side for *hysteresis_frames* consecutive checks,
    suppressing jitter from keypoint noise at the line boundary.

    Args:
        timing_line: The :class:`TimingLine` to detect crossings against.
        direction: ``"left_to_right"`` (+1 to -1), ``"right_to_left"``
            (-1 to +1), or ``"any"``.
        debounce_frames: Minimum frames between crossings for the same track.
        hysteresis_frames: Consecutive frames on the new side required to
            confirm a crossing.  Set to 1 for legacy (instant) behaviour.
    """

    def __init__(
        self,
        timing_line: TimingLine,
        direction: str = "any",
        debounce_frames: int = 60,
        hysteresis_frames: int = 3,
    ):
        self.timing_line = timing_line
        self.direction = direction
        self.debounce_frames = debounce_frames
        self.hysteresis_frames = hysteresis_frames
        # track_id -> last *confirmed* side (+1 / -1)
        self._prev_side: Dict[int, int] = {}
        # track_id -> frame number of last crossing
        self._last_crossing_frame: Dict[int, int] = {}
        # track_id -> (candidate_side, consecutive_count)
        self._pending: Dict[int, Tuple[int, int]] = {}

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

        # First observation — record side, no crossing
        if prev is None:
            self._prev_side[track_id] = side
            return False

        # Still on the same confirmed side — clear any pending transition
        if side == prev:
            self._pending.pop(track_id, None)
            return False

        # side != prev — potential transition
        pending = self._pending.get(track_id)
        if pending is not None and pending[0] == side:
            count = pending[1] + 1
        else:
            count = 1
        self._pending[track_id] = (side, count)

        if count < self.hysteresis_frames:
            return False

        # Confirmed crossing — apply debounce + direction checks
        last_frame = self._last_crossing_frame.get(track_id, -self.debounce_frames - 1)
        if frame_idx - last_frame < self.debounce_frames:
            # Debounced: still commit the side so we don't re-fire
            self._prev_side[track_id] = side
            self._pending.pop(track_id, None)
            return False

        if self.direction == "left_to_right" and not (prev == 1 and side == -1):
            self._prev_side[track_id] = side
            self._pending.pop(track_id, None)
            return False
        if self.direction == "right_to_left" and not (prev == -1 and side == 1):
            self._prev_side[track_id] = side
            self._pending.pop(track_id, None)
            return False

        self._prev_side[track_id] = side
        self._pending.pop(track_id, None)
        self._last_crossing_frame[track_id] = frame_idx
        return True

    def cleanup(self, active_track_ids: set):
        """Remove state for tracks that no longer exist."""
        for tid in list(self._prev_side.keys()):
            if tid not in active_track_ids:
                self._prev_side.pop(tid, None)
                self._last_crossing_frame.pop(tid, None)
                self._pending.pop(tid, None)


# ---------------------------------------------------------------------------
# PersistentPersonBibAssociator
# ---------------------------------------------------------------------------


class PersistentPersonBibAssociator:
    """Accumulates person→bib mapping over time via weighted majority voting.

    Each frame, spatial matching links person bboxes to bib bboxes.  For each
    matched bib, the OCR consensus number is looked up and a weighted vote is
    cast.  ``get_bib(pid)`` returns the bib number with the most weighted votes,
    subject to minimum vote and confidence thresholds.

    Vote weights by confidence level:
        HIGH: 3, MEDIUM: 2, LOW: 1, REJECT: 0 (skipped).

    Short bib numbers (fewer digits than expected) receive halved vote weight
    when ``expected_digit_counts`` is provided.

    Args:
        max_distance: Max pixel distance for bib-center-to-person-bbox fallback.
        memory_frames: Frames of inactivity before a person entry is cleaned up.
        min_votes: Minimum weighted votes for the winning bib to be emitted.
        min_confidence: Minimum vote fraction for the winning bib (0.0-1.0).
        expected_digit_counts: Set of expected digit counts (e.g. {3, 4}).
            Bibs with fewer digits than the minimum in this set get halved weight.
    """

    # Weighted votes per confidence level
    LEVEL_WEIGHTS: Dict[str, int] = {"high": 3, "medium": 2, "low": 1}

    def __init__(
        self,
        max_distance: float = 150.0,
        memory_frames: int = 120,
        min_votes: int = 1,
        min_confidence: float = 0.4,
        expected_digit_counts: Optional[set] = None,
    ):
        self.max_distance = max_distance
        self.memory_frames = memory_frames
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.expected_digit_counts = expected_digit_counts
        # Minimum expected digit count for short-bib penalty
        self._min_expected_digits: Optional[int] = None
        if expected_digit_counts:
            self._min_expected_digits = min(expected_digit_counts)
        # person_track_id → {bib_number → weighted_vote_count}
        self._votes: Dict[int, Dict[str, float]] = {}
        # person_track_id → last_frame_seen
        self._last_seen: Dict[int, int] = {}

    def _vote_weight(self, bib_number: str, level: str) -> float:
        """Compute vote weight for a bib reading.

        Applies level-based weight and short-bib penalty.
        """
        base = self.LEVEL_WEIGHTS.get(level, 0)
        if base == 0:
            return 0.0
        weight = float(base)
        # Short-bib penalty: halve weight for bibs shorter than expected
        if (
            self._min_expected_digits is not None
            and len(bib_number) < self._min_expected_digits
        ):
            weight *= 0.5
        return weight

    def update(
        self,
        person_tracked: Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]],
        bib_tracked: Dict[int, Tuple[Tuple[float, float], Tuple[int, int, int, int]]],
        final_consensus: Dict[int, Tuple[str, float, str]],
        frame_idx: int,
    ) -> None:
        """Run one frame of association and voting.

        Args:
            person_tracked: {pid: (centroid, bbox)} from person CentroidTracker.
            bib_tracked: {bid: (centroid, bbox)} from bib CentroidTracker.
            final_consensus: {bid: (number, conf, level)} from OCR voting.
            frame_idx: Current frame number.
        """
        for pid, (p_centroid, p_bbox) in person_tracked.items():
            self._last_seen[pid] = frame_idx
            px1, py1, px2, py2 = p_bbox
            best_bid: Optional[int] = None
            best_dist = float("inf")

            for bid, (b_centroid, b_bbox) in bib_tracked.items():
                bcx, bcy = b_centroid
                inside = px1 <= bcx <= px2 and py1 <= bcy <= py2
                dist = np.sqrt(
                    (p_centroid[0] - bcx) ** 2 + (p_centroid[1] - bcy) ** 2
                )
                if inside:
                    if dist < best_dist:
                        best_dist = dist
                        best_bid = bid
                elif best_bid is None and dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_bid = bid

            if best_bid is not None and best_bid in final_consensus:
                bib_number, conf, level = final_consensus[best_bid]
                weight = self._vote_weight(bib_number, level)
                if weight > 0:
                    if pid not in self._votes:
                        self._votes[pid] = {}
                    self._votes[pid][bib_number] = (
                        self._votes[pid].get(bib_number, 0.0) + weight
                    )

    def get_bib(self, person_track_id: int) -> Optional[str]:
        """Return the majority-voted bib number, or None if below thresholds.

        Returns None if:
        - No votes recorded.
        - Winning bib has fewer than ``min_votes`` weighted votes.
        - Winning bib's vote fraction is below ``min_confidence``.
        """
        votes = self._votes.get(person_track_id)
        if not votes:
            return None
        winner = max(votes, key=votes.get)  # type: ignore[arg-type]
        winner_votes = votes[winner]
        if winner_votes < self.min_votes:
            return None
        total = sum(votes.values())
        if total > 0 and winner_votes / total < self.min_confidence:
            return None
        return winner

    def get_bib_confidence(self, person_track_id: int) -> float:
        """Return the vote fraction for the winning bib (0.0-1.0)."""
        votes = self._votes.get(person_track_id)
        if not votes:
            return 0.0
        total = sum(votes.values())
        best = max(votes.values())
        return best / total if total > 0 else 0.0

    def cleanup(self, frame_idx: int) -> None:
        """Remove entries not seen for *memory_frames*."""
        stale = [
            pid
            for pid, last in self._last_seen.items()
            if frame_idx - last > self.memory_frames
        ]
        for pid in stale:
            self._votes.pop(pid, None)
            self._last_seen.pop(pid, None)


# ---------------------------------------------------------------------------
# BibCrossingDeduplicator
# ---------------------------------------------------------------------------


class BibCrossingDeduplicator:
    """Suppresses duplicate crossing events for the same bib number.

    Uses two mechanisms:
    1. **Frame debounce**: A bib that crossed within *debounce_frames* is suppressed.
    2. **Escalating confidence**: After the first emission, subsequent emissions
       require progressively higher confidence — catching partial-bib misreads
       (e.g. "65" from "1265") that appear many times minutes apart.

    ``"UNKNOWN"`` bibs always pass (cannot deduplicate unknowns).

    Args:
        debounce_frames: Minimum frames between crossings for the same bib.
        escalation_thresholds: Confidence thresholds per emission count.
            Default: 1st=0.0, 2nd>=0.7, 3rd+>=0.9.
    """

    DEFAULT_ESCALATION = {1: 0.0, 2: 0.7}  # 3+ uses fallback
    ESCALATION_FALLBACK = 0.9  # Required confidence for 3rd+ emission

    def __init__(
        self,
        debounce_frames: int = 300,
        escalation_thresholds: Optional[Dict[int, float]] = None,
    ):
        # bib_number → last frame it was reported
        self._recently_crossed: Dict[str, int] = {}
        # bib_number → total emission count
        self._emission_count: Dict[str, int] = {}
        self.debounce_frames = debounce_frames
        self.escalation_thresholds = escalation_thresholds or self.DEFAULT_ESCALATION.copy()
        # track_id → last frame an UNKNOWN was emitted (for per-track dedup)
        self._track_last_crossing: Dict[int, int] = {}

    def should_emit(
        self,
        bib_number: str,
        frame_idx: int,
        confidence: float = 1.0,
        track_id: Optional[int] = None,
    ) -> bool:
        """Return True if this crossing should be emitted (not a duplicate).

        Args:
            bib_number: The bib number string (or ``"UNKNOWN"``).
            frame_idx: Current frame number.
            confidence: Confidence score for this crossing (0.0-1.0).
            track_id: Person/bib track ID.  When provided, UNKNOWN crossings
                from the same track within the debounce window are suppressed.
        """
        if bib_number == "UNKNOWN":
            if track_id is not None:
                last = self._track_last_crossing.get(track_id)
                if last is not None and frame_idx - last < self.debounce_frames:
                    return False
                self._track_last_crossing[track_id] = frame_idx
            return True

        # Frame-based debounce
        last = self._recently_crossed.get(bib_number)
        if last is not None and frame_idx - last < self.debounce_frames:
            return False

        # Escalating confidence check
        count = self._emission_count.get(bib_number, 0)
        next_count = count + 1
        required = self.escalation_thresholds.get(next_count, self.ESCALATION_FALLBACK)
        if confidence < required:
            return False

        self._recently_crossed[bib_number] = frame_idx
        self._emission_count[bib_number] = next_count
        return True


# ---------------------------------------------------------------------------
# BackgroundSubtractorManager (deprecated)
# ---------------------------------------------------------------------------


class BackgroundSubtractorManager:
    """Wraps MOG2 background subtraction + morphological cleanup.

    .. deprecated::
        Use :class:`PoseDetector` for person detection instead.  MOG2 produces
        noisy results with merged blobs.

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
        warnings.warn(
            "BackgroundSubtractorManager is deprecated. Use PoseDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
# PersonBibAssociator (deprecated)
# ---------------------------------------------------------------------------


class PersonBibAssociator:
    """Matches person blobs to bib detections by spatial proximity.

    .. deprecated::
        Use :class:`PersistentPersonBibAssociator` for persistent association
        with temporal voting.

    A bib is associated with a person if the bib center falls inside the
    person bounding box, or if it is the closest person within a distance
    threshold.

    Args:
        max_distance: Maximum pixel distance for fallback association when
            the bib center is not inside any person bbox.
    """

    def __init__(self, max_distance: float = 150.0):
        warnings.warn(
            "PersonBibAssociator is deprecated. Use PersistentPersonBibAssociator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
# MergedBlobEstimator (deprecated)
# ---------------------------------------------------------------------------


class MergedBlobEstimator:
    """Estimates the number of persons in a merged (large) blob.

    .. deprecated::
        Use :class:`PoseDetector` which detects individual persons without
        merging.

    Uses a simple area heuristic: estimated_count = blob_area / typical_person_area.

    Args:
        typical_person_area: Expected area of a single person in pixels.
            Depends on camera distance/resolution.  A reasonable default
            for 1080p finish-line footage is ~8000-15000 px.
    """

    def __init__(self, typical_person_area: float = 10000.0):
        warnings.warn(
            "MergedBlobEstimator is deprecated. Use PoseDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    chest_point: Optional[Tuple[float, float]] = None  # (cx, cy) pixel coords
    source: str = "pose"  # "pose", "bib_tracker", or "mog2"


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
        "chest_x",
        "chest_y",
        "source",
    ]

    def __init__(self, path: Path):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def write(self, event: CrossingEvent):
        """Write a single crossing event row."""
        chest_x = f"{event.chest_point[0]:.1f}" if event.chest_point else ""
        chest_y = f"{event.chest_point[1]:.1f}" if event.chest_point else ""
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
                chest_x,
                chest_y,
                event.source,
            ]
        )

    def close(self):
        """Flush and close the file."""
        self._file.close()
