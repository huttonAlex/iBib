"""Tests for pointcam.crossing module."""

import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from pointcam.crossing import (
    BackgroundSubtractorManager,
    BibCrossingDeduplicator,
    CentroidTracker,
    CrossingDetector,
    CrossingEvent,
    CrossingEventLog,
    MergedBlobEstimator,
    PersistentPersonBibAssociator,
    PersonBibAssociator,
    PersonDetection,
    PoseDetector,
    TimingLine,
)


# ---------------------------------------------------------------------------
# TimingLine
# ---------------------------------------------------------------------------


class TestTimingLine:
    def test_vertical_line_sides(self):
        """Points left/right of a vertical center line."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        # Left of the line
        assert line.side_of_point(0.3, 0.5) != 0
        # Right of the line
        assert line.side_of_point(0.7, 0.5) != 0
        # Opposite sides should have opposite signs
        assert line.side_of_point(0.3, 0.5) != line.side_of_point(0.7, 0.5)

    def test_horizontal_line_sides(self):
        """Points above/below a horizontal center line."""
        line = TimingLine(0.0, 0.5, 1.0, 0.5)
        assert line.side_of_point(0.5, 0.3) != line.side_of_point(0.5, 0.7)

    def test_point_on_line(self):
        """A point exactly on the line returns 0."""
        line = TimingLine(0.0, 0.0, 1.0, 1.0)
        assert line.side_of_point(0.5, 0.5) == 0

    def test_to_pixel_coords(self):
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        pt1, pt2 = line.to_pixel_coords(1920, 1080)
        assert pt1 == (960, 0)
        assert pt2 == (960, 1080)

    def test_diagonal_line(self):
        """Diagonal line should still partition the plane."""
        line = TimingLine(0.0, 0.0, 1.0, 1.0)
        # (0.2, 0.8) is above the diagonal (left side)
        # (0.8, 0.2) is below the diagonal (right side)
        s1 = line.side_of_point(0.2, 0.8)
        s2 = line.side_of_point(0.8, 0.2)
        assert s1 != 0
        assert s2 != 0
        assert s1 != s2


# ---------------------------------------------------------------------------
# CrossingDetector
# ---------------------------------------------------------------------------


class TestCrossingDetector:
    def test_crossing_fires_once(self):
        """Moving from one side to the other fires after hysteresis frames."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="any", debounce_frames=5, hysteresis_frames=1
        )

        # Establish position on the left
        assert det.check(1, 0.3, 0.5, 1) is False
        # Cross to the right (hysteresis=1 → fires immediately)
        assert det.check(1, 0.7, 0.5, 2) is True
        # Stay on the right — no new crossing
        assert det.check(1, 0.8, 0.5, 3) is False

    def test_crossing_with_default_hysteresis(self):
        """With default hysteresis=10, need 10 frames on new side to fire."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=50)

        det.check(1, 0.3, 0.5, 1)  # establish left
        # Frames 2-10: on new side but below hysteresis threshold
        for f in range(2, 11):
            assert det.check(1, 0.7, 0.5, f) is False
        # Frame 11: 10th consecutive frame on new side → confirmed
        assert det.check(1, 0.7, 0.5, 11) is True

    def test_debounce(self):
        """A second crossing within debounce window is suppressed."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="any", debounce_frames=10, hysteresis_frames=1
        )

        det.check(1, 0.3, 0.5, 1)  # establish left
        assert det.check(1, 0.7, 0.5, 2) is True  # cross right (frame 2)

        # Back to left within debounce — suppressed
        assert det.check(1, 0.3, 0.5, 5) is False  # 5-2=3 < 10
        # Right again — still within debounce
        assert det.check(1, 0.7, 0.5, 6) is False  # 6-2=4 < 10

        # After debounce expires (20-2=18 > 10), next side change fires
        assert det.check(1, 0.3, 0.5, 20) is True  # crossing accepted

    def test_direction_filter_left_to_right(self):
        """Only left-to-right crossings fire."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="left_to_right", debounce_frames=0, hysteresis_frames=1
        )

        det.check(1, 0.3, 0.5, 1)  # left side
        # For a vertical line at x=0.5 going from (0.5,0) to (0.5,1):
        # side_of_point cross product = (0)(py-0) - (1)(px-0.5)
        # px=0.3 => cross = -(-0.2) = 0.2 => side=+1
        # px=0.7 => cross = -(0.2) = -0.2 => side=-1
        # left_to_right means prev=+1, side=-1
        result = det.check(1, 0.7, 0.5, 2)
        assert result is True

    def test_direction_filter_rejects_opposite(self):
        """Right-to-left crossing does not fire when direction is left_to_right."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="left_to_right", debounce_frames=0, hysteresis_frames=1
        )

        det.check(1, 0.7, 0.5, 1)  # right side
        result = det.check(1, 0.3, 0.5, 2)  # cross to left
        assert result is False

    def test_multiple_tracks(self):
        """Independent tracks are tracked separately."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="any", debounce_frames=0, hysteresis_frames=1
        )

        det.check(1, 0.3, 0.5, 1)
        det.check(2, 0.3, 0.5, 1)

        assert det.check(1, 0.7, 0.5, 2) is True
        assert det.check(2, 0.7, 0.5, 2) is True

    def test_cleanup_removes_stale_tracks(self):
        """cleanup() removes state for tracks no longer active."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=0)

        det.check(1, 0.3, 0.5, 1)
        det.check(2, 0.3, 0.5, 1)
        det.cleanup({1})  # track 2 removed

        assert 2 not in det._prev_side
        assert 1 in det._prev_side
        assert 2 not in det._pending

    def test_no_crossing_without_prior_position(self):
        """First observation of a track is not a crossing."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=0)
        assert det.check(1, 0.7, 0.5, 1) is False

    def test_hysteresis_suppresses_jitter(self):
        """Rapid side alternation (jitter) never produces a crossing."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="any", debounce_frames=0, hysteresis_frames=3
        )

        det.check(1, 0.3, 0.5, 0)  # establish left
        # Alternate sides every frame — never stays on new side long enough
        for i in range(1, 20):
            x = 0.7 if i % 2 == 1 else 0.3
            result = det.check(1, x, 0.5, i)
            assert result is False, f"Jitter fired at frame {i}"

    def test_hysteresis_clean_crossing(self):
        """A clean crossing (stay on new side for N frames) fires exactly once."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(
            line, direction="any", debounce_frames=0, hysteresis_frames=3
        )

        det.check(1, 0.3, 0.5, 0)  # establish left
        assert det.check(1, 0.7, 0.5, 1) is False  # frame 1 on right
        assert det.check(1, 0.7, 0.5, 2) is False  # frame 2 on right
        assert det.check(1, 0.7, 0.5, 3) is True   # frame 3 → confirmed
        assert det.check(1, 0.7, 0.5, 4) is False  # no re-fire


# ---------------------------------------------------------------------------
# PoseDetector (chest point computation)
# ---------------------------------------------------------------------------


class TestPoseDetector:
    def test_chest_point_four_keypoints(self):
        """Chest is average of all 4 torso keypoints when all visible."""
        kps = np.zeros((17, 3), dtype=np.float32)
        # left_shoulder(5), right_shoulder(6), left_hip(11), right_hip(12)
        kps[5] = [100, 200, 0.9]
        kps[6] = [200, 200, 0.9]
        kps[11] = [100, 400, 0.9]
        kps[12] = [200, 400, 0.9]
        bbox = (80, 180, 220, 420)

        cx, cy = PoseDetector._chest_point(kps, bbox)
        assert abs(cx - 150.0) < 1.0
        assert abs(cy - 300.0) < 1.0

    def test_chest_point_two_shoulders_only(self):
        """Fallback: average of 2 visible keypoints (shoulders only)."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [100, 200, 0.9]  # left_shoulder visible
        kps[6] = [200, 200, 0.8]  # right_shoulder visible
        # hips not visible (conf=0)
        bbox = (80, 180, 220, 420)

        cx, cy = PoseDetector._chest_point(kps, bbox)
        assert abs(cx - 150.0) < 1.0
        assert abs(cy - 200.0) < 1.0

    def test_chest_point_bbox_fallback(self):
        """Falls back to bbox centroid when <2 keypoints visible."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [100, 200, 0.9]  # only 1 visible
        bbox = (100, 200, 300, 400)

        cx, cy = PoseDetector._chest_point(kps, bbox)
        assert abs(cx - 200.0) < 1.0
        assert abs(cy - 300.0) < 1.0

    def test_chest_point_no_keypoints(self):
        """Falls back to bbox centroid when keypoints is None."""
        bbox = (100, 200, 300, 400)
        cx, cy = PoseDetector._chest_point(None, bbox)
        assert abs(cx - 200.0) < 1.0
        assert abs(cy - 300.0) < 1.0

    def test_chest_point_three_visible(self):
        """Average of 3 visible keypoints."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [100, 200, 0.9]
        kps[6] = [200, 200, 0.9]
        kps[11] = [150, 400, 0.9]
        # right_hip not visible
        bbox = (80, 180, 220, 420)

        cx, cy = PoseDetector._chest_point(kps, bbox)
        assert abs(cx - 150.0) < 1.0
        assert abs(cy - 266.67) < 1.0


# ---------------------------------------------------------------------------
# PersistentPersonBibAssociator
# ---------------------------------------------------------------------------


class TestPersistentPersonBibAssociator:
    def test_accumulation_and_voting(self):
        """Weighted votes accumulate over frames and majority wins."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=3
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}
        consensus = {10: ("1234", 0.95, "high")}

        # Cast multiple votes (HIGH = 3 weight each → 5 frames × 3 = 15 total)
        for i in range(5):
            assoc.update(person_tracked, bib_tracked, consensus, frame_idx=i)

        assert assoc.get_bib(1) == "1234"
        assert assoc.get_bib_confidence(1) == 1.0

    def test_majority_voting_weighted(self):
        """HIGH votes (weight 3) outweigh MEDIUM votes (weight 2)."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=1
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}

        # 2 HIGH votes for "1234" → 2 × 3 = 6 weighted
        consensus_a = {10: ("1234", 0.95, "high")}
        for i in range(2):
            assoc.update(person_tracked, bib_tracked, consensus_a, frame_idx=i)

        # 3 MEDIUM votes for "5678" → 3 × 2 = 6 weighted (tie, 1234 still wins by order)
        # Use 4 MEDIUM votes → 4 × 2 = 8 weighted
        consensus_b = {10: ("5678", 0.90, "medium")}
        for i in range(2, 6):
            assoc.update(person_tracked, bib_tracked, consensus_b, frame_idx=i)

        # 5678 has 8 weighted votes vs 1234's 6 weighted votes → 5678 wins
        assert assoc.get_bib(1) == "5678"

    def test_low_confidence_votes(self):
        """LOW confidence gets weight 1 (contributes but doesn't dominate)."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=1
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}

        # 5 LOW votes for "1234" → 5 × 1 = 5 weighted
        consensus_low = {10: ("1234", 0.55, "low")}
        for i in range(5):
            assoc.update(person_tracked, bib_tracked, consensus_low, frame_idx=i)

        # 2 HIGH votes for "5678" → 2 × 3 = 6 weighted
        consensus_high = {10: ("5678", 0.95, "high")}
        for i in range(5, 7):
            assoc.update(person_tracked, bib_tracked, consensus_high, frame_idx=i)

        # 5678 wins (6 > 5)
        assert assoc.get_bib(1) == "5678"

    def test_reject_filtered_out(self):
        """REJECT confidence level produces zero votes."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=1
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}
        consensus = {10: ("9999", 0.20, "reject")}

        assoc.update(person_tracked, bib_tracked, consensus, frame_idx=0)
        assert assoc.get_bib(1) is None

    def test_min_votes_threshold(self):
        """Bib not emitted if below min_votes weighted threshold."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=5
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}

        # 1 HIGH vote → 3 weighted (< min_votes=5)
        consensus = {10: ("1234", 0.95, "high")}
        assoc.update(person_tracked, bib_tracked, consensus, frame_idx=0)
        assert assoc.get_bib(1) is None

        # 2nd HIGH vote → 6 weighted (>= min_votes=5)
        assoc.update(person_tracked, bib_tracked, consensus, frame_idx=1)
        assert assoc.get_bib(1) == "1234"

    def test_min_confidence_threshold(self):
        """Bib not emitted if vote fraction is below min_confidence."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=100, min_votes=1, min_confidence=0.6
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}

        # 1 HIGH vote for "1234" → 3 weighted
        consensus_a = {10: ("1234", 0.95, "high")}
        assoc.update(person_tracked, bib_tracked, consensus_a, frame_idx=0)

        # 1 HIGH vote for "5678" → 3 weighted
        consensus_b = {10: ("5678", 0.95, "high")}
        assoc.update(person_tracked, bib_tracked, consensus_b, frame_idx=1)

        # 50/50 split → confidence = 0.5 < 0.6 → None
        assert assoc.get_bib(1) is None

    def test_short_bib_penalty(self):
        """Short bibs get halved vote weight when expected_digit_counts is set."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200,
            memory_frames=100,
            min_votes=1,
            expected_digit_counts={3, 4},
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}

        # 2 HIGH votes for "65" (2-digit, short) → 2 × 3 × 0.5 = 3.0 weighted
        consensus_short = {10: ("65", 0.90, "high")}
        for i in range(2):
            assoc.update(person_tracked, bib_tracked, consensus_short, frame_idx=i)

        # 1 HIGH vote for "1265" (4-digit, normal) → 1 × 3 = 3.0 weighted
        consensus_normal = {10: ("1265", 0.90, "high")}
        assoc.update(person_tracked, bib_tracked, consensus_normal, frame_idx=2)

        # Tied at 3.0 each — but one more normal vote tips it
        assoc.update(person_tracked, bib_tracked, consensus_normal, frame_idx=3)
        assert assoc.get_bib(1) == "1265"

    def test_cleanup_stale_entries(self):
        """Stale entries are removed after memory_frames."""
        assoc = PersistentPersonBibAssociator(
            max_distance=200, memory_frames=10, min_votes=1
        )
        person_tracked = {1: ((250.0, 300.0), (200, 100, 300, 500))}
        bib_tracked = {10: ((250.0, 250.0), (230, 230, 270, 270))}
        consensus = {10: ("1234", 0.95, "high")}

        assoc.update(person_tracked, bib_tracked, consensus, frame_idx=0)
        assert assoc.get_bib(1) == "1234"

        # Cleanup at frame 20 (0 + 10 < 20)
        assoc.cleanup(frame_idx=20)
        assert assoc.get_bib(1) is None

    def test_no_match_when_bib_too_far(self):
        """No votes cast when bib center is too far from person."""
        assoc = PersistentPersonBibAssociator(
            max_distance=50, memory_frames=100, min_votes=1
        )
        person_tracked = {1: ((100.0, 100.0), (50, 50, 150, 150))}
        bib_tracked = {10: ((500.0, 500.0), (480, 480, 520, 520))}
        consensus = {10: ("1234", 0.95, "high")}

        assoc.update(person_tracked, bib_tracked, consensus, frame_idx=0)
        assert assoc.get_bib(1) is None


# ---------------------------------------------------------------------------
# BibCrossingDeduplicator
# ---------------------------------------------------------------------------


class TestBibCrossingDeduplicator:
    def test_first_crossing_allowed(self):
        """First crossing for a bib is always emitted (threshold=0.0)."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("1234", frame_idx=100, confidence=0.5) is True

    def test_duplicate_suppressed(self):
        """Same bib within debounce window is suppressed."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("1234", frame_idx=100, confidence=0.95) is True
        assert dedup.should_emit("1234", frame_idx=200, confidence=0.95) is False

    def test_allowed_after_debounce(self):
        """Same bib allowed again after debounce expires (if confidence high enough)."""
        dedup = BibCrossingDeduplicator(debounce_frames=100)
        assert dedup.should_emit("1234", frame_idx=100, confidence=0.95) is True
        # 2nd emission requires >= 0.7 and debounce expired (250-100=150 > 100)
        assert dedup.should_emit("1234", frame_idx=250, confidence=0.8) is True

    def test_unknown_always_passes_without_track(self):
        """UNKNOWN bibs without track_id are never deduplicated."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("UNKNOWN", frame_idx=100) is True
        assert dedup.should_emit("UNKNOWN", frame_idx=101) is True
        assert dedup.should_emit("UNKNOWN", frame_idx=102) is True

    def test_unknown_dedup_by_track(self):
        """Only 1 UNKNOWN per track — subsequent UNKNOWNs permanently suppressed."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("UNKNOWN", frame_idx=100, track_id=5) is True
        assert dedup.should_emit("UNKNOWN", frame_idx=101, track_id=5) is False
        assert dedup.should_emit("UNKNOWN", frame_idx=102, track_id=5) is False
        # Even after debounce expires — still suppressed (1 per track)
        assert dedup.should_emit("UNKNOWN", frame_idx=500, track_id=5) is False

    def test_unknown_different_tracks(self):
        """Different track_ids both pass for UNKNOWN."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("UNKNOWN", frame_idx=100, track_id=5) is True
        assert dedup.should_emit("UNKNOWN", frame_idx=100, track_id=6) is True
        assert dedup.should_emit("UNKNOWN", frame_idx=101, track_id=7) is True

    def test_different_bibs_independent(self):
        """Different bibs are tracked independently."""
        dedup = BibCrossingDeduplicator(debounce_frames=300)
        assert dedup.should_emit("1234", frame_idx=100, confidence=0.95) is True
        assert dedup.should_emit("5678", frame_idx=100, confidence=0.95) is True
        assert dedup.should_emit("1234", frame_idx=150, confidence=0.95) is False
        assert dedup.should_emit("5678", frame_idx=150, confidence=0.95) is False

    def test_escalating_confidence_2nd_emission(self):
        """2nd emission of same bib requires confidence >= 0.7."""
        dedup = BibCrossingDeduplicator(debounce_frames=10)
        # 1st emission: any confidence
        assert dedup.should_emit("65", frame_idx=0, confidence=0.5) is True
        # 2nd emission: debounce expired but confidence too low
        assert dedup.should_emit("65", frame_idx=100, confidence=0.6) is False
        # 2nd emission: high enough confidence
        assert dedup.should_emit("65", frame_idx=200, confidence=0.75) is True

    def test_escalating_confidence_3rd_emission(self):
        """3rd+ emission requires confidence >= 0.9."""
        dedup = BibCrossingDeduplicator(debounce_frames=10)
        assert dedup.should_emit("65", frame_idx=0, confidence=0.95) is True   # 1st
        assert dedup.should_emit("65", frame_idx=100, confidence=0.85) is True  # 2nd (>=0.7)
        # 3rd emission: needs 0.9
        assert dedup.should_emit("65", frame_idx=200, confidence=0.85) is False
        assert dedup.should_emit("65", frame_idx=300, confidence=0.95) is True  # 3rd (>=0.9)

    def test_default_confidence_backwards_compat(self):
        """Calling should_emit without confidence defaults to 1.0 (always passes)."""
        dedup = BibCrossingDeduplicator(debounce_frames=10)
        assert dedup.should_emit("1234", frame_idx=0) is True
        assert dedup.should_emit("1234", frame_idx=100) is True  # default conf=1.0 >= 0.7


# ---------------------------------------------------------------------------
# BackgroundSubtractorManager (deprecated)
# ---------------------------------------------------------------------------


class TestBackgroundSubtractorManager:
    def test_deprecation_warning(self):
        """Constructing emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BackgroundSubtractorManager(min_area=100)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_returns_list_of_bboxes(self):
        """process() returns a list of bounding box tuples."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mgr = BackgroundSubtractorManager(min_area=100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mgr.process(frame)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# PersonBibAssociator (deprecated)
# ---------------------------------------------------------------------------


class TestPersonBibAssociator:
    def test_deprecation_warning(self):
        """Constructing emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PersonBibAssociator(max_distance=200)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_bib_inside_person_bbox(self):
        """A bib whose center is inside a person bbox gets matched."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assoc = PersonBibAssociator(max_distance=200)
        persons = {
            1: ((250.0, 300.0), (200, 100, 300, 500)),
        }
        bibs = {
            10: ((250.0, 250.0), (230, 230, 270, 270)),  # inside person 1
        }
        result = assoc.associate(persons, bibs)
        assert result[1] == 10


# ---------------------------------------------------------------------------
# MergedBlobEstimator (deprecated)
# ---------------------------------------------------------------------------


class TestMergedBlobEstimator:
    def test_deprecation_warning(self):
        """Constructing emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MergedBlobEstimator(typical_person_area=10000)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_single_person(self):
        """A blob with typical area returns 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            est = MergedBlobEstimator(typical_person_area=10000)
        assert est.estimate((0, 0, 100, 100)) == 1  # 10000 area


# ---------------------------------------------------------------------------
# CentroidTracker
# ---------------------------------------------------------------------------


class TestCentroidTracker:
    def test_register_and_track(self):
        """New detections get unique track IDs."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=100)
        detections = [(10, 10, 50, 50), (200, 200, 250, 250)]
        result = tracker.update(detections)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_track_continuity(self):
        """Same object across frames keeps the same ID."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=100)
        tracker.update([(100, 100, 150, 150)])
        result = tracker.update([(105, 105, 155, 155)])  # slight movement
        assert 0 in result

    def test_disappearance(self):
        """Object disappears after max_disappeared frames."""
        tracker = CentroidTracker(max_disappeared=2, max_distance=100)
        tracker.update([(100, 100, 150, 150)])
        tracker.update([])
        tracker.update([])
        result = tracker.update([])
        assert len(result) == 0

    def test_new_object_gets_new_id(self):
        """A distant detection gets a new track ID."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        tracker.update([(10, 10, 50, 50)])
        result = tracker.update([(10, 10, 50, 50), (500, 500, 550, 550)])
        assert len(result) == 2


class TestCentroidTrackerCustomCentroids:
    def test_custom_centroids_used_for_matching(self):
        """When centroids are provided, they are used instead of bbox centers."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        # Register with custom centroid (10, 10) — far from bbox center (125, 125)
        detections = [(100, 100, 150, 150)]
        centroids = [(10.0, 10.0)]
        result = tracker.update(detections, centroids=centroids)
        assert 0 in result
        centroid, bbox = result[0]
        assert centroid == (10.0, 10.0)
        assert bbox == (100, 100, 150, 150)

    def test_custom_centroids_affects_matching(self):
        """Custom centroids change which detection matches which track."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        # First frame: two objects
        tracker.update(
            [(0, 0, 20, 20), (100, 100, 120, 120)],
            centroids=[(10.0, 10.0), (110.0, 110.0)],
        )
        # Second frame: same objects, custom centroids close to first-frame positions
        result = tracker.update(
            [(0, 0, 20, 20), (100, 100, 120, 120)],
            centroids=[(12.0, 12.0), (112.0, 112.0)],
        )
        assert 0 in result
        assert 1 in result
        assert result[0][0] == (12.0, 12.0)
        assert result[1][0] == (112.0, 112.0)


# ---------------------------------------------------------------------------
# CrossingEventLog
# ---------------------------------------------------------------------------


class TestCrossingEventLog:
    def test_writes_csv(self):
        """CrossingEventLog writes proper CSV output."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="r") as f:
            path = Path(f.name)

        log = CrossingEventLog(path)
        event = CrossingEvent(
            sequence=1,
            frame_idx=100,
            timestamp_sec=3.333,
            person_track_id=5,
            bib_number="1234",
            confidence=0.95,
            estimated_count=1,
            person_bbox=(100, 200, 300, 500),
            chest_point=(200.0, 350.0),
            source="pose",
        )
        log.write(event)
        log.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "sequence" in lines[0]
        assert "chest_x" in lines[0]
        assert "chest_y" in lines[0]
        assert "source" in lines[0]
        assert "1234" in lines[1]
        assert "pose" in lines[1]
        assert "200.0" in lines[1]

        path.unlink()

    def test_writes_csv_without_chest_point(self):
        """CrossingEventLog handles None chest_point."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="r") as f:
            path = Path(f.name)

        log = CrossingEventLog(path)
        event = CrossingEvent(
            sequence=1,
            frame_idx=100,
            timestamp_sec=3.333,
            person_track_id=5,
            bib_number="UNKNOWN",
            confidence=0.0,
            person_bbox=(100, 200, 300, 500),
            source="bib_tracker",
        )
        log.write(event)
        log.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert "bib_tracker" in lines[1]

        path.unlink()
