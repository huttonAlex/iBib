"""Tests for pointcam.crossing module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pointcam.crossing import (
    BackgroundSubtractorManager,
    CentroidTracker,
    CrossingDetector,
    CrossingEvent,
    CrossingEventLog,
    MergedBlobEstimator,
    PersonBibAssociator,
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
        """Moving from one side to the other fires exactly once."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=5)

        # Establish position on the left
        assert det.check(1, 0.3, 0.5, 1) is False
        # Cross to the right
        assert det.check(1, 0.7, 0.5, 2) is True
        # Stay on the right — no new crossing
        assert det.check(1, 0.8, 0.5, 3) is False

    def test_debounce(self):
        """A second crossing within debounce window is suppressed."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=10)

        det.check(1, 0.3, 0.5, 1)  # left
        assert det.check(1, 0.7, 0.5, 2) is True  # cross right (frame 2)
        det.check(1, 0.3, 0.5, 5)  # back to left — suppressed (5-2=3 < 10)
        assert det.check(1, 0.7, 0.5, 6) is False  # within debounce (6-2=4 < 10)

        # After debounce expires (20-2=18 > 10), next side change fires
        assert det.check(1, 0.3, 0.5, 20) is True  # crossing accepted

    def test_direction_filter_left_to_right(self):
        """Only left-to-right crossings fire."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="left_to_right", debounce_frames=0)

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
        det = CrossingDetector(line, direction="left_to_right", debounce_frames=0)

        det.check(1, 0.7, 0.5, 1)  # right side
        result = det.check(1, 0.3, 0.5, 2)  # cross to left
        assert result is False

    def test_multiple_tracks(self):
        """Independent tracks are tracked separately."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=0)

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

    def test_no_crossing_without_prior_position(self):
        """First observation of a track is not a crossing."""
        line = TimingLine(0.5, 0.0, 0.5, 1.0)
        det = CrossingDetector(line, direction="any", debounce_frames=0)
        assert det.check(1, 0.7, 0.5, 1) is False


# ---------------------------------------------------------------------------
# BackgroundSubtractorManager
# ---------------------------------------------------------------------------


class TestBackgroundSubtractorManager:
    def test_returns_list_of_bboxes(self):
        """process() returns a list of bounding box tuples."""
        mgr = BackgroundSubtractorManager(min_area=100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mgr.process(frame)
        assert isinstance(result, list)

    def test_detects_blob_after_background_learning(self):
        """After learning a black background, a white rectangle should be detected."""
        mgr = BackgroundSubtractorManager(
            history=10,
            var_threshold=16.0,
            min_area=100,
            max_area=500000,
            min_aspect_ratio=0.2,
            max_aspect_ratio=10.0,
        )

        # Feed blank frames to establish background
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(20):
            mgr.process(blank)

        # Insert a large white rectangle (person-like)
        frame = blank.copy()
        cv2.rectangle(frame, (200, 100), (300, 400), (255, 255, 255), -1)
        bboxes = mgr.process(frame)

        assert len(bboxes) >= 1
        # The detected bbox should overlap with our rectangle
        x1, y1, x2, y2 = bboxes[0]
        assert x1 < 300 and x2 > 200 and y1 < 400 and y2 > 100

    def test_filters_small_contours(self):
        """Blobs smaller than min_area are ignored."""
        mgr = BackgroundSubtractorManager(
            history=10, var_threshold=16.0, min_area=5000
        )

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(20):
            mgr.process(blank)

        # Small blob (10x10 = 100px area)
        frame = blank.copy()
        cv2.rectangle(frame, (300, 200), (310, 210), (255, 255, 255), -1)
        bboxes = mgr.process(frame)

        assert len(bboxes) == 0


# ---------------------------------------------------------------------------
# PersonBibAssociator
# ---------------------------------------------------------------------------


class TestPersonBibAssociator:
    def test_bib_inside_person_bbox(self):
        """A bib whose center is inside a person bbox gets matched."""
        assoc = PersonBibAssociator(max_distance=200)
        persons = {
            1: ((250.0, 300.0), (200, 100, 300, 500)),
        }
        bibs = {
            10: ((250.0, 250.0), (230, 230, 270, 270)),  # inside person 1
        }
        result = assoc.associate(persons, bibs)
        assert result[1] == 10

    def test_bib_outside_but_within_distance(self):
        """A bib not inside any person but within max_distance still matches."""
        assoc = PersonBibAssociator(max_distance=200)
        persons = {
            1: ((250.0, 300.0), (200, 200, 300, 400)),
        }
        bibs = {
            10: ((350.0, 300.0), (330, 280, 370, 320)),  # outside but close
        }
        result = assoc.associate(persons, bibs)
        assert result[1] == 10

    def test_bib_too_far(self):
        """A bib beyond max_distance is not matched."""
        assoc = PersonBibAssociator(max_distance=50)
        persons = {
            1: ((100.0, 100.0), (50, 50, 150, 150)),
        }
        bibs = {
            10: ((500.0, 500.0), (480, 480, 520, 520)),  # far away
        }
        result = assoc.associate(persons, bibs)
        assert result[1] is None

    def test_no_bibs(self):
        """When there are no bibs, all persons map to None."""
        assoc = PersonBibAssociator()
        persons = {
            1: ((100.0, 100.0), (50, 50, 150, 150)),
        }
        result = assoc.associate(persons, {})
        assert result[1] is None

    def test_bib_not_reused(self):
        """Each bib can only be assigned to one person."""
        assoc = PersonBibAssociator(max_distance=500)
        persons = {
            1: ((100.0, 100.0), (50, 50, 150, 150)),
            2: ((120.0, 100.0), (70, 50, 170, 150)),
        }
        bibs = {
            10: ((110.0, 100.0), (90, 80, 130, 120)),  # close to both
        }
        result = assoc.associate(persons, bibs)
        assigned = [v for v in result.values() if v is not None]
        assert len(assigned) == 1  # only one person gets the bib
        assert 10 in assigned


# ---------------------------------------------------------------------------
# MergedBlobEstimator
# ---------------------------------------------------------------------------


class TestMergedBlobEstimator:
    def test_single_person(self):
        """A blob with typical area returns 1."""
        est = MergedBlobEstimator(typical_person_area=10000)
        assert est.estimate((0, 0, 100, 100)) == 1  # 10000 area

    def test_two_persons(self):
        """A blob with ~2x typical area returns 2."""
        est = MergedBlobEstimator(typical_person_area=10000)
        assert est.estimate((0, 0, 200, 100)) == 2  # 20000 area

    def test_minimum_one(self):
        """Even a tiny blob returns at least 1."""
        est = MergedBlobEstimator(typical_person_area=10000)
        assert est.estimate((0, 0, 10, 10)) == 1

    def test_large_group(self):
        """A blob with ~5x typical area returns 5."""
        est = MergedBlobEstimator(typical_person_area=10000)
        assert est.estimate((0, 0, 500, 100)) == 5  # 50000 area


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
        )
        log.write(event)
        log.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "sequence" in lines[0]
        assert "1234" in lines[1]

        path.unlink()
