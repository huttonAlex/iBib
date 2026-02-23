"""Integration test for the reusable pipeline module."""

from pathlib import Path

import numpy as np

from pointcam.pipeline import BibDetection, PipelineConfig, process_frames


class DummyDetector:
    def __init__(self, detections_per_frame):
        self._detections = detections_per_frame
        self._idx = 0

    def detect(self, frame, conf_threshold):
        if self._idx >= len(self._detections):
            return []
        dets = self._detections[self._idx]
        self._idx += 1
        return dets


class DummyOCR:
    def __init__(self, text="1234", conf=0.9):
        self.text = text
        self.conf = conf

    def predict_batch(self, crops):
        return [(self.text, self.conf) for _ in crops]


def test_pipeline_runs_and_writes_logs(tmp_path: Path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    detections = [
        [BibDetection(bbox=(10, 10, 30, 30), confidence=0.9)],
        [BibDetection(bbox=(10, 10, 30, 30), confidence=0.9)],
        [BibDetection(bbox=(10, 10, 30, 30), confidence=0.9)],
    ]

    config = PipelineConfig(
        conf_threshold=0.1,
        ocr_conf_threshold=0.5,
        enable_quality_filter=False,
        write_video=False,
        placement="center",
        timing_line_coords=None,
        crossing_direction="any",
        debounce_time=1.0,
        enable_person_detect=False,
        stride=1,
        start_time=0.0,
        enable_ocr_skip=False,
        crossing_mode="line",
    )

    result = process_frames(
        frames=frames,
        fps=10.0,
        detector=DummyDetector(detections),
        ocr_model=DummyOCR(),
        output_dir=tmp_path,
        output_stem="synthetic",
        config=config,
        show=False,
        print_summary=False,
    )

    assert result.stats.total_detections == 3
    assert result.outputs.detection_log_path.exists()
    assert result.outputs.review_queue_path.exists()

    log_lines = result.outputs.detection_log_path.read_text().strip().splitlines()
    assert len(log_lines) == 4  # header + 3 rows
