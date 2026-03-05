"""Reusable video pipeline for bib detection, OCR, and crossings.

This module extracts the pipeline from scripts/test_video_pipeline.py into
reusable functions that can be called from scripts or tests.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import itertools
import cv2
import numpy as np

from pointcam.crossing import (
    BibCrossingDeduplicator,
    ByteTrackAdapter,
    CentroidTracker,
    CrossingDetector,
    CrossingEvent,
    CrossingEventLog,
    PersistentPersonBibAssociator,
    PoseDetector,
    TimingLine,
)
from pointcam.recognition import (
    BibCompletenessChecker,
    BibSetValidator,
    ConfidenceLevel,
    ConfidenceManager,
    CropQualityFilter,
    DigitCountValidator,
    EnhancedTemporalVoting,
    PostOCRCleanup,
)


# Crop padding presets per camera placement (left_pct, right_pct, top_pct, bottom_pct)
PLACEMENT_PADDING = {
    "center": (0.15, 0.15, 0.10, 0.10),
    "left": (0.10, 0.25, 0.10, 0.10),
    "right": (0.25, 0.10, 0.10, 0.10),
}


@dataclass
class PipelineConfig:
    """Configuration for the video pipeline."""

    conf_threshold: float = 0.25
    ocr_conf_threshold: float = 0.5
    enable_quality_filter: bool = True
    write_video: bool = True
    write_raw_video: bool = False
    placement: str = "center"
    timing_line_coords: Optional[Tuple[float, float, float, float]] = None
    crossing_direction: str = "any"
    debounce_time: float = 2.0
    enable_person_detect: bool = True
    pose_model_path: str = "yolov8n-pose.pt"
    pose_conf: float = 0.5
    stride: int = 1
    start_time: float = 0.0
    enable_ocr_skip: bool = True
    crossing_mode: str = "line"


@dataclass
class BibDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float


class UltralyticsBibDetector:
    """Ultralytics YOLO wrapper to produce BibDetection objects."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        # .to() only works for PyTorch models; TensorRT/ONNX pass device at predict time
        if model_path.endswith(".pt"):
            self.model.to(device)

    def detect(self, frame: np.ndarray, conf_threshold: float) -> List[BibDetection]:
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        detections: List[BibDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                det_conf = float(box.conf[0])
                detections.append(
                    BibDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=det_conf,
                    )
                )
        return detections


@dataclass
class PipelineStats:
    frames_processed: int
    total_detections: int
    unique_tracks: int
    avg_det_time_ms: float
    avg_ocr_time_ms: float
    total_crossings: int
    unknown_crossings: int
    dedup_suppressed: int
    person_tracks: Optional[int]
    avg_pose_time_ms: float = 0.0


@dataclass
class PipelineOutputs:
    detection_log_path: Path
    review_queue_path: Path
    output_video_path: Optional[Path]
    crossing_log_path: Optional[Path]
    person_track_diag_path: Optional[Path]


@dataclass
class PipelineResult:
    stats: PipelineStats
    outputs: PipelineOutputs


@dataclass
class ProgressInfo:
    """Snapshot of pipeline progress, emitted periodically during processing."""

    frame_idx: int
    elapsed_sec: float
    total_crossings: int
    unknown_crossings: int
    total_detections: int
    fps: float
    avg_det_ms: float = 0.0
    avg_ocr_ms: float = 0.0
    avg_pose_ms: float = 0.0


def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _iter_video_frames(cap: cv2.VideoCapture):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


class _FramePrefetcher:
    """Pre-fetch the next frame in a background thread to overlap decode with processing."""

    def __init__(self, frame_iter):
        self._iter = iter(frame_iter)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._sentinel = object()
        self._future = self._pool.submit(next, self._iter, self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        result = self._future.result()
        if result is self._sentinel:
            self._pool.shutdown(wait=False)
            raise StopIteration
        self._future = self._pool.submit(next, self._iter, self._sentinel)
        return result

    def shutdown(self):
        self._pool.shutdown(wait=False)


def _timed_pose_detect(detector, frame):
    """Run pose detection in a thread; returns (detections, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = detector.detect(frame)
    t1 = time.perf_counter()
    return result, t1 - t0


def process_video(
    video_path: str,
    detector_path: str,
    ocr_model,
    output_dir: Path,
    config: PipelineConfig,
    bib_validator: Optional[BibSetValidator] = None,
    show: bool = False,
    detector: Optional[UltralyticsBibDetector] = None,
    pose_detector: Optional[PoseDetector] = None,
    device: Optional[str] = None,
    print_summary: bool = True,
) -> PipelineResult:
    """Process a video file and run the full pipeline."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if config.start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, config.start_time * 1000)
    start_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if detector is None:
        device = _resolve_device(device)
        detector = UltralyticsBibDetector(detector_path, device=device)

    enable_crossings = (
        config.timing_line_coords is not None or config.crossing_mode == "zone"
    )

    if enable_crossings and pose_detector is None:
        if config.enable_person_detect or config.crossing_mode == "zone":
            device = _resolve_device(device)
            pose_detector = PoseDetector(
                model_path=config.pose_model_path,
                conf=config.pose_conf,
                device=device,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(video_path).stem

    return process_frames(
        frames=_iter_video_frames(cap),
        fps=fps,
        detector=detector,
        ocr_model=ocr_model,
        output_dir=output_dir,
        output_stem=output_stem,
        config=config,
        bib_validator=bib_validator,
        pose_detector=pose_detector,
        show=show,
        total_frames=total_frames,
        frame_size=(width, height),
        start_frame_idx=start_frame_idx,
        print_summary=print_summary,
    )


def process_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    detector,
    ocr_model,
    output_dir: Path,
    output_stem: str,
    config: PipelineConfig,
    bib_validator: Optional[BibSetValidator] = None,
    pose_detector: Optional[PoseDetector] = None,
    show: bool = False,
    total_frames: Optional[int] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    start_frame_idx: int = 0,
    print_summary: bool = True,
    on_crossing: Optional[Callable[[CrossingEvent], None]] = None,
    on_progress: Optional[Callable[[ProgressInfo], None]] = None,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
) -> PipelineResult:
    """Process an iterable of frames with the full pipeline."""

    frame_iter = iter(frames)
    if frame_size is None:
        try:
            first_frame = next(frame_iter)
        except StopIteration:
            raise ValueError("No frames provided")
        height, width = first_frame.shape[:2]
        frame_iter = itertools.chain([first_frame], frame_iter)
    else:
        width, height = frame_size

    # Initialize Tier 1 components
    tracker = CentroidTracker(max_disappeared=30, max_distance=100)
    voting = EnhancedTemporalVoting(
        window_size=15,
        min_votes=1,
        stability_threshold=5,
        confidence_threshold=config.ocr_conf_threshold,
    )
    confidence_mgr = ConfidenceManager(auto_accept_validated=bib_validator is not None)

    # Initialize Tier 2 components
    quality_filter = (
        CropQualityFilter(
            min_blur_score=50.0,
            min_width=40,
            min_height=15,
            min_aspect_ratio=1.0,
            max_aspect_ratio=6.0,
            min_completeness=0.3,
            check_completeness=True,
        )
        if config.enable_quality_filter
        else None
    )

    completeness_checker = BibCompletenessChecker(edge_margin_ratio=0.02)

    max_bib_value = None
    if bib_validator:
        numeric_bibs = [int(b) for b in bib_validator.bib_set if b.isdigit()]
        if numeric_bibs:
            max_bib_value = max(numeric_bibs)

    post_cleanup = PostOCRCleanup(
        min_digits=1,
        max_digits=5,
        strip_leading_zeros=True,
        fix_letter_confusion=True,
        max_bib_value=max_bib_value,
    )

    digit_validator = None
    if bib_validator:
        digit_validator = DigitCountValidator.from_bib_set(bib_validator.bib_set)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{output_stem}_annotated.mp4"
    raw_video_path = output_dir / f"{output_stem}.mp4"
    out_writer = None
    raw_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if config.write_video:
        out_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )
    if config.write_raw_video:
        raw_writer = cv2.VideoWriter(
            str(raw_video_path), fourcc, fps, (width, height)
        )

    log_path = output_dir / f"{output_stem}_detections.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            "frame",
            "time_sec",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "det_conf",
            "ocr_raw",
            "ocr_conf",
            "validated",
            "is_valid",
            "consensus",
            "consensus_conf",
            "is_stable",
            "final_level",
        ]
    )

    frame_idx = start_frame_idx
    total_detections = 0
    total_ocr_time = 0.0
    total_det_time = 0.0
    quality_rejected = 0
    edge_rejected = 0
    partial_rejected = 0
    cleanup_modified = 0
    ocr_skip_count = 0
    digit_corrections = 0

    final_consensus: Dict[int, Tuple[str, float, str]] = {}
    final_consensus_frame: Dict[int, int] = {}
    consensus_ttl = int(30 * fps)
    all_consensus: Dict[int, Tuple[str, float, str]] = {}

    # Track-frequency filter: suppress bibs that appear on too many bib tracks
    # (phantom reads from bib design elements, systematic partial reads, etc.)
    bib_consensus_tracks: Dict[str, set] = {}  # bib_number -> set of bib track IDs
    max_bib_tracks = 5  # bibs on more than this many tracks are suppressed

    bib_track_last_pos: Dict[int, Tuple[float, float]] = {}
    bib_track_last_frame: Dict[int, int] = {}
    bib_track_first_pos: Dict[int, Tuple[float, float]] = {}
    bib_track_first_frame: Dict[int, int] = {}
    bib_overlapped_persons: Dict[int, set] = {}
    # Per-bib observations for post-processing (sampled every N frames to limit memory)
    bib_observations: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
    _bib_obs_interval = 10  # sample every 10th frame per bib track

    timing_line = None
    crossing_detector = None
    person_tracker = None
    person_bib_assoc = None
    bib_dedup = None
    crossing_log = None
    crossing_log_path = None
    crossing_seq = 0
    total_crossings = 0
    unknown_crossings = 0
    dedup_suppressed = 0
    total_pose_time = 0.0

    person_track_diag: Dict[int, dict] = {}
    total_person_dets = 0
    crossing_bib_found = 0
    crossing_unknown_pre_dedup = 0

    person_track_emitted: set = set()
    zone_min_frames = 5
    person_track_ready: set = set()
    prev_person_tracked_ids: set = set()
    all_crossing_events: List[CrossingEvent] = []  # for post-processing pass
    emitted_bibs: set = set()  # bibs already assigned to crossings

    enable_crossings = (
        config.timing_line_coords is not None or config.crossing_mode == "zone"
    )

    if enable_crossings:
        if config.timing_line_coords is not None:
            timing_line = TimingLine(*config.timing_line_coords)
            debounce_frames = int(config.debounce_time * fps)
            crossing_detector = CrossingDetector(
                timing_line=timing_line,
                direction=config.crossing_direction,
                debounce_frames=debounce_frames,
            )
        if config.crossing_mode == "line" and timing_line is None:
            raise ValueError("crossing_mode='line' requires timing_line_coords")

        bib_dedup = BibCrossingDeduplicator(debounce_frames=int(10.0 * fps))

        if config.enable_person_detect or config.crossing_mode == "zone":
            if pose_detector is None:
                raise ValueError("pose_detector required when person detection enabled")
            person_tracker = CentroidTracker(max_disappeared=20, max_distance=150)

            expected_digits = None
            if bib_validator:
                expected_digits = {len(b) for b in bib_validator.bib_set if b}

            person_bib_assoc = PersistentPersonBibAssociator(
                max_distance=350,
                memory_frames=int(30.0 * fps),
                min_votes=1,
                min_confidence=0.4,
                expected_digit_counts=expected_digits,
            )

        crossing_log_path = output_dir / f"{output_stem}_crossings.csv"
        crossing_log = CrossingEventLog(crossing_log_path)

    wall_start = time.time()

    # Async frame prefetching: decode next frame while processing current one
    prefetcher = _FramePrefetcher(frame_iter)
    # Thread pool for parallel pose detection (runs during tracking/OCR)
    _pose_pool = ThreadPoolExecutor(max_workers=1) if pose_detector is not None else None
    should_annotate = config.write_video or show or on_frame is not None

    for frame in prefetcher:
        time_sec = frame_idx / fps
        frame_idx += 1

        if raw_writer:
            raw_writer.write(frame)

        if config.stride > 1 and frame_idx % config.stride != 0:
            continue

        t0 = time.perf_counter()
        dets = detector.detect(frame, config.conf_threshold)
        t1 = time.perf_counter()
        total_det_time += t1 - t0

        # Submit pose detection in parallel — runs during tracking + OCR
        pose_future: Optional[Future] = None
        if _pose_pool is not None:
            pose_future = _pose_pool.submit(_timed_pose_detect, pose_detector, frame)

        detections = [d.bbox for d in dets]
        det_confs = [d.confidence for d in dets]

        tracked = tracker.update(detections)

        for bid, (b_centroid, b_bbox) in tracked.items():
            if bid not in bib_track_first_pos:
                bib_track_first_pos[bid] = b_centroid
                bib_track_first_frame[bid] = frame_idx
            bib_track_last_pos[bid] = b_centroid
            bib_track_last_frame[bid] = frame_idx

        ocr_batch_items = []
        for track_id, (centroid, bbox) in tracked.items():
            x1, y1, x2, y2 = bbox

            det_conf = 0.0
            for i, det in enumerate(detections):
                if det == bbox:
                    det_conf = det_confs[i]
                    break

            is_visible, _edge_reason = completeness_checker.is_fully_visible(
                bbox=(x1, y1, x2, y2),
                frame_width=width,
                frame_height=height,
            )
            if not is_visible:
                edge_rejected += 1
                continue

            bib_w = x2 - x1
            bib_h = y2 - y1
            lp, rp, tp, bp = PLACEMENT_PADDING.get(config.placement, PLACEMENT_PADDING["center"])
            pad_left = int(bib_w * lp)
            pad_right = int(bib_w * rp)
            pad_top = int(bib_h * tp)
            pad_bottom = int(bib_h * bp)
            x1_pad = max(0, x1 - pad_left)
            y1_pad = max(0, y1 - pad_top)
            x2_pad = min(width, x2 + pad_right)
            y2_pad = min(height, y2 + pad_bottom)

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            if quality_filter:
                quality = quality_filter.assess(crop)
                if not quality.is_acceptable:
                    if "partial" in (quality.rejection_reason or "").lower():
                        partial_rejected += 1
                    else:
                        quality_rejected += 1
                    continue

            if config.enable_ocr_skip:
                consensus = voting.get_consensus(track_id)
                if (
                    consensus.is_stable
                    and consensus.confidence >= 0.85
                    and track_id in final_consensus
                ):
                    bib_validated = (
                        bib_validator is not None
                        and consensus.number in bib_validator.bib_set
                    )
                    if bib_validated or bib_validator is None:
                        ocr_skip_count += 1
                        continue

            ocr_batch_items.append((track_id, bbox, det_conf, crop))

        crops = [item[3] for item in ocr_batch_items]
        ocr_detailed_results = None
        if crops and hasattr(ocr_model, "predict_batch_detailed"):
            t2 = time.perf_counter()
            ocr_detailed_results = ocr_model.predict_batch_detailed(crops)
            t3 = time.perf_counter()
            total_ocr_time += t3 - t2
            ocr_results = [(r.text, r.confidence) for r in ocr_detailed_results]
        elif crops and hasattr(ocr_model, "predict_batch"):
            t2 = time.perf_counter()
            ocr_results = ocr_model.predict_batch(crops)
            t3 = time.perf_counter()
            total_ocr_time += t3 - t2
        else:
            ocr_results = []
            for item in ocr_batch_items:
                t2 = time.perf_counter()
                result = ocr_model.predict(item[3])
                t3 = time.perf_counter()
                total_ocr_time += t3 - t2
                ocr_results.append(result)

        for idx, (track_id, bbox, det_conf, crop) in enumerate(ocr_batch_items):
            x1, y1, x2, y2 = bbox
            ocr_raw, ocr_conf = ocr_results[idx]
            total_detections += 1

            if not ocr_raw:
                continue

            cleaned_number, was_modified = post_cleanup.clean(ocr_raw)
            if was_modified:
                cleanup_modified += 1
            ocr_raw = cleaned_number

            if not ocr_raw:
                continue

            if digit_validator:
                ocr_raw, ocr_conf, _ = digit_validator.validate(ocr_raw, ocr_conf)

            validation_result = None
            validated_number = ocr_raw
            is_valid = False
            if bib_validator:
                validation_result = bib_validator.validate(ocr_raw, ocr_conf)
                validated_number = validation_result.validated
                is_valid = validation_result.is_valid

            # Digit-confusion correction: if bib not in known set, try swapping
            # low-confidence digits with their runner-up to produce a valid bib.
            # Only for 3+ digit bibs (single/double-digit reads are partial fragments).
            if (
                bib_validator is not None
                and validated_number not in bib_validator.bib_set
                and len(validated_number) >= 3
                and ocr_detailed_results is not None
            ):
                detail = ocr_detailed_results[idx]
                raw_text = detail.text
                if (
                    detail.per_digit
                    and len(detail.per_digit) == len(raw_text)
                    and len(raw_text) == len(validated_number)
                ):
                    for di, dd in enumerate(detail.per_digit):
                        if dd.prob < 0.6 and dd.runner_up_prob > 0.1:
                            candidate = (
                                validated_number[:di]
                                + dd.runner_up_digit
                                + validated_number[di + 1 :]
                            )
                            if candidate in bib_validator.bib_set:
                                validated_number = candidate
                                is_valid = True
                                digit_corrections += 1
                                break

            _is_known_bib = (
                bib_validator is not None and validated_number in bib_validator.bib_set
            )
            if not _is_known_bib:
                if len(validated_number) == 1 and ocr_conf < 0.95:
                    continue
                if len(validated_number) == 2 and ocr_conf < 0.90:
                    continue
                # Repeated-digit phantom filter: reject "111", "1111", "2222", etc.
                if (
                    len(validated_number) >= 3
                    and len(set(validated_number)) == 1
                ):
                    continue

            voting.update(track_id, validated_number, ocr_conf, frame_idx)
            consensus_result = voting.get_consensus(track_id)

            consensus_number = consensus_result.number or validated_number
            consensus_conf = (
                consensus_result.confidence if consensus_result.number else ocr_conf
            )

            classified = confidence_mgr.classify(
                bib_number=consensus_number,
                ocr_confidence=ocr_conf,
                validation_result=validation_result,
                voting_result=consensus_result,
            )

            final_consensus[track_id] = (
                consensus_number,
                classified.adjusted_confidence,
                classified.level.value,
            )
            final_consensus_frame[track_id] = frame_idx
            bib_consensus_tracks.setdefault(consensus_number, set()).add(track_id)
            all_consensus[track_id] = final_consensus[track_id]

            # Sample bib observations for post-processing
            if classified.level.value in ("high", "medium"):
                obs_list = bib_observations.setdefault(consensus_number, [])
                if not obs_list or frame_idx - obs_list[-1][0] >= _bib_obs_interval:
                    centroid_for_obs = tracked[track_id][0] if track_id in tracked else None
                    if centroid_for_obs is not None:
                        obs_list.append((frame_idx, centroid_for_obs))

            if classified.needs_review:
                confidence_mgr.add_to_review_queue(
                    prediction=classified,
                    frame_number=frame_idx,
                )

            log_writer.writerow(
                [
                    frame_idx,
                    f"{time_sec:.3f}",
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    f"{det_conf:.3f}",
                    ocr_raw,
                    f"{ocr_conf:.3f}",
                    validated_number,
                    is_valid,
                    consensus_number,
                    f"{consensus_conf:.3f}",
                    consensus_result.is_stable,
                    classified.level.value,
                ]
            )

            if should_annotate:
                if classified.level == ConfidenceLevel.HIGH:
                    color = (0, 255, 0)
                elif classified.level == ConfidenceLevel.MEDIUM:
                    color = (0, 255, 255)
                elif classified.level == ConfidenceLevel.LOW:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                status = ""
                if bib_validator:
                    if is_valid:
                        status = " [OK]"
                    elif validation_result and validation_result.is_corrected:
                        status = f" [{ocr_raw}->{validated_number}]"
                    else:
                        status = " [?]"

                if consensus_result.is_stable:
                    status += " *"

                label = (
                    f"#{track_id}: {consensus_number} "
                    f"({classified.adjusted_confidence:.2f}){status}"
                )
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        if pose_future is not None:
            person_dets, pose_elapsed = pose_future.result()
            total_pose_time += pose_elapsed
            total_person_dets += len(person_dets)

            p_bboxes = [d.bbox for d in person_dets]
            p_chests = [d.chest_point for d in person_dets]
            person_tracked = person_tracker.update(p_bboxes, centroids=p_chests)

            # Filter out phantom bibs (appearing on too many bib tracks)
            filtered_consensus = {
                tid: (num, conf, level)
                for tid, (num, conf, level) in final_consensus.items()
                if len(bib_consensus_tracks.get(num, set())) <= max_bib_tracks
            }
            person_bib_assoc.update(person_tracked, tracked, filtered_consensus, frame_idx)

            if person_tracked and tracked:
                for pid, (p_centroid, p_bbox) in person_tracked.items():
                    px1, py1, px2, py2 = p_bbox
                    for bid, (b_centroid, b_bbox) in tracked.items():
                        bx1, by1, bx2, by2 = b_bbox
                        bcx, bcy = b_centroid
                        if (px1 <= bcx <= px2 and py1 <= bcy <= py2) or \
                           not (bx2 < px1 or bx1 > px2 or by2 < py1 or by1 > py2):
                            bib_overlapped_persons.setdefault(bid, set()).add(pid)

            for pid, (p_centroid, p_bbox) in person_tracked.items():
                if pid not in person_track_diag:
                    person_track_diag[pid] = {
                        "first_frame": frame_idx,
                        "first_chest": (p_centroid[0], p_centroid[1]),
                        "frames_seen": 0,
                        "crossed": False,
                        "bib": None,
                    }
                diag = person_track_diag[pid]
                diag["last_frame"] = frame_idx
                diag["last_chest"] = (p_centroid[0], p_centroid[1])
                diag["last_bbox"] = p_bbox
                diag["frames_seen"] += 1

            tracks_to_emit: List[
                Tuple[int, Tuple[float, float], Tuple[int, int, int, int]]
            ] = []

            if config.crossing_mode == "zone":
                current_ids = set(person_tracked.keys())

                for pid in current_ids:
                    if pid not in person_track_emitted and pid not in person_track_ready:
                        diag = person_track_diag.get(pid)
                        if diag and diag["frames_seen"] >= zone_min_frames:
                            person_track_ready.add(pid)

                for pid in list(person_track_ready):
                    if pid in person_track_emitted:
                        continue
                    bib = person_bib_assoc.get_bib(pid)
                    if bib is not None and pid in current_ids:
                        p_centroid, p_bbox = person_tracked[pid]
                        tracks_to_emit.append((pid, p_centroid, p_bbox))

                disappeared = (prev_person_tracked_ids - current_ids)
                for pid in disappeared:
                    if pid in person_track_ready and pid not in person_track_emitted:
                        diag = person_track_diag.get(pid)
                        if diag:
                            lc = diag.get("last_chest", diag["first_chest"])
                            lb = diag.get("last_bbox", (0, 0, 0, 0))
                            tracks_to_emit.append((pid, lc, lb))

                prev_person_tracked_ids = current_ids.copy()
            else:
                for pid, (p_centroid, p_bbox) in person_tracked.items():
                    cx_norm = p_centroid[0] / width
                    cy_norm = p_centroid[1] / height
                    if crossing_detector.check(pid, cx_norm, cy_norm, frame_idx):
                        tracks_to_emit.append((pid, p_centroid, p_bbox))

            for pid, p_centroid, p_bbox in tracks_to_emit:
                if pid in person_track_emitted:
                    continue
                person_track_emitted.add(pid)

                if pid in person_track_diag:
                    person_track_diag[pid]["crossed"] = True

                bib_number = person_bib_assoc.get_bib(pid) or "UNKNOWN"
                confidence = person_bib_assoc.get_bib_confidence(pid)

                # Cold-start phantom check: suppress bibs that accumulated
                # early votes before exceeding the track-frequency threshold
                if (
                    bib_number != "UNKNOWN"
                    and len(bib_consensus_tracks.get(bib_number, set())) > max_bib_tracks
                ):
                    bib_number = "UNKNOWN"
                    confidence = 0.0

                # Repeated-digit phantom check at emission time:
                # suppress "111", "1111", etc. even if in valid bib range
                if (
                    bib_number != "UNKNOWN"
                    and len(bib_number) >= 3
                    and len(set(bib_number)) == 1
                ):
                    bib_number = "UNKNOWN"
                    confidence = 0.0

                if bib_number == "UNKNOWN":
                    px1, py1, px2, py2 = p_bbox
                    margin = max(50, int(0.15 * (px2 - px1)))
                    best_bib = None
                    best_conf = 0.0

                    def _is_phantom(num: str) -> bool:
                        return len(bib_consensus_tracks.get(num, set())) > max_bib_tracks

                    for bid, (b_centroid, b_bbox) in tracked.items():
                        bcx, bcy = b_centroid
                        if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                            if bid in final_consensus:
                                fc_number, fc_conf, fc_level = final_consensus[bid]
                                if (
                                    fc_level in ("high", "medium")
                                    and fc_conf > best_conf
                                    and not _is_phantom(fc_number)
                                ):
                                    best_bib = fc_number
                                    best_conf = fc_conf

                    if best_bib is None:
                        for bid, (b_centroid, b_bbox) in tracked.items():
                            bcx, bcy = b_centroid
                            if (px1 - margin) <= bcx <= (px2 + margin) and py1 <= bcy <= py2:
                                if bid in final_consensus:
                                    fc_number, fc_conf, fc_level = final_consensus[bid]
                                    if (
                                        fc_level == "high"
                                        and fc_conf > best_conf
                                        and not _is_phantom(fc_number)
                                    ):
                                        best_bib = fc_number
                                        best_conf = fc_conf

                    if best_bib is None:
                        max_overlap_persons = 3
                        cooccurrence_candidates: Dict[str, float] = {}
                        for bid, person_set in bib_overlapped_persons.items():
                            if pid not in person_set:
                                continue
                            if len(person_set) > max_overlap_persons:
                                continue
                            if bid not in all_consensus:
                                continue
                            bc_number, bc_conf, bc_level = all_consensus[bid]
                            if bc_level not in ("high", "medium"):
                                continue
                            if _is_phantom(bc_number):
                                continue
                            prev = cooccurrence_candidates.get(bc_number, 0.0)
                            if bc_conf > prev:
                                cooccurrence_candidates[bc_number] = bc_conf

                        for rc_number, rc_conf in cooccurrence_candidates.items():
                            if rc_conf > best_conf:
                                best_bib = rc_number
                                best_conf = rc_conf

                    if best_bib is None:
                        diag = person_track_diag.get(pid, {})
                        person_first_frame = diag.get("first_frame", frame_idx)
                        person_last_frame = diag.get("last_frame", frame_idx)
                        person_first_chest = diag.get("first_chest", p_centroid)
                        person_last_chest = diag.get("last_chest", p_centroid)
                        temporal_margin = 5.0 * fps
                        retro_margin = 600

                        retro_candidates: Dict[str, Tuple[float, float]] = {}

                        for bid, (bc_number, bc_conf, bc_level) in all_consensus.items():
                            if bc_level not in ("high", "medium"):
                                continue
                            if _is_phantom(bc_number):
                                continue
                            bib_first = bib_track_first_frame.get(bid)
                            bib_last = bib_track_last_frame.get(bid)
                            if bib_first is None or bib_last is None:
                                continue
                            if bib_last < person_first_frame - temporal_margin:
                                continue
                            if bib_first > person_last_frame + temporal_margin:
                                continue

                            bib_positions = []
                            bfp = bib_track_first_pos.get(bid)
                            blp = bib_track_last_pos.get(bid)
                            if bfp is not None:
                                bib_positions.append(bfp)
                            if blp is not None and blp != bfp:
                                bib_positions.append(blp)
                            if not bib_positions:
                                continue

                            person_positions = [person_first_chest, person_last_chest]
                            min_dist = float("inf")
                            for pp in person_positions:
                                for bp in bib_positions:
                                    d = ((pp[0] - bp[0]) ** 2 + (pp[1] - bp[1]) ** 2) ** 0.5
                                    if d < min_dist:
                                        min_dist = d

                            if min_dist < retro_margin:
                                prev = retro_candidates.get(bc_number)
                                if prev is None or min_dist < prev[0]:
                                    retro_candidates[bc_number] = (min_dist, bc_conf)

                        for rc_number, (rc_dist, rc_conf) in retro_candidates.items():
                            if rc_conf > best_conf or (
                                rc_conf == best_conf
                                and best_bib is not None
                                and rc_dist
                                < retro_candidates.get(best_bib, (float("inf"),))[0]
                            ):
                                best_bib = rc_number
                                best_conf = rc_conf

                    if best_bib is not None:
                        bib_number = best_bib
                        confidence = best_conf

                _crossing_bib_valid = (
                    bib_validator is not None and bib_number in bib_validator.bib_set
                )
                if bib_number != "UNKNOWN" and not _crossing_bib_valid:
                    if len(bib_number) == 1 and confidence < 0.95:
                        bib_number = "UNKNOWN"
                        confidence = 0.0
                    elif len(bib_number) == 2 and confidence < 0.90:
                        bib_number = "UNKNOWN"
                        confidence = 0.0

                if not bib_dedup.should_emit(
                    bib_number, frame_idx, confidence, track_id=pid, position=p_centroid
                ):
                    dedup_suppressed += 1
                    continue

                crossing_seq += 1
                event = CrossingEvent(
                    sequence=crossing_seq,
                    frame_idx=frame_idx,
                    timestamp_sec=time_sec,
                    person_track_id=pid,
                    bib_number=bib_number,
                    confidence=confidence,
                    person_bbox=p_bbox,
                    chest_point=p_centroid,
                    source="pose",
                )
                crossing_log.write(event)
                if on_crossing is not None:
                    on_crossing(event)
                all_crossing_events.append(event)
                total_crossings += 1
                if bib_number == "UNKNOWN":
                    unknown_crossings += 1
                else:
                    crossing_bib_found += 1
                    emitted_bibs.add(bib_number)
                if pid in person_track_diag and bib_number != "UNKNOWN":
                    person_track_diag[pid]["bib"] = bib_number

            if crossing_detector is not None:
                crossing_detector.cleanup(set(person_tracked.keys()))
            person_bib_assoc.cleanup(frame_idx)

            if should_annotate:
                for pid, (p_centroid, p_bbox) in person_tracked.items():
                    px1, py1, px2, py2 = p_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)
                    cx, cy = int(p_centroid[0]), int(p_centroid[1])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

        elif crossing_detector is not None:
            for track_id, (centroid, bbox) in tracked.items():
                cx_norm = centroid[0] / width
                cy_norm = centroid[1] / height

                if crossing_detector.check(track_id, cx_norm, cy_norm, frame_idx):
                    bib_number = "UNKNOWN"
                    confidence = 0.0
                    if track_id in final_consensus:
                        bib_number, confidence, _ = final_consensus[track_id]

                    _crossing_bib_valid = (
                        bib_validator is not None and bib_number in bib_validator.bib_set
                    )
                    if bib_number != "UNKNOWN" and not _crossing_bib_valid:
                        if len(bib_number) == 1 and confidence < 0.95:
                            bib_number = "UNKNOWN"
                            confidence = 0.0
                        elif len(bib_number) == 2 and confidence < 0.90:
                            bib_number = "UNKNOWN"
                            confidence = 0.0

                    if not bib_dedup.should_emit(
                        bib_number, frame_idx, confidence, track_id=track_id,
                        position=centroid,
                    ):
                        dedup_suppressed += 1
                        continue

                    crossing_seq += 1
                    event = CrossingEvent(
                        sequence=crossing_seq,
                        frame_idx=frame_idx,
                        timestamp_sec=time_sec,
                        person_track_id=track_id,
                        bib_number=bib_number,
                        confidence=confidence,
                        person_bbox=bbox,
                        source="bib_tracker",
                    )
                    crossing_log.write(event)
                    if on_crossing is not None:
                        on_crossing(event)
                    all_crossing_events.append(event)
                    total_crossings += 1
                    if bib_number == "UNKNOWN":
                        unknown_crossings += 1
                    else:
                        emitted_bibs.add(bib_number)

            crossing_detector.cleanup(set(tracked.keys()))

        if timing_line is not None and should_annotate:
            pt1, pt2 = timing_line.to_pixel_coords(width, height)
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

        active_track_ids = set(tracked.keys())
        stale_voting_ids = set(voting.history.keys()) - active_track_ids
        for dead_id in stale_voting_ids:
            voting.clear_track(dead_id)

        for tid in list(final_consensus.keys()):
            if tid not in active_track_ids:
                last_update = final_consensus_frame.get(tid, 0)
                if frame_idx - last_update > consensus_ttl:
                    final_consensus.pop(tid, None)
                    final_consensus_frame.pop(tid, None)

        if out_writer:
            out_writer.write(frame)

        if on_frame is not None:
            on_frame(frame)

        if show:
            cv2.imshow("Pipeline Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Periodic progress reporting
        progress_interval = 100 if total_frames else 300
        if frame_idx > 0 and frame_idx % progress_interval == 0:
            elapsed = time.time() - wall_start
            current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
            processed = max(frame_idx - start_frame_idx, 1)
            if on_progress is not None:
                on_progress(
                    ProgressInfo(
                        frame_idx=frame_idx,
                        elapsed_sec=elapsed,
                        total_crossings=total_crossings,
                        unknown_crossings=unknown_crossings,
                        total_detections=total_detections,
                        fps=current_fps,
                        avg_det_ms=1000 * total_det_time / processed,
                        avg_ocr_ms=1000 * total_ocr_time / max(total_detections, 1),
                        avg_pose_ms=1000 * total_pose_time / processed,
                    )
                )
            elif total_frames:
                print(
                    f"  Frame {frame_idx}/{total_frames} "
                    f"({100*frame_idx/total_frames:.1f}%)"
                )
            else:
                print(
                    f"  Frame {frame_idx} | {elapsed:.0f}s elapsed | "
                    f"{total_crossings} crossings"
                )

    # Cleanup parallel workers
    prefetcher.shutdown()
    if _pose_pool is not None:
        _pose_pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Post-processing pass: resolve UNKNOWN crossings using global data
    # ------------------------------------------------------------------
    postproc_resolved = 0
    if enable_crossings and crossing_log is not None and all_crossing_events:
        # Build candidate pool from sampled bib observations.
        candidate_bibs: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
        for bib_number, obs_list in bib_observations.items():
            if bib_number in emitted_bibs:
                continue
            if len(bib_consensus_tracks.get(bib_number, set())) > max_bib_tracks:
                continue
            if obs_list:
                candidate_bibs[bib_number] = obs_list

        unknown_events = [e for e in all_crossing_events if e.bib_number == "UNKNOWN"]
        temporal_margin = 15.0 * fps  # 15 seconds
        spatial_margin = 1000  # pixels

        for event in unknown_events:
            best_bib = None
            best_score = float("inf")

            for bib_number, observations in candidate_bibs.items():
                if bib_number in emitted_bibs:
                    continue
                for obs_frame, obs_pos in observations:
                    dt = abs(event.frame_idx - obs_frame)
                    if dt > temporal_margin:
                        continue
                    if event.chest_point is not None:
                        dx = event.chest_point[0] - obs_pos[0]
                        dy = event.chest_point[1] - obs_pos[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                    else:
                        px1, py1, px2, py2 = event.person_bbox
                        pcx = (px1 + px2) / 2.0
                        pcy = (py1 + py2) / 2.0
                        dx = pcx - obs_pos[0]
                        dy = pcy - obs_pos[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                    if dist > spatial_margin:
                        continue
                    # Score: prefer closer in time and space
                    score = dt + dist
                    if score < best_score:
                        best_score = score
                        best_bib = bib_number

            if best_bib is not None:
                # Validate if bib_validator available
                if bib_validator is not None and best_bib not in bib_validator.bib_set:
                    continue
                bc_conf = max(
                    (c for bid, (n, c, l) in all_consensus.items() if n == best_bib),
                    default=0.0,
                )
                event.bib_number = best_bib
                event.confidence = bc_conf
                event.source = "postproc"
                crossing_log.write(event)
                if on_crossing is not None:
                    on_crossing(event)
                emitted_bibs.add(best_bib)
                postproc_resolved += 1
                unknown_crossings -= 1

    if out_writer:
        out_writer.release()
    if raw_writer:
        raw_writer.release()
    log_file.close()
    if crossing_log is not None:
        crossing_log.close()

    if show:
        cv2.destroyAllWindows()

    review_path = output_dir / f"{output_stem}_review_queue.json"
    confidence_mgr.export_review_queue(str(review_path))

    person_track_diag_path = None
    if enable_crossings and person_track_diag:
        person_track_diag_path = output_dir / f"{output_stem}_person_tracks.csv"
        with open(person_track_diag_path, "w", newline="") as df:
            dw = csv.writer(df)
            dw.writerow(
                [
                    "track_id",
                    "first_frame",
                    "last_frame",
                    "frames_seen",
                    "first_chest_x",
                    "first_chest_y",
                    "last_chest_x",
                    "last_chest_y",
                    "crossed",
                    "bib",
                ]
            )
            for tid in sorted(person_track_diag.keys()):
                d = person_track_diag[tid]
                fc = d["first_chest"]
                lc = d.get("last_chest", fc)
                dw.writerow(
                    [
                        tid,
                        d["first_frame"],
                        d.get("last_frame", d["first_frame"]),
                        d["frames_seen"],
                        f"{fc[0]:.1f}",
                        f"{fc[1]:.1f}",
                        f"{lc[0]:.1f}",
                        f"{lc[1]:.1f}",
                        d["crossed"],
                        d["bib"] or "",
                    ]
                )

    if print_summary:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Frames processed: {frame_idx}")
        print(f"Total detections: {total_detections}")
        print(f"Unique tracks: {tracker.next_id}")
        print(
            f"Avg detection time: {1000 * total_det_time / max(frame_idx, 1):.1f} ms/frame"
        )
        print(
            f"Avg OCR time: {1000 * total_ocr_time / max(total_detections, 1):.1f} "
            "ms/detection"
        )

        print("\nTier 2 Optimizations:")
        print(f"  Edge-rejected (partial at frame edge): {edge_rejected}")
        print(f"  Partial/obstructed rejected: {partial_rejected}")
        print(f"  Quality-rejected (blur/size/etc): {quality_rejected}")
        print(
            "  Total OCR calls saved: "
            f"{edge_rejected + partial_rejected + quality_rejected + ocr_skip_count}"
        )
        print(f"  OCR stable-track skips: {ocr_skip_count}")
        print(f"  OCR cleanups applied: {cleanup_modified}")
        print(f"  Digit-confusion corrections: {digit_corrections}")

        level_counts = {"high": 0, "medium": 0, "low": 0, "reject": 0}
        bib_counts: Dict[str, int] = {}
        for track_id, (number, conf, level) in all_consensus.items():
            level_counts[level] += 1
            if level in ("high", "medium"):
                bib_counts[number] = bib_counts.get(number, 0) + 1

        print("\nConfidence distribution:")
        print(f"  HIGH:   {level_counts['high']} tracks")
        print(f"  MEDIUM: {level_counts['medium']} tracks")
        print(f"  LOW:    {level_counts['low']} tracks")
        print(f"  REJECT: {level_counts['reject']} tracks")

        print("\nUnique bibs (HIGH/MEDIUM confidence):")
        for bib, count in sorted(bib_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {bib}: {count} tracks")

        review_count = len(confidence_mgr.get_pending_reviews())
        print(f"\nItems flagged for review: {review_count}")

        if enable_crossings:
            print("\nCrossing Detection:")
            print(f"  Total crossings: {total_crossings}")
            print(f"  With bib: {total_crossings - unknown_crossings}")
            print(f"  UNKNOWN (no bib): {unknown_crossings}")
            print(f"  Post-proc resolved: {postproc_resolved}")
            print(f"  Dedup suppressed: {dedup_suppressed}")
            if person_tracker is not None:
                print(f"  Person tracks: {person_tracker.next_id}")
            if pose_detector is not None:
                print(
                    f"  Avg pose time: {1000 * total_pose_time / max(frame_idx, 1):.1f} ms/frame"
                )
                print(f"  Total person detections: {total_person_dets}")
                print(f"  Avg persons/frame: {total_person_dets / max(frame_idx, 1):.1f}")

        print("\nOutputs:")
        if config.write_video:
            print(f"  Video:  {output_video_path}")
        print(f"  Log:    {log_path}")
        print(f"  Review: {review_path}")
        if crossing_log_path is not None:
            print(f"  Crossings: {crossing_log_path}")

    stats = PipelineStats(
        frames_processed=frame_idx,
        total_detections=total_detections,
        unique_tracks=tracker.next_id,
        avg_det_time_ms=1000 * total_det_time / max(frame_idx, 1),
        avg_ocr_time_ms=1000 * total_ocr_time / max(total_detections, 1),
        total_crossings=total_crossings,
        unknown_crossings=unknown_crossings,
        dedup_suppressed=dedup_suppressed,
        person_tracks=person_tracker.next_id if person_tracker is not None else None,
        avg_pose_time_ms=1000 * total_pose_time / max(frame_idx, 1),
    )

    outputs = PipelineOutputs(
        detection_log_path=log_path,
        review_queue_path=review_path,
        output_video_path=output_video_path if config.write_video else None,
        crossing_log_path=crossing_log_path,
        person_track_diag_path=person_track_diag_path,
    )

    return PipelineResult(stats=stats, outputs=outputs)
