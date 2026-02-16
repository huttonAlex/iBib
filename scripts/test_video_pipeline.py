#!/usr/bin/env python3
"""Test full bib detection + OCR pipeline on video with Tier 1 improvements.

Runs YOLOv8 bib detector + PARSeq/CRNN OCR on each frame with:
- Bib set validation (reject impossible numbers)
- Multi-frame voting (temporal consistency)
- Confidence thresholding (flag uncertain reads)
- Person detection via YOLOv8n-pose (chest keypoint crossing)
- Persistent person-bib association with temporal voting
- Bib-level dedup to prevent double-counting

Outputs annotated video, detection log, crossing log, and review queue.

Usage:
    python scripts/test_video_pipeline.py path/to/video.mp4
    python scripts/test_video_pipeline.py path/to/video.mp4 --ocr crnn
    python scripts/test_video_pipeline.py path/to/video.mp4 --bib-set bibs.txt
    python scripts/test_video_pipeline.py path/to/video.mp4 --placement right --no-video
    python scripts/test_video_pipeline.py path/to/video.mp4 --timing-line 0.5,0.0,0.5,1.0
    python scripts/test_video_pipeline.py path/to/video.mp4 --show
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pointcam.recognition import (
    BibSetValidator,
    EnhancedTemporalVoting,
    ConfidenceManager,
    ConfidenceLevel,
    CropQualityFilter,
    PostOCRCleanup,
    DigitCountValidator,
    BibCompletenessChecker,
)
from pointcam.crossing import (
    BibCrossingDeduplicator,
    CentroidTracker,
    CrossingDetector,
    CrossingEvent,
    CrossingEventLog,
    PersistentPersonBibAssociator,
    PoseDetector,
    TimingLine,
)


# ---------------------------------------------------------------------------
# OCR Models
# ---------------------------------------------------------------------------


class PARSeqOCR:
    """PARSeq OCR from fine-tuned checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval().to(device)

        self.transform = self._make_transform

    @staticmethod
    def _make_transform(img: Image.Image) -> torch.Tensor:
        """Resize, convert to tensor, and normalize without torchvision."""
        img = img.resize((128, 32), Image.BICUBIC)
        t = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return (t - 0.5) / 0.5

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Predict bib number from BGR crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(-1)
            preds, probs_out = self.model.tokenizer.decode(probs)

        text = preds[0]
        p = probs_out[0]
        if p.numel() == 0:
            confidence = 0.5
        elif p.dim() == 1:
            confidence = p.cumprod(-1)[-1].item()
        else:
            confidence = p.cumprod(-1)[:, -1].item()

        digits = "".join(c for c in text if c.isdigit())
        return digits, confidence

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Predict bib numbers from a batch of BGR crops in a single forward pass."""
        if not crops_bgr:
            return []

        # Vectorized preprocessing: BGR→RGB, resize, normalize via numpy/cv2
        # (avoids per-crop PIL roundtrip and torchvision overhead)
        preprocessed = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 32), interpolation=cv2.INTER_CUBIC)
            t = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
            preprocessed.append((t - 0.5) / 0.5)

        batch = torch.from_numpy(np.stack(preprocessed)).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = logits.softmax(-1)
            preds, probs_out = self.model.tokenizer.decode(probs)

        results = []
        for i in range(len(crops_bgr)):
            text = preds[i]
            p = probs_out[i]
            if p.numel() == 0:
                confidence = 0.5
            elif p.dim() == 1:
                confidence = p.cumprod(-1)[-1].item()
            else:
                confidence = p.cumprod(-1)[:, -1].item()
            digits = "".join(c for c in text if c.isdigit())
            results.append((digits, confidence))

        return results


class CRNNOCR:
    """CRNN OCR from ONNX model."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Predict bib number from BGR crop."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, np.newaxis, :, :]  # (1, 1, 32, 128)

        output = self.session.run(None, {self.input_name: tensor})[0]
        logits = output[0]  # (T, num_classes)

        # Greedy CTC decode — apply softmax to get 0-1 probabilities
        indices = logits.argmax(axis=-1)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        max_probs = probs.max(axis=-1)

        result = []
        conf_values = []
        prev_idx = -1
        for t, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:  # 0 = CTC blank
                result.append(str(idx - 1))  # 1-10 -> '0'-'9'
                conf_values.append(max_probs[t])
            prev_idx = idx

        text = "".join(result)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return text, confidence


# ---------------------------------------------------------------------------
# Enhanced Pipeline
# ---------------------------------------------------------------------------


# Crop padding presets per camera placement (left_pct, right_pct, top_pct, bottom_pct)
PLACEMENT_PADDING = {
    "center": (0.15, 0.15, 0.10, 0.10),
    "left":   (0.10, 0.25, 0.10, 0.10),
    "right":  (0.25, 0.10, 0.10, 0.10),
}


def process_video(
    video_path: str,
    detector_path: str,
    ocr_model,
    output_dir: Path,
    bib_validator: Optional[BibSetValidator] = None,
    show: bool = False,
    conf_threshold: float = 0.5,
    ocr_conf_threshold: float = 0.5,
    enable_quality_filter: bool = True,
    write_video: bool = True,
    placement: str = "center",
    timing_line_coords: Optional[Tuple[float, float, float, float]] = None,
    crossing_direction: str = "any",
    debounce_time: float = 2.0,
    enable_person_detect: bool = True,
    pose_model_path: str = "yolov8n-pose.pt",
    stride: int = 1,
    start_time: float = 0.0,
    enable_ocr_skip: bool = True,
    crossing_mode: str = "line",
):
    """Process video with bib detection + OCR + Tier 1+2 improvements."""

    # Load detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(detector_path)
    # .to() only works for PyTorch models; TensorRT/ONNX pass device at predict time
    if detector_path.endswith(".pt"):
        detector.to(device)

    # Initialize Tier 1 components
    tracker = CentroidTracker(max_disappeared=30, max_distance=100)
    voting = EnhancedTemporalVoting(
        window_size=15,
        min_votes=3,
        stability_threshold=5,
        confidence_threshold=ocr_conf_threshold,
    )
    confidence_mgr = ConfidenceManager(auto_accept_validated=bib_validator is not None)

    # Initialize Tier 2 components
    quality_filter = CropQualityFilter(
        min_blur_score=50.0,
        min_width=40,
        min_height=15,
        min_aspect_ratio=1.0,
        max_aspect_ratio=6.0,
        min_completeness=0.6,
        check_completeness=True,
    ) if enable_quality_filter else None

    # Completeness checker for frame-edge detection
    completeness_checker = BibCompletenessChecker(
        edge_margin_ratio=0.02,  # 2% margin from frame edges
    )

    # Infer max bib value from bib set for range filtering
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

    # Infer expected digit counts from bib set if available
    digit_validator = None
    if bib_validator:
        digit_validator = DigitCountValidator.from_bib_set(bib_validator.bib_set)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    if bib_validator:
        print(f"Bib set: {len(bib_validator.bib_set)} valid numbers loaded")

    # Output video (optional — writing raw video is I/O-bound and slow)
    output_video_path = output_dir / f"{Path(video_path).stem}_annotated.mp4"
    out_writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )

    # Detection log with enhanced columns
    log_path = output_dir / f"{Path(video_path).stem}_detections.csv"
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

    # Stats
    frame_idx = 0
    total_detections = 0
    total_ocr_time = 0.0
    total_det_time = 0.0
    quality_rejected = 0
    edge_rejected = 0
    partial_rejected = 0
    cleanup_modified = 0
    ocr_skip_count = 0

    # Track final consensus per track (pruned when stale)
    final_consensus: Dict[int, Tuple[str, float, str]] = {}  # track_id -> (number, conf, level)
    final_consensus_frame: Dict[int, int] = {}  # track_id -> last frame_idx updated
    consensus_ttl = int(5 * fps)  # prune entries dead longer than 5 seconds
    # All-time best consensus per track for summary stats (never pruned)
    all_consensus: Dict[int, Tuple[str, float, str]] = {}  # track_id -> (number, conf, level)

    # --- Person detection + crossing detection (optional) ---
    timing_line = None
    crossing_detector = None
    pose_detector = None
    person_tracker = None
    person_bib_assoc = None
    bib_dedup = None
    crossing_log = None
    crossing_seq = 0
    total_crossings = 0
    unknown_crossings = 0
    dedup_suppressed = 0
    total_pose_time = 0.0

    # --- Diagnostic: person track lifecycle tracking ---
    # track_id -> {first_frame, last_frame, first_chest, last_chest, frames_seen, crossed, bib}
    person_track_diag: Dict[int, dict] = {}
    total_person_dets = 0  # total person detections across all frames
    crossing_bib_found = 0  # crossings where a bib was successfully associated
    crossing_unknown_pre_dedup = 0  # crossings that were UNKNOWN before dedup

    # Zone-based crossing: track which person tracks have already fired
    person_track_emitted: set = set()  # track IDs that have emitted a crossing
    ZONE_MIN_FRAMES = 5  # minimum frames to filter noise tracks
    # Deferred emission: tracks that are ready but waiting for bib identification
    person_track_ready: set = set()  # tracks past ZONE_MIN_FRAMES, awaiting bib or death
    prev_person_tracked_ids: set = set()  # track IDs from previous frame

    # Enable crossing detection for either timing-line mode or zone mode
    enable_crossings = timing_line_coords is not None or crossing_mode == "zone"

    if enable_crossings:
        if timing_line_coords is not None:
            timing_line = TimingLine(*timing_line_coords)
            debounce_frames = int(debounce_time * fps)
            crossing_detector = CrossingDetector(
                timing_line=timing_line,
                direction=crossing_direction,
                debounce_frames=debounce_frames,
            )
            print(f"Timing line: {timing_line_coords}, direction={crossing_direction}")
            print(f"Debounce: {debounce_time}s ({debounce_frames} frames)")

        if crossing_mode == "zone":
            print(f"Crossing mode: ZONE (emit once per person track after {ZONE_MIN_FRAMES} frames)")
        else:
            print(f"Crossing mode: LINE")

        # Bib-level deduplication (10s worth of frames)
        bib_dedup = BibCrossingDeduplicator(debounce_frames=int(10.0 * fps))

        if enable_person_detect or crossing_mode == "zone":
            pose_detector = PoseDetector(
                model_path=pose_model_path,
                conf=0.5,
                device=device,
            )
            person_tracker = CentroidTracker(max_disappeared=20, max_distance=150)

            # Infer expected digit counts for short-bib penalty
            expected_digits = None
            if bib_validator:
                expected_digits = {len(b) for b in bib_validator.bib_set if b}

            person_bib_assoc = PersistentPersonBibAssociator(
                max_distance=350,
                memory_frames=int(30.0 * fps),  # 30 seconds
                min_votes=1,
                min_confidence=0.4,
                expected_digit_counts=expected_digits,
            )
            print(f"Person detection: ENABLED (YOLOv8n-pose: {pose_model_path})")
        else:
            print("Person detection: DISABLED (bib-tracker crossings)")

        crossing_log_path = output_dir / f"{Path(video_path).stem}_crossings.csv"
        crossing_log = CrossingEventLog(crossing_log_path)

    # Seek past dead time at the start of the video
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Start time: {start_time:.1f}s (seeking to frame {frame_idx})")

    if stride > 1:
        print(f"Frame stride: {stride} (processing every {stride}th frame)")
    print("\nProcessing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps
        frame_idx += 1

        # Skip frames based on stride (still count for correct timestamps)
        if stride > 1 and frame_idx % stride != 0:
            continue

        # Detect bibs
        t0 = time.perf_counter()
        results = detector(frame, conf=conf_threshold, device=device, verbose=False)
        t1 = time.perf_counter()
        total_det_time += t1 - t0

        # Collect detections
        detections = []
        det_confs = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                det_conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2))
                det_confs.append(det_conf)

        # Update tracker
        tracked = tracker.update(detections)

        # Phase 1: Collect crops and metadata for batch OCR
        ocr_batch_items = []  # (track_id, bbox, det_conf, crop)
        for track_id, (centroid, bbox) in tracked.items():
            x1, y1, x2, y2 = bbox

            # Find detection confidence for this bbox
            det_conf = 0.0
            for i, det in enumerate(detections):
                if det == bbox:
                    det_conf = det_confs[i]
                    break

            # Check if bib is fully visible (not at frame edge)
            is_visible, edge_reason = completeness_checker.is_fully_visible(
                bbox=(x1, y1, x2, y2),
                frame_width=width,
                frame_height=height,
            )
            if not is_visible:
                edge_rejected += 1
                continue  # Skip bibs entering/exiting frame

            # Expand crop for OCR — padding varies by camera placement to
            # compensate for the far side of the bib being clipped by the angle
            bib_w = x2 - x1
            bib_h = y2 - y1
            lp, rp, tp, bp = PLACEMENT_PADDING.get(placement, PLACEMENT_PADDING["center"])
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

            # Tier 2: Quality filter (skip bad crops to save OCR compute)
            if quality_filter:
                quality = quality_filter.assess(crop)
                if not quality.is_acceptable:
                    if "partial" in (quality.rejection_reason or "").lower():
                        partial_rejected += 1
                    else:
                        quality_rejected += 1
                    continue  # Skip OCR for low-quality crops

            # Phase B: Skip OCR for tracks with stable high-confidence consensus
            if enable_ocr_skip:
                consensus = voting.get_consensus(track_id)
                if (
                    consensus.is_stable
                    and consensus.confidence >= 0.85
                    and track_id in final_consensus
                ):
                    # Only skip if bib is validated against the bib set.
                    # If no validator or bib not in set, keep running OCR
                    # so a wrong initial consensus can self-correct.
                    bib_validated = (
                        bib_validator is not None
                        and consensus.number in bib_validator.bib_set
                    )
                    if bib_validated or bib_validator is None:
                        ocr_skip_count += 1
                        continue

            ocr_batch_items.append((track_id, bbox, det_conf, crop))

        # Phase 2: Batch OCR (single GPU forward pass when possible)
        crops = [item[3] for item in ocr_batch_items]
        if crops and hasattr(ocr_model, "predict_batch"):
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

        # Phase 3: Process OCR results
        for idx, (track_id, bbox, det_conf, crop) in enumerate(ocr_batch_items):
            x1, y1, x2, y2 = bbox
            ocr_raw, ocr_conf = ocr_results[idx]
            total_detections += 1

            if not ocr_raw:
                continue

            # Tier 2: Post-OCR cleanup (fix common errors)
            cleaned_number, was_modified = post_cleanup.clean(ocr_raw)
            if was_modified:
                cleanup_modified += 1
            ocr_raw_original = ocr_raw
            ocr_raw = cleaned_number

            if not ocr_raw:
                continue

            # Tier 2: Digit count validation
            if digit_validator:
                ocr_raw, ocr_conf, _ = digit_validator.validate(ocr_raw, ocr_conf)

            # Tier 1: Validation
            validation_result = None
            validated_number = ocr_raw
            is_valid = False
            if bib_validator:
                validation_result = bib_validator.validate(ocr_raw, ocr_conf)
                validated_number = validation_result.validated
                is_valid = validation_result.is_valid

            # Early fragment suppression: prevent low-confidence short reads
            # from polluting the voting pipeline.  Skip only if the number
            # is NOT a known-valid bib (real 1-2 digit bibs are preserved).
            _is_known_bib = (
                bib_validator is not None and validated_number in bib_validator.bib_set
            )
            if not _is_known_bib:
                if len(validated_number) == 1 and ocr_conf < 0.95:
                    continue
                if len(validated_number) == 2 and ocr_conf < 0.90:
                    continue

            # Tier 1: Voting
            voting.update(track_id, validated_number, ocr_conf, frame_idx)
            consensus_result = voting.get_consensus(track_id)

            consensus_number = consensus_result.number or validated_number
            consensus_conf = consensus_result.confidence if consensus_result.number else ocr_conf

            # Tier 1: Confidence classification
            classified = confidence_mgr.classify(
                bib_number=consensus_number,
                ocr_confidence=ocr_conf,
                validation_result=validation_result,
                voting_result=consensus_result,
            )

            # Store final consensus for this track
            final_consensus[track_id] = (
                consensus_number,
                classified.adjusted_confidence,
                classified.level.value,
            )
            final_consensus_frame[track_id] = frame_idx
            all_consensus[track_id] = final_consensus[track_id]

            # Add to review queue if needed
            if classified.needs_review:
                confidence_mgr.add_to_review_queue(
                    prediction=classified,
                    frame_number=frame_idx,
                )

            # Log
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

            # Draw on frame with color based on confidence level
            if write_video or show:
                if classified.level == ConfidenceLevel.HIGH:
                    color = (0, 255, 0)  # Green
                elif classified.level == ConfidenceLevel.MEDIUM:
                    color = (0, 255, 255)  # Yellow
                elif classified.level == ConfidenceLevel.LOW:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with consensus and validation status
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

                label = f"#{track_id}: {consensus_number} ({classified.adjusted_confidence:.2f}){status}"
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

        # --- Person detection + crossing detection ---
        if pose_detector is not None:
            t_pose_start = time.perf_counter()
            person_dets = pose_detector.detect(frame)
            t_pose_end = time.perf_counter()
            total_pose_time += t_pose_end - t_pose_start
            total_person_dets += len(person_dets)

            # Update person tracker using chest points as centroids
            p_bboxes = [d.bbox for d in person_dets]
            p_chests = [d.chest_point for d in person_dets]
            person_tracked = person_tracker.update(p_bboxes, centroids=p_chests)

            # Accumulate person→bib associations via voting
            person_bib_assoc.update(person_tracked, tracked, final_consensus, frame_idx)

            # Diagnostic: track person lifecycle
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
                diag["frames_seen"] += 1

            # Check each person for crossing
            # Collect which tracks to emit this frame
            tracks_to_emit: List[Tuple[int, Tuple[float, float], Tuple[int, int, int, int]]] = []

            if crossing_mode == "zone":
                current_ids = set(person_tracked.keys())

                # Mark tracks as ready once they reach minimum age
                for pid in current_ids:
                    if pid not in person_track_emitted and pid not in person_track_ready:
                        diag = person_track_diag.get(pid)
                        if diag and diag["frames_seen"] >= ZONE_MIN_FRAMES:
                            person_track_ready.add(pid)

                # Deferred emission: emit ready tracks when bib found
                for pid in list(person_track_ready):
                    if pid in person_track_emitted:
                        continue
                    bib = person_bib_assoc.get_bib(pid)
                    if bib is not None and pid in current_ids:
                        # Bib identified — emit now
                        p_centroid, p_bbox = person_tracked[pid]
                        tracks_to_emit.append((pid, p_centroid, p_bbox))

                # Deferred emission: emit on track death (disappeared)
                disappeared = (prev_person_tracked_ids - current_ids)
                for pid in disappeared:
                    if pid in person_track_ready and pid not in person_track_emitted:
                        # Track died — emit with whatever we have
                        diag = person_track_diag.get(pid)
                        if diag:
                            lc = diag.get("last_chest", diag["first_chest"])
                            # Use last known position
                            tracks_to_emit.append((pid, lc, (0, 0, 0, 0)))

                prev_person_tracked_ids = current_ids.copy()
            else:
                # Line mode: use timing line crossing detector
                for pid, (p_centroid, p_bbox) in person_tracked.items():
                    cx_norm = p_centroid[0] / width
                    cy_norm = p_centroid[1] / height
                    if crossing_detector.check(pid, cx_norm, cy_norm, frame_idx):
                        tracks_to_emit.append((pid, p_centroid, p_bbox))

            for pid, p_centroid, p_bbox in tracks_to_emit:
                if pid in person_track_emitted:
                    continue
                person_track_emitted.add(pid)

                # Diagnostic: mark track as crossed
                if pid in person_track_diag:
                    person_track_diag[pid]["crossed"] = True

                bib_number = person_bib_assoc.get_bib(pid) or "UNKNOWN"
                confidence = person_bib_assoc.get_bib_confidence(pid)

                # Last-chance bib lookup: if UNKNOWN, scan tracked bibs
                # for any whose center is inside/near this person's bbox.
                if bib_number == "UNKNOWN":
                    px1, py1, px2, py2 = p_bbox
                    margin = max(50, int(0.15 * (px2 - px1)))
                    best_bib = None
                    best_conf = 0.0

                    # Check current tracked bibs (tight match: inside bbox)
                    for bid, (b_centroid, b_bbox) in tracked.items():
                        bcx, bcy = b_centroid
                        if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                            if bid in final_consensus:
                                fc_number, fc_conf, fc_level = final_consensus[bid]
                                if fc_level in ("high", "medium") and fc_conf > best_conf:
                                    best_bib = fc_number
                                    best_conf = fc_conf

                    # Check current tracked bibs (wider margin)
                    if best_bib is None:
                        for bid, (b_centroid, b_bbox) in tracked.items():
                            bcx, bcy = b_centroid
                            if (px1 - margin) <= bcx <= (px2 + margin) and py1 <= bcy <= py2:
                                if bid in final_consensus:
                                    fc_number, fc_conf, fc_level = final_consensus[bid]
                                    if fc_level == "high" and fc_conf > best_conf:
                                        best_bib = fc_number
                                        best_conf = fc_conf

                    if best_bib is not None:
                        bib_number = best_bib
                        confidence = best_conf

                # Short fragment suppression: demote low-conf short bibs
                # unless the bib is a known-valid number in the bib set.
                _crossing_bib_valid = (
                    bib_validator is not None
                    and bib_number in bib_validator.bib_set
                )
                if bib_number != "UNKNOWN" and not _crossing_bib_valid:
                    if len(bib_number) == 1 and confidence < 0.95:
                        bib_number = "UNKNOWN"
                        confidence = 0.0
                    elif len(bib_number) == 2 and confidence < 0.90:
                        bib_number = "UNKNOWN"
                        confidence = 0.0

                # Bib-level dedup (with escalating confidence)
                if not bib_dedup.should_emit(
                    bib_number, frame_idx, confidence, track_id=pid
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
                total_crossings += 1
                if bib_number == "UNKNOWN":
                    unknown_crossings += 1
                else:
                    crossing_bib_found += 1
                # Diagnostic: record bib for track
                if pid in person_track_diag and bib_number != "UNKNOWN":
                    person_track_diag[pid]["bib"] = bib_number

            # Cleanup stale state
            if crossing_detector is not None:
                crossing_detector.cleanup(set(person_tracked.keys()))
            person_bib_assoc.cleanup(frame_idx)

            # Draw person bboxes (cyan) + chest points (magenta dots)
            if write_video or show:
                for pid, (p_centroid, p_bbox) in person_tracked.items():
                    px1, py1, px2, py2 = p_bbox
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)
                    cx, cy = int(p_centroid[0]), int(p_centroid[1])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

        elif crossing_detector is not None:
            # No person detection — use bib tracker for crossings instead
            for track_id, (centroid, bbox) in tracked.items():
                cx_norm = centroid[0] / width
                cy_norm = centroid[1] / height

                if crossing_detector.check(track_id, cx_norm, cy_norm, frame_idx):
                    bib_number = "UNKNOWN"
                    confidence = 0.0
                    if track_id in final_consensus:
                        bib_number, confidence, _ = final_consensus[track_id]

                    # Short fragment suppression: demote low-conf short bibs
                    # unless the bib is a known-valid number in the bib set.
                    _crossing_bib_valid = (
                        bib_validator is not None
                        and bib_number in bib_validator.bib_set
                    )
                    if bib_number != "UNKNOWN" and not _crossing_bib_valid:
                        if len(bib_number) == 1 and confidence < 0.95:
                            bib_number = "UNKNOWN"
                            confidence = 0.0
                        elif len(bib_number) == 2 and confidence < 0.90:
                            bib_number = "UNKNOWN"
                            confidence = 0.0

                    # Bib-level dedup (with escalating confidence)
                    if not bib_dedup.should_emit(
                        bib_number, frame_idx, confidence, track_id=track_id
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
                    total_crossings += 1
                    if bib_number == "UNKNOWN":
                        unknown_crossings += 1

            crossing_detector.cleanup(set(tracked.keys()))

        # Draw timing line (magenta)
        if timing_line is not None and (write_video or show):
            pt1, pt2 = timing_line.to_pixel_coords(width, height)
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

        # Prune dead tracks from voting history to prevent memory leak.
        active_track_ids = set(tracked.keys())
        stale_voting_ids = set(voting.history.keys()) - active_track_ids
        for dead_id in stale_voting_ids:
            voting.clear_track(dead_id)

        # Prune stale final_consensus entries (dead > consensus_ttl frames)
        # to prevent ghost bib assignments from spatially overlapping old tracks.
        for tid in list(final_consensus.keys()):
            if tid not in active_track_ids:
                last_update = final_consensus_frame.get(tid, 0)
                if frame_idx - last_update > consensus_ttl:
                    final_consensus.pop(tid, None)
                    final_consensus_frame.pop(tid, None)

        # Write frame
        if out_writer:
            out_writer.write(frame)

        # Show
        if show:
            cv2.imshow("Pipeline Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Progress
        if frame_idx % 100 == 0:
            print(
                f"  Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)"
            )

    cap.release()
    if out_writer:
        out_writer.release()
    log_file.close()
    if crossing_log is not None:
        crossing_log.close()

    if show:
        cv2.destroyAllWindows()

    # Export review queue
    review_path = output_dir / f"{Path(video_path).stem}_review_queue.json"
    confidence_mgr.export_review_queue(str(review_path))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Frames processed: {frame_idx}")
    print(f"Total detections: {total_detections}")
    print(f"Unique tracks: {tracker.next_id}")
    print(f"Avg detection time: {1000 * total_det_time / frame_idx:.1f} ms/frame")
    print(
        f"Avg OCR time: {1000 * total_ocr_time / max(total_detections, 1):.1f} ms/detection"
    )

    # Tier 2 stats
    print(f"\nTier 2 Optimizations:")
    print(f"  Edge-rejected (partial at frame edge): {edge_rejected}")
    print(f"  Partial/obstructed rejected: {partial_rejected}")
    print(f"  Quality-rejected (blur/size/etc): {quality_rejected}")
    print(f"  Total OCR calls saved: {edge_rejected + partial_rejected + quality_rejected + ocr_skip_count}")
    print(f"  OCR stable-track skips: {ocr_skip_count}")
    print(f"  OCR cleanups applied: {cleanup_modified}")

    # Count by confidence level
    level_counts = {"high": 0, "medium": 0, "low": 0, "reject": 0}
    bib_counts: Dict[str, int] = {}
    for track_id, (number, conf, level) in all_consensus.items():
        level_counts[level] += 1
        if level in ("high", "medium"):
            bib_counts[number] = bib_counts.get(number, 0) + 1

    print(f"\nConfidence distribution:")
    print(f"  HIGH:   {level_counts['high']} tracks")
    print(f"  MEDIUM: {level_counts['medium']} tracks")
    print(f"  LOW:    {level_counts['low']} tracks")
    print(f"  REJECT: {level_counts['reject']} tracks")

    print(f"\nUnique bibs (HIGH/MEDIUM confidence):")
    for bib, count in sorted(bib_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {bib}: {count} tracks")

    review_count = len(confidence_mgr.get_pending_reviews())
    print(f"\nItems flagged for review: {review_count}")

    if enable_crossings:
        print(f"\nCrossing Detection:")
        print(f"  Total crossings: {total_crossings}")
        print(f"  With bib: {total_crossings - unknown_crossings}")
        print(f"  UNKNOWN (no bib): {unknown_crossings}")
        print(f"  Dedup suppressed: {dedup_suppressed}")
        if person_tracker is not None:
            print(f"  Person tracks: {person_tracker.next_id}")
        if pose_detector is not None:
            print(f"  Avg pose time: {1000 * total_pose_time / frame_idx:.1f} ms/frame")
            print(f"  Total person detections: {total_person_dets}")
            print(f"  Avg persons/frame: {total_person_dets / max(frame_idx, 1):.1f}")

        # --- Diagnostic: Person Track Funnel ---
        if person_track_diag:
            n_tracks = len(person_track_diag)
            n_crossed = sum(1 for d in person_track_diag.values() if d["crossed"])
            n_bib = sum(1 for d in person_track_diag.values() if d["bib"] is not None)
            # Tracks that spent time near the timing line
            # "Near" = chest position within 15% of frame dimension from line
            n_near_line = 0
            if timing_line is not None:
                for d in person_track_diag.values():
                    fc = d["first_chest"]
                    lc = d.get("last_chest", fc)
                    tl = timing_line
                    if abs(tl.x2 - tl.x1) > abs(tl.y2 - tl.y1):
                        line_y = (tl.y1 + tl.y2) / 2.0 * height
                        threshold = 0.15 * height
                        if (abs(fc[1] - line_y) < threshold or abs(lc[1] - line_y) < threshold):
                            n_near_line += 1
                    else:
                        line_x = (tl.x1 + tl.x2) / 2.0 * width
                        threshold = 0.15 * width
                        if (abs(fc[0] - line_x) < threshold or abs(lc[0] - line_x) < threshold):
                            n_near_line += 1

            # Track lifespan stats
            lifespans = [d["frames_seen"] for d in person_track_diag.values()]
            avg_life = sum(lifespans) / len(lifespans) if lifespans else 0
            lifespans_sorted = sorted(lifespans)
            median_life = lifespans_sorted[len(lifespans_sorted) // 2] if lifespans_sorted else 0

            print(f"\n  {'='*50}")
            print(f"  PIPELINE FUNNEL (person tracks)")
            print(f"  {'='*50}")
            print(f"  Total person detections:    {total_person_dets}")
            print(f"  Unique person tracks:       {n_tracks}")
            print(f"  Tracks near timing line:    {n_near_line} ({100*n_near_line/max(n_tracks,1):.1f}%)")
            print(f"  Tracks that crossed:        {n_crossed} ({100*n_crossed/max(n_tracks,1):.1f}%)")
            print(f"  Crossings with bib (pre-dup): {crossing_bib_found}")
            print(f"  Crossings UNKNOWN (pre-dup):  {crossing_unknown_pre_dedup + unknown_crossings}")
            print(f"  Dedup suppressed:           {dedup_suppressed}")
            print(f"  Final emitted crossings:    {total_crossings}")
            print(f"  {'='*50}")
            print(f"  Track lifespan: avg={avg_life:.1f} frames, median={median_life} frames")
            short_tracks = sum(1 for l in lifespans if l <= 3)
            print(f"  Tracks ≤3 frames:           {short_tracks} ({100*short_tracks/max(n_tracks,1):.1f}%)")

            # Write person track summary CSV
            diag_path = output_dir / f"{Path(video_path).stem}_person_tracks.csv"
            with open(diag_path, "w", newline="") as df:
                dw = csv.writer(df)
                dw.writerow([
                    "track_id", "first_frame", "last_frame", "frames_seen",
                    "first_chest_x", "first_chest_y", "last_chest_x", "last_chest_y",
                    "crossed", "bib",
                ])
                for tid in sorted(person_track_diag.keys()):
                    d = person_track_diag[tid]
                    fc = d["first_chest"]
                    lc = d.get("last_chest", fc)
                    dw.writerow([
                        tid, d["first_frame"], d.get("last_frame", d["first_frame"]),
                        d["frames_seen"],
                        f"{fc[0]:.1f}", f"{fc[1]:.1f}",
                        f"{lc[0]:.1f}", f"{lc[1]:.1f}",
                        d["crossed"], d["bib"] or "",
                    ])
            print(f"  Person track diag: {diag_path}")

    print(f"\nOutputs:")
    if write_video:
        print(f"  Video:  {output_video_path}")
    print(f"  Log:    {log_path}")
    print(f"  Review: {review_path}")
    if crossing_log is not None:
        print(f"  Crossings: {crossing_log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test bib detection + OCR pipeline on video with Tier 1 improvements"
    )
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument(
        "--ocr",
        type=str,
        default="parseq",
        choices=["parseq", "crnn"],
        help="OCR model to use (default: parseq)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="runs/detect/bib_detector/weights/best.pt",
        help="Path to YOLO detector weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/pipeline_test",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.5,
        help="OCR confidence threshold for 'good' reads",
    )
    parser.add_argument(
        "--bib-set",
        type=str,
        default=None,
        help="Path to bib set file (one number per line) for validation",
    )
    parser.add_argument(
        "--bib-range",
        type=str,
        default=None,
        help="Bib number range for validation (e.g., '1-3000' or '1000-2000')",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show video while processing",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable crop quality filtering (not recommended)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated output video (much faster)",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="center",
        choices=["left", "right", "center"],
        help="Camera placement relative to finish line (affects crop padding). "
             "See docs/CAMERA_PLACEMENT.md for guidance. (default: center)",
    )
    parser.add_argument(
        "--timing-line",
        type=str,
        default=None,
        help="Timing line as normalized coords: x1,y1,x2,y2 (e.g., '0.5,0.0,0.5,1.0'). "
             "Enables crossing detection.",
    )
    parser.add_argument(
        "--crossing-direction",
        type=str,
        default="any",
        choices=["left_to_right", "right_to_left", "any"],
        help="Valid crossing direction (default: any)",
    )
    parser.add_argument(
        "--debounce-time",
        type=float,
        default=2.0,
        help="Minimum seconds between crossings for the same track (default: 2.0)",
    )
    parser.add_argument(
        "--crossing-mode",
        type=str,
        default="line",
        choices=["line", "zone"],
        help="Crossing detection mode: 'line' = timing line crossing (requires --timing-line), "
             "'zone' = emit once per person track after minimum age (for head-on cameras). "
             "(default: line)",
    )
    parser.add_argument(
        "--no-person-detect",
        action="store_true",
        help="Disable person detection (use bib tracker for crossings instead)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolov8n-pose.pt",
        help="Path to YOLOv8-pose weights (default: yolov8n-pose.pt, auto-downloaded)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, no skipping)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Skip to this time in seconds before processing (default: 0.0)",
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "tensorrt"],
        help="OCR inference backend (default: pytorch). "
             "'tensorrt' uses ONNX Runtime + TensorRT EP for GPU acceleration.",
    )
    parser.add_argument(
        "--no-ocr-skip",
        action="store_true",
        help="Disable OCR skip optimization for stable tracks (always run OCR)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve paths
    video_path = (
        args.video
        if Path(args.video).is_absolute()
        else str(project_root / args.video)
    )
    detector_path = (
        args.detector
        if Path(args.detector).is_absolute()
        else str(project_root / args.detector)
    )
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Bib Detection + OCR Pipeline Test (with Tier 1 Improvements)")
    print("=" * 70)
    print(f"Video:    {video_path}")
    print(f"Detector: {detector_path}")
    print(f"OCR:      {args.ocr} (backend: {args.ocr_backend})")
    print(f"Placement: {args.placement}")
    print(f"OCR skip: {'disabled' if args.no_ocr_skip else 'enabled'}")
    print(f"Output:   {output_dir}")

    # Load bib validator if specified
    bib_validator = None
    if args.bib_set:
        bib_set_path = (
            args.bib_set
            if Path(args.bib_set).is_absolute()
            else str(project_root / args.bib_set)
        )
        bib_validator = BibSetValidator.from_file(bib_set_path)
        print(f"Bib set:  {bib_set_path} ({len(bib_validator.bib_set)} numbers)")
    elif args.bib_range:
        try:
            start, end = map(int, args.bib_range.split("-"))
            bib_validator = BibSetValidator.from_range(start, end)
            print(f"Bib range: {start}-{end} ({len(bib_validator.bib_set)} numbers)")
        except ValueError:
            print(f"ERROR: Invalid bib range format: {args.bib_range}")
            print("  Expected format: START-END (e.g., '1-3000')")
            sys.exit(1)
    else:
        print("Bib set:  None (validation disabled)")

    print()

    # Load OCR model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OCR model (device={device}, backend={args.ocr_backend})...")
    if args.ocr_backend == "tensorrt":
        from pointcam.inference import OnnxTensorRTParseqOCR, TensorRTCrnnOCR

        if args.ocr == "parseq":
            parseq_onnx = project_root / "models/ocr_parseq.onnx"
            ocr_model = OnnxTensorRTParseqOCR(str(parseq_onnx))
            print(f"  PARSeq TensorRT loaded from {parseq_onnx}")
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = TensorRTCrnnOCR(str(crnn_onnx))
            print(f"  CRNN TensorRT loaded from {crnn_onnx}")
    else:
        if args.ocr == "parseq":
            parseq_checkpoint = project_root / "runs/ocr_finetune/parseq_gpu_v1/best.pt"
            ocr_model = PARSeqOCR(str(parseq_checkpoint), device=device)
            print(f"  PARSeq loaded from {parseq_checkpoint}")
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = CRNNOCR(str(crnn_onnx))
            print(f"  CRNN loaded from {crnn_onnx}")

    # Parse timing line
    timing_line_coords = None
    if args.timing_line:
        try:
            parts = [float(x) for x in args.timing_line.split(",")]
            if len(parts) != 4:
                raise ValueError("need 4 values")
            timing_line_coords = tuple(parts)
        except ValueError:
            print(f"ERROR: Invalid timing line format: {args.timing_line}")
            print("  Expected format: x1,y1,x2,y2 (e.g., '0.5,0.0,0.5,1.0')")
            sys.exit(1)

    # Process
    process_video(
        video_path=video_path,
        detector_path=detector_path,
        ocr_model=ocr_model,
        output_dir=output_dir,
        bib_validator=bib_validator,
        show=args.show,
        conf_threshold=args.conf,
        ocr_conf_threshold=args.ocr_conf,
        enable_quality_filter=not args.no_quality_filter,
        write_video=not args.no_video,
        placement=args.placement,
        timing_line_coords=timing_line_coords,
        crossing_direction=args.crossing_direction,
        debounce_time=args.debounce_time,
        enable_person_detect=not args.no_person_detect,
        pose_model_path=args.pose_model,
        stride=args.stride,
        start_time=args.start_time,
        enable_ocr_skip=not args.no_ocr_skip,
        crossing_mode=args.crossing_mode,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
