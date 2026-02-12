#!/usr/bin/env python3
"""Test full bib detection + OCR pipeline on video with Tier 1 improvements.

Runs YOLOv8 bib detector + PARSeq/CRNN OCR on each frame with:
- Bib set validation (reject impossible numbers)
- Multi-frame voting (temporal consistency)
- Confidence thresholding (flag uncertain reads)
- Person detection via background subtraction (optional)
- Timing-line crossing detection

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
    CentroidTracker,
    TimingLine,
    CrossingDetector,
    BackgroundSubtractorManager,
    PersonBibAssociator,
    MergedBlobEstimator,
    CrossingEvent,
    CrossingEventLog,
)


# ---------------------------------------------------------------------------
# OCR Models
# ---------------------------------------------------------------------------


class PARSeqOCR:
    """PARSeq OCR from fine-tuned checkpoint."""

    def __init__(self, checkpoint_path: str):
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

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
        tensor = self.transform(pil_img).unsqueeze(0)

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
):
    """Process video with bib detection + OCR + Tier 1+2 improvements."""

    # Load detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(detector_path)
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

    # Track final consensus per track
    final_consensus: Dict[int, Tuple[str, float, str]] = {}  # track_id -> (number, conf, level)

    # --- Person detection + crossing detection (optional) ---
    timing_line = None
    crossing_detector = None
    bg_manager = None
    person_tracker = None
    person_bib_assoc = None
    blob_estimator = None
    crossing_log = None
    crossing_seq = 0
    total_crossings = 0
    unknown_crossings = 0

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

        if enable_person_detect:
            bg_manager = BackgroundSubtractorManager()
            person_tracker = CentroidTracker(max_disappeared=15, max_distance=150)
            person_bib_assoc = PersonBibAssociator(max_distance=150)
            blob_estimator = MergedBlobEstimator()
            print("Person detection: ENABLED (MOG2 background subtraction)")
        else:
            print("Person detection: DISABLED")

        crossing_log_path = output_dir / f"{Path(video_path).stem}_crossings.csv"
        crossing_log = CrossingEventLog(crossing_log_path)

    print("\nProcessing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_sec = frame_idx / fps

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

        # Process each tracked detection
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

            # OCR
            t2 = time.perf_counter()
            ocr_raw, ocr_conf = ocr_model.predict(crop)
            t3 = time.perf_counter()
            total_ocr_time += t3 - t2
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
        if bg_manager is not None:
            person_bboxes = bg_manager.process(frame)
            person_tracked = person_tracker.update(person_bboxes)

            # Associate persons with bibs
            associations = person_bib_assoc.associate(person_tracked, tracked)

            # Check each person for crossing
            for pid, (p_centroid, p_bbox) in person_tracked.items():
                cx_norm = p_centroid[0] / width
                cy_norm = p_centroid[1] / height

                if crossing_detector.check(pid, cx_norm, cy_norm, frame_idx):
                    bib_track_id = associations.get(pid)
                    bib_number = "UNKNOWN"
                    confidence = 0.0

                    if bib_track_id is not None and bib_track_id in final_consensus:
                        bib_number, confidence, _ = final_consensus[bib_track_id]

                    est_count = blob_estimator.estimate(p_bbox)

                    crossing_seq += 1
                    event = CrossingEvent(
                        sequence=crossing_seq,
                        frame_idx=frame_idx,
                        timestamp_sec=time_sec,
                        person_track_id=pid,
                        bib_number=bib_number,
                        confidence=confidence,
                        estimated_count=est_count,
                        person_bbox=p_bbox,
                    )
                    crossing_log.write(event)
                    total_crossings += 1
                    if bib_number == "UNKNOWN":
                        unknown_crossings += 1

            # Cleanup stale crossing detector state
            crossing_detector.cleanup(set(person_tracked.keys()))

            # Draw person blobs (cyan)
            for pid, (p_centroid, p_bbox) in person_tracked.items():
                px1, py1, px2, py2 = p_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)

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

                    crossing_seq += 1
                    event = CrossingEvent(
                        sequence=crossing_seq,
                        frame_idx=frame_idx,
                        timestamp_sec=time_sec,
                        person_track_id=track_id,
                        bib_number=bib_number,
                        confidence=confidence,
                        person_bbox=bbox,
                    )
                    crossing_log.write(event)
                    total_crossings += 1
                    if bib_number == "UNKNOWN":
                        unknown_crossings += 1

            crossing_detector.cleanup(set(tracked.keys()))

        # Draw timing line (magenta)
        if timing_line is not None:
            pt1, pt2 = timing_line.to_pixel_coords(width, height)
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

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
    print(f"  Total OCR calls saved: {edge_rejected + partial_rejected + quality_rejected}")
    print(f"  OCR cleanups applied: {cleanup_modified}")

    # Count by confidence level
    level_counts = {"high": 0, "medium": 0, "low": 0, "reject": 0}
    bib_counts: Dict[str, int] = {}
    for track_id, (number, conf, level) in final_consensus.items():
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

    if timing_line is not None:
        print(f"\nCrossing Detection:")
        print(f"  Total crossings: {total_crossings}")
        print(f"  With bib: {total_crossings - unknown_crossings}")
        print(f"  UNKNOWN (no bib): {unknown_crossings}")
        if person_tracker is not None:
            print(f"  Person tracks: {person_tracker.next_id}")

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
        default=0.5,
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
        "--no-person-detect",
        action="store_true",
        help="Disable person detection (use bib tracker for crossings instead)",
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
    print(f"OCR:      {args.ocr}")
    print(f"Placement: {args.placement}")
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
    print("Loading OCR model...")
    if args.ocr == "parseq":
        parseq_checkpoint = project_root / "runs/ocr_finetune/parseq_gpu_v1/best.pt"
        ocr_model = PARSeqOCR(str(parseq_checkpoint))
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
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
