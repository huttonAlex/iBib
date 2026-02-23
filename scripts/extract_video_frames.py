#!/usr/bin/env python3
"""
Extract representative frames from a video for detector labeling.

If a detections CSV is provided, sampling favors frames with detections and
optionally includes high-density frames (top-k by detection count).
If --auto-label is enabled, YOLO labels are populated from the detections CSV
so you can correct/verify instead of labeling from scratch.

Output structure matches existing unlabeled batches:
  <output>/annotated/         # extracted frames (jpg)
  <output>/yolo_annotations/  # empty label files (one per image)
  <output>/manifest.csv       # metadata for each extracted frame

Example:
  python scripts/extract_video_frames.py \
    --video data/raw_videos/REC-0004-A.mp4 \
    --detections REC-0004-A_detections.csv \
    --output data/unlabeled_rec0004a \
    --uniform-every 2.0 \
    --topk 200 \
    --max-frames 800
    --auto-label \
    --label-filter non_reject \
    --min-det-conf 0.25
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2


def _load_detections(path: Path) -> tuple[Counter, dict[int, float], dict[int, list[dict]]]:
    frame_counts: Counter = Counter()
    frame_time: dict[int, float] = {}
    by_frame: dict[int, list[dict]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "frame" not in row:
                continue
            try:
                frame = int(float(row["frame"]))
            except (TypeError, ValueError):
                continue
            frame_counts[frame] += 1
            if "time_sec" in row and row["time_sec"]:
                try:
                    frame_time.setdefault(frame, float(row["time_sec"]))
                except ValueError:
                    pass
            by_frame.setdefault(frame, []).append(row)
    return frame_counts, frame_time, by_frame


def _select_uniform_frames(
    frames_sorted: list[int], frame_time: dict[int, float], interval_sec: float, fps: float
) -> list[int]:
    selected: list[int] = []
    last_t = None
    for frame in frames_sorted:
        t = frame_time.get(frame, frame / fps)
        if last_t is None or (t - last_t) >= interval_sec:
            selected.append(frame)
            last_t = t
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames for detector labeling")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--detections", help="Optional detections CSV with frame/time_sec")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--uniform-every", type=float, default=2.0, help="Seconds per frame")
    parser.add_argument("--topk", type=int, default=200, help="Top-K dense frames to include")
    parser.add_argument("--max-frames", type=int, default=800, help="Cap total frames")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality (0-100)")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix")
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Populate YOLO labels from detections CSV (verify/correct instead of labeling from scratch)",
    )
    parser.add_argument(
        "--label-filter",
        choices=["all", "non_reject", "high_medium", "high_only"],
        default="non_reject",
        help="Which detections to keep when auto-labeling",
    )
    parser.add_argument(
        "--min-det-conf",
        type=float,
        default=0.0,
        help="Optional minimum detector confidence for auto-labeling",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = Path(args.output)
    images_dir = output_dir / "annotated"
    labels_dir = output_dir / "yolo_annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_counts: Counter = Counter()
    frame_time: dict[int, float] = {}
    detections_by_frame: dict[int, list[dict]] = {}

    if args.detections:
        det_path = Path(args.detections)
        if not det_path.exists():
            raise SystemExit(f"Detections CSV not found: {det_path}")
        frame_counts, frame_time, detections_by_frame = _load_detections(det_path)
        if args.auto_label and not detections_by_frame:
            raise SystemExit("Auto-label requested, but detections CSV has no usable rows.")
    elif args.auto_label:
        raise SystemExit("--auto-label requires --detections")

    selected_uniform: set[int] = set()
    selected_topk: set[int] = set()

    if frame_counts:
        frames_sorted = sorted(frame_counts.keys())
        uniform = _select_uniform_frames(frames_sorted, frame_time, args.uniform_every, fps)
        selected_uniform = set(uniform)

        if args.topk > 0:
            topk = sorted(frame_counts.items(), key=lambda x: (-x[1], x[0]))[: args.topk]
            selected_topk = {f for f, _ in topk}
    else:
        # No detections: fall back to uniform sampling across entire video
        stride = max(1, int(round(args.uniform_every * fps)))
        selected_uniform = set(range(0, total_frames, stride))

    selected = sorted(selected_uniform | selected_topk)

    if args.max_frames and len(selected) > args.max_frames:
        step = len(selected) / args.max_frames
        selected = [selected[int(i * step)] for i in range(args.max_frames)]

    prefix = args.prefix or video_path.stem.lower().replace(" ", "_")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "filename",
                "frame",
                "time_sec",
                "detections",
                "selection",
                "labels",
            ]
        )

        for frame in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                continue
            t = frame_time.get(frame, frame / fps)
            det_count = frame_counts.get(frame, 0)
            if frame in selected_uniform and frame in selected_topk:
                sel = "both"
            elif frame in selected_topk:
                sel = "topk"
            else:
                sel = "uniform"

            filename = f"{prefix}_f{frame:06d}_t{t:07.2f}.jpg"
            out_path = images_dir / filename
            cv2.imwrite(
                str(out_path),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)],
            )

            # Create YOLO label file (auto-labeled or empty placeholder)
            label_path = labels_dir / f"{Path(filename).stem}.txt"
            labels_written = 0
            if args.auto_label:
                h, w = img.shape[:2]
                rows = detections_by_frame.get(frame, [])
                lines = []
                for row in rows:
                    level = (row.get("final_level") or "").strip().lower()
                    if args.label_filter == "non_reject" and level == "reject":
                        continue
                    if args.label_filter == "high_medium" and level not in {"high", "medium"}:
                        continue
                    if args.label_filter == "high_only" and level != "high":
                        continue
                    try:
                        det_conf = float(row.get("det_conf") or 0.0)
                    except ValueError:
                        det_conf = 0.0
                    if det_conf < args.min_det_conf:
                        continue

                    try:
                        x1 = float(row["x1"])
                        y1 = float(row["y1"])
                        x2 = float(row["x2"])
                        y2 = float(row["y2"])
                    except (KeyError, ValueError, TypeError):
                        continue

                    # Clamp to image bounds
                    x1 = max(0.0, min(x1, w - 1))
                    x2 = max(0.0, min(x2, w - 1))
                    y1 = max(0.0, min(y1, h - 1))
                    y2 = max(0.0, min(y2, h - 1))

                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw <= 1.0 or bh <= 1.0:
                        continue

                    xc = x1 + bw / 2.0
                    yc = y1 + bh / 2.0

                    # YOLO normalized
                    lines.append(
                        f"0 {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                    )
                if lines:
                    label_path.write_text("\n".join(lines) + "\n")
                    labels_written = len(lines)
                else:
                    label_path.write_text("")
            else:
                label_path.write_text("")

            writer.writerow([filename, frame, f"{t:.3f}", det_count, sel, labels_written])

    cap.release()
    print(f"Extracted {len(selected)} frames -> {images_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
