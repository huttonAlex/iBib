#!/usr/bin/env python3
"""Test the completeness filter on existing OCR test images or video frames.

This script helps validate that the completeness filter is correctly
identifying partial/obstructed bibs.

Usage:
    # Test on existing OCR dataset (see which would be rejected)
    python scripts/test_completeness_filter.py --dataset data/ocr_dataset/test

    # Test on video with debug output (saves rejected crops)
    python scripts/test_completeness_filter.py --video path/to/video.mp4 --save-rejected

    # Compare with/without completeness filter
    python scripts/test_completeness_filter.py --video path/to/video.mp4 --compare
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pointcam.recognition import (
    CropQualityFilter,
    BibCompletenessChecker,
    CropQuality,
)


def test_on_dataset(dataset_dir: Path, output_dir: Optional[Path] = None):
    """
    Test completeness filter on existing labeled OCR images.

    Shows which images would be rejected and why.
    """
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.tsv"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return

    # Load labels if available
    labels = {}
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                labels[row['filename']] = row['label']

    # Initialize filter with improved partial bib detection
    quality_filter = CropQualityFilter(
        min_blur_score=50.0,
        min_width=40,
        min_height=15,
        min_aspect_ratio=0.7,
        min_completeness=0.45,
        min_content_extent=0.5,
        check_completeness=True,
    )

    # Create output directory for rejected images
    if output_dir:
        rejected_dir = output_dir / "rejected"
        accepted_dir = output_dir / "accepted"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        accepted_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    results = {
        'total': 0,
        'accepted': 0,
        'rejected_blur': 0,
        'rejected_size': 0,
        'rejected_partial': 0,
        'rejected_other': 0,
    }

    rejected_samples = []

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    print(f"Testing {len(image_files)} images...")
    print()

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results['total'] += 1

        # Assess quality
        quality = quality_filter.assess(img)
        label = labels.get(img_path.name, "?")

        if quality.is_acceptable:
            results['accepted'] += 1
            if output_dir:
                cv2.imwrite(str(accepted_dir / img_path.name), img)
        else:
            reason = quality.rejection_reason or "unknown"

            if "blur" in reason.lower():
                results['rejected_blur'] += 1
            elif "narrow" in reason.lower() or "short" in reason.lower():
                results['rejected_size'] += 1
            elif "partial" in reason.lower() or "obstruct" in reason.lower():
                results['rejected_partial'] += 1
            else:
                results['rejected_other'] += 1

            rejected_samples.append({
                'file': img_path.name,
                'label': label,
                'reason': reason,
                'completeness': quality.completeness_score,
            })

            if output_dir:
                # Save with reason in filename
                reason_short = reason.split(':')[0].replace(' ', '_')[:20]
                out_name = f"{img_path.stem}_{reason_short}{img_path.suffix}"
                cv2.imwrite(str(rejected_dir / out_name), img)

    # Print summary
    print("=" * 60)
    print("COMPLETENESS FILTER TEST RESULTS")
    print("=" * 60)
    print(f"Total images: {results['total']}")
    print(f"Accepted: {results['accepted']} ({100*results['accepted']/max(results['total'],1):.1f}%)")
    print(f"Rejected: {results['total'] - results['accepted']}")
    print(f"  - Blur: {results['rejected_blur']}")
    print(f"  - Size: {results['rejected_size']}")
    print(f"  - Partial/Obstructed: {results['rejected_partial']}")
    print(f"  - Other: {results['rejected_other']}")

    if rejected_samples:
        print(f"\nSample rejected images:")
        for sample in rejected_samples[:10]:
            print(f"  {sample['file']} (label={sample['label']})")
            print(f"    Reason: {sample['reason']}")
            print(f"    Completeness: {sample['completeness']:.2f}")

    if output_dir:
        print(f"\nRejected images saved to: {rejected_dir}")
        print(f"Accepted images saved to: {accepted_dir}")

    return results


def test_on_video(
    video_path: str,
    detector_path: str,
    output_dir: Path,
    save_rejected: bool = False,
    compare_mode: bool = False,
    max_frames: int = 500,
):
    """
    Test completeness filter on video frames.

    Args:
        video_path: Path to video file
        detector_path: Path to YOLO detector
        output_dir: Output directory
        save_rejected: Save rejected crop images
        compare_mode: Run with and without filter, compare results
        max_frames: Maximum frames to process
    """
    from ultralytics import YOLO

    # Load detector
    detector = YOLO(detector_path)

    # Initialize filters with improved partial bib detection
    quality_filter_on = CropQualityFilter(
        min_blur_score=50.0,
        min_width=40,
        min_height=15,
        min_aspect_ratio=0.7,
        min_completeness=0.45,
        min_content_extent=0.5,
        check_completeness=True,
    )

    quality_filter_off = CropQualityFilter(
        min_blur_score=50.0,
        min_width=40,
        min_height=15,
        min_aspect_ratio=0.7,
        check_completeness=False,
    )

    completeness_checker = BibCompletenessChecker(edge_margin_ratio=0.02)

    # Output directories
    if save_rejected:
        rejected_dir = output_dir / "rejected_crops"
        rejected_dir.mkdir(parents=True, exist_ok=True)

        edge_dir = rejected_dir / "edge"
        partial_dir = rejected_dir / "partial"
        quality_dir = rejected_dir / "quality"
        edge_dir.mkdir(exist_ok=True)
        partial_dir.mkdir(exist_ok=True)
        quality_dir.mkdir(exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {total_frames} frames")
    print(f"Processing up to {max_frames} frames...")

    # Stats
    stats_on = {'detections': 0, 'edge_rejected': 0, 'partial_rejected': 0, 'quality_rejected': 0, 'passed': 0}
    stats_off = {'detections': 0, 'rejected': 0, 'passed': 0}

    frame_idx = 0
    crop_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Detect bibs
        results = detector(frame, conf=0.5, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                crop_idx += 1

                stats_on['detections'] += 1
                stats_off['detections'] += 1

                # Get crop
                pad = 5
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(width, x2 + pad)
                y2_pad = min(height, y2 + pad)
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if crop.size == 0:
                    continue

                # Test WITH completeness checking
                is_visible, edge_reason = completeness_checker.is_fully_visible(
                    (x1, y1, x2, y2), width, height
                )

                if not is_visible:
                    stats_on['edge_rejected'] += 1
                    if save_rejected:
                        cv2.imwrite(str(edge_dir / f"crop_{crop_idx:05d}_edge.jpg"), crop)
                else:
                    quality_on = quality_filter_on.assess(crop)
                    if not quality_on.is_acceptable:
                        if "partial" in (quality_on.rejection_reason or "").lower():
                            stats_on['partial_rejected'] += 1
                            if save_rejected:
                                cv2.imwrite(str(partial_dir / f"crop_{crop_idx:05d}_partial.jpg"), crop)
                        else:
                            stats_on['quality_rejected'] += 1
                            if save_rejected:
                                cv2.imwrite(str(quality_dir / f"crop_{crop_idx:05d}_quality.jpg"), crop)
                    else:
                        stats_on['passed'] += 1

                # Test WITHOUT completeness checking (for comparison)
                if compare_mode:
                    quality_off = quality_filter_off.assess(crop)
                    if quality_off.is_acceptable:
                        stats_off['passed'] += 1
                    else:
                        stats_off['rejected'] += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{min(max_frames, total_frames)}")

    cap.release()

    # Print results
    print()
    print("=" * 60)
    print("COMPLETENESS FILTER VIDEO TEST")
    print("=" * 60)

    print(f"\nWITH completeness checking:")
    print(f"  Total detections: {stats_on['detections']}")
    print(f"  Edge-rejected: {stats_on['edge_rejected']} ({100*stats_on['edge_rejected']/max(stats_on['detections'],1):.1f}%)")
    print(f"  Partial-rejected: {stats_on['partial_rejected']} ({100*stats_on['partial_rejected']/max(stats_on['detections'],1):.1f}%)")
    print(f"  Quality-rejected: {stats_on['quality_rejected']} ({100*stats_on['quality_rejected']/max(stats_on['detections'],1):.1f}%)")
    print(f"  Passed to OCR: {stats_on['passed']} ({100*stats_on['passed']/max(stats_on['detections'],1):.1f}%)")

    if compare_mode:
        print(f"\nWITHOUT completeness checking:")
        print(f"  Total detections: {stats_off['detections']}")
        print(f"  Quality-rejected: {stats_off['rejected']} ({100*stats_off['rejected']/max(stats_off['detections'],1):.1f}%)")
        print(f"  Passed to OCR: {stats_off['passed']} ({100*stats_off['passed']/max(stats_off['detections'],1):.1f}%)")

        diff = stats_off['passed'] - stats_on['passed']
        print(f"\nDifference: {diff} fewer crops sent to OCR with completeness checking")
        print(f"  These are likely partial/obstructed bibs that would cause errors")

    if save_rejected:
        print(f"\nRejected crops saved to: {rejected_dir}")
        print(f"  - edge/: {stats_on['edge_rejected']} crops at frame edges")
        print(f"  - partial/: {stats_on['partial_rejected']} partial/obstructed crops")
        print(f"  - quality/: {stats_on['quality_rejected']} blur/size issues")


def main():
    parser = argparse.ArgumentParser(description="Test completeness filter")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to OCR dataset directory (with images/ and labels.tsv)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file"
    )
    parser.add_argument(
        "--detector", type=str, default="runs/detect/bib_detector/weights/best.pt",
        help="Path to YOLO detector (for video mode)"
    )
    parser.add_argument(
        "--output", type=str, default="runs/completeness_test",
        help="Output directory"
    )
    parser.add_argument(
        "--save-rejected", action="store_true",
        help="Save rejected crop images for inspection"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare with/without completeness checking"
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Maximum frames to process (video mode)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_dir = Path(args.dataset)
        if not dataset_dir.is_absolute():
            dataset_dir = project_root / args.dataset

        print("Testing on OCR dataset...")
        test_on_dataset(dataset_dir, output_dir if args.save_rejected else None)

    elif args.video:
        video_path = args.video
        if not Path(video_path).is_absolute():
            video_path = str(project_root / args.video)

        detector_path = args.detector
        if not Path(detector_path).is_absolute():
            detector_path = str(project_root / args.detector)

        print("Testing on video...")
        test_on_video(
            video_path=video_path,
            detector_path=detector_path,
            output_dir=output_dir,
            save_rejected=args.save_rejected,
            compare_mode=args.compare,
            max_frames=args.max_frames,
        )

    else:
        print("Please specify --dataset or --video")
        parser.print_help()


if __name__ == "__main__":
    main()
