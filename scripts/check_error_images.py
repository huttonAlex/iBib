#!/usr/bin/env python3
"""Check if the known OCR error images would be filtered by completeness check.

This helps validate if the completeness filter catches the problematic cases.

Usage:
    python scripts/check_error_images.py
"""

import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pointcam.recognition import CropQualityFilter, SuspiciousPredictionFilter


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Load error cases
    errors_path = project_root / "runs/ocr_eval_finetuned/parseq_errors.csv"

    if not errors_path.exists():
        print(f"Error file not found: {errors_path}")
        return

    # Initialize pre-OCR quality filter
    quality_filter = CropQualityFilter(
        min_blur_score=50.0,
        min_width=40,
        min_height=15,
        min_aspect_ratio=0.7,
        min_completeness=0.45,
        min_content_extent=0.5,
        check_completeness=True,
    )

    # Initialize post-OCR suspicious prediction filter
    # Configure based on expected bib format (adjust for your race)
    suspicious_filter = SuspiciousPredictionFilter(
        min_confidence_short=0.97,  # High bar for 1-2 digit predictions
        min_confidence_medium=0.7,
        reject_confidence=0.35,
        expected_digit_counts={3, 4},  # Most races use 3-4 digit bibs
        strict_mode=True,
    )

    print("=" * 70)
    print("Checking if OCR error images would be filtered")
    print("Using TWO-STAGE filtering:")
    print("  Stage 1: Pre-OCR quality filter (blur, size, content extent)")
    print("  Stage 2: Post-OCR suspicious prediction filter (confidence, digit count)")
    print("=" * 70)
    print()

    # Identify truncation errors (where prediction is shorter than ground truth)
    truncation_errors = []
    other_errors = []

    with open(errors_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row['ground_truth']
            pred = row['predicted']

            # Check if this looks like truncation
            if len(pred) < len(gt):
                truncation_errors.append(row)
            else:
                other_errors.append(row)

    print(f"Total errors: {len(truncation_errors) + len(other_errors)}")
    print(f"Truncation errors (pred shorter than GT): {len(truncation_errors)}")
    print(f"Other errors: {len(other_errors)}")
    print()

    # Check truncation errors
    print("TRUNCATION ERRORS (likely partial bibs):")
    print("-" * 70)

    stage1_filtered = 0
    stage2_filtered = 0
    passed_count = 0

    for row in truncation_errors:
        img_path = row['img_path']
        gt = row['ground_truth']
        pred = row['predicted']
        conf = float(row['confidence'])

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Cannot read: {img_path}")
            continue

        # Stage 1: Pre-OCR quality filter
        quality = quality_filter.assess(img)

        # Stage 2: Post-OCR suspicious prediction filter
        suspicious = suspicious_filter.check(pred, conf)

        filename = Path(img_path).name
        print(f"  {filename}: GT={gt} → Pred={pred} (conf={conf:.2f})")

        if not quality.is_acceptable:
            stage1_filtered += 1
            print(f"    STAGE 1 FILTERED: {quality.rejection_reason}")
        elif suspicious.is_suspicious:
            stage2_filtered += 1
            reject_str = " (REJECT)" if suspicious.should_reject else " (FLAG)"
            print(f"    STAGE 2 CAUGHT{reject_str}: {suspicious.reason}")
        else:
            passed_count += 1
            print(f"    PASSED BOTH STAGES")
        print(f"    Completeness: {quality.completeness_score:.2f}")
        print()

    total_caught = stage1_filtered + stage2_filtered
    print(f"Truncation errors caught: {total_caught}/{len(truncation_errors)}")
    print(f"  Stage 1 (quality): {stage1_filtered}")
    print(f"  Stage 2 (suspicious): {stage2_filtered}")
    print()

    # Check other errors
    print("OTHER ERRORS (digit confusion, etc):")
    print("-" * 70)

    other_stage1 = 0
    other_stage2 = 0
    for row in other_errors:
        img_path = row['img_path']
        gt = row['ground_truth']
        pred = row['predicted']
        conf = float(row['confidence'])

        img = cv2.imread(img_path)
        if img is None:
            continue

        quality = quality_filter.assess(img)
        suspicious = suspicious_filter.check(pred, conf)

        if not quality.is_acceptable:
            other_stage1 += 1
            filename = Path(img_path).name
            print(f"  {filename}: GT={gt} → Pred={pred} (conf={conf:.2f})")
            print(f"    STAGE 1: {quality.rejection_reason}")
            print()
        elif suspicious.is_suspicious and suspicious.should_reject:
            other_stage2 += 1
            filename = Path(img_path).name
            print(f"  {filename}: GT={gt} → Pred={pred} (conf={conf:.2f})")
            print(f"    STAGE 2: {suspicious.reason}")
            print()

    other_total = other_stage1 + other_stage2
    if other_total == 0:
        print("  (None caught - these are clear images with digit confusion)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Truncation errors caught: {total_caught}/{len(truncation_errors)}")
    print(f"  Pre-OCR quality filter: {stage1_filtered}")
    print(f"  Post-OCR suspicious filter: {stage2_filtered}")
    print()
    print(f"Other errors caught: {other_total}/{len(other_errors)}")
    print(f"  Pre-OCR quality filter: {other_stage1}")
    print(f"  Post-OCR suspicious filter: {other_stage2}")
    print()

    if total_caught == len(truncation_errors):
        print("SUCCESS: All truncation errors would be caught by the two-stage filter!")
    elif total_caught > 0:
        print(f"PARTIAL: {total_caught}/{len(truncation_errors)} truncation errors caught.")
        print("Remaining errors may need manual review or adjusted thresholds.")
    else:
        print("WARNING: No truncation errors caught. Filters may need tuning.")


if __name__ == "__main__":
    main()
