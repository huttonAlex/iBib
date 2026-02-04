#!/usr/bin/env python3
"""
Process unlabeled race photos to extract bib crops and OCR predictions.

This script:
1. Runs bib detection on all images
2. Extracts and saves bib crops
3. Runs OCR on each crop
4. Generates a CSV for manual review/correction
5. After review, can generate training data

Usage:
    python scripts/process_unlabeled.py --source /path/to/photos --output data/unlabeled_processing

After manual review of the CSV:
    python scripts/process_unlabeled.py --generate-training data/unlabeled_processing
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


def preprocess_for_ocr(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Preprocess bib crop for better OCR."""
    h, w = image.shape[:2]

    # Upscale small crops
    if h < target_height:
        scale = target_height / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def extract_bib_number(ocr_results: list, min_conf: float = 0.3) -> tuple[str, float]:
    """Extract the most likely bib number from OCR results."""
    if not ocr_results:
        return "", 0.0

    candidates = []
    for bbox, text, confidence in ocr_results:
        if confidence < min_conf:
            continue

        # Extract only digits
        digits = ''.join(c for c in text if c.isdigit())

        # Valid bib numbers are 1-4 digits
        if digits and 1 <= len(digits) <= 4:
            purity_bonus = 0.1 if text.strip() == digits else 0
            candidates.append((digits, confidence + purity_bonus, len(digits)))

    if not candidates:
        return "", 0.0

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0], candidates[0][1]


def process_images(
    source_dirs: list[Path],
    output_dir: Path,
    model_path: Path,
    conf_threshold: float = 0.5,
    padding: int = 15,
):
    """Process all images and extract bib crops with OCR predictions."""

    # Setup output directories
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading detection model: {model_path}")
    model = YOLO(str(model_path))

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Collect all images
    images = []
    for source_dir in source_dirs:
        if source_dir.is_file():
            images.append(source_dir)
        else:
            # Search recursively
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                images.extend(source_dir.rglob(ext))

    images = sorted(set(images))
    print(f"\nFound {len(images)} images to process\n")

    # Results for CSV
    results = []
    total_crops = 0

    for idx, image_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {image_path.name}", end=" ")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("- FAILED to load")
            continue

        h, w = image.shape[:2]

        # Run detection
        detections = model(image, verbose=False)[0]

        num_detections = 0
        for i, box in enumerate(detections.boxes):
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Add padding
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)

            # Extract crop
            crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

            # Skip tiny crops
            crop_h, crop_w = crop.shape[:2]
            if crop_h < 20 or crop_w < 20:
                continue

            # Save crop
            crop_filename = f"{image_path.stem}_crop{i:02d}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            # Run OCR
            preprocessed = preprocess_for_ocr(crop)
            ocr_results = reader.readtext(
                preprocessed,
                allowlist='0123456789',
                paragraph=False,
            )
            bib_number, ocr_conf = extract_bib_number(ocr_results)

            # Store result
            results.append({
                'source_image': str(image_path),
                'source_filename': image_path.name,
                'crop_filename': crop_filename,
                'crop_path': str(crop_path),
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2,
                'detection_conf': round(conf, 3),
                'ocr_prediction': bib_number,
                'ocr_conf': round(ocr_conf, 3),
                'verified_number': '',  # For manual review
                'status': 'pending',  # pending, verified, rejected
            })

            # Draw on annotated image
            color = (0, 255, 0) if bib_number else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"#{bib_number}" if bib_number else "?"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            num_detections += 1
            total_crops += 1

        # Save annotated image
        annotated_path = annotated_dir / image_path.name
        cv2.imwrite(str(annotated_path), image)

        print(f"- {num_detections} bibs")

    # Write CSV for review
    csv_path = output_dir / "review.csv"
    fieldnames = [
        'source_filename', 'crop_filename', 'detection_conf',
        'ocr_prediction', 'ocr_conf', 'verified_number', 'status'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    # Write full results JSON (for later processing)
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Images processed: {len(images)}")
    print(f"  Bib crops extracted: {total_crops}")
    print(f"  Crops with OCR prediction: {sum(1 for r in results if r['ocr_prediction'])}")
    print(f"\nOutputs:")
    print(f"  Crops: {crops_dir}")
    print(f"  Annotated images: {annotated_dir}")
    print(f"  Review CSV: {csv_path}")
    print(f"  Full results: {json_path}")
    print("\nNext steps:")
    print("  1. Open review.csv in a spreadsheet")
    print("  2. Check 'ocr_prediction' column against crop images")
    print("  3. Fill in 'verified_number' with correct number")
    print("  4. Set 'status' to 'verified' or 'rejected'")
    print("  5. Run: python scripts/process_unlabeled.py --generate-training <output_dir>")


def generate_training_data(output_dir: Path):
    """Generate training data from reviewed CSV."""

    json_path = output_dir / "results.json"
    csv_path = output_dir / "review.csv"

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    # Load full results
    with open(json_path) as f:
        results = json.load(f)

    # Load reviewed CSV
    reviewed = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewed[row['crop_filename']] = row

    # Update results with review data
    verified_count = 0
    rejected_count = 0

    for r in results:
        crop_fn = r['crop_filename']
        if crop_fn in reviewed:
            rev = reviewed[crop_fn]
            r['verified_number'] = rev.get('verified_number', '').strip()
            r['status'] = rev.get('status', 'pending').strip().lower()

            if r['status'] == 'verified' and r['verified_number']:
                verified_count += 1
            elif r['status'] == 'rejected':
                rejected_count += 1

    print(f"Review status:")
    print(f"  Verified: {verified_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Pending: {len(results) - verified_count - rejected_count}")

    if verified_count == 0:
        print("\nNo verified entries found. Please review the CSV first.")
        return

    # Generate OCR training data (crop -> number pairs)
    ocr_training_dir = output_dir / "ocr_training"
    ocr_training_dir.mkdir(exist_ok=True)

    ocr_labels = []
    for r in results:
        if r['status'] == 'verified' and r['verified_number']:
            # Copy crop to training directory
            src_crop = Path(r['crop_path'])
            if src_crop.exists():
                dst_crop = ocr_training_dir / src_crop.name
                if not dst_crop.exists():
                    import shutil
                    shutil.copy(src_crop, dst_crop)

                ocr_labels.append({
                    'filename': src_crop.name,
                    'number': r['verified_number'],
                })

    # Write OCR labels
    ocr_labels_path = ocr_training_dir / "labels.json"
    with open(ocr_labels_path, 'w') as f:
        json.dump(ocr_labels, f, indent=2)

    # Also write as simple TSV for easy use
    ocr_tsv_path = ocr_training_dir / "labels.tsv"
    with open(ocr_tsv_path, 'w') as f:
        f.write("filename\tnumber\n")
        for item in ocr_labels:
            f.write(f"{item['filename']}\t{item['number']}\n")

    print(f"\nOCR training data generated:")
    print(f"  Directory: {ocr_training_dir}")
    print(f"  Samples: {len(ocr_labels)}")
    print(f"  Labels: {ocr_labels_path}")

    # Generate YOLO detection annotations for verified bibs
    yolo_dir = output_dir / "yolo_annotations"
    yolo_dir.mkdir(exist_ok=True)

    # Group by source image
    by_image = {}
    for r in results:
        if r['status'] != 'rejected':
            src = r['source_filename']
            if src not in by_image:
                by_image[src] = {
                    'path': r['source_image'],
                    'boxes': []
                }
            by_image[src]['boxes'].append({
                'x1': r['bbox_x1'],
                'y1': r['bbox_y1'],
                'x2': r['bbox_x2'],
                'y2': r['bbox_y2'],
            })

    # Write YOLO format annotations
    yolo_count = 0
    for img_name, data in by_image.items():
        img = cv2.imread(data['path'])
        if img is None:
            continue

        h, w = img.shape[:2]

        # YOLO format: class x_center y_center width height (normalized)
        lines = []
        for box in data['boxes']:
            x_center = ((box['x1'] + box['x2']) / 2) / w
            y_center = ((box['y1'] + box['y2']) / 2) / h
            box_w = (box['x2'] - box['x1']) / w
            box_h = (box['y2'] - box['y1']) / h
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # Write annotation file
        ann_path = yolo_dir / (Path(img_name).stem + ".txt")
        with open(ann_path, 'w') as f:
            f.write('\n'.join(lines))
        yolo_count += 1

    print(f"\nYOLO annotations generated:")
    print(f"  Directory: {yolo_dir}")
    print(f"  Images with annotations: {yolo_count}")

    print("\nTo use YOLO annotations:")
    print("  1. Copy source images to a training directory")
    print("  2. Copy .txt files alongside them")
    print("  3. Update your dataset YAML")


def main():
    parser = argparse.ArgumentParser(description="Process unlabeled race photos")
    parser.add_argument(
        "--source",
        type=str,
        nargs='+',
        help="Source directories or files with images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/unlabeled_processing",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/bib_detector/weights/best.pt",
        help="Path to trained YOLOv8 model",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--generate-training",
        type=str,
        metavar="DIR",
        help="Generate training data from reviewed CSV in DIR",
    )

    args = parser.parse_args()

    if args.generate_training:
        generate_training_data(Path(args.generate_training))
    elif args.source:
        source_dirs = [Path(s) for s in args.source]
        process_images(
            source_dirs=source_dirs,
            output_dir=Path(args.output),
            model_path=Path(args.model),
            conf_threshold=args.conf,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
