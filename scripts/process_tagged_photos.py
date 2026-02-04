#!/usr/bin/env python3
"""
Process tagged race photos from scoring provider CSV exports.

Downloads images from S3, runs bib detection + OCR, and auto-validates
against the known bib numbers from the provider's tagging system.

Input CSV format (from scoring provider):
    camera_tag, image_url, event_image_id, associated_bibs

Auto-validation logic:
    - If OCR result matches a tagged bib for that image → auto-verified
    - If no match → flagged for manual review (with expected bibs as hints)

Usage:
    # Process tagged photos (download, detect, OCR, auto-validate)
    python scripts/process_tagged_photos.py \\
        --csv /path/to/export.csv \\
        --output data/tagged_event_88679

    # Resume interrupted download (skips already-downloaded images)
    python scripts/process_tagged_photos.py \\
        --csv /path/to/export.csv \\
        --output data/tagged_event_88679

    # Generate training data from results
    python scripts/process_unlabeled.py --generate-training data/tagged_event_88679
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


def parse_provider_csv(csv_path: str) -> list[dict]:
    """Parse the scoring provider's CSV export.

    Handles multiline quoted fields for multiple bibs per image.
    Returns list of {url, image_id, bibs: [str, ...]}.
    """
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bibs_raw = row["associated_bibs"].strip()
            bibs = [b.strip() for b in bibs_raw.split("\n") if b.strip()]
            records.append(
                {
                    "url": row["image_url"].strip(),
                    "image_id": row["event_image_id"].strip(),
                    "camera_tag": row.get("camera_tag", "").strip(),
                    "bibs": bibs,
                }
            )
    return records


def download_image(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download a single image from URL. Returns True on success."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # Already downloaded

    try:
        req = Request(url, headers={"User-Agent": "PointCam/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest_path.write_bytes(data)
        return True
    except (URLError, HTTPError, TimeoutError, OSError) as e:
        print(f"  Download failed: {dest_path.name}: {e}")
        return False


def download_all(
    records: list[dict], images_dir: Path, max_workers: int = 8
) -> dict[str, Path]:
    """Download all images in parallel. Returns {image_id: local_path}."""
    images_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for rec in records:
        # Use original filename from URL
        filename = rec["url"].split("/")[-1]
        dest = images_dir / filename
        tasks.append((rec["image_id"], rec["url"], dest))

    # Check how many already exist
    existing = sum(1 for _, _, dest in tasks if dest.exists() and dest.stat().st_size > 0)
    to_download = len(tasks) - existing
    print(f"Images: {len(tasks)} total, {existing} already downloaded, {to_download} to fetch")

    if to_download == 0:
        return {tid: dest for tid, _, dest in tasks}

    downloaded = {}
    failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for tid, url, dest in tasks:
            if dest.exists() and dest.stat().st_size > 0:
                downloaded[tid] = dest
                continue
            future = executor.submit(download_image, url, dest)
            futures[future] = (tid, dest)

        for future in as_completed(futures):
            tid, dest = futures[future]
            completed += 1
            if future.result():
                downloaded[tid] = dest
            else:
                failed += 1

            if completed % 50 == 0 or completed == to_download:
                print(f"  Downloaded {completed}/{to_download} ({failed} failed)")

    # Add pre-existing
    for tid, _, dest in tasks:
        if tid not in downloaded and dest.exists():
            downloaded[tid] = dest

    print(f"Download complete: {len(downloaded)} images available, {failed} failed")
    return downloaded


def preprocess_for_ocr(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Preprocess bib crop for better OCR."""
    h, w = image.shape[:2]
    if h < target_height:
        scale = target_height / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        digits = "".join(c for c in text if c.isdigit())
        if digits and 1 <= len(digits) <= 4:
            purity_bonus = 0.1 if text.strip() == digits else 0
            candidates.append((digits, confidence + purity_bonus, len(digits)))

    if not candidates:
        return "", 0.0

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0], candidates[0][1]


def process_tagged_photos(
    csv_path: str,
    output_dir: Path,
    model_path: Path,
    conf_threshold: float = 0.5,
    padding: int = 15,
    max_download_workers: int = 8,
    limit: int = 0,
):
    """Main processing pipeline for tagged photos."""

    # Parse CSV
    print(f"Parsing CSV: {csv_path}")
    records = parse_provider_csv(csv_path)
    print(f"  {len(records)} images, {len(set(b for r in records for b in r['bibs']))} unique bibs")

    if limit > 0:
        records = records[:limit]
        print(f"  Limited to first {limit} images")

    # Build lookup: image_id -> expected bibs
    expected_bibs = {r["image_id"]: r["bibs"] for r in records}

    # Download images
    print(f"\nDownloading images...")
    images_dir = output_dir / "source_images"
    downloaded = download_all(records, images_dir, max_workers=max_download_workers)

    # Setup output directories
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"\nLoading detection model: {model_path}")
    model = YOLO(str(model_path))

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Process each image
    results = []
    total_crops = 0
    auto_verified = 0
    auto_flagged = 0

    # Map image_id -> record for lookup
    id_to_record = {r["image_id"]: r for r in records}

    for idx, rec in enumerate(records):
        image_id = rec["image_id"]
        if image_id not in downloaded:
            continue

        image_path = downloaded[image_id]
        image_expected_bibs = rec["bibs"]

        print(
            f"[{idx+1}/{len(records)}] {image_path.name} "
            f"(expect bibs: {','.join(image_expected_bibs)})",
            end=" ",
        )

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("- FAILED to load")
            continue

        h, w = image.shape[:2]

        # Run detection
        detections = model(image, verbose=False)[0]

        num_detections = 0
        matched_bibs = set()

        for i, box in enumerate(detections.boxes):
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)

            crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
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
                preprocessed, allowlist="0123456789", paragraph=False
            )
            bib_number, ocr_conf = extract_bib_number(ocr_results)

            # Auto-validation: check if OCR result matches any expected bib
            status = "pending"
            verified_number = ""

            if bib_number and bib_number in image_expected_bibs:
                # OCR matches a tagged bib → auto-verify
                status = "verified"
                verified_number = bib_number
                auto_verified += 1
                matched_bibs.add(bib_number)
            else:
                auto_flagged += 1

            results.append(
                {
                    "source_image": str(image_path),
                    "source_filename": image_path.name,
                    "crop_filename": crop_filename,
                    "crop_path": str(crop_path),
                    "image_id": image_id,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "detection_conf": round(conf, 3),
                    "ocr_prediction": bib_number,
                    "ocr_conf": round(ocr_conf, 3),
                    "expected_bibs": ",".join(image_expected_bibs),
                    "verified_number": verified_number,
                    "status": status,
                }
            )

            num_detections += 1
            total_crops += 1

        status_icon = "✓" if matched_bibs else "?"
        print(f"- {num_detections} bibs, matched: {matched_bibs or 'none'}")

    # Write CSV for review (compatible with review_ui.py)
    csv_out_path = output_dir / "review.csv"
    fieldnames = [
        "source_filename",
        "crop_filename",
        "detection_conf",
        "ocr_prediction",
        "ocr_conf",
        "expected_bibs",
        "verified_number",
        "status",
    ]

    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    # Write full results JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"  Images processed: {len(records)}")
    print(f"  Bib crops extracted: {total_crops}")
    print(f"  Auto-verified (OCR matched tag): {auto_verified}")
    print(f"  Needs review (no match): {auto_flagged}")
    if total_crops > 0:
        print(f"  Auto-verify rate: {auto_verified/total_crops*100:.1f}%")
    print(f"\nOutputs:")
    print(f"  Source images: {images_dir}")
    print(f"  Crops: {crops_dir}")
    print(f"  Review CSV: {csv_out_path}")
    print(f"  Full results: {json_path}")
    print(f"\nNext steps:")
    print(f"  1. Review flagged items: python scripts/review_ui.py {output_dir}")
    print(f"     (auto-verified items can be skipped, focus on 'pending' ones)")
    print(f"  2. Generate training data: python scripts/process_unlabeled.py --generate-training {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process tagged race photos from scoring provider"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to scoring provider CSV export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tagged_processing",
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
        "--limit",
        type=int,
        default=0,
        help="Limit to first N images (for testing)",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel download threads",
    )

    args = parser.parse_args()

    process_tagged_photos(
        csv_path=args.csv,
        output_dir=Path(args.output),
        model_path=Path(args.model),
        conf_threshold=args.conf,
        max_download_workers=args.download_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
