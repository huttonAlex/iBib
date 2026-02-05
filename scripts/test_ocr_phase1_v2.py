#!/usr/bin/env python3
"""
Phase 1 OCR Test v2: Enhanced EasyOCR with Super-Resolution + Multi-pass voting

Improvements over baseline:
1. Aggressive upscaling for small crops (min 100px height)
2. Multiple preprocessing variants
3. Digit-only allowlist
4. Voting across preprocessing methods
5. Better confidence calibration
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import re

from ultralytics import YOLO
import easyocr


def upscale_image(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Upscale image to target height using INTER_LANCZOS4."""
    h, w = image.shape[:2]
    if h >= target_height:
        return image

    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)


def preprocess_for_ocr(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate preprocessing variants optimized for digit recognition."""
    variants = []

    # Ensure color image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 1. Original (upscaled)
    variants.append(("original", image))

    # 2. Grayscale + CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", enhanced))

    # 3. Otsu threshold (binary)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", binary))

    # 4. Inverted binary (for dark text on light bg)
    inverted = cv2.bitwise_not(binary)
    variants.append(("inverted", inverted))

    # 5. Sharpened + CLAHE
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    sharp_clahe = clahe.apply(sharpened)
    variants.append(("sharp_clahe", sharp_clahe))

    # 6. Morphological cleaning
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)
    variants.append(("morph_clean", cleaned))

    return variants


def extract_bib_number(results: list, min_conf: float = 0.2) -> tuple[str, float]:
    """Extract bib number from EasyOCR results."""
    if not results:
        return "", 0.0

    candidates = []
    for bbox, text, conf in results:
        if conf < min_conf:
            continue

        # Extract digits
        digits = ''.join(c for c in str(text) if c.isdigit())

        # Valid bib: 1-4 digits
        if digits and 1 <= len(digits) <= 4:
            # Bonus for pure digit match
            purity = 1.0 if text.strip() == digits else 0.9
            candidates.append((digits, conf * purity))

    if not candidates:
        return "", 0.0

    # Return highest confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def vote_results(results: list[tuple[str, float]], min_votes: int = 1) -> tuple[str, float]:
    """Vote among OCR results from different preprocessing methods."""
    if not results:
        return "", 0.0

    # Group by number
    votes = {}
    for number, conf in results:
        if not number:
            continue
        if number not in votes:
            votes[number] = []
        votes[number].append(conf)

    if not votes:
        return "", 0.0

    # Score: vote_count * avg_confidence
    scored = []
    for number, confs in votes.items():
        if len(confs) >= min_votes:
            avg_conf = sum(confs) / len(confs)
            # Boost score by number of votes
            score = avg_conf * (1 + 0.1 * len(confs))
            scored.append((number, score, avg_conf, len(confs)))

    if not scored:
        return "", 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    return best[0], best[2]  # Return number and avg confidence


def process_image(
    image_path: Path,
    detector: YOLO,
    reader: easyocr.Reader,
    output_dir: Path,
    conf_threshold: float = 0.5,
    padding: int = 20,
    min_crop_height: int = 100,
) -> list[dict]:
    """Process single image."""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return []

    h, w = image.shape[:2]
    results = []

    # Detect bibs
    detections = detector(image, verbose=False)[0]

    for box in detections.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        # Bounding box with padding
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        # Crop and upscale
        bib_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        bib_crop = upscale_image(bib_crop, target_height=min_crop_height)

        # Multi-pass OCR
        ocr_results = []
        for variant_name, variant_img in preprocess_for_ocr(bib_crop):
            try:
                # Run OCR with digit allowlist
                raw_results = reader.readtext(
                    variant_img,
                    allowlist='0123456789',
                    paragraph=False,
                    min_size=10,
                    text_threshold=0.5,
                )
                number, ocr_conf = extract_bib_number(raw_results)
                if number:
                    ocr_results.append((number, ocr_conf))
            except Exception as e:
                continue

        # Vote for final result
        bib_number, final_conf = vote_results(ocr_results, min_votes=1)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": conf,
            "bib_number": bib_number,
            "ocr_conf": final_conf,
            "num_votes": len([r for r in ocr_results if r[0] == bib_number]),
        })

        # Draw
        color = (0, 255, 0) if bib_number else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if bib_number:
            label = f"#{bib_number} ({final_conf:.2f})"
        else:
            label = "?"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save
    cv2.imwrite(str(output_dir / image_path.name), image)
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 OCR v2")
    parser.add_argument("--model", default="runs/detect/bib_detector/weights/best.pt")
    parser.add_argument("--source", default="data/images/val")
    parser.add_argument("--output", default="runs/ocr_phase1_v2")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="0 for all images")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)

    print(f"Loading detector: {args.model}")
    detector = YOLO(args.model)

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Get images
    if source_path.is_file():
        images = [source_path]
    else:
        images = sorted(source_path.glob("*.jpg")) + sorted(source_path.glob("*.png"))

    if args.limit > 0:
        images = images[:args.limit]

    print(f"\nProcessing {len(images)} images with enhanced preprocessing...\n")

    total_detections = 0
    successful_ocr = 0

    for image_path in images:
        results = process_image(
            image_path, detector, reader, output_dir,
            conf_threshold=args.conf
        )

        total_detections += len(results)

        print(f"{image_path.name}:")
        for r in results:
            if r['bib_number']:
                print(f"  Bib #{r['bib_number']} (det: {r['det_conf']:.2f}, ocr: {r['ocr_conf']:.2f}, votes: {r['num_votes']})")
                successful_ocr += 1
            else:
                print(f"  [No number] (det: {r['det_conf']:.2f})")
        if not results:
            print("  [No bibs detected]")
        print()

    # Summary
    print("=" * 60)
    print(f"Total bibs detected: {total_detections}")
    print(f"Successful OCR: {successful_ocr} ({100*successful_ocr/max(1,total_detections):.1f}%)")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
