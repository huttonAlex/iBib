#!/usr/bin/env python3
"""
Test script for bib number OCR using EasyOCR with preprocessing optimizations.

This script:
1. Loads the trained YOLOv8 bib detector
2. Detects bibs in test images
3. Preprocesses bib crops for better OCR
4. Runs OCR to extract numbers
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from ultralytics import YOLO
import easyocr


def preprocess_bib_crop(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """
    Preprocess bib crop for better OCR accuracy.

    Steps:
    1. Upscale small images
    2. Convert to grayscale
    3. Apply CLAHE (contrast enhancement)
    4. Denoise

    Args:
        image: BGR image crop of bib region
        target_height: Minimum height to upscale to

    Returns:
        Preprocessed grayscale image
    """
    h, w = image.shape[:2]

    # Upscale small crops
    if h < target_height:
        scale = target_height / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return denoised


def preprocess_bib_crop_v2(image: np.ndarray, target_height: int = 100) -> np.ndarray:
    """
    Alternative preprocessing - sharpen and threshold.
    """
    h, w = image.shape[:2]

    # Upscale small crops
    if h < target_height:
        scale = target_height / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary


def extract_bib_number(ocr_results: list, min_conf: float = 0.3) -> tuple[str, float]:
    """
    Extract the most likely bib number from OCR results.

    Args:
        ocr_results: List of (bbox, text, confidence) tuples from EasyOCR
        min_conf: Minimum confidence threshold

    Returns:
        tuple of (bib_number, confidence)
    """
    if not ocr_results:
        return "", 0.0

    candidates = []
    for bbox, text, confidence in ocr_results:
        # Skip low confidence
        if confidence < min_conf:
            continue

        # Extract only digits from text
        digits = ''.join(c for c in text if c.isdigit())

        # Valid bib numbers are typically 1-4 digits (5 is usually noise)
        if digits and 1 <= len(digits) <= 4:
            # Prefer results that are purely numeric (text == digits)
            purity_bonus = 0.1 if text.strip() == digits else 0
            candidates.append((digits, confidence + purity_bonus, len(digits)))

    if not candidates:
        return "", 0.0

    # Sort by: confidence first, then prefer longer numbers (more specific)
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0], candidates[0][1]


def try_multiple_preprocessing(
    bib_crop: np.ndarray,
    reader: easyocr.Reader,
) -> tuple[str, float]:
    """
    Try multiple preprocessing methods and return best result.
    """
    methods = [
        ("original", lambda img: img),
        ("clahe", preprocess_bib_crop),
        ("binary", preprocess_bib_crop_v2),
    ]

    best_number = ""
    best_conf = 0.0

    for name, preprocess_fn in methods:
        try:
            processed = preprocess_fn(bib_crop)
            ocr_results = reader.readtext(
                processed,
                allowlist='0123456789',  # Only look for digits
                paragraph=False,
            )
            number, conf = extract_bib_number(ocr_results)

            if conf > best_conf:
                best_number = number
                best_conf = conf

        except Exception as e:
            continue

    return best_number, best_conf


def process_image(
    image_path: Path,
    model: YOLO,
    reader: easyocr.Reader,
    output_dir: Path,
    conf_threshold: float = 0.5,
    padding: int = 15,
    use_multi_preprocess: bool = True,
) -> list[dict]:
    """
    Process a single image: detect bibs and extract numbers.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return []

    h, w = image.shape[:2]
    results = []

    # Run detection
    detections = model(image, verbose=False)[0]

    # Process each detection
    for i, box in enumerate(detections.boxes):
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Add padding for OCR (but stay within image bounds)
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        # Crop bib region
        bib_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # Run OCR with preprocessing
        if use_multi_preprocess:
            bib_number, ocr_conf = try_multiple_preprocessing(bib_crop, reader)
        else:
            ocr_results = reader.readtext(
                bib_crop,
                allowlist='0123456789',
                paragraph=False,
            )
            bib_number, ocr_conf = extract_bib_number(ocr_results)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": conf,
            "bib_number": bib_number,
            "ocr_conf": ocr_conf,
        })

        # Draw on image
        color = (0, 255, 0) if bib_number else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"#{bib_number} ({ocr_conf:.2f})" if bib_number else "?"
        cv2.putText(
            image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    # Save annotated image
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test bib OCR on images")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/bib_detector/weights/best.pt",
        help="Path to trained YOLOv8 model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/val",
        help="Path to test images (file or directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ocr_test",
        help="Output directory for annotated images",
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
        default=10,
        help="Maximum number of images to process (0 for all)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple OCR without multi-preprocessing",
    )
    args = parser.parse_args()

    # Setup paths
    model_path = Path(args.model)
    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading detection model: {model_path}")
    model = YOLO(str(model_path))

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(
        ['en'],
        gpu=False,
        verbose=False,
    )

    # Get image list
    if source_path.is_file():
        images = [source_path]
    else:
        images = sorted(source_path.glob("*.jpg")) + sorted(source_path.glob("*.png"))

    if args.limit > 0:
        images = images[:args.limit]

    use_multi = not args.simple
    print(f"\nProcessing {len(images)} images (multi-preprocess: {use_multi})...\n")

    # Process images
    total_detections = 0
    successful_ocr = 0

    for image_path in images:
        results = process_image(
            image_path, model, reader, output_dir,
            conf_threshold=args.conf,
            use_multi_preprocess=use_multi,
        )

        total_detections += len(results)

        # Print results
        print(f"{image_path.name}:")
        for r in results:
            status = f"  Bib #{r['bib_number']}" if r['bib_number'] else "  [No number detected]"
            print(f"{status} (det: {r['det_conf']:.2f}, ocr: {r['ocr_conf']:.2f})")
            if r['bib_number']:
                successful_ocr += 1
        if not results:
            print("  [No bibs detected]")
        print()

    # Summary
    print("=" * 50)
    print(f"Total bibs detected: {total_detections}")
    print(f"Successful OCR: {successful_ocr} ({100*successful_ocr/max(1,total_detections):.1f}%)")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
