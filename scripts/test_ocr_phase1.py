#!/usr/bin/env python3
"""
Phase 1 OCR Test: TrOCR + Super-Resolution + Multi-pass voting

This script implements the Phase 1 improvements:
1. AI super-resolution for small crops (OpenCV DNN)
2. TrOCR transformer-based text recognition
3. Multi-pass with different preprocessing
4. Ensemble voting for final result
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import re

from ultralytics import YOLO

# TrOCR imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class SuperResolution:
    """OpenCV DNN-based super-resolution."""

    def __init__(self, model_path: str = None, scale: int = 2):
        self.scale = scale
        self.sr = None

        # Try to initialize OpenCV super-resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            # Use ESPCN model (fast and good quality)
            # Model needs to be downloaded separately
            if model_path and Path(model_path).exists():
                self.sr.readModel(model_path)
                self.sr.setModel("espcn", scale)
        except Exception as e:
            print(f"Super-resolution not available: {e}")
            self.sr = None

    def upscale(self, image: np.ndarray, target_height: int = 64) -> np.ndarray:
        """Upscale image to target height."""
        h, w = image.shape[:2]

        if h >= target_height:
            return image

        scale = target_height / h

        # Use super-resolution if available, else bicubic
        if self.sr is not None and scale <= self.scale:
            try:
                return self.sr.upsample(image)
            except:
                pass

        # Fallback to bicubic interpolation
        new_h = int(h * scale)
        new_w = int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class TrOCRRecognizer:
    """TrOCR-based text recognition optimized for printed text."""

    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str = "cpu"):
        print(f"Loading TrOCR model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """
        Recognize text in image.

        Args:
            image: BGR or grayscale numpy array

        Returns:
            (text, confidence) tuple
        """
        # Convert to RGB PIL Image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Process
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=10,  # Bib numbers are short
                num_beams=3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode
        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Calculate confidence from sequence scores
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            confidence = torch.exp(outputs.sequences_scores).item()
        else:
            confidence = 0.5  # Default if no scores

        return text, confidence


def preprocess_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate multiple preprocessing variants for voting."""
    variants = []

    # Original
    variants.append(("original", image.copy()))

    # Grayscale + CLAHE
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))

    # Sharpened
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    variants.append(("sharpened", sharpened))

    # High contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_l = clahe.apply(l)
    enhanced_lab = cv2.merge([clahe_l, a, b])
    high_contrast = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    variants.append(("high_contrast", high_contrast))

    return variants


def extract_bib_number(text: str) -> str:
    """Extract digits from OCR text, filtering to valid bib number."""
    # Extract all digit sequences
    digits = re.findall(r'\d+', text)

    if not digits:
        return ""

    # Filter to valid lengths (1-4 digits)
    valid = [d for d in digits if 1 <= len(d) <= 4]

    if not valid:
        return ""

    # Return longest valid sequence (most likely to be the bib number)
    return max(valid, key=len)


def vote_results(results: list[tuple[str, float]]) -> tuple[str, float]:
    """Vote among multiple OCR results."""
    if not results:
        return "", 0.0

    # Count occurrences weighted by confidence
    votes = {}
    for text, conf in results:
        number = extract_bib_number(text)
        if number:
            if number not in votes:
                votes[number] = []
            votes[number].append(conf)

    if not votes:
        return "", 0.0

    # Score = count * avg_confidence
    scored = []
    for number, confs in votes.items():
        score = len(confs) * (sum(confs) / len(confs))
        avg_conf = sum(confs) / len(confs)
        scored.append((number, score, avg_conf))

    # Return highest scored
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored[0][2]


def process_image(
    image_path: Path,
    detector: YOLO,
    recognizer: TrOCRRecognizer,
    superres: SuperResolution,
    output_dir: Path,
    conf_threshold: float = 0.5,
    padding: int = 15,
) -> list[dict]:
    """Process single image with Phase 1 pipeline."""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return []

    h, w = image.shape[:2]
    results = []

    # Run detection
    detections = detector(image, verbose=False)[0]

    for box in detections.boxes:
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

        # Crop bib region
        bib_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # Upscale if small
        crop_h = bib_crop.shape[0]
        if crop_h < 64:
            bib_crop = superres.upscale(bib_crop, target_height=64)

        # Multi-pass OCR with voting
        ocr_results = []
        for variant_name, variant_img in preprocess_variants(bib_crop):
            try:
                text, ocr_conf = recognizer.recognize(variant_img)
                if text.strip():
                    ocr_results.append((text, ocr_conf))
            except Exception as e:
                continue

        # Vote for final result
        bib_number, final_conf = vote_results(ocr_results)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": conf,
            "bib_number": bib_number,
            "ocr_conf": final_conf,
        })

        # Draw on image
        color = (0, 255, 0) if bib_number else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"#{bib_number} ({final_conf:.2f})" if bib_number else "?"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save result
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 OCR Test")
    parser.add_argument("--model", default="runs/detect/bib_detector/weights/best.pt")
    parser.add_argument("--source", default="data/images/val")
    parser.add_argument("--output", default="runs/ocr_phase1")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--trocr-model", default="microsoft/trocr-base-printed",
                        help="TrOCR model name")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)

    # Load models
    print(f"Loading detector: {args.model}")
    detector = YOLO(args.model)

    print(f"Loading TrOCR: {args.trocr_model}")
    recognizer = TrOCRRecognizer(args.trocr_model, device="cpu")

    print("Initializing super-resolution...")
    superres = SuperResolution()

    # Get images
    if source_path.is_file():
        images = [source_path]
    else:
        images = sorted(source_path.glob("*.jpg")) + sorted(source_path.glob("*.png"))

    if args.limit > 0:
        images = images[:args.limit]

    print(f"\nProcessing {len(images)} images...\n")

    # Process
    total_detections = 0
    successful_ocr = 0

    for image_path in images:
        results = process_image(
            image_path, detector, recognizer, superres,
            output_dir, conf_threshold=args.conf
        )

        total_detections += len(results)

        print(f"{image_path.name}:")
        for r in results:
            if r['bib_number']:
                print(f"  Bib #{r['bib_number']} (det: {r['det_conf']:.2f}, ocr: {r['ocr_conf']:.2f})")
                successful_ocr += 1
            else:
                print(f"  [No number] (det: {r['det_conf']:.2f})")
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
