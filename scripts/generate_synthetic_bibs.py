#!/usr/bin/env python3
"""Generate synthetic bib number images for OCR training.

Creates training data targeting specific weaknesses:
- Confusing digit pairs (1/7, 3/8, 6/8, 0/6)
- Motion blur and degradation
- Various fonts and styles
- Different aspect ratios and sizes

Usage:
    python scripts/generate_synthetic_bibs.py --count 5000 --output data/synthetic_ocr
    python scripts/generate_synthetic_bibs.py --count 1000 --bib-range 1-3000 --hard-mode
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font Configuration
# ---------------------------------------------------------------------------

# Common fonts that resemble race bib numbers
# Will fall back to default if not available
FONT_CANDIDATES = [
    "arial.ttf",
    "arialbd.ttf",  # Arial Bold
    "impact.ttf",
    "verdana.ttf",
    "verdanab.ttf",  # Verdana Bold
    "tahoma.ttf",
    "calibri.ttf",
    "calibrib.ttf",  # Calibri Bold
    "consola.ttf",   # Consolas (monospace)
    "courbd.ttf",    # Courier Bold
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def find_available_fonts() -> List[str]:
    """Find fonts available on the system."""
    available = []
    for font_path in FONT_CANDIDATES:
        try:
            ImageFont.truetype(font_path, 40)
            available.append(font_path)
        except (OSError, IOError):
            pass

    if not available:
        print("Warning: No TrueType fonts found, using default")
        available = [None]  # Will use default font

    return available


# ---------------------------------------------------------------------------
# Color Schemes (typical race bib colors)
# ---------------------------------------------------------------------------

COLOR_SCHEMES = [
    # (background, text) - common bib color combinations
    ((255, 255, 255), (0, 0, 0)),       # White bg, black text
    ((255, 255, 0), (0, 0, 0)),         # Yellow bg, black text
    ((255, 200, 0), (0, 0, 0)),         # Orange-yellow bg, black text
    ((0, 255, 0), (0, 0, 0)),           # Green bg, black text
    ((255, 150, 150), (0, 0, 0)),       # Pink bg, black text
    ((200, 200, 255), (0, 0, 0)),       # Light blue bg, black text
    ((255, 255, 255), (255, 0, 0)),     # White bg, red text
    ((255, 255, 255), (0, 0, 255)),     # White bg, blue text
    ((255, 255, 0), (255, 0, 0)),       # Yellow bg, red text
    ((0, 0, 0), (255, 255, 255)),       # Black bg, white text (rare but exists)
]


# ---------------------------------------------------------------------------
# Synthetic Image Generation
# ---------------------------------------------------------------------------

class SyntheticBibGenerator:
    """Generate synthetic bib number images."""

    def __init__(
        self,
        fonts: Optional[List[str]] = None,
        color_schemes: Optional[List[Tuple]] = None,
        output_size: Tuple[int, int] = (128, 32),
    ):
        self.fonts = fonts or find_available_fonts()
        self.color_schemes = color_schemes or COLOR_SCHEMES
        self.output_size = output_size

        print(f"Using {len(self.fonts)} fonts, {len(self.color_schemes)} color schemes")

    def generate(
        self,
        number: str,
        apply_degradation: bool = True,
        hard_mode: bool = False,
    ) -> np.ndarray:
        """
        Generate a synthetic bib image.

        Args:
            number: Bib number string
            apply_degradation: Apply blur/noise/etc
            hard_mode: More aggressive degradation

        Returns:
            BGR image as numpy array
        """
        # Random parameters
        font_path = random.choice(self.fonts)
        bg_color, text_color = random.choice(self.color_schemes)

        # Create base image (larger, then resize)
        base_width = random.randint(200, 400)
        base_height = random.randint(60, 120)

        img = Image.new('RGB', (base_width, base_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Find font size that fits
        font_size = base_height - 20
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Center text
        bbox = draw.textbbox((0, 0), number, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (base_width - text_width) // 2
        y = (base_height - text_height) // 2 - bbox[1]

        # Add slight position variation
        x += random.randint(-10, 10)
        y += random.randint(-5, 5)

        # Draw text
        draw.text((x, y), number, fill=text_color, font=font)

        # Convert to numpy
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Apply degradation
        if apply_degradation:
            img_bgr = self._apply_degradation(img_bgr, hard_mode)

        # Resize to output size
        img_bgr = cv2.resize(img_bgr, self.output_size, interpolation=cv2.INTER_LINEAR)

        return img_bgr

    def _apply_degradation(self, img: np.ndarray, hard_mode: bool = False) -> np.ndarray:
        """Apply realistic degradation to image."""

        # Probability multiplier for hard mode
        p_mult = 1.5 if hard_mode else 1.0

        # 1. Motion blur (running subjects)
        if random.random() < 0.3 * p_mult:
            kernel_size = random.choice([3, 5, 7, 9])
            if random.random() < 0.7:
                # Horizontal motion blur (most common)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
            else:
                # Diagonal motion blur
                kernel = np.eye(kernel_size) / kernel_size
            img = cv2.filter2D(img, -1, kernel)

        # 2. Gaussian blur (out of focus)
        if random.random() < 0.2 * p_mult:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 3. Brightness/contrast variation
        if random.random() < 0.4 * p_mult:
            alpha = random.uniform(0.7, 1.3)  # Contrast
            beta = random.randint(-30, 30)     # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 4. Gaussian noise
        if random.random() < 0.2 * p_mult:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. JPEG compression artifacts
        if random.random() < 0.3 * p_mult:
            quality = random.randint(50, 90)
            _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # 6. Slight rotation (tilted bibs)
        if random.random() < 0.3 * p_mult:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 7. Perspective distortion (viewing angle)
        if random.random() < 0.2 * p_mult:
            img = self._apply_perspective(img)

        # 8. Partial occlusion (hard mode only)
        if hard_mode and random.random() < 0.15:
            img = self._apply_occlusion(img)

        return img

    def _apply_perspective(self, img: np.ndarray) -> np.ndarray:
        """Apply slight perspective distortion."""
        h, w = img.shape[:2]

        # Small random offsets for corners
        offset = int(min(w, h) * 0.1)

        src_pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])

        dst_pts = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return img

    def _apply_occlusion(self, img: np.ndarray) -> np.ndarray:
        """Apply partial occlusion (hand, other runner, etc)."""
        h, w = img.shape[:2]

        # Random rectangle occlusion
        occ_w = random.randint(w // 6, w // 3)
        occ_h = random.randint(h // 3, h)

        # Usually from sides
        if random.random() < 0.5:
            x = 0  # Left side
        else:
            x = w - occ_w  # Right side
        y = random.randint(0, h - occ_h)

        # Occlude with skin-tone or dark color
        if random.random() < 0.5:
            color = (random.randint(150, 220), random.randint(120, 180), random.randint(100, 160))
        else:
            color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))

        cv2.rectangle(img, (x, y), (x + occ_w, y + occ_h), color, -1)

        return img


# ---------------------------------------------------------------------------
# Number Generation Strategies
# ---------------------------------------------------------------------------

def generate_random_numbers(
    count: int,
    bib_range: Tuple[int, int] = (1, 9999),
    digit_counts: List[int] = [2, 3, 4],
) -> List[str]:
    """Generate random bib numbers."""
    numbers = []
    for _ in range(count):
        num_digits = random.choice(digit_counts)
        min_val = max(bib_range[0], 10 ** (num_digits - 1))
        max_val = min(bib_range[1], 10 ** num_digits - 1)

        if min_val <= max_val:
            numbers.append(str(random.randint(min_val, max_val)))
        else:
            numbers.append(str(random.randint(bib_range[0], bib_range[1])))

    return numbers


def generate_hard_numbers(count: int) -> List[str]:
    """Generate numbers targeting common OCR confusion patterns."""
    hard_patterns = []

    # Confusing digit pairs
    confusing_digits = [
        ('1', '7'),  # Very common confusion
        ('3', '8'),
        ('6', '8'),
        ('0', '6'),
        ('5', '6'),
        ('1', '4'),
        ('2', '7'),
    ]

    for _ in range(count):
        pattern_type = random.choice([
            'confusing_pair',
            'repeated_digits',
            'leading_one',
            'all_same',
            'sequential',
        ])

        if pattern_type == 'confusing_pair':
            # Include confusing digit pairs
            d1, d2 = random.choice(confusing_digits)
            num_digits = random.choice([3, 4])
            number = ''.join(random.choice([d1, d2, random.choice('0123456789')])
                           for _ in range(num_digits))

        elif pattern_type == 'repeated_digits':
            # 1111, 7777, etc
            digit = random.choice('0123456789')
            num_digits = random.choice([3, 4])
            number = digit * num_digits

        elif pattern_type == 'leading_one':
            # Numbers starting with 1 (often confused)
            num_digits = random.choice([3, 4])
            number = '1' + ''.join(random.choice('0123456789') for _ in range(num_digits - 1))

        elif pattern_type == 'all_same':
            # Same digit repeated
            digit = random.choice('0123456789')
            number = digit * random.choice([2, 3, 4])

        elif pattern_type == 'sequential':
            # 1234, 4321, etc
            start = random.randint(1, 6)
            if random.random() < 0.5:
                number = ''.join(str(start + i) for i in range(4))
            else:
                number = ''.join(str(start + 3 - i) for i in range(4))

        # Avoid leading zeros for multi-digit
        if len(number) > 1 and number[0] == '0':
            number = str(random.randint(1, 9)) + number[1:]

        hard_patterns.append(number)

    return hard_patterns


# ---------------------------------------------------------------------------
# Main Generation
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: Path,
    count: int,
    bib_range: Optional[Tuple[int, int]] = None,
    hard_mode: bool = False,
    hard_ratio: float = 0.3,
):
    """Generate synthetic OCR training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    generator = SyntheticBibGenerator()

    # Generate numbers
    if bib_range:
        # Mix of random from range + hard examples
        n_random = int(count * (1 - hard_ratio))
        n_hard = count - n_random

        numbers = generate_random_numbers(n_random, bib_range)
        numbers += generate_hard_numbers(n_hard)
    else:
        # Default mix
        n_random = int(count * 0.5)
        n_hard = count - n_random

        numbers = generate_random_numbers(n_random)
        numbers += generate_hard_numbers(n_hard)

    random.shuffle(numbers)

    # Generate images
    labels = []

    for i, number in enumerate(numbers):
        # Generate image
        img = generator.generate(
            number,
            apply_degradation=True,
            hard_mode=hard_mode or (random.random() < hard_ratio),
        )

        # Save image
        filename = f"syn_{i:06d}.jpg"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), img)

        labels.append({
            "filename": filename,
            "label": number,
        })

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{count} images...")

    # Save labels
    labels_path = output_dir / "labels.tsv"
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"], delimiter='\t')
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nGenerated {count} synthetic images")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_path}")

    # Show sample numbers
    print(f"\nSample numbers generated:")
    for num in random.sample(numbers, min(10, len(numbers))):
        print(f"  {num}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bib images for OCR training")
    parser.add_argument(
        "--output", type=str, default="data/synthetic_ocr",
        help="Output directory"
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--bib-range", type=str, default=None,
        help="Bib number range (e.g., '1-3000')"
    )
    parser.add_argument(
        "--hard-mode", action="store_true",
        help="Generate more challenging examples"
    )
    parser.add_argument(
        "--hard-ratio", type=float, default=0.3,
        help="Ratio of hard examples (default: 0.3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse bib range
    bib_range = None
    if args.bib_range:
        try:
            start, end = map(int, args.bib_range.split('-'))
            bib_range = (start, end)
        except ValueError:
            print(f"Invalid bib range: {args.bib_range}")
            return

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output

    print("=" * 60)
    print("Synthetic Bib Image Generator")
    print("=" * 60)
    print(f"Output:     {output_dir}")
    print(f"Count:      {args.count}")
    print(f"Bib range:  {bib_range or 'default (1-9999)'}")
    print(f"Hard mode:  {args.hard_mode}")
    print(f"Hard ratio: {args.hard_ratio}")
    print()

    generate_dataset(
        output_dir=output_dir,
        count=args.count,
        bib_range=bib_range,
        hard_mode=args.hard_mode,
        hard_ratio=args.hard_ratio,
    )

    print("\nDone!")
    print("\nTo use this data for training:")
    print(f"  1. Add to your OCR dataset")
    print(f"  2. Re-run: python scripts/finetune_ocr.py --model parseq")


if __name__ == "__main__":
    main()
