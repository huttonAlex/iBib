#!/usr/bin/env python3
"""
Dataset Validation Script

Validates a COCO-format dataset for PointCam bib detection training.
Checks structure, annotations, and provides statistics.

Usage:
    python scripts/validate_dataset.py --images data/images --annotations data/annotations/instances.json
    python scripts/validate_dataset.py --images data/images --annotations data/annotations/instances.json --show-samples
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    from pycocotools.coco import COCO
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not installed. Install with: pip install pycocotools")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def validate_coco_structure(annotations_path: Path) -> dict:
    """Validate COCO JSON structure and return parsed data."""
    print(f"\n{'='*60}")
    print("COCO Structure Validation")
    print('='*60)

    if not annotations_path.exists():
        print(f"ERROR: Annotations file not found: {annotations_path}")
        return None

    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return None

    # Check required fields
    required_fields = ['images', 'annotations', 'categories']
    missing = [f for f in required_fields if f not in data]

    if missing:
        print(f"ERROR: Missing required fields: {missing}")
        return None

    print(f"  Images:      {len(data['images']):,}")
    print(f"  Annotations: {len(data['annotations']):,}")
    print(f"  Categories:  {len(data['categories'])}")

    # Show categories
    print(f"\n  Categories:")
    for cat in data['categories']:
        print(f"    - {cat['id']}: {cat['name']}")

    return data


def validate_images(images_path: Path, coco_data: dict) -> dict:
    """Validate image files exist and match annotations."""
    print(f"\n{'='*60}")
    print("Image Validation")
    print('='*60)

    if not images_path.exists():
        print(f"ERROR: Images directory not found: {images_path}")
        return {'found': 0, 'missing': [], 'extra': []}

    # Get annotated image filenames
    annotated_files = {img['file_name'] for img in coco_data['images']}

    # Get actual files on disk
    actual_files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        actual_files.update(f.name for f in images_path.glob(ext))
        actual_files.update(f.name for f in images_path.glob(ext.upper()))

    # Compare
    found = annotated_files & actual_files
    missing = annotated_files - actual_files
    extra = actual_files - annotated_files

    print(f"  Annotated images: {len(annotated_files):,}")
    print(f"  Files on disk:    {len(actual_files):,}")
    print(f"  Matched:          {len(found):,}")
    print(f"  Missing files:    {len(missing):,}")
    print(f"  Unannotated:      {len(extra):,}")

    if missing:
        print(f"\n  First 5 missing files:")
        for f in list(missing)[:5]:
            print(f"    - {f}")

    return {'found': len(found), 'missing': list(missing), 'extra': list(extra)}


def validate_annotations(coco_data: dict) -> dict:
    """Validate annotation quality and statistics."""
    print(f"\n{'='*60}")
    print("Annotation Validation")
    print('='*60)

    annotations = coco_data['annotations']
    images = {img['id']: img for img in coco_data['images']}

    # Statistics
    stats = {
        'total_annotations': len(annotations),
        'images_with_annotations': len(set(a['image_id'] for a in annotations)),
        'annotations_per_image': defaultdict(int),
        'bbox_sizes': [],
        'issues': []
    }

    # Count annotations per image
    for ann in annotations:
        stats['annotations_per_image'][ann['image_id']] += 1

    ann_counts = list(stats['annotations_per_image'].values())

    print(f"  Total annotations:        {stats['total_annotations']:,}")
    print(f"  Images with annotations:  {stats['images_with_annotations']:,}")
    print(f"  Images without annotations: {len(images) - stats['images_with_annotations']:,}")
    print(f"  Annotations per image:")
    print(f"    Min: {min(ann_counts) if ann_counts else 0}")
    print(f"    Max: {max(ann_counts) if ann_counts else 0}")
    print(f"    Avg: {sum(ann_counts)/len(ann_counts):.1f}" if ann_counts else "    Avg: 0")

    # Analyze bounding boxes
    print(f"\n  Bounding Box Analysis:")

    widths = []
    heights = []
    areas = []
    invalid_boxes = 0

    for ann in annotations:
        if 'bbox' not in ann:
            stats['issues'].append(f"Annotation {ann['id']} missing bbox")
            continue

        bbox = ann['bbox']  # COCO format: [x, y, width, height]

        if len(bbox) != 4:
            stats['issues'].append(f"Annotation {ann['id']} invalid bbox length")
            continue

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            invalid_boxes += 1
            stats['issues'].append(f"Annotation {ann['id']} invalid bbox dimensions: {bbox}")
            continue

        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        stats['bbox_sizes'].append({'w': w, 'h': h, 'area': w * h})

    if widths:
        print(f"    Width  - Min: {min(widths):.0f}, Max: {max(widths):.0f}, Avg: {sum(widths)/len(widths):.0f}")
        print(f"    Height - Min: {min(heights):.0f}, Max: {max(heights):.0f}, Avg: {sum(heights)/len(heights):.0f}")
        print(f"    Area   - Min: {min(areas):.0f}, Max: {max(areas):.0f}, Avg: {sum(areas)/len(areas):.0f}")

    if invalid_boxes:
        print(f"    Invalid boxes: {invalid_boxes}")

    # Check for common issues
    print(f"\n  Quality Checks:")

    # Very small boxes (might be labeling errors)
    tiny_boxes = sum(1 for a in areas if a < 100)
    print(f"    Very small boxes (<100px area): {tiny_boxes}")

    # Very large boxes (might span full image)
    huge_boxes = sum(1 for a in areas if a > 500000)
    print(f"    Very large boxes (>500k area): {huge_boxes}")

    if stats['issues']:
        print(f"\n  Issues Found: {len(stats['issues'])}")
        for issue in stats['issues'][:5]:
            print(f"    - {issue}")
        if len(stats['issues']) > 5:
            print(f"    ... and {len(stats['issues']) - 5} more")

    return stats


def show_sample_images(images_path: Path, coco_data: dict, num_samples: int = 4):
    """Display sample images with bounding boxes."""
    if not HAS_MATPLOTLIB or not HAS_CV2:
        print("\nSkipping visualization (matplotlib or cv2 not available)")
        return

    print(f"\n{'='*60}")
    print("Sample Images")
    print('='*60)

    # Build image_id -> annotations mapping
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Get images that have annotations
    images_with_anns = [img for img in coco_data['images'] if img['id'] in img_to_anns]

    if not images_with_anns:
        print("No images with annotations found")
        return

    # Sample evenly
    step = max(1, len(images_with_anns) // num_samples)
    samples = images_with_anns[::step][:num_samples]

    fig, axes = plt.subplots(1, min(num_samples, len(samples)), figsize=(4 * num_samples, 4))
    if len(samples) == 1:
        axes = [axes]

    for ax, img_info in zip(axes, samples):
        img_path = images_path / img_info['file_name']

        if not img_path.exists():
            ax.set_title(f"Missing: {img_info['file_name']}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.set_title(f"{img_info['file_name']}\n{len(img_to_anns[img_info['id']])} bibs")
        ax.axis('off')

        # Draw bounding boxes
        for ann in img_to_anns[img_info['id']]:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(images_path.parent / 'dataset_samples.png', dpi=150)
    print(f"  Saved sample visualization to: {images_path.parent / 'dataset_samples.png'}")
    plt.show()


def print_summary(coco_data: dict, image_results: dict, ann_stats: dict):
    """Print final summary and recommendations."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    total_images = len(coco_data['images'])
    total_anns = len(coco_data['annotations'])

    # Overall status
    issues = []
    warnings = []

    if image_results['missing']:
        issues.append(f"{len(image_results['missing'])} missing image files")

    if total_images < 500:
        warnings.append(f"Small dataset ({total_images} images) - consider augmentation")

    if total_anns / total_images < 1:
        warnings.append("Many images have no annotations")

    if ann_stats['issues']:
        warnings.append(f"{len(ann_stats['issues'])} annotation issues found")

    # Print status
    if not issues:
        print("\n  STATUS: READY FOR TRAINING")
    else:
        print("\n  STATUS: ISSUES FOUND")
        for issue in issues:
            print(f"    ERROR: {issue}")

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    # Recommendations
    print("\n  Recommendations:")
    print(f"    - Train/Val split: ~{int(total_images * 0.8)} train / ~{int(total_images * 0.2)} val")

    if total_images >= 800:
        print("    - Dataset size is good for initial training")
    else:
        print("    - Consider data augmentation to increase effective dataset size")

    print("\n  Next steps:")
    print("    1. Run: python scripts/prepare_yolo_dataset.py")
    print("    2. Review sample images for labeling quality")
    print("    3. Begin baseline evaluation with pre-trained YOLOv8")


def main():
    parser = argparse.ArgumentParser(description='Validate COCO dataset for PointCam')
    parser.add_argument('--images', type=str, default='data/images',
                        help='Path to images directory')
    parser.add_argument('--annotations', type=str, default='data/annotations/instances.json',
                        help='Path to COCO annotations JSON')
    parser.add_argument('--show-samples', action='store_true',
                        help='Display sample images with annotations')

    args = parser.parse_args()

    images_path = Path(args.images)
    annotations_path = Path(args.annotations)

    print("\n" + "="*60)
    print("PointCam Dataset Validation")
    print("="*60)
    print(f"Images:      {images_path}")
    print(f"Annotations: {annotations_path}")

    # Validate COCO structure
    coco_data = validate_coco_structure(annotations_path)
    if coco_data is None:
        sys.exit(1)

    # Validate images
    image_results = validate_images(images_path, coco_data)

    # Validate annotations
    ann_stats = validate_annotations(coco_data)

    # Show samples if requested
    if args.show_samples:
        show_sample_images(images_path, coco_data)

    # Print summary
    print_summary(coco_data, image_results, ann_stats)


if __name__ == '__main__':
    main()
