#!/usr/bin/env python3
"""
Consolidate YOLO training data from multiple sources into a single dataset.

Scans 5 data sources for matching image/annotation pairs, performs a stratified
85/15 train/val split by event, and copies files into data/images/{train,val}
and data/labels/{train,val} with event-prefixed filenames to avoid collisions.

Usage:
    python scripts/consolidate_detector_data.py [--dry-run]
"""

import argparse
import random
import shutil
from pathlib import Path

# Data source definitions: (directory_name, image_subdir, annotation_subdir)
SOURCES = [
    ("tagged_event_86638", "source_images", "yolo_annotations"),
    ("tagged_event_89536", "source_images", "yolo_annotations"),
    ("tagged_event_88679", "source_images", "yolo_annotations"),
    ("unlabeled_batch1", "annotated", "yolo_annotations"),
    ("unlabeled_batch2", "annotated", "yolo_annotations"),
    ("unlabeled_rec0004a", "annotated", "yolo_annotations"),
    ("unlabeled_rec0011a", "annotated", "yolo_annotations"),
]

# Short prefixes to keep filenames manageable
PREFIX_MAP = {
    "tagged_event_86638": "e86638",
    "tagged_event_89536": "e89536",
    "tagged_event_88679": "e88679",
    "unlabeled_batch1": "ub1",
    "unlabeled_batch2": "ub2",
    "unlabeled_rec0004a": "rec4a",
    "unlabeled_rec0011a": "rec11a",
}

SPLIT_RATIO = 0.85  # train fraction
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def find_pairs(data_root: Path, source_name: str, img_subdir: str, ann_subdir: str):
    """Find matching image/annotation pairs for a source."""
    img_dir = data_root / source_name / img_subdir
    ann_dir = data_root / source_name / ann_subdir

    if not img_dir.exists():
        print(f"  WARNING: Image directory not found: {img_dir}")
        return []
    if not ann_dir.exists():
        print(f"  WARNING: Annotation directory not found: {ann_dir}")
        return []

    # Build lookup of annotation stems
    ann_stems = {p.stem: p for p in ann_dir.iterdir() if p.suffix == ".txt"}

    # Find images that have matching annotations
    pairs = []
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
            if img_path.stem in ann_stems:
                pairs.append((img_path, ann_stems[img_path.stem]))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Consolidate YOLO detector training data")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stats without copying files"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = project_root / "data"

    # --- Scan all sources ---
    print("Scanning data sources...\n")
    source_pairs = {}  # source_name -> list of (img_path, ann_path)

    for source_name, img_subdir, ann_subdir in SOURCES:
        pairs = find_pairs(data_root, source_name, img_subdir, ann_subdir)
        source_pairs[source_name] = pairs
        print(f"  {source_name}: {len(pairs)} image/annotation pairs")

    total = sum(len(p) for p in source_pairs.values())
    print(f"\n  Total: {total} pairs\n")

    if total == 0:
        print("ERROR: No data found. Check that data sources exist.")
        return

    # --- Stratified split by event ---
    print(f"Splitting {SPLIT_RATIO:.0%} train / {1 - SPLIT_RATIO:.0%} val (by event)...\n")
    random.seed(RANDOM_SEED)

    train_files = []  # list of (source_name, img_path, ann_path)
    val_files = []

    for source_name, pairs in source_pairs.items():
        if not pairs:
            continue
        shuffled = list(pairs)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * SPLIT_RATIO)
        train_pairs = shuffled[:split_idx]
        val_pairs = shuffled[split_idx:]

        for img, ann in train_pairs:
            train_files.append((source_name, img, ann))
        for img, ann in val_pairs:
            val_files.append((source_name, img, ann))

        print(f"  {source_name}: {len(train_pairs)} train / {len(val_pairs)} val")

    print(f"\n  Total: {len(train_files)} train / {len(val_files)} val\n")

    if args.dry_run:
        print("Dry run — no files copied.")
        return

    # --- Back up existing data ---
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    images_old = data_root / "images_old"
    labels_old = data_root / "labels_old"

    if images_dir.exists():
        if images_old.exists():
            print(f"Removing previous backup {images_old}...")
            shutil.rmtree(images_old)
        print(f"Backing up {images_dir} -> {images_old}")
        images_dir.rename(images_old)

    if labels_dir.exists():
        if labels_old.exists():
            print(f"Removing previous backup {labels_old}...")
            shutil.rmtree(labels_old)
        print(f"Backing up {labels_dir} -> {labels_old}")
        labels_dir.rename(labels_old)

    # --- Create output directories ---
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # --- Copy files ---
    print("\nCopying files...")

    def copy_split(file_list, split):
        for source_name, img_path, ann_path in file_list:
            prefix = PREFIX_MAP[source_name]
            # Normalize extension to lowercase
            ext = img_path.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"
            dest_stem = f"{prefix}_{img_path.stem}"

            shutil.copy2(img_path, images_dir / split / f"{dest_stem}{ext}")
            shutil.copy2(ann_path, labels_dir / split / f"{dest_stem}.txt")

    copy_split(train_files, "train")
    copy_split(val_files, "val")

    # --- Verify ---
    train_imgs = len(list((images_dir / "train").iterdir()))
    train_lbls = len(list((labels_dir / "train").iterdir()))
    val_imgs = len(list((images_dir / "val").iterdir()))
    val_lbls = len(list((labels_dir / "val").iterdir()))

    print(f"\nDone! Output in {images_dir} and {labels_dir}")
    print(f"  Train: {train_imgs} images, {train_lbls} labels")
    print(f"  Val:   {val_imgs} images, {val_lbls} labels")

    if train_imgs != train_lbls or val_imgs != val_lbls:
        print("  WARNING: Image/label count mismatch!")
    else:
        print("  All counts match.")


if __name__ == "__main__":
    main()
