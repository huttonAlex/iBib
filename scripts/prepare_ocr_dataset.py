#!/usr/bin/env python3
"""Prepare unified OCR dataset from multiple data sources.

Consolidates all 5 ocr_training directories into a single dataset with
stratified train/val/test splits. Exports in Common (symlinked images + TSV),
HuggingFace Dataset, and LMDB formats.

Usage:
    python scripts/prepare_ocr_dataset.py [--output data/ocr_dataset] [--no-augment]
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Data sources: (directory, event_name)
DATA_SOURCES = [
    ("data/tagged_event_86638/ocr_training", "event_86638"),
    ("data/tagged_event_88679/ocr_training", "event_88679"),
    ("data/tagged_event_89536/ocr_training", "event_89536"),
    ("data/unlabeled_batch1/ocr_training", "batch1"),
    ("data/unlabeled_batch2/ocr_training", "batch2"),
]


def load_samples(project_root: Path) -> list[dict]:
    """Load all samples from the 5 data sources."""
    samples = []
    for rel_dir, event_name in DATA_SOURCES:
        ocr_dir = project_root / rel_dir
        tsv_path = ocr_dir / "labels.tsv"
        if not tsv_path.exists():
            print(f"  WARNING: {tsv_path} not found, skipping")
            continue

        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            count = 0
            skipped = 0
            for row in reader:
                filename = row["filename"]
                label = str(row["number"])
                img_path = ocr_dir / filename

                # Validate: image exists
                if not img_path.exists():
                    skipped += 1
                    continue

                # Validate: label is digits-only, 1-4 chars
                if not re.fullmatch(r"\d{1,4}", label):
                    skipped += 1
                    continue

                samples.append({
                    "img_path": str(img_path),
                    "label": label,
                    "event": event_name,
                    "digit_count": len(label),
                    "filename": filename,
                })
                count += 1

            print(f"  {event_name}: {count} samples loaded, {skipped} skipped")

    return samples


def stratified_split(samples: list[dict], val_ratio=0.1, test_ratio=0.1, seed=42):
    """Stratified 80/10/10 split within each event."""
    train, val, test = [], [], []

    # Group by event
    by_event = defaultdict(list)
    for s in samples:
        by_event[s["event"]].append(s)

    for event, event_samples in sorted(by_event.items()):
        n = len(event_samples)
        if n < 10:
            # Too few samples, put all in train
            train.extend(event_samples)
            continue

        # Stratify by digit_count within each event
        strat_labels = [s["digit_count"] for s in event_samples]

        # First split: train+val vs test
        test_size = max(int(n * test_ratio), 1)
        remaining_size = n - test_size

        try:
            splitter1 = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            train_val_idx, test_idx = next(splitter1.split(event_samples, strat_labels))
        except ValueError:
            # Fall back to random split if stratification fails
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n)
            test_idx = indices[:test_size]
            train_val_idx = indices[test_size:]

        test_samples = [event_samples[i] for i in test_idx]
        train_val_samples = [event_samples[i] for i in train_val_idx]

        # Second split: train vs val
        val_size = max(int(n * val_ratio), 1)
        tv_labels = [s["digit_count"] for s in train_val_samples]

        try:
            splitter2 = StratifiedShuffleSplit(
                n_splits=1, test_size=val_size, random_state=seed
            )
            train_idx, val_idx = next(splitter2.split(train_val_samples, tv_labels))
        except ValueError:
            rng = np.random.RandomState(seed + 1)
            indices = rng.permutation(len(train_val_samples))
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

        train.extend([train_val_samples[i] for i in train_idx])
        val.extend([train_val_samples[i] for i in val_idx])
        test.extend(test_samples)

        print(
            f"  {event}: {len(train_idx)} train, "
            f"{len(val_idx)} val, {len(test_idx)} test"
        )

    return train, val, test


def get_augmentation_pipeline():
    """Define albumentations augmentation pipeline for OCR training."""
    import albumentations as A

    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
    ])


def augment_and_oversample(
    train_samples: list[dict],
    output_dir: Path,
    aug_pipeline,
) -> list[dict]:
    """Oversample minority digit counts with augmentation.

    Multipliers: 1-digit: 10x, 2-digit: 5x, 3-digit: 2x, 4-digit: 0x (no extra).
    """
    multipliers = {1: 10, 2: 5, 3: 2, 4: 0}
    aug_dir = output_dir / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    augmented = []
    counts = Counter()

    for sample in train_samples:
        dc = sample["digit_count"]
        n_aug = multipliers.get(dc, 0)
        if n_aug == 0:
            continue

        img = cv2.imread(sample["img_path"])
        if img is None:
            continue

        for i in range(n_aug):
            result = aug_pipeline(image=img)
            aug_img = result["image"]

            aug_filename = f"aug_{dc}d_{counts[dc]:05d}.jpg"
            aug_path = aug_dir / aug_filename
            cv2.imwrite(str(aug_path), aug_img)

            augmented.append({
                "img_path": str(aug_path),
                "label": sample["label"],
                "event": sample["event"],
                "digit_count": dc,
                "filename": aug_filename,
                "augmented": True,
            })
            counts[dc] += 1

    print(f"\n  Augmented samples created: {dict(counts)}")
    return augmented


def export_common(
    splits: dict[str, list[dict]], output_dir: Path, use_symlinks: bool = True
):
    """Export Common format: symlinked images + TSV labels."""
    for split_name, samples in splits.items():
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        tsv_path = split_dir / "labels.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["filename", "label", "event", "digit_count"])

            for i, sample in enumerate(samples):
                src = Path(sample["img_path"]).resolve()
                ext = src.suffix
                dst_name = f"{i:06d}{ext}"
                dst = images_dir / dst_name

                if use_symlinks:
                    try:
                        if dst.exists() or dst.is_symlink():
                            dst.unlink()
                        dst.symlink_to(src)
                    except OSError:
                        # Symlinks may fail on Windows/WSL cross-filesystem
                        import shutil

                        shutil.copy2(str(src), str(dst))
                else:
                    import shutil

                    shutil.copy2(str(src), str(dst))

                writer.writerow([
                    dst_name,
                    sample["label"],
                    sample["event"],
                    sample["digit_count"],
                ])

        print(f"  Common/{split_name}: {len(samples)} samples -> {tsv_path}")


def export_huggingface(splits: dict[str, list[dict]], output_dir: Path):
    """Export HuggingFace Dataset format for TrOCR fine-tuning."""
    try:
        from datasets import Dataset, DatasetDict, Image as HFImage
    except ImportError:
        print("  WARNING: 'datasets' not installed, skipping HuggingFace export")
        return

    hf_dir = output_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)

    ds_dict = {}
    for split_name, samples in splits.items():
        ds_dict[split_name] = Dataset.from_dict({
            "image": [s["img_path"] for s in samples],
            "label": [s["label"] for s in samples],
            "event": [s["event"] for s in samples],
        }).cast_column("image", HFImage())

    dataset = DatasetDict(ds_dict)
    dataset.save_to_disk(str(hf_dir))
    print(f"  HuggingFace dataset saved to {hf_dir}")


def export_lmdb(splits: dict[str, list[dict]], output_dir: Path):
    """Export LMDB format for PARSeq fine-tuning."""
    try:
        import lmdb
    except ImportError:
        print("  WARNING: 'lmdb' not installed, skipping LMDB export")
        return

    for split_name, samples in splits.items():
        lmdb_dir = output_dir / "lmdb" / split_name
        lmdb_dir.mkdir(parents=True, exist_ok=True)

        # Estimate map size: ~500KB per image avg, plus overhead
        map_size = len(samples) * 512 * 1024

        env = lmdb.open(str(lmdb_dir), map_size=map_size)

        with env.begin(write=True) as txn:
            for i, sample in enumerate(samples):
                img_data = open(sample["img_path"], "rb").read()

                # Store image
                img_key = f"image-{i+1:09d}".encode()
                txn.put(img_key, img_data)

                # Store label
                label_key = f"label-{i+1:09d}".encode()
                txn.put(label_key, sample["label"].encode())

            # Store count
            txn.put(b"num-samples", str(len(samples)).encode())

        env.close()
        print(f"  LMDB/{split_name}: {len(samples)} samples -> {lmdb_dir}")


def print_statistics(splits: dict[str, list[dict]]):
    """Print distribution statistics table."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    # Overall counts
    total = sum(len(s) for s in splits.values())
    print(f"\nTotal samples: {total}")
    for name, samples in splits.items():
        print(f"  {name}: {len(samples)} ({100 * len(samples) / total:.1f}%)")

    # Per digit count
    print(f"\n{'Split':<10} {'1-digit':>10} {'2-digit':>10} {'3-digit':>10} {'4-digit':>10}")
    print("-" * 50)
    for name, samples in splits.items():
        dc = Counter(s["digit_count"] for s in samples)
        print(
            f"{name:<10} {dc.get(1, 0):>10} {dc.get(2, 0):>10} "
            f"{dc.get(3, 0):>10} {dc.get(4, 0):>10}"
        )

    # Per event
    all_events = sorted({s["event"] for ss in splits.values() for s in ss})
    print(f"\n{'Split':<10}", end="")
    for ev in all_events:
        print(f" {ev:>12}", end="")
    print()
    print("-" * (10 + 13 * len(all_events)))

    for name, samples in splits.items():
        ec = Counter(s["event"] for s in samples)
        print(f"{name:<10}", end="")
        for ev in all_events:
            print(f" {ec.get(ev, 0):>12}", end="")
        print()

    # Label distribution (top 20 most common)
    all_labels = [s["label"] for ss in splits.values() for s in ss]
    label_counts = Counter(all_labels)
    print(f"\nUnique labels: {len(label_counts)}")
    print("Top 20 most common:")
    for label, count in label_counts.most_common(20):
        print(f"  '{label}': {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare unified OCR dataset")
    parser.add_argument(
        "--output", type=str, default="data/ocr_dataset",
        help="Output directory (default: data/ocr_dataset)",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Skip augmentation/oversampling",
    )
    parser.add_argument(
        "--no-symlinks", action="store_true",
        help="Copy files instead of symlinking",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splits (default: 42)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output

    print("=" * 70)
    print("OCR Dataset Preparation")
    print("=" * 70)

    # Step 1: Load all samples
    print("\n[1/5] Loading samples from all sources...")
    samples = load_samples(project_root)
    print(f"  Total: {len(samples)} valid samples")

    if len(samples) == 0:
        print("ERROR: No samples found. Check data directories.")
        sys.exit(1)

    # Step 2: Stratified split
    print("\n[2/5] Creating stratified train/val/test splits...")
    train, val, test = stratified_split(samples, seed=args.seed)
    print(f"  Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Step 3: Augmentation & oversampling
    if not args.no_augment:
        print("\n[3/5] Augmenting and oversampling minority classes...")
        aug_pipeline = get_augmentation_pipeline()
        augmented = augment_and_oversample(train, output_dir, aug_pipeline)
        train_aug = train + augmented
        print(f"  Training set: {len(train)} original + {len(augmented)} augmented = {len(train_aug)}")
    else:
        print("\n[3/5] Skipping augmentation (--no-augment)")
        train_aug = train

    splits = {"train": train_aug, "val": val, "test": test}

    # Step 4: Export all formats
    print("\n[4/5] Exporting datasets...")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  --- Common format (symlinked images + TSV) ---")
    export_common(splits, output_dir, use_symlinks=not args.no_symlinks)

    print("\n  --- HuggingFace Dataset ---")
    export_huggingface(splits, output_dir)

    print("\n  --- LMDB ---")
    export_lmdb(splits, output_dir)

    # Step 5: Statistics
    print("\n[5/5] Computing statistics...")
    # Show stats for original splits (without augmentation) for clarity
    original_splits = {"train": train, "val": val, "test": test}
    print_statistics(original_splits)

    if not args.no_augment:
        print(f"\n(Training set has {len(train_aug)} total samples after augmentation)")

    print(f"\nDataset exported to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
