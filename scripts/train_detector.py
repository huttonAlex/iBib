#!/usr/bin/env python3
"""
Train YOLOv8 bib detector for PointCam.

Usage:
    python scripts/train_detector.py [--model yolov8n.pt] [--epochs 100] [--batch 16]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 bib detector")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (yolov8n/s/m/l/x.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device (cuda:0, cpu, etc)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "configs" / "dataset.yaml"

    # Check dataset config exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project=str(project_root / "runs" / "detect"),
        name="bib_detector",
        exist_ok=True,
        resume=args.resume,
        # Augmentation settings suitable for bib detection
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10,   # Rotation (bibs usually upright)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,   # Horizontal flip
        flipud=0.0,   # No vertical flip (bibs have orientation)
        mosaic=1.0,
        mixup=0.0,
        # Performance
        workers=4,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {project_root / 'runs' / 'detect' / 'bib_detector' / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    main()
