#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for YOLOv8 training.

YOLO format: <class_id> <x_center> <y_center> <width> <height>
All values normalized to [0, 1].
"""

import json
from pathlib import Path


def convert_coco_to_yolo(coco_json_path: Path, output_dir: Path) -> None:
    """Convert COCO annotations to YOLO format txt files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image id -> info mapping
    images = {img["id"]: img for img in coco["images"]}

    # Build image id -> annotations mapping
    img_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each image's annotations
    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = Path(img_info["file_name"]).stem

        # YOLO label file
        label_path = output_dir / f"{file_name}.txt"

        annotations = img_annotations.get(img_id, [])
        lines = []

        for ann in annotations:
            # COCO bbox: [x, y, width, height] (top-left corner)
            x, y, w, h = ann["bbox"]
            # Map category_id to 0-indexed class (source data uses 1 for Bib)
            class_id = 0  # All bibs map to class 0

            # Convert to YOLO format (center, normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamp values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Converted {len(images)} images to YOLO format in {output_dir}")


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Convert train annotations
    convert_coco_to_yolo(
        data_dir / "annotations" / "train.json",
        data_dir / "labels" / "train",
    )

    # Convert val annotations
    convert_coco_to_yolo(
        data_dir / "annotations" / "val.json",
        data_dir / "labels" / "val",
    )

    print("\nConversion complete!")
    print("Labels saved to data/labels/train and data/labels/val")


if __name__ == "__main__":
    main()
