# Dataset Directory

This directory contains the training data for PointCam bib detection.

## Current Dataset

- **Total images**: 221
- **Train split**: 176 images (80%), 294 annotations
- **Val split**: 45 images (20%), 78 annotations
- **Category**: "Bib" (id: 0)
- **Image size**: 398x600 pixels

## Directory Structure

```
data/
├── images/
│   ├── train/           # Training images (176)
│   └── val/             # Validation images (45)
├── annotations/
│   ├── annotations.json # Full dataset annotations
│   ├── train.json       # Training split annotations
│   └── val.json         # Validation split annotations
└── README.md
```

## COCO Annotation Format

All annotation files follow COCO format:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image001.jpg",
      "width": 398,
      "height": 600
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "Bib"
    }
  ]
}
```

## Usage

For YOLOv8 training, use the pre-split data:
- Images: `data/images/train/` and `data/images/val/`
- Annotations: `data/annotations/train.json` and `data/annotations/val.json`

## Notes

- Images and annotations are excluded from git (see .gitignore)
- Supported image formats: .jpg, .jpeg, .png, .bmp
- Bounding boxes use COCO format: [x, y, width, height] (top-left corner)
