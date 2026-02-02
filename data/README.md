# Dataset Directory

This directory contains the training data for PointCam bib detection.

## Expected Structure

```
data/
├── images/              # All image files
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── annotations/         # COCO format annotations
│   └── instances.json   # Main annotations file
└── README.md           # This file
```

## COCO Annotation Format

The `instances.json` file should follow COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "bib",
      "supercategory": "race"
    }
  ]
}
```

## Loading Your Dataset

1. Copy your images to `data/images/`
2. Copy your COCO annotations to `data/annotations/instances.json`
3. Run validation:
   ```bash
   python scripts/validate_dataset.py --images data/images --annotations data/annotations/instances.json
   ```

## Notes

- Images and annotations are excluded from git (see .gitignore)
- Supported image formats: .jpg, .jpeg, .png, .bmp
- Bounding boxes use COCO format: [x, y, width, height] (top-left corner)
