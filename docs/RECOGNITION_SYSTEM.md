# Recognition System Technical Specification

This document provides detailed technical specifications for the PointCam bib detection and number recognition pipeline.

---

## Overview

The recognition system is responsible for:
1. Detecting race bibs in video frames
2. Recognizing the numbers on those bibs
3. Tracking bibs across frames
4. Determining when a bib crosses the timing line

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Recognition Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Frame (1920x1080 @ 60fps)                                        │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────┐                                                 │
│   │ STAGE 1      │  YOLOv8n (TensorRT)                            │
│   │ Detection    │  Input: 640x640                                 │
│   │              │  Output: [(bbox, confidence), ...]              │
│   └──────┬───────┘  Target: <10ms, mAP > 0.90                     │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                 │
│   │ STAGE 2      │  ByteTrack or Centroid Tracking                │
│   │ Tracking     │  Input: detections                              │
│   │              │  Output: [(track_id, bbox), ...]                │
│   └──────┬───────┘  Purpose: consistent ID across frames          │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                 │
│   │ STAGE 3      │  Crop → Preprocess → OCR                       │
│   │ Recognition  │  PaddleOCR PP-OCRv4 or Custom CRNN             │
│   │              │  Output: (bib_number, confidence)               │
│   └──────┬───────┘  Target: <15ms/bib, accuracy > 85%             │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                 │
│   │ STAGE 4      │  Multi-frame voting                            │
│   │ Temporal     │  Collect N frames, vote on consensus           │
│   │ Voting       │  Output: (final_number, combined_conf)         │
│   └──────┬───────┘  Target: final accuracy > 90%                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐                                                 │
│   │ STAGE 5      │  Detect line crossing                          │
│   │ Crossing     │  Emit CrossingEvent                            │
│   │ Detection    │                                                 │
│   └──────────────┘                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Bib Detection

### Model Selection

**Primary: YOLOv8n (nano)**

| Property | Value |
|----------|-------|
| Parameters | 3.2M |
| Input size | 640x640 |
| Speed (Jetson) | 60+ fps with TensorRT |
| Expected mAP | 0.90+ on bib dataset |

**Alternatives:**
- YOLOv8s: If nano accuracy insufficient
- YOLOv10n: Newer, NMS-free architecture
- RT-DETR: Transformer-based, higher accuracy but slower

### Training Configuration

```yaml
# bib_dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: bib  # Single class
```

```python
# Training hyperparameters
training_config = {
    'model': 'yolov8n.pt',
    'data': 'bib_dataset.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 20,  # Early stopping

    # Augmentation (critical for robustness)
    'augment': True,
    'hsv_h': 0.015,      # Hue variation
    'hsv_s': 0.7,        # Saturation (different bib colors)
    'hsv_v': 0.4,        # Brightness (lighting conditions)
    'degrees': 10,       # Rotation (runners lean)
    'translate': 0.1,    # Position shift
    'scale': 0.5,        # Size variation
    'shear': 5,          # Slight shear
    'perspective': 0.001,
    'flipud': 0.0,       # NO vertical flip
    'fliplr': 0.5,       # Horizontal flip OK
    'mosaic': 1.0,
    'mixup': 0.1,
    'copy_paste': 0.1,
}
```

### Additional Augmentation (Albumentations)

For motion blur and lighting conditions:

```python
import albumentations as A

augmentation_pipeline = A.Compose([
    # Motion blur (critical for running subjects)
    A.MotionBlur(blur_limit=(3, 15), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),

    # Lighting conditions
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.4
    ),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        p=0.2
    ),
    A.CLAHE(clip_limit=4.0, p=0.2),

    # Weather simulation
    A.RandomRain(p=0.1),
    A.RandomSunFlare(p=0.1),

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### Deployment

```python
# Export to TensorRT (Jetson)
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(
    format='engine',
    device=0,
    half=True,  # FP16 for speed
    imgsz=640,
    workspace=4,  # GB
    simplify=True,
)
```

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| mAP@0.5 | > 0.90 | On test set |
| Precision | > 0.92 | Low false positives |
| Recall | > 0.88 | Catch most bibs |
| Inference | < 10ms | TensorRT FP16 |

---

## Stage 2: Tracking

### Purpose

Track detected bibs across frames to:
1. Maintain consistent identity for temporal voting
2. Detect timing line crossings
3. Avoid duplicate counting

### Algorithm Options

| Algorithm | Speed | Accuracy | Complexity | Recommendation |
|-----------|-------|----------|------------|----------------|
| Centroid Tracking | <1ms | Good for sparse | Simple | Start here |
| SORT | ~1ms | Good | Moderate | If centroid fails |
| ByteTrack | ~2ms | Excellent | Moderate | Best balance |
| DeepSORT | ~10ms | Excellent | Complex | Overkill |

### Implementation (Centroid Tracking)

```python
from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = OrderedDict()  # id -> centroid
        self.disappeared = OrderedDict()  # id -> frames disappeared
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bboxes):
        """
        Update tracker with new detections.

        Args:
            bboxes: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            Dict of {object_id: centroid}
        """
        if len(bboxes) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroids of input bboxes
        input_centroids = np.zeros((len(bboxes), 2))
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            input_centroids[i] = ((x1 + x2) / 2, (y1 + y2) / 2)

        # If no existing objects, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Match using Hungarian algorithm (greedy approximation)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 100:  # Max distance threshold
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
```

### ByteTrack Integration

For better performance with occlusions:

```python
# Using ByteTrack library
from bytetrack import BYTETracker

tracker = BYTETracker(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    min_box_area=100,
)

# Update with detections
tracks = tracker.update(
    detections,  # (N, 5) array: x1, y1, x2, y2, conf
    img_size=(1080, 1920),
)

# tracks contains: track_id, bbox, score
```

---

## Stage 3: Number Recognition (OCR)

### Approach Comparison

| Approach | Accuracy | Speed | Complexity | When to Use |
|----------|----------|-------|------------|-------------|
| PaddleOCR (pretrained) | 70-85% | Fast | Low | Start here |
| PaddleOCR (fine-tuned) | 85-95% | Fast | Medium | If pretrained < 85% |
| Custom CRNN | 90-98% | Fast | High | If fonts very specific |
| EasyOCR | 70-85% | Medium | Low | Alternative baseline |

### Primary: PaddleOCR PP-OCRv4

**Architecture:**
- Text Detection: DB (Differentiable Binarization)
- Text Recognition: SVTR (Scene Visual Text Recognition)
- For bib crops: Skip detection, use recognition only

```python
from paddleocr import PaddleOCR

class BibOCR:
    def __init__(self, use_gpu=True):
        self.ocr = PaddleOCR(
            use_angle_cls=False,  # Bibs are roughly horizontal
            lang='en',
            det=False,  # Skip detection, we have crops
            rec=True,
            use_gpu=use_gpu,
            rec_algorithm='SVTR_LCNet',
            rec_model_dir=None,  # Use default or fine-tuned
        )

    def recognize(self, bib_crop):
        """
        Recognize number from cropped bib image.

        Args:
            bib_crop: BGR image of cropped bib

        Returns:
            (bib_number, confidence) or (None, 0.0)
        """
        result = self.ocr.ocr(bib_crop, det=False, cls=False)

        if result and result[0]:
            text, confidence = result[0][0]
            # Post-process: extract only digits
            bib_number = self._extract_number(text)
            return bib_number, confidence

        return None, 0.0

    def _extract_number(self, text):
        """Extract valid bib number from OCR text."""
        # Remove non-digits
        digits = ''.join(filter(str.isdigit, text))

        # Validate: 3-4 digits expected
        if 3 <= len(digits) <= 4:
            return digits
        elif len(digits) > 4:
            # Take first 4 digits (bib number might have extra text)
            return digits[:4]
        elif len(digits) > 0:
            # Partial read, return anyway
            return digits

        return None
```

### Color vs Grayscale Processing

Different stages of the pipeline have different requirements:

| Stage | Format | Reason |
|-------|--------|--------|
| **Bib Detection** | Color (BGR) | YOLO pretrained on color, bib colors aid detection |
| **OCR** | Grayscale | Text is contrast/shape problem, better enhancement |
| **Evidence Storage** | Color | Human review needs full context |

**Why grayscale for OCR:**
- Text recognition is fundamentally about contrast, not color
- Reduces input complexity (1 channel vs 3)
- CLAHE and other enhancements work better on single channel
- Handles problematic cases (red text on red background)
- Many OCR models convert internally anyway

**Why color for detection:**
- YOLOv8 pretrained on color images (ImageNet/COCO)
- Bibs often have distinctive colors (yellow, green, orange)
- Color contrast helps separate bib from clothing
- No accuracy gain from grayscale detection

```
Visual Example - Why Grayscale Helps OCR:

Original (red text on orange bib):     Grayscale + CLAHE:
┌─────────────────┐                    ┌─────────────────┐
│ ████████████████│                    │                 │
│ █ 1 2 3 4 █████│  ← Hard to read    │   1 2 3 4       │  ← Clear contrast
│ ████████████████│                    │                 │
└─────────────────┘                    └─────────────────┘
```

### Preprocessing (Critical for Accuracy)

```python
import cv2
import numpy as np

class BibPreprocessor:
    """
    Preprocess bib crops for optimal OCR performance.
    """

    def __init__(self, target_height=48):
        self.target_height = target_height  # PP-OCRv4 expects 48px height

    def preprocess(self, bib_crop, method='standard'):
        """
        Preprocess bib crop for OCR.

        Args:
            bib_crop: BGR image
            method: 'standard', 'high_contrast', 'binary'

        Returns:
            Preprocessed BGR image
        """
        if method == 'standard':
            return self._preprocess_standard(bib_crop)
        elif method == 'high_contrast':
            return self._preprocess_high_contrast(bib_crop)
        elif method == 'binary':
            return self._preprocess_binary(bib_crop)
        else:
            return bib_crop

    def _preprocess_standard(self, img):
        """
        Standard preprocessing pipeline.

        Key insight: Convert to grayscale for OCR.
        Text recognition is about contrast/shape, not color.
        Grayscale allows better contrast enhancement.
        """
        # 1. Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = self.target_height / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(img, (new_w, self.target_height))

        # 2. Convert to grayscale (IMPORTANT: improves OCR accuracy)
        #    - Removes color variations that don't carry text info
        #    - Enables better contrast enhancement
        #    - Handles problematic color combinations (red on orange, etc.)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #    - Works on grayscale only
        #    - Enhances local contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Slight sharpening to enhance text edges
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 5. Convert back to BGR (PaddleOCR expects 3-channel input)
        #    This is just format conversion, still grayscale visually
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _preprocess_high_contrast(self, img):
        """High contrast preprocessing for difficult cases."""
        h, w = img.shape[:2]
        scale = self.target_height / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(img, (new_w, self.target_height))

        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)

        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _preprocess_binary(self, img):
        """Binary threshold preprocessing."""
        h, w = img.shape[:2]
        scale = self.target_height / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(img, (new_w, self.target_height))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Otsu's binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def try_multiple(self, bib_crop, ocr_func):
        """
        Try multiple preprocessing methods, return best result.

        Tries grayscale methods first (usually better), then color fallback.

        Args:
            bib_crop: Original BGR image
            ocr_func: Function that takes image, returns (text, confidence)

        Returns:
            Best (text, confidence) result
        """
        # Grayscale methods (preferred)
        methods = ['standard', 'high_contrast', 'binary']
        results = []

        for method in methods:
            processed = self.preprocess(bib_crop, method)
            text, conf = ocr_func(processed)
            if text:
                results.append((text, conf, method))

        # Color fallback for edge cases (colored text that disappears in grayscale)
        if not results or max(r[1] for r in results) < 0.7:
            color_processed = self._preprocess_color_only(bib_crop)
            text, conf = ocr_func(color_processed)
            if text:
                results.append((text, conf, 'color'))

        if not results:
            return None, 0.0

        # Return result with highest confidence
        best = max(results, key=lambda x: x[1])
        return best[0], best[1]

    def _preprocess_color_only(self, img):
        """
        Color-preserving preprocessing (fallback for colored text).

        Use when grayscale fails - some bibs have colored numbers
        that lose contrast when converted to grayscale.
        """
        h, w = img.shape[:2]
        scale = self.target_height / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(img, (new_w, self.target_height))

        # Enhance in LAB color space (preserves color while improving contrast)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

### Alternative: Custom CRNN Model

For cases where PaddleOCR doesn't perform well:

```python
import torch
import torch.nn as nn

class BibNumberCRNN(nn.Module):
    """
    CRNN (CNN + RNN) architecture for bib number recognition.

    Uses CTC (Connectionist Temporal Classification) loss for sequence learning.
    """

    def __init__(self, img_height=32, img_width=128, num_classes=11):
        """
        Args:
            img_height: Input image height (will be resized)
            img_width: Input image width (will be resized)
            num_classes: 11 = digits 0-9 + CTC blank
        """
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        # CNN Feature Extractor
        # Input: (B, 1, 32, 128) -> Output: (B, 512, 1, 32)
        self.cnn = nn.Sequential(
            # Block 1: 32x128 -> 16x64
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x64 -> 8x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x32 -> 4x32 (pool height only)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4: 4x32 -> 2x32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5: 2x32 -> 1x32
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # RNN Sequence Modeling
        # Input: (B, 32, 512) -> Output: (B, 32, 512)
        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 1, H, W) grayscale images

        Returns:
            (B, T, num_classes) log probabilities for CTC
        """
        # CNN: (B, 1, 32, 128) -> (B, 512, 1, 32)
        conv = self.cnn(x)

        # Reshape: (B, 512, 1, 32) -> (B, 32, 512)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  # (B, 512, 32)
        conv = conv.permute(0, 2, 1)  # (B, 32, 512)

        # RNN: (B, 32, 512) -> (B, 32, 512)
        rnn_out, _ = self.rnn1(conv)
        rnn_out, _ = self.rnn2(rnn_out)

        # Output: (B, 32, num_classes)
        output = self.fc(rnn_out)

        return output


class CTCDecoder:
    """Decode CTC output to text."""

    def __init__(self, blank_idx=0):
        self.blank_idx = blank_idx
        self.idx_to_char = {i: str(i-1) for i in range(1, 11)}  # 1-10 -> '0'-'9'
        self.idx_to_char[0] = ''  # blank

    def decode(self, output):
        """
        Greedy CTC decoding.

        Args:
            output: (T, num_classes) log probabilities

        Returns:
            Decoded string
        """
        # Greedy: take argmax at each timestep
        indices = output.argmax(dim=-1)  # (T,)

        # Remove consecutive duplicates and blanks
        result = []
        prev_idx = None

        for idx in indices.tolist():
            if idx != self.blank_idx and idx != prev_idx:
                result.append(self.idx_to_char[idx])
            prev_idx = idx

        return ''.join(result)


# Training
def train_crnn(model, train_loader, optimizer, device):
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    for images, labels, label_lengths in train_loader:
        images = images.to(device)  # (B, 1, 32, 128)

        optimizer.zero_grad()

        output = model(images)  # (B, T, C)
        output = output.permute(1, 0, 2)  # (T, B, C) for CTC
        output = output.log_softmax(2)

        input_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long)

        loss = ctc_loss(output, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
```

---

## Stage 4: Temporal Voting

### Purpose

Since we're processing video, we can use multiple frames to improve accuracy through voting:

- Same bib seen across 5-10 frames
- OCR might vary slightly frame-to-frame
- Voting gives consensus answer with higher confidence

### Implementation

```python
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class TemporalVoting:
    """
    Combine OCR results across multiple frames for higher accuracy.
    """

    def __init__(self, window_size: int = 10, min_votes: int = 3):
        """
        Args:
            window_size: Number of recent frames to consider
            min_votes: Minimum votes required for confident result
        """
        self.window_size = window_size
        self.min_votes = min_votes
        self.history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

    def update(self, track_id: int, ocr_result: Optional[str], confidence: float):
        """
        Add OCR result for a tracked bib.

        Args:
            track_id: Unique ID from tracker
            ocr_result: Recognized number (or None)
            confidence: OCR confidence score
        """
        if ocr_result is not None:
            self.history[track_id].append((ocr_result, confidence))

        # Keep only recent results
        self.history[track_id] = self.history[track_id][-self.window_size:]

    def get_consensus(self, track_id: int) -> Tuple[Optional[str], float]:
        """
        Get consensus number from multiple frames.

        Args:
            track_id: Unique ID from tracker

        Returns:
            (bib_number, combined_confidence) or (None, 0.0)
        """
        if track_id not in self.history:
            return None, 0.0

        results = self.history[track_id]

        if len(results) < self.min_votes:
            return None, 0.0

        # Count votes for each number
        votes: Dict[str, Dict] = {}
        for number, conf in results:
            if number not in votes:
                votes[number] = {'count': 0, 'total_conf': 0.0, 'confs': []}
            votes[number]['count'] += 1
            votes[number]['total_conf'] += conf
            votes[number]['confs'].append(conf)

        if not votes:
            return None, 0.0

        # Get number with most votes
        best_number = max(votes.keys(), key=lambda k: (votes[k]['count'], votes[k]['total_conf']))
        best_data = votes[best_number]

        # Calculate combined confidence
        vote_ratio = best_data['count'] / len(results)
        avg_conf = best_data['total_conf'] / best_data['count']
        max_conf = max(best_data['confs'])

        # Combined confidence formula
        # Weight: 50% vote ratio, 30% avg confidence, 20% max confidence
        combined_conf = 0.5 * vote_ratio + 0.3 * avg_conf + 0.2 * max_conf

        # Only return if we have enough agreement
        if vote_ratio >= 0.5:  # At least 50% agreement
            return best_number, combined_conf

        return None, 0.0

    def clear_track(self, track_id: int):
        """Remove history for a track (after crossing event)."""
        if track_id in self.history:
            del self.history[track_id]

    def get_all_consensus(self) -> Dict[int, Tuple[Optional[str], float]]:
        """Get consensus for all tracked bibs."""
        return {
            track_id: self.get_consensus(track_id)
            for track_id in self.history.keys()
        }
```

### Confidence Levels

| Vote Ratio | Avg Confidence | Combined | Interpretation |
|------------|----------------|----------|----------------|
| 1.0 | 0.95 | 0.97 | HIGH - Very reliable |
| 0.8 | 0.85 | 0.82 | MEDIUM - Likely correct |
| 0.6 | 0.70 | 0.61 | LOW - May need review |
| < 0.5 | any | 0 | UNKNOWN - Insufficient agreement |

---

## Stage 5: Crossing Detection

### Timing Line Representation

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TimingLine:
    """
    Virtual timing line in frame coordinates.

    Defined as a line segment from (x1, y1) to (x2, y2).
    Coordinates are normalized (0.0 to 1.0).
    """
    x1: float  # Start X (0-1)
    y1: float  # Start Y (0-1)
    x2: float  # End X (0-1)
    y2: float  # End Y (0-1)

    def to_pixels(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (
            int(self.x1 * frame_width),
            int(self.y1 * frame_height),
            int(self.x2 * frame_width),
            int(self.y2 * frame_height),
        )
```

### Crossing Detection Algorithm

```python
import numpy as np
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CrossingEvent:
    track_id: int
    bib_number: Optional[str]
    confidence: float
    timestamp: datetime
    position_in_frame: Tuple[float, float]  # Normalized centroid


class CrossingDetector:
    """
    Detect when tracked bibs cross the timing line.
    """

    def __init__(self, timing_line: TimingLine):
        self.timing_line = timing_line
        self.previous_positions: Dict[int, Tuple[float, float]] = {}

    def check_crossing(
        self,
        track_id: int,
        centroid: Tuple[float, float],  # Normalized (0-1)
        bib_number: Optional[str],
        confidence: float,
        timestamp: datetime,
    ) -> Optional[CrossingEvent]:
        """
        Check if a bib has crossed the timing line.

        Uses line segment intersection to detect crossing.

        Returns:
            CrossingEvent if crossed, None otherwise
        """
        if track_id not in self.previous_positions:
            self.previous_positions[track_id] = centroid
            return None

        prev_pos = self.previous_positions[track_id]
        curr_pos = centroid

        # Check if movement segment intersects timing line
        if self._segments_intersect(prev_pos, curr_pos):
            # Crossing detected!
            event = CrossingEvent(
                track_id=track_id,
                bib_number=bib_number,
                confidence=confidence,
                timestamp=timestamp,
                position_in_frame=curr_pos,
            )

            # Update position (but mark as crossed)
            self.previous_positions[track_id] = curr_pos

            return event

        # Update position
        self.previous_positions[track_id] = curr_pos
        return None

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> bool:
        """
        Check if line segment p1->p2 intersects the timing line.

        Uses cross product method for line intersection.
        """
        # Timing line endpoints
        l1 = (self.timing_line.x1, self.timing_line.y1)
        l2 = (self.timing_line.x2, self.timing_line.y2)

        # Cross product helper
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Check intersection
        return (
            ccw(p1, l1, l2) != ccw(p2, l1, l2) and
            ccw(p1, p2, l1) != ccw(p1, p2, l2)
        )

    def clear_track(self, track_id: int):
        """Remove track from position history."""
        if track_id in self.previous_positions:
            del self.previous_positions[track_id]
```

---

## Complete Pipeline Integration

```python
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class FrameResult:
    crossing_events: List[CrossingEvent]
    active_tracks: Dict[int, Tuple[str, float]]  # track_id -> (number, confidence)


class RecognitionPipeline:
    """
    Complete recognition pipeline integrating all stages.
    """

    def __init__(
        self,
        detector_path: str,
        timing_line: TimingLine,
        confidence_threshold: float = 0.7,
    ):
        # Stage 1: Detection
        self.detector = YOLO(detector_path)

        # Stage 2: Tracking
        self.tracker = CentroidTracker(max_disappeared=30)

        # Stage 3: OCR
        self.preprocessor = BibPreprocessor(target_height=48)
        self.ocr = BibOCR(use_gpu=True)

        # Stage 4: Temporal Voting
        self.voting = TemporalVoting(window_size=10, min_votes=3)

        # Stage 5: Crossing Detection
        self.crossing_detector = CrossingDetector(timing_line)

        self.confidence_threshold = confidence_threshold

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime,
    ) -> FrameResult:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: BGR image (H, W, 3)
            timestamp: GPS-synchronized timestamp

        Returns:
            FrameResult with crossing events and active tracks
        """
        h, w = frame.shape[:2]
        crossing_events = []

        # Stage 1: Detect bibs
        detections = self.detector(frame, verbose=False)[0]
        bboxes = []

        for det in detections.boxes:
            if det.conf >= 0.5:  # Detection confidence threshold
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                bboxes.append((x1, y1, x2, y2))

        # Stage 2: Track bibs
        tracked = self.tracker.update(bboxes)

        # Stage 3 & 4: OCR and voting for each track
        for track_id, centroid in tracked.items():
            # Find bbox for this track (simplified - in practice, track includes bbox)
            bbox = self._find_bbox_for_centroid(centroid, bboxes)
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Crop bib
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Preprocess and OCR
            processed = self.preprocessor.preprocess(crop)
            ocr_result, ocr_conf = self.ocr.recognize(processed)

            # Update voting
            self.voting.update(track_id, ocr_result, ocr_conf)

            # Get consensus
            consensus_number, consensus_conf = self.voting.get_consensus(track_id)

            # Stage 5: Check for crossing
            normalized_centroid = (centroid[0] / w, centroid[1] / h)

            crossing = self.crossing_detector.check_crossing(
                track_id=track_id,
                centroid=normalized_centroid,
                bib_number=consensus_number,
                confidence=consensus_conf,
                timestamp=timestamp,
            )

            if crossing:
                # Determine final bib number
                if crossing.confidence >= self.confidence_threshold:
                    final_number = crossing.bib_number
                else:
                    final_number = "UNKNOWN"

                crossing.bib_number = final_number
                crossing_events.append(crossing)

                # Clear this track's voting history
                self.voting.clear_track(track_id)
                self.crossing_detector.clear_track(track_id)

        # Get all active tracks with consensus
        active_tracks = self.voting.get_all_consensus()

        return FrameResult(
            crossing_events=crossing_events,
            active_tracks=active_tracks,
        )

    def _find_bbox_for_centroid(self, centroid, bboxes, max_dist=50):
        """Find the bbox that contains or is nearest to centroid."""
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = ((cx - centroid[0])**2 + (cy - centroid[1])**2)**0.5
            if dist < max_dist:
                return bbox
        return None
```

---

## Performance Targets

### Jetson Orin Nano

| Stage | Target | Method |
|-------|--------|--------|
| Detection | <10ms | TensorRT FP16 |
| Tracking | <2ms | Centroid/ByteTrack |
| OCR (per bib) | <15ms | PaddleOCR Lite |
| Voting | <1ms | Simple Python |
| Crossing | <1ms | Simple Python |
| **Total** | <30ms/frame | **30+ fps** |

### Accuracy Targets

| Stage | Metric | Target |
|-------|--------|--------|
| Detection | mAP@0.5 | > 0.90 |
| OCR (per frame) | Accuracy | > 85% |
| Voting (consensus) | Accuracy | > 90% |
| Position ordering | Accuracy | > 99% |

---

## Testing Strategy

### Unit Tests

1. Detection accuracy on test set
2. OCR accuracy on cropped bibs
3. Tracking consistency
4. Crossing detection correctness

### Integration Tests

1. End-to-end on recorded video
2. Multiple simultaneous bibs
3. Fast-moving subjects
4. Poor lighting conditions

### Regression Tests

1. Known failure cases
2. Edge cases (partial bibs, unusual fonts)
3. Performance benchmarks

---

## Troubleshooting Guide

### Low Detection Accuracy

1. Check training data quality
2. Add more augmentation (blur, brightness)
3. Increase model size (YOLOv8s)
4. Collect more training data

### Low OCR Accuracy

1. Improve preprocessing
2. Try multiple preprocessing methods
3. Fine-tune PaddleOCR on your fonts
4. Train custom CRNN model

### Tracking Issues

1. Increase max_disappeared threshold
2. Switch to ByteTrack
3. Adjust matching distance threshold
4. Check frame rate consistency

### Missed Crossings

1. Check timing line calibration
2. Verify centroid tracking
3. Adjust crossing detection sensitivity
4. Check timestamp synchronization
