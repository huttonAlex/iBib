# OCR Improvement Plan

## Goal
Achieve 95%+ bib number recognition accuracy for the PointCam race timing system.

## Current State (2026-02-04)
- **Bib Detection**: 97.3% mAP50 (YOLOv8 Nano) - Excellent, production-ready
- **OCR Recognition**: 52.7% success rate (EasyOCR) - **Bottleneck**
- **Training Data**: 10,853 verified bib crops with ground truth numbers (4 events) - **Ready for Phase 2**

## Root Cause Analysis

OCR failures occur due to:
1. **Small crops** - Distant bibs have insufficient pixel resolution
2. **Header text confusion** - OCR reads "HOME RUN 5K" instead of number
3. **Angle/perspective** - Tilted bibs distort numbers
4. **General-purpose OCR** - EasyOCR optimized for documents, not race bibs

## Solution: Fine-Tuned Scene Text Recognition

### Approach: Fine-Tune, Not Build From Scratch
- Building from scratch requires 50,000+ samples; we have 10,853
- Fine-tuning a pretrained model (TrOCR or PARSeq) needs hundreds to low thousands
- Pretrained models already understand digit shapes, fonts, perspective
- We teach them "this is what bib numbers look like"

### Target Pipeline
```
Image → YOLO (detect bib) → Crop + Preprocess → Fine-tuned model (read number) → Validation
```

This replaces EasyOCR entirely with a specialized model.

### Candidate Models
- **TrOCR** (Microsoft) - Vision Transformer encoder + text decoder, strong on scene text
- **PARSeq** - Permutation-based autoregressive model, state-of-the-art STR benchmark
- **CRNN + CTC** - Classic approach, lighter weight, good for Jetson deployment

### Training Strategy
1. Freeze pretrained backbone (image encoder)
2. Fine-tune decoder/head layers on bib crop → number pairs
3. Augment training data (rotation, brightness, blur, noise)
4. Validate on held-out set from each event

## Data Status

### Available Now
| Source | Images | Verified Crops | Bib Range | Cameras | Event |
|--------|--------|---------------|-----------|---------|-------|
| Batch 1 | 296 | 457 | 1-4 digit | 1 | NY Giants Foundation 5K |
| Batch 2 | 749 | 1,108 | 1-4 digit | 1 | NY Giants Foundation 5K |
| Event 88679 | 998 | 828 | 2-500 | 1 (photo) | Tagged photos (scoring provider) |
| Event 86638 | 4,000 | 5,901 | 1000-3999 | 3 (fincam, fincam2, splitcam) | Tagged photos (scoring provider) |
| Event 89536 | 2,000 | 2,559 | 2-2501 | 1 (finishphoto) | Tagged photos (scoring provider) |
| **Total** | **8,043** | **10,853** | **1-4 digits** | **4 events** |

Data locations:
- `data/unlabeled_batch1/ocr_training/` - crops + `labels.tsv`
- `data/unlabeled_batch2/ocr_training/` - crops + `labels.tsv`
- `data/tagged_event_88679/ocr_training/` - crops + `labels.tsv`
- `data/tagged_event_86638/ocr_training/` - crops + `labels.tsv`
- `data/tagged_event_89536/ocr_training/` - crops + `labels.tsv`

### Data Diversity (Achieved)

Data now spans 4 events with:
- Multiple bib designs, fonts, and colors
- Different lighting conditions
- Multiple camera angles and distances (6 camera positions total)
- Full digit range coverage (1-digit through 4-digit bib numbers)

### Original Data Collection Targets (Met)

| Priority | What | Status |
|----------|------|--------|
| **P0** | 4+ events, different bib designs | Done (4 events) |
| **P1** | Different lighting (sun, shade, evening) | Done (mixed across events) |
| **P2** | Different camera angles/distances | Done (6 camera positions) |
| **P3** | Events with tagged data for auto-validation | Done (3 events via scoring provider) |

**Original target: 3,000+ verified crops from 4+ events. Achieved: 10,853 from 4 events.**

### Adding New Data (Workflows)

**For unlabeled photos (no bib tags):**
```bash
# 1. Process new images (detects bibs, extracts crops, runs initial OCR)
python scripts/process_unlabeled.py --source /path/to/new/photos --output data/unlabeled_batchN

# 2. Review and correct in browser UI
python scripts/review_ui.py data/unlabeled_batchN

# 3. Generate training data from verified reviews
python scripts/process_unlabeled.py --generate-training data/unlabeled_batchN
```

**For tagged photos from scoring provider (CSV with image URLs + bib numbers):**
```bash
# 1. Download images, detect bibs, OCR, auto-validate against tags (~67% auto-verify rate)
python scripts/process_tagged_photos.py --csv /path/to/export.csv --output data/tagged_eventN

# 2. Review only the flagged items (auto-verified items are skipped)
python scripts/review_ui.py data/tagged_eventN

# 3. Generate training data
python scripts/process_unlabeled.py --generate-training data/tagged_eventN
```

The tagged photo workflow auto-verifies ~67% of crops by matching OCR results against the
scoring provider's bib tags. The review UI shows expected bib numbers for flagged items,
and pre-fills the expected bib when only one is tagged for that image.

### Scoring Provider Integration

CSV export format (from scoring provider):
```
camera_tag,image_url,event_image_id,associated_bibs
fincam,https://s3.amazonaws.com/.../IMG_xxx.jpg,12345,1234
```

- Images hosted on public S3, downloaded automatically
- `associated_bibs` can contain multiple bibs (newline-separated in quoted field)
- Auto-validation: if OCR reads a number matching a tagged bib, crop is auto-verified
- Scales to hundreds of thousands of photos across many events

## Implementation Phases (Revised)

### Phase 1: Data Collection (COMPLETE)
**Status: 10,853 verified crops from 4 events (target was 3,000)**

Tasks:
- [x] Build auto-labeling pipeline (`scripts/process_unlabeled.py`)
- [x] Build review UI (`scripts/review_ui.py`)
- [x] Build tagged photo pipeline (`scripts/process_tagged_photos.py`)
- [x] Process and verify NY Giants Foundation 5K (1,565 crops)
- [x] Process and verify Event 88679 via scoring provider (828 crops)
- [x] Process and verify Event 86638 via scoring provider (5,901 crops)
- [x] Process and verify Event 89536 via scoring provider (2,559 crops)
- [ ] Combine all verified data into unified training set
- [ ] Create train/val/test split (stratified by event)

### Phase 2: Fine-Tune Digit Recognition Model
**Target: 90%+ accuracy**
**Prerequisite: 3,000+ diverse verified crops (ACHIEVED: 10,853)**

Tasks:
- [ ] Evaluate TrOCR vs PARSeq vs CRNN on our data
- [ ] Fine-tune best candidate on bib crops
- [ ] Augmentation: rotation (+-15 deg), brightness, blur, noise, perspective warp
- [ ] Evaluate per-event accuracy (must generalize across events)
- [ ] Benchmark inference speed on target hardware

### Phase 3: Pipeline Integration & Optimization
**Target: 95%+ accuracy, <200ms per bib**

Tasks:
- [ ] Replace EasyOCR with fine-tuned model in pipeline
- [ ] Add preprocessing: upscale small crops, contrast normalization
- [ ] Add validation: digit count check, confidence thresholding
- [ ] Optimize for Jetson (TensorRT export) and Pi (ONNX Runtime)
- [ ] Temporal consistency: same bib across multiple frames → vote

### Phase 4: Continuous Improvement
**Target: 97%+ accuracy**

Tasks:
- [ ] Deploy and collect real-world failure cases
- [ ] Add hard examples to training set
- [ ] Periodic retraining with expanded data
- [ ] A/B test against EasyOCR baseline

## Reference Systems

This approach mirrors proven real-world systems:

1. **License Plate Recognition (LPR)**: 97-99% accuracy
   - Two-stage detection (plate → characters)
   - Custom digit classifiers
   - Format validation

2. **Sports Jersey Recognition**: 90-95% accuracy
   - Region detection + specialized OCR
   - Temporal consistency across frames

## Success Metrics

- **Primary**: OCR accuracy on held-out test set (target: 95%+)
- **Secondary**: Per-event accuracy (must be >90% on unseen events)
- **Tertiary**: Processing speed (target: <200ms per bib on Jetson)
- **Quaternary**: Confidence calibration (high confidence = high accuracy)

## Hardware Constraints

- All models must run on Jetson Orin Nano (8GB) with TensorRT
- CPU fallback required for Raspberry Pi 5 (ONNX Runtime)
- Evidence images saved for failed/low-confidence reads
