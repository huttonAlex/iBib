# Photo Tagging Service Plan

Automatically detect bib numbers in race photos and tag them to participant results, so participants see personalized photos on their results page.

## Current State

- Race photos uploaded to per-race R2 buckets (bucket name contains race ID)
- Results service: Python FastAPI + Postgres, Docker containers on VPS
- Bib detection: YOLOv8n (nano, 3M params) — 97.8% mAP50, trained on 5,662 images
- OCR: Fine-tuned PARSeq (23.8M params) — 97.4% exact match on bib crops
- Fallback OCR: Fine-tuned CRNN (8.3M params) — 91.7% exact match, faster
- Post-OCR cleanup, bib set validation, confidence classification exist in `pointcam`
- **No photo-specific inference path exists yet** — pipeline.py is video-oriented

## Scope

This plan covers the **models and processing pipeline only** — getting the highest quality bib tagging from single photos. VPS integration, review UI, and access control are out of scope.

## Architecture Overview

```
R2 bucket (per race)
         │
    photo bytes
         │
         v
  Photo Tagger Pipeline
  ┌─────────────────────────────────┐
  │ Detect (YOLOv8m @ 1280px)       │
  │         │                       │
  │    N bounding boxes             │
  │         │                       │
  │ For each bbox:                  │
  │   Crop → OCR (PARSeq) → Clean  │
  │         │                       │
  │   Validate against bib roster   │
  │         │                       │
  │ Multi-crop ensemble (optional)  │
  └─────────────────────────────────┘
         │
  list of {bib, confidence, bbox}
```

**GPU compute**: Modal (serverless, per-second billing, scale-to-zero). See DD-023.

---

## Model Strategy: Cloud vs Edge

The Jetson pipeline is constrained to YOLOv8n + PARSeq ONNX for real-time 30fps video. The photo tagging service has no real-time requirement — seconds per photo is fine. This unlocks larger, more accurate models.

### Detection: YOLOv8n → YOLOv8m (or YOLOv8l)

| Model | Params | GFLOPs | mAP50 (COCO) | Inference T4 @640 | Inference T4 @1280 |
|---|---|---|---|---|---|
| YOLOv8n (current) | 3.2M | 8.1 | 37.3 | ~2ms | ~8ms |
| YOLOv8s | 11.2M | 28.6 | 44.9 | ~4ms | ~15ms |
| YOLOv8m | 25.9M | 78.9 | 50.2 | ~8ms | ~30ms |
| YOLOv8l | 43.7M | 165.2 | 52.9 | ~14ms | ~50ms |

**Recommendation: Train YOLOv8m on the existing 5,662-image dataset.**

- 3.4x more parameters than YOLOv8n, significantly better at small/distant objects
- On a T4 at 1280px input, inference is ~30ms — trivial for batch photo processing
- YOLOv8l gives diminishing returns (+2.7 mAP for 2x the compute)
- The existing training data and pipeline (`scripts/train_detector.py`) works as-is, just change the base weights

**Higher input resolution is equally important.** Running at 1280px instead of 640px dramatically improves small bib detection. In race photos, runners can be far from the camera — small bibs are the primary failure mode. On the Jetson this would kill FPS; on a T4 it adds ~20ms.

### OCR: Keep Fine-Tuned PARSeq

PARSeq is already at **97.4% exact match** with excellent confidence calibration (0.990 correct vs 0.624 wrong). There's little room to improve the model itself. The wins come from:

1. **Better crops** — larger detector finds more bibs, higher-res input means sharper crops
2. **Multi-crop ensemble** — for each detected bib, generate 2-3 crop variants (different padding ratios, slight perspective corrections) and take the highest-confidence OCR result
3. **No need for CRNN fallback** — on cloud GPU, PARSeq latency doesn't matter. CRNN's main advantage was speed (33ms vs 124ms CPU), irrelevant here

### What About Larger OCR Models?

| Option | Accuracy | Cost | Verdict |
|---|---|---|---|
| PARSeq (current, fine-tuned) | 97.4% | Trained, ready | **Use this** |
| PARSeq-large / ViTSTR-large | ~98-99% (estimated) | Need to find/train, marginal gain | Not worth it yet |
| TrOCR-small (62M params) | Not benchmarked (failed to evaluate) | Would need fine-tuning | Skip — PARSeq is better studied in this codebase |
| GPT-4V / vision LLM | ~99%+ | API cost, latency, overkill | Unnecessary |

**Verdict**: PARSeq is the right OCR model. The accuracy ceiling is the detection step, not OCR. Invest in a better detector and higher-res input first. If OCR becomes the bottleneck after that, revisit.

---

## Software Changes Required

### 1. Extract Shared Utilities from Pipeline

**Problem**: `PostOCRCleanup`, `BibSetValidator`, `CropQualityFilter`, and crop extraction logic are embedded in `pipeline.py`'s 1100-line `process_frames()`.

**Approach**: Create `src/pointcam/ocr_utils.py` with the portable classes. Don't refactor `pipeline.py` — just extract copies that both can use.

| Component | Extract? | Notes |
|---|---|---|
| `PostOCRCleanup` | Yes | Stateless text transforms, no dependencies |
| `BibSetValidator` | Yes | Takes bib list, does exact + fuzzy matching |
| `CropQualityFilter` | Yes | Single-crop quality check |
| Crop extraction + padding | Yes | Geometry math, placement-aware |
| `UltralyticsBibDetector` | Yes | Clean class, just wraps ultralytics |

### 2. Train YOLOv8m Detector

Retrain the bib detector with YOLOv8m base weights on the existing dataset.

- Use `scripts/train_detector.py` with `yolov8m.pt` instead of `yolov8n.pt`
- Same dataset (5,662 images, COCO format)
- Same augmentation pipeline
- Export to ONNX for Modal deployment
- Compare mAP50 against v2 nano baseline (97.8%)
- Test at both 640px and 1280px input resolution

### 3. New Module: `photo_tagger.py`

Single-image bib detection + OCR pipeline.

```python
class PhotoTagger:
    def __init__(self, detector_path, ocr_path, bib_list=None, input_size=1280):
        self.detector = UltralyticsBibDetector(detector_path, conf=0.25)
        self.ocr = OnnxParseqOCR(ocr_path)  # or PyTorch on GPU
        self.cleanup = PostOCRCleanup()
        self.validator = BibSetValidator(bib_list) if bib_list else None

    def tag(self, image_bytes: bytes) -> list[PhotoTag]:
        """Detect all bibs in a single photo, return tags."""
        ...

@dataclass
class PhotoTag:
    bib_number: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in original image coords
    validated: bool                   # True if bib is in the race roster
    ocr_raw: str                     # pre-cleanup OCR text
```

**Key design points**:
- Accepts raw image bytes (JPEG/PNG from R2)
- Returns all detected bibs with confidence and location
- Optionally validates against a bib roster
- No side effects — pure function, easy to test
- Configurable input resolution (1280 default for photos)

### 4. Multi-Crop Ensemble (Accuracy Boost)

For each detected bib bbox, generate multiple crop variants and take the best OCR result:

1. **Standard crop** — bbox + 10% padding
2. **Tight crop** — bbox + 5% padding (less background noise)
3. **Wide crop** — bbox + 20% padding (catches cut-off digits)

Run PARSeq on all 3, take the result with highest confidence. This compensates for the lack of temporal voting in the video pipeline.

**Expected impact**: +1-3% accuracy on borderline cases where padding affects OCR. Low cost — 3x OCR calls per bib is still <10ms on GPU.

### 5. Validation Script (Phase 0)

Before building the full pipeline, validate that existing models work on race photos:

```bash
python scripts/test_photo_tagger.py \
    --photos /path/to/race/photos/ \
    --detector models/bib_detector_v2.pt \
    --ocr models/ocr_parseq.onnx \
    --bib-list bibs.txt \
    --output results.csv
```

Outputs per-photo results: detected bibs, confidence, bbox. Manual review against the actual photos to measure precision and recall.

## Repo Organization

```
src/
  pointcam/
    inference.py          # OCR models (no changes needed)
    pipeline.py           # video pipeline (no changes needed)
    recognition.py        # existing
    crossing.py           # existing
    photo_tagger.py       # NEW: single-image detect + OCR pipeline
    ocr_utils.py          # NEW: extracted PostOCRCleanup, BibSetValidator, etc.

scripts/
    train_detector.py     # existing (use for YOLOv8m training)
    test_photo_tagger.py  # NEW: validate models on race photos

services/
  photo-tagger/
    modal_app.py          # NEW: Modal worker (thin wrapper around PhotoTagger)
    requirements.txt      # deps for Modal container
```

## Confidence Strategy

Without temporal voting, confidence thresholds must be stricter:

| Confidence | Bib Valid? | Action |
|---|---|---|
| >= 0.85 | Yes (exact roster match) | Auto-tag |
| >= 0.85 | No (not in roster) | Discard — likely OCR error |
| 0.60 - 0.85 | Yes (exact or fuzzy match) | Tag with `needs_review` flag |
| < 0.60 | Any | Discard |

PARSeq's calibration is a major asset here — wrong predictions average 0.624 confidence vs 0.990 for correct. The 0.85 threshold should cleanly separate them.

## Phasing

### Phase 0: Validate Models on Race Photos
- Collect 50-100 actual DSLR race photos from a past event
- Run current YOLOv8n + PARSeq on them (quick script)
- Measure: how many bibs detected, OCR accuracy on detected bibs
- Identify failure modes (small bibs? angle? occlusion? lighting?)
- **Exit gate**: Baseline accuracy numbers on photos, list of failure modes

### Phase 1: Train YOLOv8m Detector
- Retrain on existing dataset with `yolov8m.pt` base weights
- Evaluate at 640px and 1280px input
- Compare mAP50 and recall against YOLOv8n baseline
- Export to ONNX
- **Exit gate**: YOLOv8m trained and showing improvement over nano

### Phase 2: Build Photo Tagger Module
- Extract shared utilities into `ocr_utils.py`
- Build `photo_tagger.py` with `PhotoTagger` class
- Add multi-crop ensemble
- Unit tests with real race photo samples
- **Exit gate**: `PhotoTagger.tag()` works reliably on test photos

### Phase 3: Modal Deployment
- Create `modal_app.py` wrapping PhotoTagger
- Bake model weights into Modal image
- Test end-to-end: photo bytes in → tag list out
- Benchmark throughput
- **Exit gate**: Modal endpoint processing photos correctly

### Phase 4: Accuracy Iteration
- Run against a full race's photos, score against known results
- Tune confidence thresholds
- Consider test-time augmentation (horizontal flip, slight rotation) if accuracy needs improvement
- Document precision/recall numbers
- **Exit gate**: Meets quality bar for production tagging
