# Technical Notes

This document captures technical findings, experiment results, and lessons learned during development.

## Purpose

- Document experimental results
- Track library/technology evaluations
- Record performance measurements
- Capture field test observations
- Note gotchas and lessons learned

## Format

Entries are organized by phase and date. Each entry should include:
- **Date**: When the finding was made
- **Phase/Milestone**: Context within project
- **Topic**: What was investigated
- **Findings**: What was learned
- **Implications**: How this affects the project
- **References**: Links, papers, code samples

---

## Phase 1: Proof of Concept

### Milestone 1.1: Dataset & Research

### Entry 2026-02-03: Bib Detection Training Results

**Phase/Milestone**: 1.1 - Dataset & Research

**Objective**:
Train a bib detector and evaluate OCR accuracy on detected crops.

**Method**:
- Trained YOLOv8 Nano on 127 annotated images (COCO format converted to YOLO)
- 80/20 train/val split
- 50 epochs, imgsz=640, batch=16
- OCR via EasyOCR with digit-only allowlist and multiple preprocessing passes

**Findings**:
- **Detection**: 97.3% mAP50, 79.1% mAP50-95 (excellent)
- **OCR (baseline EasyOCR)**: 42.9% on 10-image sample
- **OCR (with preprocessing)**: 52.7% on 45-image validation set
- Preprocessing improvements: CLAHE contrast enhancement, adaptive thresholding, upscaling small crops
- Failures concentrated on: small/distant bibs, angled bibs, header text confusion

**Implications for Project**:
- Detection is production-ready; OCR is the bottleneck
- General-purpose OCR (EasyOCR) insufficient for bib numbers
- Need specialized digit recognition model fine-tuned on bib crops

### Entry 2026-02-03: Unlabeled Data Processing & Annotation

**Phase/Milestone**: 1.1 - Dataset & Research

**Objective**:
Build a large labeled dataset of bib crops for training a custom digit recognizer.

**Method**:
- Processed 1,045 unlabeled race photos from one event (NY Giants Foundation 5K)
- Auto-detected bibs with trained YOLOv8 model (conf > 0.5)
- Ran EasyOCR on crops to generate initial predictions
- Built web-based review UI for manual verification/correction
- Human reviewer verified each crop and corrected OCR errors

**Findings**:
- 2,112 bib crops extracted from 1,045 images
- 1,565 verified correct (bib crop + ground truth number)
- 545 rejected (false detections, unreadable, or not bibs)
- OCR pre-fill was correct ~60-70% of the time, speeding review
- All data from single event: same bib design, font, lighting, camera

**Data Location**:
- Batch 1: `data/unlabeled_batch1/` (457 verified from 296 images)
- Batch 2: `data/unlabeled_batch2/` (1,108 verified from 749 images)
- OCR training pairs: `*/ocr_training/labels.tsv` (crop filename + number)
- YOLO annotations: `*/yolo_annotations/` (auto-generated)

**Implications for Project**:
- 1,565 labeled crops is sufficient to fine-tune a pretrained model
- Single-event data risks overfitting; need 2-3 more events for robustness
- Review workflow (detect -> OCR predict -> human verify) scales well to new data

### Entry 2026-02-04: Scoring Provider Tagged Photo Processing

**Phase/Milestone**: 1.1 - Dataset & Research

**Objective**:
Scale up training data collection using pre-tagged photos from scoring provider CSV exports,
achieving diversity across multiple events.

**Method**:
- Built `scripts/process_tagged_photos.py` to process scoring provider CSV exports
- CSV format: `camera_tag, image_url, event_image_id, associated_bibs`
- Images downloaded from public S3 in parallel (8 threads)
- YOLO detection + EasyOCR on each image, then auto-validation against tagged bibs
- Auto-verify: if OCR result exactly matches a tagged bib, mark as verified automatically
- Flagged items reviewed manually via `scripts/review_ui.py` (updated to show expected bibs)
- Processed 3 events: 88679 (998 images), 86638 (4,000 images), 89536 (2,000 images)

**Findings**:
- **Event 88679**: 967 crops, 828 verified, 139 rejected. Bibs 2-500, 1 camera.
- **Event 86638**: 7,290 crops, 5,901 verified, 1,389 rejected. Bibs 1000-3999, 3 cameras.
- **Event 89536**: 2,924 crops, 2,559 verified, 363 rejected. Bibs 2-2501, 1 camera.
- Auto-verify rate: ~63-67% across all events (OCR matched tagged bib)
- Combined with NY Giants 5K data: **10,853 total verified crops from 4 events**
- Running two processing jobs in parallel halved per-job throughput due to CPU contention

**Data Location**:
- `data/tagged_event_88679/` - 828 verified OCR training samples, 731 YOLO annotations
- `data/tagged_event_86638/` - 5,901 verified OCR training samples, 3,533 YOLO annotations
- `data/tagged_event_89536/` - 2,559 verified OCR training samples, 1,581 YOLO annotations

**Implications for Project**:
- Phase 1 data collection target (3,000 crops from 4+ events) exceeded by 3.6x
- Scoring provider integration enables rapid dataset expansion for future events
- Data diversity achieved: 4 events, 6 camera positions, 1-4 digit bib ranges
- Ready to proceed to Phase 2 (model fine-tuning)

---

### Milestone 1.2: Static Image Processing

### Entry 2026-02-04: Phase 2.1 OCR Fine-Tuning Pipeline Setup

**Phase/Milestone**: 1.2 - Static Image Processing / OCR Model Evaluation

**Objective**:
Build the tooling infrastructure for evaluating and fine-tuning specialized OCR models
on the 10,853 verified bib crop dataset, targeting 90%+ accuracy (up from 52.7% EasyOCR baseline).

**Method**:
- Created 4 scripts for the full evaluate-train-export workflow
- Added `ocr-eval` dependency group to `pyproject.toml` (torch, transformers, etc.)
- Designed stratified dataset split ensuring all events in every split
- Selected 3 candidate models spanning the accuracy-vs-speed tradeoff space

**Scripts Created**:
| Script | Purpose |
|--------|---------|
| `scripts/prepare_ocr_dataset.py` | Consolidate 5 sources into unified train/val/test with augmentation |
| `scripts/evaluate_ocr_models.py` | Baseline + fine-tuned model evaluation with full metrics |
| `scripts/finetune_ocr.py` | Fine-tune CRNN, PARSeq, TrOCR with model-specific strategies |
| `scripts/export_ocr_model.py` | ONNX export with accuracy verification and speed benchmarking |

**Key Design Choices**:
- **Stratified split within each event**: Prevents train/test leakage and ensures
  generalization is measurable. Every event appears in train, val, and test.
- **Three model tiers**: CRNN (8.3M, edge-friendly), PARSeq (23.8M, SOTA accuracy),
  TrOCR (62M, strong pretrained). All three fine-tuned since CPU training time is not a constraint.
- **Online augmentation**: Applied during training (not pre-generated) to maximize
  effective diversity. Pipeline: rotation, motion blur, brightness/contrast, perspective,
  noise, downscale.
- **Minority oversampling**: 1-digit bibs (50 samples) get 10x augmentation, 2-digit 5x,
  3-digit 2x. 4-digit bibs (76.7% of data) get no extra copies.
- **Common model interface**: `predict(image) -> (str, float)` allows uniform evaluation
  and easy model swapping in the pipeline.
- **ONNX export**: Opset 17 with dynamic batch axis, accuracy parity check (within 1%),
  speed benchmarking. Targets RPi 5 (ONNX Runtime) and Jetson (TensorRT via ONNX).

**Execution Plan**:
1. Install deps: `pip install -e .[ocr-eval]`
2. Prepare dataset: `python scripts/prepare_ocr_dataset.py`
3. Baseline eval: `python scripts/evaluate_ocr_models.py --models trocr parseq`
4. Fine-tune: `python scripts/finetune_ocr.py --model {crnn,parseq,trocr}`
5. Re-evaluate fine-tuned checkpoints
6. Export winner to ONNX

**Implications for Project**:
- Phase 2.1 tooling is complete; ready to execute the evaluation/training pipeline
- EasyOCR and PaddleOCR will be replaced by the fine-tuned model in Phase 3
- CRNN is the most deployment-friendly candidate (smallest, grayscale input, fastest)
  but PARSeq may win on accuracy
- CPU-only training means fine-tuning will be slow but feasible for all three models

---

### Milestone 1.3: Timing Line Detection

### Entry 2026-02-13: YOLOv8n-Pose Crossing Detection + Bib-Time Association

**Phase/Milestone**: 1.3 - Timing Line Detection

**Objective**:
Replace MOG2 background subtraction with real person detection for timing-line
crossing, and implement persistent bib-person association so bibs detected in
earlier frames follow the person to the crossing moment.

**Background**:
MOG2 blob detection on REC-0011 produced 1,199 crossings vs ~800-900 actual
finishers.  Over 50% were merged blobs — two runners side-by-side appeared as
one large contour.  Blob centroids also had no anatomical correspondence,
making the "crossing moment" imprecise.

**Method**:
- Added `PoseDetector` wrapping YOLOv8n-pose (6MB weights, auto-downloaded)
- Chest point = average of visible torso keypoints (COCO: left\_shoulder=5,
  right\_shoulder=6, left\_hip=11, right\_hip=12). Fallback: bbox centroid
- `CentroidTracker.update()` extended with optional `centroids` param so person
  tracker matches on chest points instead of bbox centers
- `PersistentPersonBibAssociator` accumulates person→bib votes over time:
  each frame spatially matches person bboxes to bib bboxes, looks up OCR
  consensus, casts a vote.  `get_bib(pid)` returns the majority winner.
- `BibCrossingDeduplicator` suppresses same-bib duplicates within 10s window
  (UNKNOWN always passes)
- `CrossingEvent` gains `chest_point` and `source` fields; CSV log updated

**Architecture**:
```
Frame → YOLO bib detector (GPU) → bib tracks + OCR → final_consensus
Frame → YOLOv8n-pose (GPU)     → person tracks (chest keypoint)
               ↓
   PersistentPersonBibAssociator: person_track → bib_number (voted)
               ↓
   CrossingDetector: chest keypoint crosses timing line
               ↓
   BibCrossingDeduplicator: suppress same-bib duplicates
               ↓
   Emit CrossingEvent(bib_number, timestamp, confidence)
```

**Expected Performance** (Jetson Orin Nano):

| Operation | Cost | Type |
|-----------|------|------|
| YOLO bib detection | ~8-12ms | GPU |
| OCR per crop | ~5-15ms | GPU/CPU |
| YOLOv8n-pose | ~25-33ms | GPU |
| Association + crossing | <1ms | CPU |
| **Total** | **~40-60ms (16-25fps)** | |

**Test Coverage**:
- 41 tests (17 new): PoseDetector chest computation (5), PersistentPersonBibAssociator
  voting/cleanup (5), BibCrossingDeduplicator (5), CentroidTracker custom centroids (2)
- Deprecated classes (BackgroundSubtractorManager, PersonBibAssociator, MergedBlobEstimator)
  emit DeprecationWarning — verified by tests

**Implications for Project**:
- Ready for validation on Jetson with REC-0011 video
- Expected significantly fewer false crossings (no merged blobs)
- Expected more bib-identified crossings (persistent association vs single-frame)
- Bib-level dedup should eliminate all duplicate bib entries in output CSV
- GPU shared between two YOLO models — monitor memory on Jetson (~25MB total)

---

### Milestone 1.4: Basic Video Processing

*Awaiting first entries*

---

## Phase 2: Real-World Testing

*Future entries*

---

## Phase 3: Backup System Deployment

*Future entries*

---

## Phase 4: Production Hardening

*Future entries*

---

## Template for New Entries

```markdown
### Entry YYYY-MM-DD: [Topic]

**Phase/Milestone**: X.Y - Name
**Investigator**: Name (if applicable)

**Objective**:
What we were trying to find out

**Method**:
How we investigated (test setup, approach)

**Findings**:
- Finding 1
- Finding 2
- Data/metrics

**Analysis**:
What the findings mean

**Implications for Project**:
How this affects our approach, decisions, or roadmap

**Recommendations**:
What should be done based on this

**References**:
- Links
- Papers
- Code samples

**Attachments**:
- Path to test data
- Screenshots
- Result files
```

---

## Quick Notes Section

Use this section for quick observations that don't warrant a full entry yet.

**2026-02-01**:
- Project documentation structure created
- Awaiting Phase 1.1 dataset collection to begin technical work

---

## Known Issues / Watchlist

Track things that need investigation or monitoring:

*None yet*

---

## Useful Code Snippets

Track reusable code patterns discovered during development:

*None yet*

---

## Performance Baseline

Track performance measurements across phases:

### Phase 1 Baselines
*To be populated during Phase 1.2*

### Phase 2 Baselines
*To be populated during Phase 2*

### Phase 3+ Baselines
*To be populated during Phase 3+*

---

## Experiment Log

| ID | Date | Phase | Topic | Result | Notes |
|----|------|-------|-------|--------|-------|
| E001 | 2026-02-03 | 1.1 | YOLOv8 bib detection | 97.3% mAP50 | 127 training images |
| E002 | 2026-02-03 | 1.1 | EasyOCR baseline | 42.9% accuracy | No preprocessing |
| E003 | 2026-02-03 | 1.1 | EasyOCR + preprocessing | 52.7% accuracy | CLAHE, threshold, upscale |
| E004 | 2026-02-03 | 1.1 | Unlabeled data annotation | 1,565 verified crops | Single event, review UI |
| E005 | 2026-02-04 | 1.1 | Tagged photo processing | 9,288 verified crops | 3 events via scoring provider |
| E006 | 2026-02-04 | 1.1 | Combined dataset | 10,853 total verified | 4 events, 6 cameras, 1-4 digit bibs |
| E007 | 2026-02-04 | 1.2 | Phase 2.1 pipeline setup | Tooling complete | 4 scripts, 3 candidate models |
| E008 | 2026-02-13 | 1.3 | YOLOv8n-pose crossing detection | Implementation complete | Replaces MOG2, 41 tests passing |
