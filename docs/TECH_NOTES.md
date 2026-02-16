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

### Entry 2026-02-13: Pipeline Tuning — UNKNOWN Rate & Duplicate Bibs (Run 1)

**Phase/Milestone**: 1.3 - Pipeline Integration

**Objective**:
Reduce 60% UNKNOWN rate and 31 duplicate bibs seen on REC-0011-A (30min 5K, 1121 bibs, CRNN OCR, Jetson Orin Nano).

**Root Causes Identified**:
1. **Bib detector miss (73% of UNKNOWNs)**: People too small/distant. Camera fix needed.
2. **Association failure (27% of UNKNOWNs)**: Bibs detected near person but not linked. ~24 had stable detections (clear bug).
3. **Broken confidence system**: `auto_accept_validated` forced any valid bib to 0.90 (HIGH). Voting stability forced `max(conf, 0.85)` (HIGH). Result: 100% of detections were `final_level=high` — confidence levels useless.
4. **Duplicate bibs**: OCR reads partial numbers (e.g. "65" from "1265"). Since "65" is valid (in 1-1678 range), it passes validation, gets boosted to HIGH, and the 10s dedup window can't catch occurrences minutes apart.

**Changes Made (commit 8a62e6d)**:
1. `ConfidenceManager.classify()`: replaced floor boosts (`max(conf, 0.90)`) with additive boosts (`+0.15`, `+0.10`)
2. `PersistentPersonBibAssociator`: weighted votes (HIGH=3, MEDIUM=2, LOW=1), min_votes=3, min_confidence=0.4, short-bib penalty (halved weight for fewer digits than expected)
3. `BibCrossingDeduplicator`: escalating confidence (1st=any, 2nd>=0.7, 3rd+>=0.9)
4. Last-chance bib lookup: scan tracked bibs inside person bbox at crossing time

**Results (Run 1: min_votes=3)**:

| Metric | Before | After | Change |
|---|---|---|---|
| Total crossings | 600 | 564 | -36 |
| UNKNOWN | 360 (60%) | 394 (70%) | +34 (worse) |
| Identified | 240 (40%) | 170 (30%) | -70 (worse) |
| Unique bibs | 161 | 145 | -16 |
| Duplicate bibs (>1) | 31 | 19 | -12 (better) |
| Worst duplicate | 65: 16x | 1115: 4x | much better |

Confidence distribution (after fix — was 100% HIGH before):
- HIGH: 519, MEDIUM: 209, LOW: 217, REJECT: 153

Identified crossing confidence: min=0.40, median=0.84, mean=0.80.

**Analysis**:
- Duplicates greatly improved: 31→19, worst case 16x→4x. Escalating confidence works.
- UNKNOWN rate got **worse** (60%→70%): `min_votes=3` is too aggressive. With weighted voting (HIGH=3 per frame), a single high-confidence frame already gives 3 weighted votes, making the threshold redundant for good reads but blocking marginal ones.
- The confidence system now actually produces meaningful levels (was 100% HIGH, now distributed).
- Net: traded 70 identifications for 12 fewer duplicates — bad tradeoff.

**Next Step**: Lower `min_votes` from 3 → 1 and rerun. The weighted voting and escalating dedup already handle quality filtering; the min_votes threshold is over-filtering.

### Entry 2026-02-13: Pipeline Tuning — Run 2 (min_votes=1)

**Phase/Milestone**: 1.3 - Pipeline Integration

**Objective**:
Test whether lowering `min_votes` from 3 → 1 recovers the lost identifications from Run 1.

**Changes (commit 2d812f5)**:
- `PersistentPersonBibAssociator` default `min_votes`: 3 → 1
- Pipeline script: explicit `min_votes=1`

**Results (Run 2: min_votes=1)**:

| Metric | Before | Run 1 (mv=3) | Run 2 (mv=1) | Run 2 vs Before |
|---|---|---|---|---|
| Total crossings | 600 | 564 | 563 | -37 |
| UNKNOWN | 360 (60%) | 394 (70%) | 388 (69%) | +28 (worse) |
| Identified | 240 (40%) | 170 (30%) | 175 (31%) | -65 (worse) |
| Unique bibs | 161 | 145 | 147 | -14 |
| Duplicates (>1) | 31 | 19 | 20 | -11 (better) |
| Worst dup | 65: 16x | 1115: 4x | 15: 5x | much better |

**Analysis**:
- Lowering `min_votes` was marginal: only +5 identifications (170→175).
- The identification drop is NOT caused by `min_votes`. The bottleneck is elsewhere — likely the confidence changes (additive boosts instead of floors) and/or escalating dedup suppressing real bibs.
- Duplicate fix remains solid: 31→20 with worst case 16x→5x.
- Need deeper investigation to find which specific change is dropping identifications.

### Entry 2026-02-13: Root Cause Analysis — Identification Drop (240→175)

**Phase/Milestone**: 1.3 - Pipeline Integration

**Objective**:
Understand why identified crossings dropped from 240→175 despite only targeting duplicates.

**Method**:
Time-matched crossings between old and new runs (±5s window). Cross-referenced with detection CSVs to trace where bibs disappeared.

**Findings**:

1. **Broken confidence floor was the primary inflator in the old run.**
   Old run: 100% of tracks (1098/1098) classified as `final_level=high`. New run: HIGH=519, MEDIUM=209, LOW=217, REJECT=153. The old code had no real tiering — everything was forced to HIGH.

2. **57% of "lost" crossings were phantom duplicates.**
   Of 150 lost crossings matched by time, 85 came from bibs with multiple crossings in the old run (65: 16x, 15: 10x, 35: 9x, 45: 8x). These are clearly OCR fragment misreads that got accepted due to the broken confidence floor.

3. **63 of 93 "lost" bib numbers don't appear in new detections at all.**
   Numbers like 25, 35, 55, 51 — suspiciously common 2-digit OCR fragments (e.g. "35" from "1350"). The old floor (conf=1.0) accepted them; the new code correctly rejects them.

4. **Old conf=1.0 floor directly responsible for 33 lost crossings.**
   50/240 old identified crossings had confidence exactly 1.0. Of those, 33 were lost. Old confidence range: 0.129-1.0. New range: 0.400-1.0.

5. **~30 bibs are still detected but not identified at crossing time.**
   Bibs like 65, 15, 45 have thousands of detection rows with high OCR confidence, but fail to get assigned to a person at crossing. Possible causes: short-bib penalty reducing their vote weight, or these are OCR misreads on other people's bibs that are correctly filtered.

**Analysis**:
The drop from 240→175 is **mostly correct behavior**. The old 240 was inflated by:
- Broken confidence floor accepting everything as HIGH
- No tiering, so garbage OCR reads passed
- Massive duplicate crossings (65 crossing 16x)

Adjusted for duplicates, the old run had ~161 unique identified bibs → new has 147. True net loss: ~14 unique bibs, not 65.

**Implications for Project**:
- The confidence and dedup fixes are working as intended
- The ~30 bibs detected but not crossing as identified may benefit from tuning the short-bib penalty or the last-chance lookup
- Camera angle improvement remains the highest-impact change for UNKNOWN rate (73% of UNKNOWNs are detector misses)

---

### Entry 2026-02-14: False Positive Sources & Maturity Assessment

**Phase/Milestone**: 1.3 - Pipeline Integration

**Objective**:
Assess false positive risks from non-bib number sources (jersey numbers, finish line clocks) ahead of Giants 5K test video, and document realistic accuracy expectations across project maturity phases.

**False Positive Sources Identified**:

| Source | Risk Level | Existing Defense | Gap |
|--------|-----------|-----------------|-----|
| Jersey numbers (1-2 digit) | High | BibSetValidator, SuspiciousPredictionFilter (≥0.85 for short), digit count | On chest like bibs, persistent, reinforced by voting if matches a valid bib |
| Finish line clock | Medium | PersistentPersonBibAssociator (no person match) | Runner passing in front of clock may cause brief spatial match |
| Sponsor banners / signage | Low | Person association, aspect ratio | Static, rarely matched to person |
| Spectator clothing | Low | Detection zone, person tracking | Spectators rarely in detection zone |

**Jersey Number Analysis**:
- Giants 5K: many runners wearing NFL jerseys with 1-2 digit numbers
- Jersey numbers are on the chest — same location as bibs, same spatial match
- They move with the runner — not filtered by static region suppression
- Temporal voting actually reinforces them over time
- Key defenses: (1) bib set validation rejects numbers not in registration list, (2) digit count filtering penalizes 1-2 digit reads, (3) bib detection model should learn paper bibs vs jersey printing if trained properly
- Remaining risk: jersey number that happens to match a valid bib number

**Planned Mitigations**:
- Static region suppression (automatic, no UI needed) — heatmap of detection locations, suppress persistent stationary detections (clocks, signs)
- ROI exclusion zones (deferred to Phase 3 UI) — operator masks known false positive areas
- Negative training examples — include jerseys, clocks, signage in bib detector training as negatives

**Accuracy Expectations by Phase**:

| Phase | Conditions | Expected Read Rate | Key Limiting Factor |
|-------|-----------|-------------------|-------------------|
| Phase 1 (PoC, recorded video) | Controlled, good footage | 80-85% | OCR model quality |
| Phase 2 (on-device, live) | Real-time constraints | 75-85% | Inference speed tradeoffs |
| Phase 3 (field, backup) | Real race conditions | 70-85% | Occlusion in packs, lighting, bib damage |
| Phase 4+ (hardened) | All conditions | 80-90% | Fundamental camera limitations |

**Camera vs RFID — Realistic Role Assessment**:

Camera-based detection has fundamental limitations that prevent full RFID replacement for large/competitive races:
- **Occlusion**: Runners in packs hide each other's bibs. RFID reads through bodies.
- **Bib damage**: Folded, covered by jackets, wet, pinned wrong. RFID chips are embedded.
- **Lighting**: Dawn/dusk starts, backlighting, shadows. RFID is light-independent.
- **Weather**: Rain degrades image quality. RFID is weather-independent.

Camera advantages RFID lacks:
- **No per-participant cost** (no chips/bibs)
- **Visual evidence** of crossing
- **Position ordering** visible even without bib reads
- **Any location** — point a camera anywhere, no antenna infrastructure
- **Finish photo equivalent** — evidence store provides automatic finish photos

**Projected maturity trajectory**:
1. **Near-term**: Backup to RFID — catches missed reads, provides visual evidence
2. **Medium-term**: Co-primary — camera + RFID combined read rate higher than either alone
3. **Possible**: Primary for small/casual races (<500 runners) where RFID cost isn't justified
4. **Unlikely**: Full RFID replacement for large/competitive races

**Network Architecture Decision**:
Race timers connect laptops via cellular hotspot to cloud-based scoring. RFID readers have independent 4G connections to ChronoTrack. No local network switch exists at the finish line. Jetson joins the operator's hotspot — both devices on the same network, zero workflow change. See DD-020.

**Next Steps**:
- Run Giants 5K test video to measure actual jersey/clock false positive rates
- Implement static region suppression if clock false positives appear
- Assess whether bib detector model needs negative training examples for jerseys

### Entry 2026-02-14: Giants 5K Full Video Run — Performance Fixes & Ground Truth Comparison (Run 3)

**Phase/Milestone**: 1.4 - Basic Video Processing

**Objective**:
Run the full 30min Giants 5K video (REC-0006-A, 54,002 frames) on Jetson Orin Nano Super with performance fixes, validate against ground truth finish order (480 finishers).

**Performance Fixes Applied (commit b7dff26)**:
1. **Batch PARSeq OCR**: Single GPU forward pass for all crops per frame (was N individual calls)
2. **Drawing guard**: Skip all cv2 annotation calls when `--no-video` and not showing
3. **Frame stride option**: `--stride N` to process every Nth frame (not used this run)
4. **Dead track pruning**: Clear voting/consensus dicts for deregistered tracks each frame — fixes unbounded memory growth that caused silent crash at ~22k frames

**Pipeline Configuration**:
- `--no-video --bib-set data/race_bibs_5k.txt --timing-line 0.52,0.0,0.52,1.0 --crossing-direction left_to_right --placement right`
- PARSeq OCR on GPU (CUDA), YOLOv8n bib detector, YOLOv8n-pose person detector

**Performance Results**:

| Metric | Run 2 (REC-0011) | Run 3 (REC-0006) |
|--------|-------------------|-------------------|
| Frames processed | ~22,000 (crash) | **54,002 (100%)** |
| Total runtime | ~22min (crash) | ~45min |
| Avg throughput | ~17 fps | ~20 fps overall |
| Sparse sections | — | **36.5 fps** |
| Dense sections | — | ~11 fps |
| Crash at ~22k frames | Yes | **No** |
| Avg detection time | — | 31.3 ms/frame |
| Avg OCR time | — | 37.6 ms/detection |
| Avg pose time | — | 37.2 ms/frame |

The crash fix works. Throughput exceeds 30 fps in sparse sections but drops to ~11 fps during dense packs (more bibs = more OCR crops per frame). Batch OCR helps but doesn't fully overcome the serial YOLO+PARSeq+Pose pipeline.

**Detection Results**:

| Metric | Value |
|--------|-------|
| Total detections (OCR calls) | 71,493 |
| Unique bib tracks | 1,679 |
| Edge-rejected crops | 22,163 |
| Quality-rejected crops | 13,790 |
| Partial/obstructed rejected | 1,777 |
| OCR calls saved by filters | 37,730 |
| OCR cleanups applied | 19,896 |
| HIGH confidence tracks | 407 |
| MEDIUM confidence tracks | 65 |
| LOW confidence tracks | 132 |
| REJECT tracks | 749 |

**Crossing Detection Results**:

| Metric | Value |
|--------|-------|
| Total crossings detected | 207 |
| With bib identified | 106 (51%) |
| UNKNOWN (no bib) | 101 (49%) |
| Dedup suppressed | 46 |
| Person tracks created | 2,612 |

**Ground Truth Comparison** (480 finishers in bib_order.txt):

| Metric | Value |
|--------|-------|
| **Crossing recall** | 207/480 = **43%** detected |
| **Bib recall** | 37/480 = **7.7%** correctly identified |
| **Precision** | 47/106 = **44%** of identified bibs are real |
| **False positive bibs** | 54 unique bib numbers not in ground truth |
| **Order accuracy (LCS)** | 9/37 = **24%** of correct bibs in right order |

**Deep-Dive: Why Recall is Low**

*Analysis of first 50 ground truth finishers*:
- 31/50 (62%) were detected by OCR at some point in the detections CSV
- 11/50 (22%) made it to a crossing record
- 20/50 (40%) were detected by OCR but never had a crossing event — **crossing detection is the gap, not OCR**
- 19/50 (38%) were never detected at all — bib detector misses

*Multi-crossing tracks*:
- 179 distinct person tracks produced 207 crossings
- 15 tracks crossed more than once (28 extra crossings)
- Worst: track 223 crossed **10 times** over 158 seconds, all UNKNOWN — person lingering near timing line
- Tracks 335, 430, 460, 505, 554 also had repeated crossings

**Root Cause Analysis — Five Failure Modes**:

1. **Persons oscillating at timing line** (inflates crossing count):
   Track 223 has chest_x ≈ 0.52 (exactly on the timing line) across 10 crossings spanning 158 seconds. Small jitter in chest keypoint detection causes the side-of-line to flip repeatedly. Each flip after the 2s debounce fires a new crossing. This single person accounts for 10/101 UNKNOWN crossings.

2. **UNKNOWN crossings never deduplicated** (inflates UNKNOWN count):
   `BibCrossingDeduplicator.should_emit()` has `if bib_number == "UNKNOWN": return True` — UNKNOWNs always pass. Track 223's 10 UNKNOWN crossings are all emitted. Per-track debounce only prevents same track within 2 seconds.

3. **Bib detector misses** (38% of GT bibs never detected):
   19 of the first 50 GT bibs never appear in the detections CSV at all. These runners' bibs were never detected by YOLO — too small, occluded, or bib folded/covered.

4. **Crossing detector misses detected bibs** (40% detected but no crossing):
   20 of the first 50 GT bibs have OCR detections but no crossing event. The person tracker failed to link the chest keypoint crossing to these bib observations — association failure or person track fragmentation.

5. **OCR misreads create false positives** (54 phantom bibs):
   Short OCR fragments (e.g., "65" from "1265", "39" from "1039") happen to be valid bib numbers. They pass validation, accumulate votes, and get emitted as crossings. Evidence: bibs 1, 2, 6 appear as the most common "detections" with 35, 13, and 13 tracks respectively — these are almost certainly single-digit OCR fragments.

**UNKNOWN Rate by Time Segment**:

| Segment | Crossings | Identified | ID Rate |
|---------|-----------|------------|---------|
| 0-10 min | 38 | 13 | 34% |
| 10-20 min | 100 | 64 | **64%** |
| 20-30 min | 69 | 29 | 42% |

Best identification rate in the densest section (10-20 min) where more bibs are visible for longer. Early (fast leaders) and late (spread out) runners have lower rates.

**Analysis**:
The pipeline successfully processes the full 30-min video without crashing, confirming the memory leak fix. However, accuracy is far below usable levels. The primary bottleneck is **crossing detection**, not OCR — 62% of bibs are seen by OCR at some point but only 22% make it to a crossing event. The secondary issue is **crossing inflation** from lingering tracks.

**Implications for Project**:
- Performance fixes validated (full video, no crash, 36+ fps in sparse sections)
- Accuracy requires fundamental crossing detection improvements (see DD-022)
- The 43% crossing recall means the pipeline misses more than half of finishers — unacceptable even for a backup system
- False positive rate (56% of identifications wrong) would corrupt race results
- Priority order for improvement: (1) crossing hysteresis, (2) UNKNOWN dedup by track, (3) bib-person association, (4) false positive filtering

**Attachments**:
- Ground truth: `bib_order.txt` (480 finishers in order)
- Crossings: `runs/pipeline_test/REC-0006-A_crossings.csv`
- Detections: `runs/pipeline_test/REC-0006-A_detections.csv`

### Entry 2026-02-15: Real-Time Inference Optimization (TensorRT + OCR Skip)

**Phase/Milestone**: 1.4 - Basic Video Processing

**Objective**:
Optimize pipeline from ~5 fps (serial PyTorch, single-threaded) toward ~30 fps target for real-time operation on Jetson Orin Nano.

**Background**:
Run 3 showed ~20 fps overall (36 fps sparse, 11 fps dense) with three serial PyTorch/ONNX models: bib detection (31ms), pose detection (37ms), PARSeq OCR (38ms/batch). TensorRT 10.3.0 is installed on the Jetson. `tegrastats` showed CPU clocks at 729 MHz — far below the ~1.5 GHz max.

**Optimization Phases Implemented**:

**Phase A: TensorRT Model Exports**

| Model | Backend | Export Method | Expected Speedup |
|-------|---------|---------------|-----------------|
| YOLOv8n bib detector | ultralytics `.engine` | `YOLO.export(format="engine", half=True)` | 31ms → ~8-10ms |
| YOLOv8n-pose | ultralytics `.engine` | `YOLO.export(format="engine", half=True)` | 37ms → ~10-12ms |
| PARSeq OCR | ONNX RT + TensorRT EP | Runtime compilation, cached | 38ms → ~12-15ms |
| CRNN OCR | `trtexec` → `.engine` | Direct conversion (simple CTC arch) | ~17ms → ~5-8ms |

- YOLOv8 models: trivial — ultralytics auto-detects `.engine` extension, zero code changes to `crossing.py:PoseDetector` or pipeline detector loading
- PARSeq: autoregressive decoder has dynamic control flow incompatible with static TRT engines. ONNX Runtime TRT EP compiles compatible subgraphs to TensorRT, falls back to CUDA EP for the rest. Engine cache built on first run, reused thereafter.
- CRNN: simple CNN+RNN+CTC pipeline, direct `trtexec` conversion works

Files:
- `scripts/export_tensorrt.py` — run on Jetson to build device-specific engines
- `src/pointcam/inference.py` — `OnnxTensorRTParseqOCR` and `TensorRTCrnnOCR` classes
- `scripts/test_video_pipeline.py` — `--ocr-backend tensorrt` flag

**Phase B: Conditional OCR Skip for Stable Tracks**

After quality filter, before OCR, check if a track already has a stable high-confidence consensus:
- `is_stable` requires 5 consecutive identical reads (`EnhancedTemporalVoting`)
- Consensus confidence >= 0.85
- Track must already be in `final_consensus` dict
- If all conditions met, skip OCR for this crop (save ~12-38ms per crop)
- If track loses stability (different number appears), `is_stable` goes False and OCR resumes

Expected: 30-50% fewer OCR calls in dense sections where 3-8 bibs are visible and 40-60% are already stable.

Flags: `--no-ocr-skip` disables for benchmarking/regression testing.

**Phase C: System Tuning**

`sudo jetson_clocks` — locks CPU/GPU/EMC to max frequencies. Must run after every boot.
```bash
# One-time (per boot):
sudo jetson_clocks

# Verify with:
sudo jetson_clocks --show
# CPU should show ~1.5GHz (was 729MHz), GPU should show max freq

# To make persistent across reboots:
sudo systemctl enable jetson_clocks
```

Frame stride (`--stride 2`) already implemented. At 30fps source, processes 15fps input. With hysteresis_frames=3, a crossing needs 3 processed frames on new side = 6 real frames (200ms) — well within ±1s timing tolerance.

Drawing already gated by `if write_video or show:` — `--no-video` without `--show` skips all annotation.

**Expected Combined Performance**:

| Configuration | Est. FPS | Limiting Factor |
|--------------|----------|----------------|
| Baseline (PyTorch, stride 1) | ~5 fps | Serial GPU inference |
| + TensorRT (all models) | ~12-15 fps | GPU total ~30-37ms |
| + OCR skip (stable tracks) | ~15-20 fps | Fewer OCR calls in dense |
| + stride 2 | ~25-33 fps | Half the frames to process |
| + jetson_clocks | ~28-36 fps | CPU overhead reduced |

**Verification Plan**:
1. Export on Jetson: `python scripts/export_tensorrt.py`
2. Accuracy parity: compare crossing CSVs between PyTorch and TensorRT on same clip
3. Performance: compare fps with `--start-time 460` on REC-0006-A.mp4
4. OCR skip regression: compare `--no-ocr-skip` vs default, verify identical crossings
5. End-to-end: full run with all optimizations vs ground truth `bib_order.txt`

**Deferred — Phase D: Threaded Pipeline**:
On a single GPU, neural network inference is inherently serial (shared CUDA stream). Threading benefit is limited to overlapping I/O with GPU work. Only worth implementing if Phases A-C don't reach ~25-30 fps target.

### Entry 2026-02-15: Accuracy Root-Cause Analysis & Improvement Roadmap

**Phase/Milestone**: 1.4 - Basic Video Processing

**Objective**:
Understand why end-to-end bib accuracy is 7.1% (target: ≥95%) and map out a prioritized path to close the gap.

**Context**:
Runs 3-4 on REC-0006-A (Giants 5K, 480 finishers, head-on camera at finish line) consistently show ~34 correct bibs out of 480 (7.1% recall) and 32-44% precision. Camera placement is directly facing the timing line at bib height — this is good placement. The problem is software.

**The Accuracy Chain**:

Accuracy is the product of three serial stages. All three must be near-perfect to reach 95%:

```
Detection (YOLO)  ×  Association (person↔bib)  ×  OCR+Emission  =  End-to-End
   62%            ×         ~60%               ×     ~19%       =     7.1%

Target:  ≥98%     ×        ≥98%               ×     ≥99%       =    ≥95%
```

**Five Failure Modes (from analysis of first 50 GT finishers)**:

| # | Failure Mode | Impact | Cause | Software Fix? |
|---|-------------|--------|-------|--------------|
| 1 | YOLO never detects bib | 38% of GT bibs lost | Small/distant/occluded bibs, conf threshold too high | Yes — lower threshold, larger model, more training data |
| 2 | Detected bib not associated at crossing | 40% of detected bibs lost | 4-second association memory too short, spatial distance threshold too tight | Yes — extend memory window, widen distance |
| 3 | Oscillation at timing line | Inflates UNKNOWN count 5-10x per lingering track | 3-frame hysteresis too lenient for keypoint jitter | Yes — increase hysteresis |
| 4 | UNKNOWN dedup broken | All UNKNOWN crossings emitted | Per-track 2s debounce only, no location-based dedup | Yes — add spatial UNKNOWN dedup |
| 5 | OCR fragments create false positives | 54 phantom bibs in Run 3 | "65" from "1265" is valid in bib range 1-1678 | Yes — stricter fragment rejection |

**Improvement Roadmap** (prioritized by impact):

**Tier 1 — Association & Crossing Architecture (biggest gap, software-only)**

These fix the 40% of bibs that ARE detected but never make it to a crossing event:

1. Extend association memory: 4s → 30s. Bibs detected early must persist until person crosses.
2. Increase crossing hysteresis: 3 → 8-10 frames. Eliminates oscillation from keypoint jitter.
3. Add spatial UNKNOWN dedup: suppress duplicate UNKNOWNs from same location within 10s.
4. Widen association distance: 250px → 300-350px. Handles larger persons / wider camera angles.

**Tier 2 — Detection Recall (38% of bibs never seen by YOLO)**

Camera placement is good, so this is a model/threshold problem:

5. Lower YOLO confidence threshold: 0.5 → 0.25-0.3. Accept more detections, let OCR + voting filter noise.
6. Evaluate YOLOv8s (small) vs current YOLOv8n (nano). 3x more parameters, better small-object recall.
7. More training data: current 127 images is small. Add video frame extractions from REC-0006-A as training data (with GT bibs as labels).
8. Multi-scale inference: run YOLO at multiple resolutions and merge detections (trades speed for recall).

**Tier 3 — OCR & False Positive Reduction**

9. Stricter fragment rejection: require higher confidence for 1-2 digit bibs (0.95 for 1-digit, 0.90 for 2-digit).
10. Dynamic short-bib penalty: only penalize low-confidence short reads, not confident ones.
11. OCR model improvement: current PARSeq fine-tuned on 10K crops is decent (~70% on video crops) — more training data from video frames could help.

**Expected Impact (cumulative)**:

| After | Est. Recall | Est. Precision | Key Change |
|-------|------------|---------------|------------|
| Current | 7.1% | 33-44% | Baseline |
| Tier 1 (association fixes) | 20-35% | 50-60% | Recover detected-but-unassociated bibs |
| Tier 2 (detection recall) | 50-70% | 60-75% | See bibs that YOLO currently misses |
| Tier 3 (OCR + FP reduction) | 55-75% | 80-90% | Reduce false positives |
| Full pipeline maturity | 80-90% | 90-95% | Iteration, more data, tuning |
| 95% target | ≥95% | ≥95% | May require architectural changes (see below) |

**Open Questions for 95%**:

Reaching 95% may require more than tuning — potential architectural changes:
- **Re-identification**: if a bib is read at any point, bind it permanently to a person's appearance embedding (not just spatial proximity). Person re-ID survives occlusion and track fragmentation.
- **Multi-camera**: side-angle camera catches bibs that head-on misses (turned torso, pack occlusion).
- **Retroactive association**: after a crossing, search back through the full detection history for the best bib match to that person track, not just the real-time voting window.
- **Higher resolution**: 4K camera at finish line significantly increases bib readability at distance.

**Implications for Project**:
- Speed optimization (TensorRT, stride) was premature — accuracy is the blocker, not speed
- Tier 1 fixes are pure software, low effort, high impact — should be next sprint
- Tier 2 fixes require model retraining but are well-understood
- The 95% target is ambitious but achievable with systematic improvements
- Current video (head-on at bib height) is good test data — failures are software, not camera

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

### Evaluation Scorecard

Standard metrics for comparing pipeline runs. Every benchmark run should report all of these.

**Test configuration** (record for each run):
- Video file, start time, frame range
- Hardware (Jetson Orin Nano Super, CPU clock, GPU clock)
- Model backends (PyTorch .pt / TensorRT .engine / ONNX)
- OCR model + backend (PARSeq/CRNN, pytorch/tensorrt)
- Flags: stride, OCR skip, video writing, etc.

#### Performance Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Wall clock (s)** | Total processing time from first to last frame | — |
| **Throughput (fps)** | Frames processed / wall clock seconds | ≥30 fps |
| **Avg detection time (ms)** | YOLO bib detector per-frame latency | ≤10 ms |
| **Avg pose time (ms)** | YOLOv8n-pose per-frame latency | ≤12 ms |
| **Avg OCR time (ms)** | OCR per-detection latency | ≤15 ms |
| **OCR calls** | Total OCR forward passes | lower = better |
| **OCR skip count** | OCR calls avoided by stable-track skip | higher = better |
| **OCR skip rate** | skip_count / (skip_count + OCR calls) | 30-50% target |

#### Accuracy Metrics (vs ground truth)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Crossing recall** | Crossings detected / GT finishers | ≥95% |
| **Bib recall** | Correct unique bibs / GT finishers | ≥95% |
| **Precision** | Correct bibs / all unique bibs detected | ≥95% |
| **False positive bibs** | Unique bibs not in GT | ≤5 |
| **UNKNOWN rate** | UNKNOWN crossings / total crossings | ≤10% |
| **Duplicate bibs** | Bibs appearing in >1 crossing | ≤3 |

#### Secondary Metrics

| Metric | Definition | Notes |
|--------|-----------|-------|
| **Identified crossings** | Non-UNKNOWN crossings | Numerator for ID rate |
| **ID rate** | Identified / total crossings | Higher = better |
| **Confidence distribution** | HIGH / MEDIUM / LOW / REJECT track counts | Should be majority HIGH |
| **Edge/quality rejected** | Crops filtered before OCR | Compute savings |
| **Cleanup modifications** | Post-OCR corrections applied | Indicates OCR error rate |

### Phase 1 Baselines

**Standard test: REC-0006-A.mp4 (Giants 5K, 30fps, 1800s, 480 finishers)**

| Run | Date | Config | FPS | Wall (min) | Crossings | ID'd | Correct Bibs | Precision | UNKNOWN% |
|-----|------|--------|-----|-----------|-----------|------|-------------|-----------|----------|
| Run 3 | 2026-02-14 | PyTorch, full video, stride 1 | ~20 avg | ~45 | 207 | 106 | 37 | 44% | 49% |
| Run 4a | 2026-02-15 | PyTorch, from 460s, no OCR skip | ~7 | 96 | 178 | 102 | 34 | 36.6% | 43% |
| Run 4b | 2026-02-15 | TRT YOLO + OCR skip, from 460s | ~10.5 | 64 | 183 | 112 | 34 | 32.7% | 39% |

**Notes on Run 4 comparison:**
- TRT YOLO engines cut detection+pose from ~68ms to ~20ms per frame
- OCR skip reduced OCR calls by 51% (49,950 → 24,352)
- Wall clock 1.5x faster (96→64 min), bottleneck is now PyTorch PARSeq OCR (38ms/batch)
- Correct bib count identical (34), but TRT run has more false positives (+11)
- jetson_clocks locked CPU at 1.34 GHz (was 729 MHz) for both runs
- Next step: TensorRT OCR backend to address remaining OCR bottleneck

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
| E009 | 2026-02-13 | 1.3 | Pipeline tuning run 1 (min_votes=3) | Dupes 31→19, UNKNOWN 60%→70% | min_votes too aggressive |
| E010 | 2026-02-13 | 1.3 | Pipeline tuning run 2 (min_votes=1) | Dupes 31→20, UNKNOWN 60%→69% | min_votes not the bottleneck |
| E011 | 2026-02-13 | 1.3 | Identification drop root cause | 57% were phantom dupes, rest OCR misreads | Drop is mostly correct behavior |
| E012 | 2026-02-14 | 1.3 | False positive source analysis | Jersey numbers high risk, clocks medium | Bib set validation is strongest defense |
| E013 | 2026-02-14 | 1.4 | Pipeline perf fixes (batch OCR, pruning) | Full video, no crash, 36fps sparse | Dead track pruning fixes OOM crash |
| E014 | 2026-02-14 | 1.4 | Giants 5K ground truth comparison | 7.7% bib recall, 44% precision | Crossing detection is the bottleneck, not OCR |
| E015 | 2026-02-15 | 1.4 | TensorRT YOLO + OCR skip benchmark | 1.5x faster, 34 correct bibs (same) | OCR still bottleneck at 38ms PyTorch |
