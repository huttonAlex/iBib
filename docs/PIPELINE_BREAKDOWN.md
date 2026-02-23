# Pipeline Breakdown

A comprehensive guide to every stage of the PointCam processing pipeline — what it does, how it works, the key thresholds, and why the approach was chosen.

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Bib Detection](#2-bib-detection)
3. [Bib Tracking](#3-bib-tracking)
4. [Pre-OCR Filtering](#4-pre-ocr-filtering)
5. [OCR Inference](#5-ocr-inference)
6. [Post-OCR Processing](#6-post-ocr-processing)
7. [Temporal Voting](#7-temporal-voting)
8. [Confidence Classification](#8-confidence-classification)
9. [Person Detection and Tracking](#9-person-detection-and-tracking)
10. [Person-Bib Association](#10-person-bib-association)
11. [Crossing Detection](#11-crossing-detection)
12. [Bib Recovery](#12-bib-recovery)
13. [Deduplication](#13-deduplication)
14. [Outputs](#14-outputs)
15. [Performance Profile](#15-performance-profile)
16. [Key Thresholds Reference](#16-key-thresholds-reference)
17. [Known Bottlenecks](#17-known-bottlenecks)
18. [Evolution and Key Lessons](#18-evolution-and-key-lessons)

---

## 1. Pipeline Overview

### High-level flow

```
Frame (30fps video)
  │
  ├─► Bib Detection (YOLOv8n) ─► Bib Tracking (CentroidTracker)
  │     │
  │     ├─► Edge Rejection (frame-edge bibs filtered)
  │     ├─► Crop Padding (asymmetric, based on camera placement)
  │     ├─► Crop Quality Filter (blur, size, aspect ratio, completeness)
  │     ├─► OCR Skip Check (stable tracks skip OCR)
  │     └─► OCR Inference (PARSeq or CRNN)
  │           │
  │           ├─► Post-OCR Cleanup (letter→digit, strip zeros, max value)
  │           ├─► Digit Count Validation (penalize unexpected lengths)
  │           ├─► Short-Bib Filtering (high-conf required for 1-2 digit bibs)
  │           ├─► Bib Set Validation (exact/fuzzy match against known bibs)
  │           ├─► Temporal Voting (multi-frame consensus per track)
  │           └─► Confidence Classification (HIGH/MEDIUM/LOW/REJECT)
  │
  ├─► Person Detection (YOLOv8n-pose) ─► Person Tracking (CentroidTracker)
  │
  ├─► Person-Bib Association (weighted majority voting)
  │
  ├─► Bib-Person Overlap Tracking (per-frame containment recording)
  │
  └─► Crossing Detection (zone or line mode)
        │
        ├─► Bib Recovery (4 fallback levels for UNKNOWN crossings)
        ├─► Deduplication (5 mechanisms)
        └─► Emit CrossingEvent
```

### Source files

| File | Role |
|------|------|
| `src/pointcam/pipeline.py` | Main orchestrator — `process_video()`, `process_frames()`, config, detection wrapper |
| `src/pointcam/inference.py` | OCR model classes (PARSeq, CRNN) with ONNX/TensorRT backends |
| `src/pointcam/crossing.py` | Person detection, tracking, timing line, crossing detection, association, dedup |
| `src/pointcam/recognition.py` | All OCR post-processing — quality filter, cleanup, validation, voting, confidence |
| `scripts/test_video_pipeline.py` | CLI entry point for running the pipeline on video files |

---

## 2. Bib Detection

**Component**: `UltralyticsBibDetector` in `pipeline.py`

### What it does

Locates bib number regions in each video frame, outputting bounding boxes with confidence scores.

### Approach

- **Model**: YOLOv8 Nano (3.2M parameters), fine-tuned on race imagery
- **Training data**: 6,664 images from 5 race events (detector v2)
- **Input**: Full video frame (resized internally to 640x640 by Ultralytics)
- **Output**: List of `BibDetection(bbox, confidence)` per frame

### Key parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `conf_threshold` | 0.25 | Deliberately low — maximizes recall, downstream filters handle noise |

### Performance

- **Accuracy**: mAP50 = 0.978, recall = 0.934 on test set
- **Speed**: ~31ms PyTorch, ~8-10ms TensorRT (Jetson Orin Nano)

### Why this approach

YOLOv8 Nano was chosen for its balance of accuracy and speed on edge hardware. The very low confidence threshold (0.25) is intentional — it's better to let noisy detections through and rely on downstream OCR + voting to reject bad reads, than to miss bibs at the detection stage. Detection recall is the single biggest bottleneck in the entire pipeline (~38% of ground-truth bibs are never detected at all).

---

## 3. Bib Tracking

**Component**: `CentroidTracker` (bib instance) in `crossing.py`

### What it does

Assigns persistent track IDs to bib detections across frames, so the same physical bib is recognized over time.

### Approach

- **Algorithm**: Centroid-based tracking with Hungarian algorithm (`scipy.linear_sum_assignment`)
- **Centroids**: Computed as bounding box centers
- **Matching**: Frame-to-frame centroid distance, optimal assignment via Hungarian method

### Key parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_disappeared` | 30 frames | How long a track survives without a detection before being deregistered |
| `max_distance` | 100 px | Maximum centroid movement between frames for a match |

### Cleanup

When a bib track is deregistered (disappeared for 30 frames), its voting history and consensus entries are pruned to prevent memory leaks. The `all_consensus` dict is never pruned — it's used for retroactive bib recovery.

---

## 4. Pre-OCR Filtering

Three filters run before sending a bib crop to OCR. Each prevents a specific class of bad input that would waste OCR compute and pollute voting.

### 4.1 Edge Rejection

**Component**: `BibCompletenessChecker` in `recognition.py`

Rejects any bib whose bounding box is within **2% of the frame edge**. A bib at the frame edge is being clipped by the camera, producing partial digit reads (e.g., reading "23" from "1234" because the leading digits are off-screen).

**Impact**: ~47,000 rejections in a typical run. Disabling this filter (tested in Run 10) caused a 50x detection explosion and recall regression — partial-bib OCR reads overwhelmed the voting system.

### 4.2 Crop Padding

**Component**: `PLACEMENT_PADDING` dict in `pipeline.py`

Adds asymmetric padding around the detected bib bbox before cropping for OCR. The padding direction depends on camera placement since bibs on the far side of the runner may have their leading/trailing digits slightly outside the detection box.

| Placement | Left | Right | Top | Bottom |
|-----------|------|-------|-----|--------|
| `center` | 15% | 15% | 10% | 10% |
| `left` | 10% | 25% | 10% | 10% |
| `right` | 25% | 10% | 10% | 10% |

Padding percentages are relative to bib bbox dimensions, clamped to frame boundaries.

### 4.3 Crop Quality Filter

**Component**: `CropQualityFilter` in `recognition.py`

Multi-criteria filter that rejects crops unlikely to produce useful OCR reads:

| Check | Threshold | What it catches |
|-------|-----------|-----------------|
| **Blur** (Laplacian variance) | >= 50.0 | Motion blur, out-of-focus bibs |
| **Min size** | 40px wide, 15px tall | Bibs too small to read |
| **Aspect ratio** | 1.0 – 6.0 (w/h) | Vertically squished or absurdly wide crops |
| **Contrast** (grayscale std dev) | >= 20.0 | Washed-out or uniform crops |
| **Completeness** (multi-heuristic) | >= 0.3 overall, >= 0.5 content extent | Partially obstructed bibs, bibs with too much background |

The completeness score combines five sub-heuristics: content extent (how much of the crop width has ink), margin balance (asymmetric margins suggest clipping), ink density, horizontal distribution uniformity, and vertical extent.

**Impact**: ~13,800 quality rejections + ~1,800 partial/obstructed rejections per run.

---

## 5. OCR Inference

**Component**: `inference.py`

### What it does

Reads the bib number from a cropped image. Two model architectures are available.

### 5.1 PARSeq (Primary)

- **Architecture**: Permuted Autoregressive Sequence model, 23.8M parameters
- **Input**: 128x32 RGB image, normalized to [-1, 1]
- **Output**: (batch, seq_len, vocab_size) logits → softmax → `ParseqTokenizer.decode()`
- **Decoding**: Greedy decode using offline tokenizer JSON (vocab, BOS/EOS/PAD tokens). Only digit characters are extracted from the output.
- **Confidence**: Product of per-character probabilities
- **Accuracy**: 97.4% on test set
- **Calibration**: Well-calibrated — avg confidence 0.62 on errors vs 0.99 on correct reads
- **Speed**: ~124ms PyTorch, ~38ms ONNX batch, ~12-15ms TensorRT

### 5.2 CRNN (Alternative)

- **Architecture**: CNN + BiLSTM + CTC, 8.3M parameters
- **Input**: 128x32 grayscale, normalized to [0, 1]
- **Output**: CTC logits → greedy CTC decode (collapse repeats, remove blanks)
- **Confidence**: Product of per-timestep max probabilities
- **Accuracy**: 91.7% on test set
- **Calibration**: Poorly calibrated — avg confidence 0.87 on wrong predictions (overconfident on errors)
- **Speed**: ~17ms ONNX, ~5-8ms TensorRT

### Why PARSeq is preferred

PARSeq is ~6% more accurate and has much better confidence calibration, which is critical since the entire downstream pipeline (voting, confidence classification, dedup escalation) relies on meaningful confidence scores. CRNN is faster but its overconfidence on errors makes thresholding unreliable.

### OCR Skip Optimization

Before running OCR, the pipeline checks if a bib track already has a stable, high-confidence consensus. If the track's voting result is stable (`is_stable=True`), confidence >= 0.85, and the consensus has already been recorded in `final_consensus` with a validated bib (or no validator is active), OCR is skipped entirely. This saves ~30-50% of OCR calls during dense sections of a race.

### Batch inference

Both models support `predict_batch(crops)` for processing multiple bib crops in a single forward pass. The pipeline collects all crops for a frame and batches them together.

---

## 6. Post-OCR Processing

Four steps clean up and validate the raw OCR output before it enters the voting system.

### 6.1 Post-OCR Cleanup

**Component**: `PostOCRCleanup` in `recognition.py`

Fixes common OCR misreads and enforces basic sanity:

1. **Letter-to-digit substitution**: 26 mappings (O→0, I→1, l→1, S→5, B→8, Z→2, etc.)
2. **Extract digits only**: Strip any remaining non-digit characters
3. **Strip leading zeros**: "0123" → "123"
4. **Digit count validation**: Reject results with <1 or >5 digits
5. **Max bib value check**: If a `max_bib_value` is configured (derived from bib set), reject numbers above it

### 6.2 Digit Count Validation

**Component**: `DigitCountValidator` in `recognition.py`

Infers expected digit counts from the bib set (e.g., if all bibs are 3-4 digits, expected = `{3, 4}`). Applies a **0.15 confidence penalty** to OCR results with unexpected digit counts. Does not reject outright — just makes unexpected lengths less competitive in voting.

### 6.3 Short-Bib Filtering

**Component**: Inline logic in `pipeline.py` `process_frames()`

Guards against OCR fragments (e.g., reading just "6" from a bib that actually says "1265"):

| Bib length | Not in bib set | Required OCR confidence |
|------------|----------------|------------------------|
| 1 digit | Yes | >= 0.95 |
| 2 digits | Yes | >= 0.90 |

Bibs that are in the bib set pass regardless. This filter is applied both during normal OCR processing and again after bib recovery at emission time.

### 6.4 Bib Set Validation

**Component**: `BibSetValidator` in `recognition.py`

Validates OCR results against the known set of race bibs. Uses a priority cascade:

1. **Exact match**: Bib is in the set → `is_valid=True`, confidence boost **+0.10**
2. **Leading digit recovery**: Prepend digits 1-9, check if result is in set (catches cases where the leading digit was cropped or misread)
3. **Trailing digit recovery**: Append digits 0-9, check if result is in set
4. **Prefix match**: Find bibs sharing the first 2 digits, use `SequenceMatcher` similarity >= 0.75
5. **Length-similar match**: Check all bibs with |length difference| <= 1

Corrected matches: `is_corrected=True`, confidence penalty **-0.10**, always flagged for review.
No match found: confidence penalty **-0.05**.

---

## 7. Temporal Voting

**Component**: `EnhancedTemporalVoting` in `recognition.py`

### What it does

Aggregates OCR reads across multiple frames for each bib track to produce a stable consensus. A single OCR read is noisy; averaging over 5-15 frames dramatically improves accuracy.

### Approach

- **Vote history**: Stores up to 15 frames of (bib_number, confidence) votes per track
- **Recency weighting**: Each vote is weighted by `0.95^(age_in_frames)` × `ocr_confidence` — newer, higher-confidence reads count more
- **Winner selection**: The bib number with the highest total weighted score wins
- **Combined confidence**: `0.5 × agreement_ratio + 0.5 × avg_confidence` where agreement_ratio = winner votes / total votes

### Key parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_size` | 15 frames | How many frames of history to keep |
| `min_votes` | 1 | Minimum votes to produce a consensus (very permissive) |
| `stability_threshold` | 5 | Consecutive identical reads to declare `is_stable=True` |
| `confidence_threshold` | 0.5 | Below this, result is flagged as `needs_review` |
| `recency_decay` | 0.95 | Per-frame weight decay |

### Why so permissive

`min_votes=1` is deliberately low. Earlier experiments with `min_votes=3` caused too many valid bibs to be rejected — runners who are only visible for a few frames would never reach 3 votes. The combination of downstream confidence classification and deduplication provides sufficient noise suppression without requiring a high vote minimum.

---

## 8. Confidence Classification

**Component**: `ConfidenceManager` in `recognition.py`

### What it does

Classifies each consensus result into a confidence tier, which controls downstream behavior (association vote weight, dedup escalation, review flagging).

### Tiers

| Level | Threshold | Meaning |
|-------|-----------|---------|
| HIGH | >= 0.85 | Auto-accept, highest association weight |
| MEDIUM | >= 0.70 | Acceptable, moderate association weight |
| LOW | >= 0.50 | Flagged for review |
| REJECT | < 0.50 | Minimal weight, flagged for review |

### Adjustments

Confidence is adjusted **additively** (not floored — this was a critical fix from Run 1 where flooring distorted the signal):

| Condition | Adjustment |
|-----------|------------|
| Valid bib (exact match) | +0.15 |
| Stable voting (5+ consecutive) | +0.10 |
| High agreement ratio (>= 0.80) | +0.05 |
| Bib set exact match | +0.10 |
| Bib set corrected match | -0.10 |
| No bib set match | -0.05 |

### Review queue

LOW and REJECT predictions are added to a review queue. Corrected predictions (fuzzy-matched to a different bib) are always added regardless of confidence level. The review queue is exported as JSON at the end of a run.

---

## 9. Person Detection and Tracking

### 9.1 Person Detection

**Component**: `PoseDetector` in `crossing.py`

- **Model**: YOLOv8n-pose (6MB, auto-downloaded)
- **Input**: Full video frame
- **Output**: Person bounding boxes + 17 COCO keypoints per person
- **Conf threshold**: 0.5
- **Chest point extraction**: Average of visible torso keypoints (left/right shoulders + left/right hips). Keypoint visibility threshold: 0.3. Fallback: bbox centroid if no torso keypoints are visible.

The chest point is the key output — it serves as the person's representative position for tracking, association, and crossing detection.

### 9.2 Person Tracking

**Component**: `CentroidTracker` (person instance) in `crossing.py`

Same algorithm as bib tracking but with different parameters tuned for person motion:

| Parameter | Value | vs. Bib Tracker | Rationale |
|-----------|-------|-----------------|-----------|
| `max_disappeared` | 20 frames | 30 for bibs | Shorter — prevents phantom long-lived person tracks |
| `max_distance` | 150 px | 100 for bibs | Larger — people move faster than bibs (arms swinging, stride) |

Uses chest keypoints as centroids (passed via the `centroids` parameter) rather than bbox centers, for more stable tracking when arms and legs swing.

---

## 10. Person-Bib Association

**Component**: `PersistentPersonBibAssociator` in `crossing.py`

### What it does

Links bib numbers (from OCR) to person tracks (from pose detection). This is necessary because a bib detection and a person detection are separate objects — the system must figure out which bib belongs to which person.

### Approach: Weighted Majority Voting

Each frame, the associator:

1. **Spatially matches** persons to bibs (two-pass):
   - **Pass 1**: For each person, find the best bib candidate. Prefer bib centroid inside person bbox, then bbox overlap, then proximity within 350px.
   - **Pass 2**: Enforce exclusive assignment — if multiple persons claim the same bib, the closest person wins.

2. **Casts weighted votes**: Look up the OCR consensus for the matched bib track. Cast a vote for the bib number on the person track, weighted by confidence level:

   | Confidence | Vote Weight |
   |------------|-------------|
   | HIGH | 3.0 |
   | MEDIUM | 2.0 |
   | LOW | 1.0 |
   | REJECT | 0.5 |

   Short-bib penalty: If the bib has fewer digits than the expected minimum, the vote weight is halved.

3. **Determines winner**: The bib number with the highest total weighted votes wins, subject to:
   - Minimum 1 weighted vote total
   - Minimum 0.4 vote fraction (winner votes / total votes) — prevents weak plurality wins

### Cross-track vote seeding

When a new person track appears near a recently-dead track (within `max_distance=150px`), accumulated votes from the dead track are transferred to the new one. This handles track fragmentation — when the same person gets re-detected as a new track after a brief occlusion.

### Memory

Vote history expires after **30 seconds** (at video fps). Stale associations for long-gone tracks are pruned each frame.

---

## 11. Crossing Detection

### What it does

Determines when a person has crossed the finish line, triggering a crossing event emission.

### 11.1 Zone Mode (primary)

**Used for**: Head-on camera angles (camera pointing at approaching runners)

Zone mode does not use a timing line. Instead, it treats the entire frame as a "crossing zone" and emits one crossing per person track:

1. After a person track persists for **5 frames** (`zone_min_frames`), it is marked as "ready"
2. Emission is deferred until a bib is identified via the person-bib association
3. If the track disappears (deregistered) before getting a bib, it emits with whatever it has (UNKNOWN if no bib was ever associated)
4. Each track emits at most once (tracked via `person_track_emitted` set)

### 11.2 Line Mode (alternative)

**Used for**: Side-angle cameras (runners crossing left-to-right or right-to-left)

Uses a virtual `TimingLine` defined in normalized frame coordinates [0,1]:

1. Each frame, compute which side of the line the person's chest point is on (via cross product)
2. **Hysteresis**: The person must stay on the new side for **3 consecutive frames** before a crossing is confirmed
3. **Direction filter**: Can require `left_to_right`, `right_to_left`, or `any`
4. **Debounce**: Minimum frames between crossings for the same track

### Why zone mode exists

Line mode was designed for side-angle cameras where runners physically cross a line in the frame. But for head-on cameras (runners approaching the camera), there is no lateral crossing — movement is along the Z-axis (depth). A vertical timing line doesn't work in this scenario (tested and failed in early runs). Zone mode was introduced in Run 7-8 to handle this geometry.

---

## 12. Bib Recovery

**Component**: Inline logic in `pipeline.py`

### What it does

When a person track is about to emit a crossing as UNKNOWN (no bib was associated), the pipeline attempts to recover a bib through four increasingly speculative fallback levels.

### Level 1: Direct Containment

Scan all currently tracked bibs. If a bib's centroid is inside the person's bounding box, use that bib's OCR consensus. Requires HIGH or MEDIUM confidence.

**When this helps**: Bib was detected and OCR'd but the person-bib associator failed to link them (e.g., spatial matching was off by a few frames).

### Level 2: Expanded Containment

Expand the person's bounding box by a margin (`max(50px, 15% of person width)`) and scan bibs again. Requires HIGH confidence only (stricter because the match is less certain).

**When this helps**: Bib detection was slightly outside the person bbox due to arm swing, loose clothing, or bbox estimation error.

### Level 3: Co-occurrence Recovery

Uses the `bib_overlapped_persons` dictionary, which records (every frame) which person track IDs each bib track was ever spatially inside. For the current person track, find all bibs that were ever overlapping with it.

**Uniqueness filter**: Skip bibs that overlapped with more than 3 person tracks. In dense crowds, a bib in the middle of a group can "touch" many person bboxes — these associations are unreliable.

Requires HIGH or MEDIUM confidence.

**When this helps**: The bib was visible earlier in the person's trajectory but is no longer visible at emission time. Historical containment data recovers it.

### Level 4: Retroactive Spatial+Temporal Recovery

The most speculative level. Scans ALL historical bib consensus entries (the `all_consensus` dict, which is never pruned). For each bib:

1. **Temporal check**: Did the bib's lifetime overlap the person's lifetime (with 5-second margin)?
2. **Spatial check**: Is the minimum distance between any combination of (person first/last chest point) and (bib first/last centroid) within **600 pixels**?
3. **Selection**: Use the spatially closest match with HIGH or MEDIUM confidence.

**When this helps**: The person and bib were never simultaneously tracked (one appeared after the other was deregistered), but they occupied similar spatial/temporal territory.

### Post-recovery short-bib filter

After recovery, the short-bib filter is re-applied: 1-digit bibs not in the bib set need >= 0.95 OCR confidence, 2-digit bibs need >= 0.90. Failing this reverts the crossing back to UNKNOWN.

---

## 13. Deduplication

**Component**: `BibCrossingDeduplicator` in `crossing.py`

### What it does

Prevents the same physical crossing from being emitted multiple times. Track fragmentation, recovery fallbacks, and zone-mode can all generate redundant crossings for the same runner.

### Five mechanisms

| # | Mechanism | Behavior |
|---|-----------|----------|
| 1 | **UNKNOWN per-track limit** | Each person track can emit at most 1 UNKNOWN crossing |
| 2 | **Per-track-per-bib dedup** | Same person track re-crossing with same bib is always suppressed |
| 3 | **Proximity suppression** | If a bib emission is within 200px of a previous emission for the same bib number, it is suppressed (does not count toward the emission cap). Handles fragmented tracks for the same physical person. |
| 4 | **Frame debounce** | Same bib number within 10 seconds (at video fps) is suppressed |
| 5 | **Escalating confidence** | 1st emission: any confidence. 2nd: >= 0.70. 3rd+: >= 0.90. Hard cap at 5 emissions per bib number. |

### Why 5 mechanisms

Each mechanism catches a different class of duplicate:

- **#1** prevents UNKNOWN floods from fragmented tracks
- **#2** is a basic safety net for zone-mode re-emission
- **#3** handles the case where track fragmentation creates two person tracks for the same runner — they're close together spatially, so proximity catches it
- **#4** is a time-based backstop in case proximity doesn't apply (e.g., runner loops back)
- **#5** handles legitimate multi-lap races while preventing runaway duplicates. The escalating confidence requirement means later emissions need stronger evidence.

---

## 14. Outputs

Each pipeline run produces five output files:

| File | Content |
|------|---------|
| `{stem}_detections.csv` | Per-detection log: frame, time, track_id, bbox, det_conf, ocr_raw, ocr_conf, validated, consensus, confidence_level |
| `{stem}_crossings.csv` | Crossing events: sequence, frame, time, person_track_id, bib_number, confidence, person_bbox, chest_point, source |
| `{stem}_person_tracks.csv` | Diagnostic: every person track with first/last frame, chest positions, crossed status, assigned bib |
| `{stem}_review_queue.json` | Low-confidence and corrected predictions flagged for human review |
| `{stem}_annotated.mp4` | Annotated video with bounding boxes, labels, timing line (disabled with `--no-video`) |

---

## 15. Performance Profile

Measured on Jetson Orin Nano (8GB):

| Operation | PyTorch | TensorRT | Notes |
|-----------|---------|----------|-------|
| Bib detection (YOLOv8n) | 31ms | ~8-10ms | Per frame |
| Pose detection (YOLOv8n-pose) | 37ms | ~10-12ms | Per frame |
| OCR PARSeq (batch) | 38ms | ~12-15ms | Per batch of crops |
| OCR CRNN (batch) | 17ms | ~5-8ms | Per batch of crops |
| Association + crossing | <1ms | <1ms | CPU-only |
| **Total (sparse scene)** | **36 fps** | **~28-36 fps** | Few bibs visible |
| **Total (dense scene)** | **11 fps** | **~15-20 fps** | Many bibs visible |

The system targets 30fps real-time. Sparse scenes (1-3 runners) achieve this easily. Dense scenes (10+ runners simultaneously) are currently below real-time in PyTorch mode and borderline with TensorRT.

---

## 16. Key Thresholds Reference

All tunable thresholds in one place, organized by pipeline stage.

### Detection

| Threshold | Value | Location |
|-----------|-------|----------|
| YOLO confidence | 0.25 | `PipelineConfig.conf_threshold` |

### Pre-OCR Filtering

| Threshold | Value | Location |
|-----------|-------|----------|
| Frame edge margin | 2% | `BibCompletenessChecker` |
| Blur (Laplacian var) | 50.0 | `CropQualityFilter` |
| Min crop size | 40×15 px | `CropQualityFilter` |
| Aspect ratio range | 1.0–6.0 | `CropQualityFilter` |
| Contrast (std dev) | 20.0 | `CropQualityFilter` |
| Completeness score | 0.3 | `CropQualityFilter` |
| Content extent | 0.5 | `CropQualityFilter` |

### OCR & Post-Processing

| Threshold | Value | Location |
|-----------|-------|----------|
| 1-digit non-bib-set conf | 0.95 | `pipeline.py` inline |
| 2-digit non-bib-set conf | 0.90 | `pipeline.py` inline |
| Fuzzy match similarity | 0.75 | `BibSetValidator` |
| Digit count penalty | -0.15 | `DigitCountValidator` |
| Exact match boost | +0.10 | `BibSetValidator` |
| Corrected match penalty | -0.10 | `BibSetValidator` |
| No-match penalty | -0.05 | `BibSetValidator` |

### Temporal Voting

| Threshold | Value | Location |
|-----------|-------|----------|
| Vote window size | 15 frames | `EnhancedTemporalVoting` |
| Min votes | 1 | `EnhancedTemporalVoting` |
| Stability threshold | 5 consecutive | `EnhancedTemporalVoting` |
| Recency decay | 0.95/frame | `EnhancedTemporalVoting` |
| Review confidence | 0.5 | `EnhancedTemporalVoting` |

### Confidence Classification

| Threshold | Value | Location |
|-----------|-------|----------|
| HIGH | >= 0.85 | `ConfidenceManager` |
| MEDIUM | >= 0.70 | `ConfidenceManager` |
| LOW | >= 0.50 | `ConfidenceManager` |
| Valid bib boost | +0.15 | `ConfidenceManager` |
| Stable voting boost | +0.10 | `ConfidenceManager` |
| High agreement boost | +0.05 | `ConfidenceManager` |
| OCR skip confidence | 0.85 | `pipeline.py` |

### Person Detection & Tracking

| Threshold | Value | Location |
|-----------|-------|----------|
| Pose confidence | 0.5 | `PoseDetector` |
| Keypoint visibility | 0.3 | `PoseDetector` |
| Person max_disappeared | 20 frames | Person `CentroidTracker` |
| Person max_distance | 150 px | Person `CentroidTracker` |
| Bib max_disappeared | 30 frames | Bib `CentroidTracker` |
| Bib max_distance | 100 px | Bib `CentroidTracker` |

### Person-Bib Association

| Threshold | Value | Location |
|-----------|-------|----------|
| Max association distance | 350 px | `PersistentPersonBibAssociator` |
| Vote memory | 30 seconds | `PersistentPersonBibAssociator` |
| Min weighted votes | 1 | `PersistentPersonBibAssociator` |
| Min vote fraction | 0.4 | `PersistentPersonBibAssociator` |
| HIGH vote weight | 3.0 | `PersistentPersonBibAssociator` |
| MEDIUM vote weight | 2.0 | `PersistentPersonBibAssociator` |
| LOW vote weight | 1.0 | `PersistentPersonBibAssociator` |
| REJECT vote weight | 0.5 | `PersistentPersonBibAssociator` |

### Crossing & Recovery

| Threshold | Value | Location |
|-----------|-------|----------|
| Zone min frames | 5 | `pipeline.py` |
| Line hysteresis | 3 frames | `CrossingDetector` |
| Overlap uniqueness cap | 3 persons | `pipeline.py` co-occurrence |
| Retroactive spatial radius | 600 px | `pipeline.py` |
| Retroactive temporal margin | 5 seconds | `pipeline.py` |
| Consensus TTL | 30 seconds | `pipeline.py` |

### Deduplication

| Threshold | Value | Location |
|-----------|-------|----------|
| Proximity suppression radius | 200 px | `BibCrossingDeduplicator` |
| Frame debounce | 10 seconds | `BibCrossingDeduplicator` |
| 2nd emission confidence | >= 0.70 | `BibCrossingDeduplicator` |
| 3rd+ emission confidence | >= 0.90 | `BibCrossingDeduplicator` |
| Max emissions per bib | 5 | `BibCrossingDeduplicator` |

---

## 17. Known Bottlenecks

### 1. Detection recall ceiling (biggest bottleneck)

~38% of ground-truth bibs are never detected by YOLO at all. These bibs are too small (distant runners), occluded by arms/other runners, folded, or otherwise invisible to the detector. No amount of downstream processing can recover a bib that was never detected.

**Potential mitigations**: Upgrade to YOLOv8s (larger model), more diverse training data, 4K camera resolution, multi-camera coverage.

### 2. Person track fragmentation

The pipeline produces ~2.4 person tracks per actual finisher (e.g., 790 crossings for ~933 finishers in Run 13c). The same person gets multiple track IDs due to brief occlusions, pose model misses, and dense crowd confusion. Cross-track vote seeding helps but doesn't fully solve this.

### 3. OCR-to-crossing conversion gap

Many bibs that OCR successfully reads never make it to a crossing emission. The conversion rate from "OCR-seen bib" to "emitted crossing" is ~46% in some tests. This gap is caused by association failures (bib was read but not linked to the right person at crossing time) and dedup over-suppression.

### 4. Head-on camera geometry

Head-on cameras see bibs very small until runners are close, giving only a brief window for detection + OCR. Angled cameras have wider temporal coverage but tested worse overall (22.5% recall vs 64.5% head-on) due to increased foreshortening and occlusion.

### 5. Wide bib ranges hurt precision

When the valid bib range is wide (e.g., 801-7350 for a half marathon), the bib set validator has less filtering power. More OCR errors coincidentally match valid bibs. Precision dropped from ~96% to ~55% in half-marathon testing.

---

## 18. Evolution and Key Lessons

### Run-by-run progression

| Run | Recall | Precision | UNKNOWN % | Key change |
|-----|--------|-----------|-----------|------------|
| 3 | 7.7% | 44% | — | First crossing-aware pipeline |
| 7-8 | 28.5% | — | — | Zone mode for head-on cameras |
| 9 | 32.9% | — | — | Deferred emission (emit on bib ID or track death) |
| 10 | 29.0% | — | 14.7% | Cross-track vote seeding (regressed recall via dedup cascade) |
| 11b | 62.7% | 96.1% | 4.8% | Restored edge rejection + proximity dedup + detector v2 |
| 12 | 63.8% | 96.0% | — | Bbox overlap matching + enhanced retroactive recovery |
| 13c | 64.5% | 95.9% | 5.1% | Same-frame containment recording + uniqueness filter |

### Key lessons learned

1. **Crossing detection, not OCR, was the original bottleneck** (Run 3): 62% of bibs were seen by OCR but only 22% made it to a crossing. The pipeline architecture needed to bridge the gap between "bib was read" and "person crossed."

2. **Camera geometry dictates the crossing algorithm** (Run 7-8): A timing line doesn't work for head-on cameras. Zone mode was necessary but required a completely different emission strategy.

3. **Most false positives are correct bibs on wrong people** (Run 9): Only 7/330 FPs were actual OCR errors. 97.9% were correctly-read bibs associated with the wrong runner. Association accuracy matters more than OCR accuracy.

4. **Edge rejection is critical** (Run 10 vs 11b): Disabling frame-edge bib filtering caused a 50x detection explosion. Partial bibs at frame edges produce confidently-wrong OCR reads that pollute voting.

5. **Additive confidence adjustment, not flooring** (Run 1): Setting confidence floors (e.g., valid bib → confidence = max(conf, 0.80)) distorted the signal. Additive adjustments (+0.15 for valid bibs) preserved the relative ordering of predictions.

6. **Cross-track bridging is harmful in dense crowds** (Run 13): Transferring votes between nearby person tracks works in sparse scenes but causes vote contamination in dense crowds where many tracks are near each other. The feature was tried and reverted.
