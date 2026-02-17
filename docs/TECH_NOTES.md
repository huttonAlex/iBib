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

## 2026-02-17: Bib Detector v2 Retraining & Pipeline Improvement Roadmap

### Bib Detector v2 — Training Results

**Phase/Milestone**: 1.4 - Basic Video Processing

**Objective**:
Retrain the YOLOv8n bib detector on the full annotated dataset (~6,664 images from 5 data sources) to improve detection recall, which was identified as a primary bottleneck capping bib recall at 32.9%.

**Method**:
- Consolidated YOLO annotations from 5 data sources using `scripts/consolidate_detector_data.py`
- Stratified 85/15 train/val split by event (5,662 train / 1,002 val)
- Trained YOLOv8n for 100 epochs on vast.ai RTX 4090 (24GB VRAM)
- Early stopping triggered at epoch 87 (best at epoch 67, patience=20)
- Training time: 2.29 hours

**Data Sources**:

| Source | Pairs | Train | Val |
|--------|-------|-------|-----|
| tagged_event_86638 | 3,533 | 3,003 | 530 |
| tagged_event_89536 | 1,581 | 1,343 | 238 |
| tagged_event_88679 | 731 | 621 | 110 |
| unlabeled_batch1 | 202 | 171 | 31 |
| unlabeled_batch2 | 617 | 524 | 93 |
| **Total** | **6,664** | **5,662** | **1,002** |

**Results — v1 vs v2 Comparison**:

| Metric | v1 (old) | v2 (new) | Change |
|--------|----------|----------|--------|
| Training images | 176 | 5,662 | **32x more data** |
| Validation images | 45 | 1,002 | **22x more data** |
| Events in training | 1 | 5 | +4 events |
| mAP50 | 0.966 | **0.978** | +1.2pp |
| mAP50-95 | 0.791 | **0.824** | **+3.3pp** |
| Precision | 0.973 | 0.926 | -4.7pp |
| Recall | 0.916 | **0.934** | +1.8pp |
| Training epochs | 50 | 87 (early stop) | — |
| Best epoch | — | 67 | — |

**Analysis**:

The headline mAP50 improvement (+1.2pp) appears modest because v1 already scored 96.6% — but that was on only 45 validation images from a single event, meaning it was likely overfit. Key improvements:

1. **Generalization**: v2 validates against 1,002 images from 5 different events (different bib designs, camera angles, lighting conditions). The 97.8% mAP50 on this diverse set is more trustworthy than the old 96.6% on a homogeneous 45-image set.

2. **Localization quality**: mAP50-95 improved 79.1% → 82.4% (+3.3pp). This measures tighter bounding boxes across IoU thresholds, meaning the detector produces more precise crops for OCR — directly improving downstream bib reading accuracy.

3. **Recall improvement**: 91.6% → 93.4% (+1.8pp). More bibs are found, especially small/distant ones and bibs in varied conditions that v1 never saw during training.

4. **Precision tradeoff**: 97.3% → 92.6% (-4.7pp). The model is slightly more permissive — but 92.6% precision is still excellent, and false positive detections are filtered by downstream OCR + voting. The recall gain is worth the precision cost.

**Expected Pipeline Impact**:

The detection recall improvement means more bibs enter the OCR+association pipeline:
- Detection recall estimated improvement: ~67% → 80-90% of finishers get at least one bib detection
- End-to-end bib recall: 32.9% → potentially 40-50% (detection alone won't fix association/OCR issues)
- The detector is no longer the primary bottleneck — OCR accuracy and association now limit recall

**Files**:
- Model: `runs/detect/bib_detector/weights/best.pt` (active, used by all pipeline scripts)
- Backup: `models/bib_detector_v2.pt`
- Training metrics: `models/bib_detector_v2_results.csv`
- Training artifacts: `runs/detect/bib_detector/` (plots, confusion matrices, sample batches)
- Consolidation script: `scripts/consolidate_detector_data.py`

---

### Pipeline Improvement Roadmap — OCR & Recognition Approaches to Test

**Context**:
With bib detector v2 deployed, the bottleneck shifts to OCR accuracy and bib-person association. Current OCR models (CRNN, PARSeq) were designed for edge inference on Jetson Orin Nano. The question is whether we should invest in better OCR models or fundamentally different approaches, and what accuracy/speed tradeoffs exist.

**Current State**:
- OCR reads bibs correctly ~97.9% of the time when it gets a good crop (only 7/330 false positives were actual OCR errors in Run 9)
- The real problem is that many crops are too small, blurry, angled, or partially occluded for current models to read
- PaddleOCR/PARSeq/CRNN all struggle with the same failure cases: distant runners, motion blur, folded bibs

**Approaches to Test** (ordered by implementation effort):

#### Approach 1: Multi-Frame Consensus Voting (Low Effort)

Instead of relying on a single OCR read per crop, accumulate reads across multiple frames as the runner approaches. Take the majority vote.

- **How**: Already partially implemented in `EnhancedTemporalVoting`. Needs tuning: longer voting windows, weighted voting by crop quality (size, sharpness), require N agreeing frames before accepting.
- **Expected impact**: +5-10pp bib recall by recovering bibs that are readable in some frames but not others
- **Speed cost**: None (already processing every frame)
- **Hardware**: No change needed

#### Approach 2: YOLOv8s Instead of YOLOv8n (Low Effort)

Upgrade from YOLOv8 Nano (3M params, 8.1 GFLOPs) to YOLOv8 Small (11M params, 28.6 GFLOPs).

- **How**: Retrain with `yolov8s.pt` base weights on the same 6,664-image dataset
- **Expected impact**: Better small/distant bib detection, ~2-5pp mAP improvement
- **Speed cost**: ~3x slower inference (~30ms vs ~10ms per frame on Jetson)
- **Hardware**: Jetson Orin Nano can handle it at stride 2; desktop GPU no issue
- **Test**: Train and compare mAP50/mAP50-95, then run on test video and compare bib recall

#### Approach 3: Foundation Model OCR — Florence-2 / GOT-OCR (Medium Effort)

Replace lightweight OCR with a vision-language foundation model that understands context.

- **Florence-2** (0.23B/0.77B params): Microsoft's vision foundation model. Can do OCR + region captioning. The 0.23B variant runs on 4GB VRAM.
- **GOT-OCR** (580M params): Scene text specialist model, good with arbitrary fonts/angles.

- **How**: After YOLO detection, crop bib region, feed to Florence-2/GOT-OCR instead of PARSeq
- **Expected impact**: Significant improvement on difficult crops (angled, partially occluded, unusual fonts)
- **Speed cost**: ~50-200ms per crop vs ~5-15ms for PARSeq. Not viable for real-time on Jetson Nano.
- **Hardware**: Requires Jetson Orin NX (16GB) or desktop GPU. Or use deferred reading architecture (Approach 5).
- **Test**: Run on the same test video crops, compare character-level accuracy vs PARSeq

#### Approach 4: VLM API Calls for Difficult Crops (Medium Effort)

Use a vision-language model API (Claude, GPT-4V) for crops that local OCR can't read confidently.

- **How**: After local OCR returns low confidence (<0.5), send the crop to a VLM API with prompt: "What number is on this race bib?"
- **Expected impact**: Near-human accuracy on readable crops; solves almost all OCR failures
- **Speed cost**: 200-500ms per API call + network latency. Only called for low-confidence crops (maybe 10-20% of total)
- **Per-call cost**: ~$0.01-0.05 per crop. For 480 finishers with 10-20% fallback: $5-50 per race
- **Hardware**: Needs network connectivity (cellular/WiFi at finish line)
- **Test**: Collect low-confidence crops from test video, batch-send to API, measure improvement

#### Approach 5: Deferred Reading Architecture (Higher Effort)

Separate real-time crossing detection from bib reading. Detect crossings in real-time (just person tracking + zone detection), then read bibs offline from saved evidence frames.

- **How**:
  1. Real-time: Run pose detection only, emit crossing events as UNKNOWN with evidence frame saved
  2. Post-race: Run bib detection + OCR on saved evidence frames with larger/better models (no real-time constraint)
  3. Match bib reads back to crossing events by track ID
- **Expected impact**: Removes all real-time OCR constraints. Can use the best available models with no speed penalty.
- **Speed cost**: Real-time pipeline is faster (no OCR at all). Post-race processing adds a few minutes.
- **Hardware**: Post-race processing can run on any hardware (laptop, cloud GPU, etc.)
- **Test**: Save evidence frames from test video, process offline with Florence-2/GOT-OCR, compare to real-time results

**Recommended Testing Order**:

1. **Multi-frame consensus tuning** — Free, no new dependencies, immediate test
2. **VLM API on low-confidence crops** — Quick to prototype, measures ceiling of what better OCR can achieve
3. **YOLOv8s retraining** — Low effort, addresses detection side
4. **Florence-2 offline** — Test on saved crops to measure accuracy before committing to architecture change
5. **Deferred reading** — Only if offline model accuracy justifies the architecture change

**Success Criteria**:
For each approach, measure on the Giants 5K test video (REC-0006-A, 480 finishers):
- Bib recall (correct bibs / 480 GT finishers)
- Precision (correct bibs / all bibs emitted)
- Inference time per frame/crop
- Any new hardware or network requirements

**Attachments**:
- Training plots: `runs/detect/bib_detector/results.png`
- Confusion matrix: `runs/detect/bib_detector/confusion_matrix.png`
- Full metrics CSV: `models/bib_detector_v2_results.csv`

---

## 2026-02-17: Edge Rejection + Proximity Dedup + Bib Detector v2 (Run 11b)

### Changes

**Fix 1 — Restore edge rejection** (`scripts/test_video_pipeline.py`):
Restored the `continue` after `edge_rejected += 1` that was removed in the Run 10 user changes. Without it, OCR detections had exploded ~50x (1.5k → 75k rows), flooding consensus with partial/wrong readings from frame-clipped bibs.

**Fix 2 — Proximity-aware bib dedup** (`src/pointcam/crossing.py`):
Replaced the blunt `MAX_EMISSIONS_PER_BIB=3` hard cap with position-aware suppression in `BibCrossingDeduplicator`. When a same-bib emission is attempted within 200px of a previous emission, it's silently suppressed *without* incrementing the emission count (same-person track fragment). Only spatially distant emissions count toward the cap (raised to 5 as safety net).

**Bib detector v2**: New YOLOv8n model trained on 6,664 images from 5 events (see separate v2 entry above). Better recall (93.4% vs 91.6%) and localization (mAP50-95 82.4% vs 79.1%).

### Ground Truth Change

Previous runs (9, 10) were scored against `bib_order.txt` (480 bibs) which contained many non-finishers and partial reads from earlier pipeline runs. Starting with Run 11b, all runs are scored against **official race results** from the scoring provider (`5k-run-walk-overall-results-20260214163650-0500.csv`, 2,622 finishers).

**Important caveat on recall**: The test video (REC-0006-A.mp4) covers only ~25 minutes of finish line footage (~30 min video, 54,002 frames at 30fps), not the entire race (~90 min of finishers). Manual verification confirmed bib 2675 (official position 933) as one of the last bibs visible in the video, placing the **estimated visible finishers at ~933**. The recall denominator of 2,622 is all official finishers, so the 22.3% recall represents coverage of the full race from a partial video. Against the ~933 visible finishers, the effective recall is approximately **585/933 = 62.7%**.

All previous runs have been rescored against the official results for apples-to-apples comparison.

### Results (scored against official race results, 2,622 finishers)

| Metric | Run 9 | Run 10 | **Run 11b** |
|--------|-------|--------|-------------|
| Recall (full race) | 16.4% (431/2622) | 14.3% (375/2622) | **22.3% (585/2622)** |
| Recall (visible, ~933) | 46.2% (431/933) | 40.2% (375/933) | **62.7% (585/933)** |
| Precision | 88.3% (431/488) | 88.0% (375/426) | **96.1% (585/609)** |
| True positives | 431 | 375 | **585** |
| False positives | 57 | 51 | **24** |
| Total crossings | 1,091 | 714 | 819 |
| UNKNOWN | 392 (35.9%) | 105 (14.7%) | **87 (10.6%)** |
| Unique bibs emitted | 488 | 426 | 609 |
| Dedup suppressed | 1,032 | — | 1,457 |
| Edge-rejected | — | 0 (removed) | 47,115 |

### Configuration
```
python scripts/test_video_pipeline.py REC-0006-A.mp4 \
  --no-video --bib-range 2-3000 --crossing-mode zone \
  --placement right --ocr parseq
```
Hardware: Jetson Orin Nano Super, PyTorch models, PARSeq OCR.

### Analysis

**Recall improved significantly**: 585 correct bibs vs 431 (Run 9) and 375 (Run 10) — a **+36% increase** over the previous best. Against the ~933 visible finishers in the video, effective recall is **62.7%** (up from 46.2% in Run 9). This is driven by the combination of:
1. Bib detector v2 finding more bibs (better recall/localization)
2. Edge rejection preventing OCR noise from corrupting consensus votes
3. Proximity dedup allowing more legitimate emissions (less aggressive than hard cap)

**Precision reached 96.1%**: Only 24 false positive bibs, down from 57 (Run 9). The remaining FPs are mostly short partial reads (1, 2, 3, 15, 23, 25) — classic OCR fragments of longer bib numbers.

**UNKNOWN rate at 10.6%**: Continued improvement from Run 10's cross-track seeding and REJECT vote weight, now combined with cleaner consensus from edge rejection.

**Edge rejection is critical**: 47,115 partial bibs were filtered before OCR. Without this filter (Run 10), these flood the consensus system with wrong readings.

### Comparison Note — Old vs New Ground Truth

The previous scoring against `bib_order.txt` (480 bibs) showed inflated recall numbers (Run 9: 32.9%, Run 10: 29.0%) because that file contained many non-finisher bibs and OCR fragments. Against official results, Run 9 was actually 16.4% recall with 88.3% precision — not 32.9% recall with 32.4% precision. The pipeline was more precise than previously thought, but also identifying fewer real finishers.

### Next Steps
1. **Refine visible finisher count**: Manual spot-check places ~933 finishers in the video (bib 2675, position 933, near end of video). A more precise count could be obtained by cross-referencing finish times with the video time window
2. **Short-bib FP cleanup**: The 24 remaining FPs are almost all 1-2 digit partial reads — tighter fragment rejection could eliminate most
3. **Larger YOLO model (YOLOv8s)**: May improve detection of small/distant bibs
4. **OCR model improvement**: Fine-tune PARSeq on more diverse data

---

## 2026-02-16: Cross-Track Bib Association & UNKNOWN Recovery (Run 10)

### Changes (Issues #1-3 from Run 9 analysis)

**Issue #1 — Consensus filter too strict** (user fix):
- `min_votes` 3 → 1 (temporal voting)
- `min_completeness` 0.6 → 0.3 (quality filter)
- `consensus_ttl` 5s → 30s (stale bib pruning)
- Validation penalty -0.2 → -0.05
- Removed edge-rejection skip (bibs at frame edges now processed)

**Issue #2 — Orphaned votes from track fragmentation**:
- Cross-track vote seeding: when a new person track appears near a recently-dead track, transfer accumulated bib votes (prevents fragmentation from losing bib identity)
- Added `_positions` dict to `PersistentPersonBibAssociator` for spatial matching

**Issue #3 — UNKNOWN recovery**:
- REJECT-level bibs now get 0.5 vote weight (was 0 — 64.7% of nearby detections were wasted)
- Retroactive bib recovery at dead-track emission: scan all historical bib positions for HIGH/MEDIUM matches within 400px of person's chest, filtered by 2s temporal overlap
- Dead-track emissions use real last_bbox instead of (0,0,0,0)

### Results

| Metric | Run 9 (deferred) | Run 10 (Issues #1-3) | Change |
|--------|-------------------|----------------------|--------|
| Bib Recall | **158/480 (32.9%)** | 139/480 (29.0%) | **-3.9pp** |
| Bib Precision | 32.4% (158/488) | 32.6% (139/426) | +0.3pp |
| Total crossings | 1,091 | 714 | -34.6% |
| UNKNOWN | 392 (35.9%) | **105 (14.7%)** | **-21.2pp** |
| With bib | 699 | 609 | -90 |
| Unique bibs emitted | 488 | 426 | -62 |
| OCR hallucinations | 7 | 7 | Same |
| Person tracks (max ID) | 2,121 | 2,269 | +148 |
| Tracks emitting | 1,091 (51.4%) | 714 (31.5%) | -20pp |

### Analysis

**UNKNOWN reduction succeeded**: 392 → 105 (-73%). The combination of REJECT vote weight, retroactive recovery, and cross-track seeding gives far more tracks a bib identity. This was the primary goal and it worked.

**Recall regressed (-3.9pp)**: Lost 42 finisher bibs, gained 23. All 42 lost bibs had high confidence in Run 9 (mean 0.988). The gained bibs include lower-confidence recoveries (0.42-0.47) showing retroactive recovery is working.

**Root cause of recall regression — dedup cascade**:
The 377 fewer total crossings are the key. In zone mode, every person track emits exactly 1 crossing. Run 10 has MORE person tracks (2,269 vs 2,121) but FEWER emissions (714 vs 1,091). This means the bib-level deduplicator is suppressing 1,555 tracks in Run 10 vs 1,030 in Run 9 — **525 more suppressions**.

The mechanism:
1. Cross-track seeding + REJECT votes + retroactive recovery → more tracks get bib IDs
2. Multiple fragmented tracks for the same person now all get the same bib (instead of 1 bib + N UNKNOWNs)
3. Bib dedup allows only 3 emissions per bib → the 4th+ fragment is suppressed entirely
4. If a non-finisher track claims a bib first (e.g., spectator near a runner), the finisher's track for that bib gets dedup-suppressed

**Edge rejection removal amplifies the problem**: With edge bibs no longer skipped, partial OCR reads from frame-clipped bibs enter the voting system. With `min_votes=1` and `consensus_ttl=30s`, these noisy readings persist longer and can pollute bib assignments.

### Positive Signals
- Precision held (32.4% → 32.6%) — we're not introducing false bibs
- OCR hallucinations unchanged (7) — quality filters still effective
- UNKNOWN rate dropped from 35.9% to 14.7% — the pipeline now identifies most person tracks
- 23 new finisher bibs gained that Run 9 missed entirely

### Diagnosis: Where the 42 Lost Bibs Went
The lost bibs weren't caused by a single mechanism. Likely contributors:
1. **Dedup competition**: Non-finisher track claims bib before finisher track (cross-track seeding assigns bib to wrong fragment)
2. **Vote pollution**: Longer TTL (30s) keeps stale/wrong bib consensus alive, which retroactive recovery then matches to wrong person tracks
3. **Edge noise**: Partial bibs from frame edges produce wrong OCR readings that pollute voting at `min_votes=1`

### Next Steps
1. **Restore edge rejection** (`continue` after `edge_rejected`) — the ~6x increase in detections (1.5k → 75k OCR rows) suggests massive noise injection
2. **Tune consensus_ttl**: 30s may be too long; try 10-15s as compromise between 5s and 30s
3. **Add dedup awareness to retroactive recovery**: before assigning a bib via retroactive recovery, check if that bib has already been emitted (skip if so)
4. **Consider per-bib-set enforcement at emission**: if a recovered bib is not in the bib set, demote to UNKNOWN
5. **Profile cross-track seeding accuracy**: log donor→recipient transfers and check if they match the same person

---

## 2026-02-16: Deferred Emission (Run 9)

### Change
Instead of emitting at a fixed frame count (30 frames), tracks are marked "ready" after 5 frames. Emission is deferred until either:
1. **Bib identified** via association system → emit immediately
2. **Track dies** (disappears from tracker) → emit with whatever we have (UNKNOWN if no bib)

### Results

| Metric | Run 8 (fixed 30-frame) | Run 9 (deferred) | Change |
|--------|----------------------|-------------------|--------|
| Bib Recall | 137/480 (28.5%) | **158/480 (32.9%)** | +4.4pp |
| Bib Precision | 30.6% | 32.4% (158/488) | +1.8pp |
| Total crossings | 448 | 1,091 | +143% |
| UNKNOWN rate | 47.7% | 35.9% (392/1091) | -11.8pp |
| Unique bibs emitted | — | 488 | — |
| False positive bibs | 310 | 330 | — |
| Person tracks | 2123 | 2124 | Same |
| Dedup suppressed | — | 1,032 | — |

### False Positive Deep Dive

Of 330 false positive bibs (emitted but not in ground truth):
- **323 (97.9%)**: Valid race bib numbers correctly read by OCR — real racers not in GT (spectators, non-finishers, pre-race warm-up passes)
- **7 (2.1%)**: True OCR hallucinations — all single-character substitutions (e.g., 802 for 807, 946 for 916)
- **Confidence is NOT discriminative**: Both TP and FP have mean confidence ~0.99

### Key Insight
**OCR accuracy is not the bottleneck.** The pipeline reads bibs correctly 97.9% of the time. The problems are:
1. **Missing finishers** (322/480 = 67% never emitted): YOLO + pose detection + tracking simply doesn't detect two-thirds of finishers
2. **Scene control**: Every person track emits a crossing, including spectators, non-finishers, and approach passes
3. **Person track fragmentation**: 2124 tracks for ~480 finishers (4.4x)

### Next Steps
- **Model improvements**: Better bib detection recall (YOLO training data augmentation for head-on angles)
- **Spatial filtering**: Use Y-position range to limit crossings to actual finish zone
- **Bib whitelist enforcement**: Reject non-bib-set emissions (eliminates all 7 hallucinations, zero TP cost)

---

## 2026-02-16: Zone-Based Crossing Detection (Runs 7-8)

### Problem
Timing line crossing detection fundamentally fails with head-on cameras:
- **Vertical line** (x=0.52): Runners move in Z-axis toward camera, not X-axis. Crossings are keypoint jitter (h=3: 151 random crossings, h=5: 0, h=10: 0).
- **Horizontal line** (y=0.65): Pose detector only detects people when already AT/PAST the line. Only 2 crossings in 180s.

### Solution: Zone-Based Crossing
Instead of detecting a line crossing, emit one crossing per person track after it persists for N frames. The pose detector's detection range IS the crossing zone.

### Results

| Metric | Run 4a (baseline) | Run 7 (zone=5) | Run 8 (zone=30, cap=3) |
|--------|-------------------|-----------------|------------------------|
| Bib Recall | 34/480 (7.1%) | 117/480 (24.4%) | **137/480 (28.5%)** |
| Bib Precision | 36.6% | 33.9% | 30.6% |
| Total crossings | 93 | 1095 | 1096 |
| UNKNOWN rate | — | 47.2% | 47.7% |
| FP bibs | — | 228 | 310 |
| Max dup per bib | — | 56x (bib 185) | 3x (capped) |
| Person tracks | — | 1742 | 2123 |

### Key Changes (Run 8 vs Run 7)
- `ZONE_MIN_FRAMES`: 5 → 30 (1 second delay for bib association)
- `MAX_EMISSIONS_PER_BIB`: unlimited → 3 (hard cap eliminates phantom bibs)
- `max_disappeared`: 45 → 20 (user fix: prevents phantom long-lived tracks)
- `consensus_ttl`: 5 seconds (user fix: prunes stale bib consensus)

### Remaining Bottlenecks (Priority Order)
1. **OCR accuracy**: 310 FP bibs — OCR reads wrong numbers (fragments, partial digits)
2. **Person track fragmentation**: 2123 tracks for 480 finishers (4.4x). max_disappeared=20 means brief occlusions create new tracks.
3. **Bib association timing**: 47% UNKNOWN — zone fires before bib voting converges. Need "emit on identification or track death" strategy.
4. **Bib detection recall**: Not all bibs are detected by YOLO, especially small/distant ones.

### Next Steps
- **Deferred emission**: Emit crossing when bib is identified OR when track dies (maximizes bib association window)
- **Better OCR**: Fine-tune PARSeq on race-specific data, or try larger model
- **YOLO retraining**: More training data for bib detection in various conditions

---

## 2026-02-16: Run 5c Results & Timing Line Geometry Mismatch

### Run 5c Configuration
- Hysteresis=3 (reverted from 5→3 after 5 also produced zero crossings)
- CSV flush after each write (for real-time monitoring)
- All accuracy-refactor changes from Run 5: wider association (350px, 30s memory), YOLO conf 0.25, max_disappeared=45, 1-UNKNOWN-per-track, fragment rejection, stop pruning final_consensus
- TRT YOLO engines, PyTorch OCR

### Run 5c Results (vs Run 4a baseline)

| Metric | Run 4a | Run 5c | Change |
|--------|--------|--------|--------|
| Bib Recall | 34/480 (7.1%) | 31/480 (6.5%) | -0.6pp |
| Bib Precision | 36.6% | 34.1% (31/91) | -2.5pp |
| Total crossings | 93 | 151 | +62% |
| UNKNOWN crossings | — | 36 (23.8%) | — |
| False positive bibs | — | 60 | — |
| Bib 185 duplicates | — | 17x | — |
| Pairwise order accuracy | — | 62.6% | — |
| Person tracks detected | — | 1742 | — |
| Crossing rate (crossings/tracks) | — | 8.7% | — |
| Avg pose time | — | 19.0ms/frame | — |

### Root Cause: Timing Line Geometry Mismatch

**The vertical timing line (x=0.52) fundamentally does not work with a head-on camera.**

Evidence:
- All 151 crossing chest_x values cluster within 10-20px of the timing line (998px)
  - 85/151 (56%) within 10px, 122/151 (81%) within 20px
  - Mean chest_x: 1013.1px, std dev: 19.0px
- Runners approach TOWARD the camera (Z-axis), not left-to-right (X-axis)
- The 151 "crossings" are actually keypoint jitter events, not real left-to-right motion
- This explains the hysteresis sensitivity: h=10→0 crossings, h=5→0 crossings, h=3→151 crossings
  - Random jitter rarely sustains 3+ consecutive frames on one side of a thin line

### Hysteresis Tuning History

| Hysteresis | Crossings | Explanation |
|-----------|-----------|-------------|
| 10 (Run 5) | 0 | Random jitter never sustains 10 frames |
| 5 (Run 5b) | 0 | Still too strict for random jitter |
| 3 (Run 5c) | 151 | Some jitter sequences align for 3 frames |
| 3 (original Run 4) | 93 | Fewer crossings with original params |

### Proposed Fix: Horizontal Timing Line

For head-on cameras, runners move top-to-bottom (Y-axis) as they approach. Chest_y ranges from ~700 (far) to ~1080 (close). A horizontal timing line at the appropriate Y-value would detect actual physical motion rather than keypoint noise.

### Other Issues Identified
1. **Bib 185 emitted 17 times**: Escalating confidence dedup (0.7 for 2nd, 0.9 for 3rd+) insufficient — bib 185 has conf 0.9-1.0 consistently
2. **60 false positive bibs**: OCR reading incorrect numbers (e.g., partial reads like "1065" for actual bibs)
3. **Long-lived phantom tracks**: Person track 201 existed for 51 seconds with 5 crossings, reading 5 different bib numbers — suggests max_disappeared=45 is too high, allowing track ID reuse across different people

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

**Standard test: REC-0006-A.mp4 (Giants 5K, 30fps, 1800s)**

**Ground truth**: Official race results (2,622 finishers). Video covers ~25 min of finish line footage (~933 visible finishers based on manual verification that bib 2675, official position 933, is one of the last visible bibs). "Recall (visible)" uses the ~933 denominator for a more meaningful comparison.

| Run | Date | Config | Crossings | Correct Bibs | FP Bibs | Recall (visible, ~933) | Recall (full, 2622) | Precision | UNKNOWN% |
|-----|------|--------|-----------|-------------|---------|----------------------|--------------------|-----------|---------|
| Run 3 | 2026-02-14 | PyTorch, full video, stride 1 | 207 | — | — | — | — | — | 49% |
| Run 4a | 2026-02-15 | PyTorch, from 460s, no OCR skip | 178 | — | — | — | — | — | 43% |
| Run 4b | 2026-02-15 | TRT YOLO + OCR skip, from 460s | 183 | — | — | — | — | — | 39% |
| Run 9 | 2026-02-16 | Zone, deferred emission | 1,091 | 431 | 57 | 46.2% | 16.4% | 88.3% | 35.9% |
| Run 10 | 2026-02-16 | + cross-track seed, REJECT votes | 714 | 375 | 51 | 40.2% | 14.3% | 88.0% | 14.7% |
| **Run 11b** | **2026-02-17** | **+ edge reject, prox dedup, YOLO v2** | **819** | **585** | **24** | **62.7%** | **22.3%** | **96.1%** | **10.6%** |

**Notes:**
- Runs 3-4b were scored against a different ground truth (480 bibs from bib_order.txt) and have not been rescored against official results
- Run 11b uses bib detector v2 (trained on 6,664 images), restored edge rejection, and proximity-aware dedup
- Visible finisher estimate (~933) based on manual confirmation that bib 2675 (official position 933) appears near the end of the video

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
| E016 | 2026-02-16 | 1.4 | Zone crossing Runs 7-8 | 28.5% recall, 30.6% precision | Zone mode, min_frames=30, cap=3 |
| E017 | 2026-02-16 | 1.4 | Deferred emission Run 9 | 32.9% recall, 32.4% precision | Only 7/330 FP are OCR errors; rest are real bibs |
| E018 | 2026-02-16 | 1.4 | Cross-track bib + UNKNOWN recovery Run 10 | 29.0% recall (-3.9pp), 14.7% UNKNOWN (-21.2pp) | UNKNOWN fix works; recall regressed from dedup cascade |
| E019 | 2026-02-17 | 1.4 | Bib detector v2 retraining (6,664 images) | mAP50=0.978, mAP50-95=0.824, R=0.934 | 32x more data, 5 events, early stop epoch 87 |
| E020 | 2026-02-17 | 1.4 | Edge rejection + proximity dedup + v2 detector Run 11b | 22.3% recall, 96.1% precision (vs 2622 GT) | +36% more correct bibs vs Run 9; FPs down 57→24; scored against official results |
