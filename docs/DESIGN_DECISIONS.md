# Design Decisions Log

This document tracks key technical decisions, alternatives considered, and rationale.

---

## Accepted Decisions

### DD-001: On-Board vs Cloud Processing

**Decision**: Process video frames on-board (edge device) rather than streaming to cloud

**Context**: Need to determine where CV processing happens

**Alternatives Considered**:
1. Cloud processing (stream frames to server)
2. Hybrid (some on-board, some cloud)
3. Full on-board processing

**Rationale**:
- Eliminates network dependency (critical for cellular-only connectivity)
- Reduces latency for real-time timing
- No bandwidth costs for video streaming
- Better privacy (no video streaming)
- Backup role requires independent operation

**Consequences**:
- Requires capable edge hardware (~$500-800)
- More complex on-site setup
- Processing power limits model complexity
- Updates require OTA mechanism

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-002: Position Accuracy vs Absolute Time Priority

**Decision**: Prioritize position/order accuracy over absolute timestamp accuracy

**Context**: System cannot guarantee sub-second precision, must choose priority

**Alternatives Considered**:
1. Prioritize absolute timing precision
2. Equal priority for both
3. Prioritize position/order accuracy

**Rationale**:
- Backup system role means RFID provides primary times
- Race results critically depend on finish order
- ±1 second time error with correct order is acceptable
- Wrong order is unacceptable even with precise times

**Consequences**:
- Algorithm design focuses on relative timing
- Conflict resolution prioritizes order maintenance
- May need manual review for photo-finish scenarios
- Clear documentation needed for users

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-003: Camera Positioning - Bib Height on Tripod

**Decision**: Position camera at bib height on standard tripod

**Context**: Need to determine optimal camera placement

**Alternatives Considered**:
1. Overhead camera (bird's eye view)
2. Ground level (looking up)
3. Bib height (horizontal view)
4. Elevated angle (45° down)

**Rationale**:
- Bib height provides clearest bib view with least distortion
- Standard tripod is portable and familiar to race operators
- Avoids infrastructure requirements (poles, mounting points)
- Easier to maintain consistent framing

**Consequences**:
- May have occlusion issues with taller/shorter runners
- Limited vertical field of view
- Tripod stability is critical
- Setup requires clear line of sight to timing line

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-004: Primary Development Language - Python

**Decision**: Use Python as primary development language

**Context**: Need to select implementation language

**Alternatives Considered**:
1. Python - Rich CV/ML ecosystem, rapid development
2. C++ - Maximum performance, more complex
3. Rust - Memory safe, good performance
4. Hybrid - Python + C++ for critical paths

**Rationale**:
- Extensive CV/ML library support (OpenCV, PyTorch, YOLO, PaddleOCR)
- Rapid prototyping for Phase 1-2
- Easier to iterate on algorithms
- Can optimize critical paths later with TensorRT/ONNX
- Good edge device support (Jetson, RPi)
- Team familiarity

**Consequences**:
- May need optimization for real-time performance
- TensorRT/ONNX for inference optimization
- Easier to find contributors/maintainers

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-005: CTP01 Protocol for Timing Integration

**Decision**: Implement ChronoTrack Socket Protocol 1 (CTP01) for scoring software integration

**Context**: Need standard protocol for timing data transmission

**Alternatives Considered**:
1. CTP01 (ChronoTrack standard)
2. Custom REST API
3. MQTT/WebSocket custom protocol
4. File-based exchange

**Rationale**:
- CTP01 is established standard in timing industry
- Compatible with MYLAPS/ChronoTrack scoring software
- Protocol documentation available
- Built-in sequence numbers for recovery
- Human-readable for debugging

**Consequences**:
- Must implement CTP01 client role
- Tilde-separated, CRLF-terminated format
- Push mode required for real-time streaming
- Need to handle reconnection with sequence resume

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-006: GPS Module for Time Synchronization

**Decision**: Use GPS module for time synchronization (not NTP)

**Context**: Need accurate timestamps, cellular-only connectivity

**Alternatives Considered**:
1. GPS module with PPS
2. NTP over cellular
3. Manual sync with RFID system
4. Local clock only

**Rationale**:
- GPS provides accurate time without network dependency
- Works in cellular dead zones
- PPS pulse gives ±10ms accuracy
- Consistent with race timing standards
- Relatively low cost ($30-50)

**Consequences**:
- Additional hardware component
- Need clear sky view (may need external antenna)
- Must handle GPS signal loss gracefully
- Adds ~$40 to BOM

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-007: Web-Based Operator Interface

**Decision**: Implement web-based UI for operator interface

**Context**: Need operator interface for setup and monitoring

**Alternatives Considered**:
1. Web-based (browser access)
2. Native mobile app (iOS/Android)
3. Dedicated display on unit
4. Command-line only

**Rationale**:
- No app installation required
- Works on any device (phone, tablet, laptop)
- Easier to develop and maintain
- Can access via phone hotspot or local network
- FastAPI + simple HTML/JS is sufficient

**Consequences**:
- Need to run web server on edge device
- Must work well on mobile browsers
- Security considerations (local network only)
- No offline app capability (but not needed)

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-008: Simple Naming for Unit Identification

**Decision**: Use simple operator-assigned naming for unit identification (not central registry)

**Context**: Need to identify units when multiple deployed

**Alternatives Considered**:
1. Simple naming (operator assigns: "Finish-1", "Split-2")
2. Central registration server
3. MAC address/serial number only
4. Automatic discovery

**Rationale**:
- Keeps system simple and standalone
- No central server dependency
- Familiar pattern for timing operators
- Easy to set up and change
- Can add central management later if needed

**Consequences**:
- No automatic fleet management
- Operator responsible for unique names
- Location name used in CTP01 protocol
- Future: Could add optional central registry

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-009: Offline-First Data Architecture

**Decision**: Design system as offline-first with sync when connected

**Context**: Cellular connectivity may be unreliable at race venues

**Alternatives Considered**:
1. Online-required (block if no connection)
2. Offline-first with sync
3. Store-and-forward only (no real-time)

**Rationale**:
- Race venues often have poor cellular coverage
- Cannot lose timing data due to connectivity
- SQLite queue provides persistence
- Sequence numbers enable reliable replay
- Matches CTP01 protocol recovery capabilities

**Consequences**:
- Must implement robust queue
- Must handle sequence number gaps on server
- Timestamps must be accurate without network
- Queue size must accommodate full race

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-010: Configurable Evidence Capture

**Decision**: Make evidence frame capture configurable per race (ALL, LOW_CONFIDENCE, NONE)

**Context**: Need to balance storage vs dispute resolution capability

**Alternatives Considered**:
1. Always capture all frames
2. Never capture frames
3. Configurable per race

**Rationale**:
- Different races have different dispute likelihood
- Storage constraints vary
- "ALL" for high-stakes races
- "LOW_CONFIDENCE" for typical races (captures UNKNOWN bibs)
- "NONE" for minimal storage situations

**Consequences**:
- Configuration UI needed
- Storage management required
- Evidence retrieval API needed
- Must define confidence threshold

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-011: Unknown Bib Handling - Log with Timestamp

**Decision**: Log crossing events even when bib number cannot be read, marked as "UNKNOWN"

**Context**: Need to decide behavior when OCR fails

**Alternatives Considered**:
1. Skip unreadable bibs entirely
2. Log as "UNKNOWN" with timestamp
3. Alert operator for manual entry
4. Buffer until bib can be confirmed

**Rationale**:
- Crossing happened - timing data valuable
- Position can be verified against other systems
- UNKNOWN can be resolved via evidence review
- Better to have data than not
- CTP01 can transmit any string as bib

**Consequences**:
- Need to handle UNKNOWN in data pipeline
- Evidence capture critical for UNKNOWN bibs
- May need post-race resolution workflow
- Scoring software must handle UNKNOWN values

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-012: Product for Sale Business Model

**Decision**: Develop PointCam as a product for sale/license to timing companies

**Context**: Need to determine business model

**Alternatives Considered**:
1. Internal tool only
2. Product for sale/license
3. Open source project
4. Service/rental model

**Rationale**:
- Timing market has commercial value
- Small team can support product sales
- Licensing provides recurring revenue potential
- Clear value proposition as RFID backup

**Consequences**:
- Requires production-grade quality
- Needs customer documentation
- Support infrastructure required
- OTA updates essential
- Pricing strategy needed

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-013: Road Running Focus (Initial)

**Decision**: Focus initially on road running events (5K, 10K, marathon)

**Context**: Need to scope initial target market

**Alternatives Considered**:
1. All race types immediately
2. Road running only
3. Triathlon focus
4. Trail/ultra focus

**Rationale**:
- Road running has most consistent bib visibility
- Largest market segment
- Most standardized bib formats
- Clear timing line (finish arch)
- Easier initial validation

**Consequences**:
- Bib detection can assume front-facing bibs
- May not handle wet/muddy bibs well (trail)
- Triathlon (multiple disciplines) deferred
- Can expand to other types in Phase 6+

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-014: YOLOv8 for Bib Detection

**Decision**: Use YOLOv8 (nano variant) for bib detection

**Context**: Need to select object detection model

**Alternatives Considered**:
1. YOLOv8n (nano) - Fast, good accuracy
2. YOLOv8s (small) - Better accuracy, slower
3. RT-DETR - Newer architecture
4. MobileNet-SSD - Very fast, lower accuracy
5. Custom architecture

**Rationale**:
- YOLOv8n provides good speed/accuracy tradeoff
- Excellent TensorRT support on Jetson
- Active development and community
- Easy to train on COCO format dataset
- Can upgrade to 's' variant if needed

**Consequences**:
- Training pipeline uses Ultralytics
- Export to TensorRT for production
- May need data augmentation for edge cases
- Model updates straightforward

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-015: PaddleOCR for Number Recognition

**Decision**: Use PaddleOCR for bib number recognition

**Context**: Need to select OCR solution

**Alternatives Considered**:
1. PaddleOCR - Fast, accurate, flexible
2. EasyOCR - Good accuracy, slower
3. Tesseract - Established, needs training
4. Custom CNN - Maximum control

**Rationale**:
- PaddleOCR handles varied fonts well
- Good speed on edge devices
- Pre-trained models work well on numbers
- Active development
- Lightweight deployment option (PP-OCRv3)

**Consequences**:
- PaddlePaddle dependency
- May need fine-tuning for specific bib fonts
- Lightweight model for edge deployment
- Consider custom training if accuracy insufficient

**Status**: Superseded by DD-018
**Date**: 2026-02-01

---

### DD-016: Virtual Timing Line (Calibrated Position)

**Decision**: Use virtual timing line (operator-calibrated position) rather than physical markers

**Context**: Need method to detect timing line crossing

**Alternatives Considered**:
1. Physical painted line
2. Colored tape marker
3. ArUco markers
4. Virtual line (calibrated position in frame)

**Rationale**:
- No physical setup required at timing point
- Works with any existing race infrastructure
- Operator can calibrate via Web UI
- No dependency on line visibility/condition
- Simpler deployment

**Consequences**:
- Camera must not move after calibration
- Operator training required for calibration
- May need recalibration if tripod bumped
- Consider ArUco as optional enhancement

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-017: Jetson Orin Nano for Production Hardware

**Decision**: Use NVIDIA Jetson Orin Nano (8GB) as primary production platform

**Context**: Need to select edge computing hardware

**Alternatives Considered**:
1. Jetson Orin Nano - GPU, good performance
2. Raspberry Pi 5 - Cheaper, no GPU
3. Intel NUC - High performance, expensive, power hungry
4. Coral Dev Board - TPU, limited compatibility

**Rationale**:
- GPU acceleration crucial for 60fps YOLO+OCR
- TensorRT provides significant speedup
- Good power efficiency for battery operation
- Strong ecosystem (JetPack)
- ~$500 acceptable for commercial product

**Consequences**:
- Higher BOM cost than RPi
- JetPack/TensorRT expertise needed
- Power consumption 7-15W
- RPi 5 as development/budget alternative

**Status**: Accepted
**Date**: 2026-02-01

---

### DD-018: Fine-Tuned Scene Text Recognition for OCR

**Decision**: Replace general-purpose OCR (EasyOCR/PaddleOCR) with a fine-tuned scene text
recognition model, evaluated across CRNN, PARSeq, and TrOCR candidates

**Context**: EasyOCR baseline achieved only 52.7% accuracy on bib crops (Phase 1 testing).
PaddleOCR (DD-015) was the original plan but was never deployed. With 10,853 verified bib
crops now available, fine-tuning a specialized model is feasible and likely to exceed 90%.

**Alternatives Considered**:
1. Keep EasyOCR with better preprocessing - Limited ceiling (~60-65%)
2. PaddleOCR PP-OCRv4 (pretrained) - Better than EasyOCR but still general-purpose
3. Fine-tune PaddleOCR SVTR - Tied to PaddlePaddle ecosystem
4. Fine-tune PyTorch scene text models (CRNN/PARSeq/TrOCR) - Best accuracy, ONNX portable

**Rationale**:
- General-purpose OCR fails on bib-specific challenges (small crops, angled text, header confusion)
- 10,853 labeled crops is sufficient for fine-tuning (exceeds 3,000 minimum by 3.6x)
- PyTorch models export to ONNX, deployable on both Jetson (TensorRT) and RPi 5 (ONNX Runtime)
- Three candidates cover the accuracy-vs-speed tradeoff: CRNN (8.3M, fastest), PARSeq
  (23.8M, best accuracy/speed), TrOCR (62M, strongest pretrained features)
- Digit-only vocabulary (10 digits + control tokens) dramatically reduces output space
- Fine-tuned model eliminates PaddlePaddle dependency (simplifies deployment)

**Consequences**:
- Adds PyTorch + transformers as training dependencies (in `ocr-eval` optional group)
- Deployment uses only ONNX Runtime (no PyTorch needed on device)
- Model selection depends on Phase 2.1 evaluation results
- May need retraining as new event data is collected (continuous improvement pipeline)

**Status**: Accepted
**Date**: 2026-02-04

---

### DD-019: YOLOv8n-Pose for Person Detection (Replacing MOG2)

**Decision**: Use YOLOv8n-pose for person detection and timing-line crossing detection, replacing MOG2 background subtraction

**Context**: MOG2 blob-based person detection tested on REC-0011 produced noisy results (1,199 crossings vs ~800-900 actual finishers). Over 50% were merged blobs — MOG2 cannot distinguish two runners side-by-side from one large blob. Additionally, blob centroids don't correspond to any anatomical landmark, making crossing timing imprecise.

**Alternatives Considered**:
1. MOG2 background subtraction (baseline) - Simple, zero GPU cost, but noisy and inaccurate
2. YOLOv8n (standard object detection) - Detects persons but no keypoints for precise crossing
3. YOLOv8n-pose (pose estimation) - Detects persons with skeleton keypoints, ~25-33ms on Jetson
4. MediaPipe Pose - Good keypoints but poor multi-person support

**Rationale**:
- Pose model provides skeleton keypoints (shoulders, hips) to compute chest point
- Chest point crossing is the official rule for finish-line timing
- Individual person bounding boxes eliminate merged-blob problem entirely
- ~25MB GPU memory — fits alongside bib detector on Jetson's 8GB
- Total pipeline ~40-60ms/frame (16-25fps) — sufficient for crossing detection

**Architecture**:
- Separate identity (bib OCR) from timing (chest crosses line)
- `PersistentPersonBibAssociator` links person tracks to bib numbers via voting
- `BibCrossingDeduplicator` prevents same bib from being reported twice
- Fallback: `--no-person-detect` uses bib tracker centroids for crossings

**Consequences**:
- Adds YOLOv8n-pose model download (~6MB weights, auto-downloaded by ultralytics)
- GPU is now shared between bib detector and pose model (sequential per frame)
- Frame rate drops from ~30fps (bib-only) to ~16-25fps (bib + pose)
- MOG2 classes deprecated but kept for backward compatibility

**Status**: Accepted
**Date**: 2026-02-13

---

### DD-020: Network Architecture - Join Operator's Hotspot

**Decision**: Jetson joins the operator's existing cellular hotspot rather than creating its own network or requiring a dedicated switch

**Context**: Need to determine how the operator accesses the PointCam web UI during races. Race timers typically use a cellular hotspot for internet access (scoring software is cloud-based, RFID readers connect via 4G to ChronoTrack servers). There is no local network switch at the finish line. The operator's laptop WiFi is already connected to the hotspot for scoring — they cannot switch to a separate PointCam network.

**Alternatives Considered**:
1. Jetson creates its own WiFi AP — operator must disconnect from hotspot to access UI, breaking scoring software connectivity
2. Dedicated network switch at finish line — adds hardware, changes operator workflow, most timers don't use one
3. Remote server — Jetson sends data to cloud server for UI. Adds latency, network dependency, more load (encoding + uploading), and a failure point
4. Jetson joins operator's existing hotspot — both devices on same network, no workflow change

**Rationale**:
- Zero change to operator's existing workflow (hotspot already running)
- Laptop accesses PointCam UI via browser tab on the same network it's already on
- Jetson gets internet access for CTP01 delivery to cloud scoring software
- No additional hardware needed
- Consistent with DD-001 (on-board processing) — all CV processing stays local, only timing data traverses the network
- Web server load on Jetson is negligible (~2-3% CPU, ~50MB RAM) compared to CV inference

**Network Topology**:
```
[Phone / MiFi Hotspot]
├── WiFi → Scoring laptop (scoring software + browser → pointcam.local)
├── WiFi → Jetson (CV pipeline + web UI + CTP01 client)
└── 4G  → Internet (ChronoTrack, cloud scoring)

[RFID Readers] → 4G → ChronoTrack (independent network)
```

**Implementation Notes**:
- mDNS/Avahi for `pointcam.local` hostname resolution
- Fallback: display IP address on Jetson boot (OLED or serial console)
- Support DHCP (default) and static IP configuration
- Watch for hotspot AP isolation (client isolation) — some carriers enable it by default, preventing device-to-device communication on the hotspot

**Consequences**:
- Depends on hotspot allowing device-to-device traffic (most phone hotspots do)
- PointCam UI requires same network as operator's device
- No UI access if hotspot is down (but scoring software is also down in that case)
- Future option: Jetson can run WiFi AP simultaneously via wlan0 for tablet access during setup, while connected to hotspot via ethernet adapter

**Status**: Accepted
**Date**: 2026-02-14

---

### DD-021: False Positive Mitigation Strategy

**Decision**: Address false positive bib detections (jersey numbers, finish line clocks, signage) through a layered approach: existing pipeline filters first, static region suppression planned, ROI exclusion zones deferred to UI phase

**Context**: Testing with race videos (e.g., Giants 5K) identified two major false positive sources: (1) sports jersey numbers worn by runners — same chest location as bibs, persistently visible, and reinforced by temporal voting; (2) finish line clock digits — static numeric display in the frame. Additional sources include sponsor banners, course markers, and spectator clothing.

**Existing Defenses** (already implemented):
- `BibSetValidator` — rejects numbers not in registration list (-0.2 confidence penalty)
- `SuspiciousPredictionFilter` — short predictions (1-2 digits, typical of jerseys) require ≥0.85 confidence
- `PersistentPersonBibAssociator` — requires spatial match to a tracked person (helps with static clocks)
- `CropQualityFilter` — aspect ratio, blur, contrast checks filter non-bib regions
- `EnhancedTemporalVoting` — noise from single frames gets outvoted

**Planned: Static Region Suppression**:
- Detections appearing at the same pixel coordinates across many frames without moving are likely signs/clocks
- Heatmap of detection locations over time, suppress regions with persistent stationary detections
- No operator configuration needed — fully automatic

**Deferred: ROI Exclusion Zones** (to Phase 3 - UI):
- Operator draws exclusion masks over known false positive areas (clocks, banners)
- Operator defines detection zones to restrict where in the frame to look for runners
- Requires web UI for setup — deferred until Milestone 3.1

**Remaining Gap: Jersey Numbers**:
- Jerseys are worn on the chest (same as bibs), move with the runner, and are persistent
- Best long-term defense is the bib detection model itself — trained on paper bibs, it should learn to distinguish from jersey printing
- Including negative examples (jerseys, signage) in training data would strengthen this
- Digit count helps (jerseys: 1-2 digits; bibs: typically 3-5) but isn't foolproof
- Bib set validation is the strongest existing defense for jersey numbers

**Status**: Accepted
**Date**: 2026-02-14

---

### DD-022: Crossing Detection Overhaul — Hysteresis, Track Dedup, and Association Fixes

**Decision**: Address the five failure modes identified in Giants 5K Run 3 ground truth comparison (7.7% bib recall, 44% precision) through targeted fixes to crossing detection, deduplication, and bib-person association.

**Context**: Running the full 30-min Giants 5K video (480 finishers) against ground truth revealed that crossing detection — not OCR — is the primary bottleneck. 62% of bibs are seen by OCR at some point but only 22% make it to a crossing event. Additionally, persons lingering at the timing line generate repeated false crossings, and UNKNOWN crossings are never deduplicated.

**Problem Breakdown**:

| Failure Mode | Impact | Evidence |
|---|---|---|
| Timing line oscillation | 15 tracks cross multiple times (28 extra crossings) | Track 223: 10 crossings over 158s, all at chest_x ≈ 0.52 |
| UNKNOWN never deduped | 101 UNKNOWN crossings, many from same person | `should_emit()` returns True immediately for UNKNOWN |
| Bib-person association fails | 40% of detected bibs never link to a crossing | 20/50 GT bibs seen by OCR but no crossing event |
| Bib detector misses | 38% of GT bibs never detected at all | 19/50 GT bibs absent from detections CSV |
| OCR misreads as valid bibs | 54 false positive bib numbers | Single-digit fragments (1, 2, 6) are top "detections" |

**Fixes — Priority Order**:

**Fix 1: Crossing hysteresis (High priority)**
- Current: `CrossingDetector.check()` fires when `prev_side != current_side` — a single frame flip triggers a crossing
- Problem: chest keypoint jitter at the timing line causes repeated flips
- Solution: require the track to remain on the new side for N consecutive frames (e.g., 3-5 frames = 100-170ms) before confirming a crossing. This is a standard hysteresis/debounce pattern for noisy signals
- Implementation: add a `_pending_crossing` dict tracking `{track_id: (new_side, first_frame, consecutive_count)}`; only fire when `consecutive_count >= hysteresis_frames`
- Expected impact: eliminates track 223's 10 crossings → 1 or 0

**Fix 2: UNKNOWN deduplication by track ID (High priority)**
- Current: `BibCrossingDeduplicator.should_emit()` returns True immediately for UNKNOWN
- Problem: same person crosses multiple times with no bib → multiple UNKNOWN events
- Solution: add per-track debounce for UNKNOWN crossings using `person_track_id`. Maintain a `_track_last_crossing` dict; suppress UNKNOWN if same track crossed within `debounce_frames`
- Implementation: in `should_emit()`, accept an optional `track_id` parameter; for UNKNOWN, debounce by track_id instead of bib_number
- Expected impact: reduces 101 UNKNOWN → ~70 (removing multi-crossing duplicates)

**Fix 3: Wider bib-person association radius (Medium priority)**
- Current: `max_distance=150px` for matching bibs to persons. Bib must be inside person bbox or within 150px
- Problem: at 1920px width, runners at the timing line may have bibs detected 200-300px from their person bbox center (different YOLO models, different bbox scales)
- Solution: increase `max_distance` to 200-250px, or scale it relative to person bbox height
- Also: improve `_find_best_match()` to prefer bibs that overlap the person's torso region (upper half of person bbox) rather than just proximity to bbox center
- Expected impact: recovers some of the 40% of detected-but-unlinked bibs

**Fix 4: Suppress short OCR fragments at crossing time (Medium priority)**
- Current: bibs like "1", "2", "6" pass validation because they're in the bib set (1-1678 range)
- Problem: these are almost always single-digit OCR fragments, not real bibs. Bib "1" has 35 tracks, "6" has 13 — wildly more than any real bib
- Solution: at crossing emission time, require bibs with fewer digits than the race's typical digit count to have consensus confidence ≥ 0.9 (or require stability). Alternatively, tag short-digit consensus results that were never validated by the bib set's fuzzy matching
- Implementation: add a `min_crossing_confidence` per digit count (e.g., 1-digit: 0.95, 2-digit: 0.85, 3-digit: 0.60, 4-digit: 0.50)
- Expected impact: eliminates most of the 54 false positive bib numbers

**Fix 5: Timing line zone instead of line (Low priority, future)**
- Current: timing line is a 1D line; crossing = side transition
- Problem: a person must cross exactly that pixel column. If they cross between frames (30fps = 33ms gaps), they may never be seen transitioning
- Solution: define a timing *zone* (e.g., 50px wide band). Enter-zone + exit-zone = crossing. More forgiving than exact line crossing
- Expected impact: may recover some of the 57% of finishers not detected crossing

**Alternatives Considered**:
1. **Increase pose detection confidence threshold**: Would reduce false person tracks but also miss real people. Not recommended — person detection recall is already the bottleneck.
2. **Use bib tracker for crossings instead of person tracker**: Reverts to the old approach. Problem: bib tracker has worse position accuracy (bib centroid ≠ person position) and bibs may not be visible at crossing moment.
3. **Add a second timing line for confirmation**: Two lines N pixels apart, require both crossed. Reduces false crossings but also reduces recall. Too aggressive given current 43% recall.

**Implementation Order**:
1. Fixes 1+2 together (hysteresis + UNKNOWN dedup) — quick code changes, biggest impact
2. Fix 4 (short fragment suppression) — simple confidence threshold at emission
3. Fix 3 (wider association) — tune parameters
4. Fix 5 (timing zone) — requires more design work, defer to next iteration

**Consequences**:
- Hysteresis adds 3-5 frame latency (~100-170ms) to crossing detection — acceptable given ±1s timing tolerance
- UNKNOWN dedup by track means some legitimate re-crossings (e.g., relay runners) would be suppressed — acceptable for 5K (no re-crossings)
- Short fragment suppression may filter legitimate short bibs (e.g., bib "1") if confidence is marginal — acceptable tradeoff given the 54 false positives currently

**Status**: Accepted
**Date**: 2026-02-14

---

## Pending Decisions

### DD-P01: Multi-Camera Coordination (Future)

**Status**: Deferred to Phase 6

**Context**: May want multiple cameras at same timing point

**Considerations**:
- Redundancy for reliability
- Different angles for occlusion handling
- Consensus algorithm for conflicts
- Increased cost and complexity

---

### DD-P02: Lap Counting Implementation (Future)

**Status**: Deferred to Phase 6

**Context**: Multi-lap races need lap counting

**Considerations**:
- Re-identification of same bib
- Minimum lap time validation
- Display/reporting of lap data
- CTP01 lap field usage

---

## Template for New Decisions

```markdown
### DD-XXX: [Decision Title]

**Decision**: [What was decided]

**Context**: [Why this decision was needed]

**Alternatives Considered**:
1. Option 1
2. Option 2
3. Option 3

**Rationale**: [Why this decision was made]

**Consequences**: [Trade-offs and implications]

**Status**: [Proposed | Accepted | Deprecated | Superseded]
**Date**: YYYY-MM-DD
```

---

## Decision Status Key

- **Proposed**: Under consideration, not yet decided
- **Accepted**: Decision made and being implemented
- **Deprecated**: No longer valid, kept for historical reference
- **Superseded**: Replaced by a newer decision (reference DD-XXX)
- **Deferred**: Postponed to a future phase
