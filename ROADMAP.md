# Development Roadmap

## Overview

This roadmap outlines the development path for PointCam from proof-of-concept to commercial product. The project follows an iterative approach with clear phase gates.

**Team**: Small team (2-5 people)
**Business Model**: Product for sale/license
**Current Status**: Phase 1.2 - Custom Model Training (OCR fine-tuning)

---

## Phase 1: Proof of Concept
**Goal**: Validate core CV technologies using existing dataset

### Milestone 1.1: Baseline Evaluation (COMPLETE)
**Status**: Complete
**Assets**: 1000 tagged images in COCO format, 10,853 verified bib crops from 4 events

**Deliverables:**
- [x] Evaluate existing dataset for training/test split
- [x] Benchmark off-the-shelf bib detection models (YOLOv8 Nano: 97.3% mAP50)
- [x] Benchmark OCR solutions on cropped bibs (EasyOCR: 52.7%, TrOCR: tested)
- [x] Document baseline accuracy metrics
- [x] Identify dataset gaps → collected 10,853 crops from 4 events (3.6x target)

**Results:**
- Detection: 97.3% mAP50 — production-ready
- OCR: 52.7% (EasyOCR) — bottleneck, requires fine-tuned model
- Data: 10,853 verified crops, 4 events, 6 camera positions, 1-4 digit bibs

---

### Milestone 1.2: Custom Model Training (IN PROGRESS)
**Status**: OCR fine-tuning pipeline built, ready to execute

**Deliverables:**
- [x] Train YOLOv8 bib detector on dataset (97.3% mAP50 — done)
- [x] Build OCR evaluation framework (`scripts/evaluate_ocr_models.py`)
- [x] Build dataset preparation pipeline (`scripts/prepare_ocr_dataset.py`)
- [x] Build fine-tuning pipeline (`scripts/finetune_ocr.py`)
- [x] Build ONNX export tool (`scripts/export_ocr_model.py`)
- [ ] Prepare unified dataset with stratified splits
- [ ] Evaluate pretrained baselines (TrOCR, PARSeq)
- [ ] Fine-tune CRNN, PARSeq, TrOCR on 10,853 bib crops
- [ ] Select best model and export to ONNX
- [ ] Measure end-to-end detection + OCR accuracy
- [ ] Document model performance vs requirements

**Success Criteria:**
- Bib detection mAP > 0.85 on test set (ACHIEVED: 97.3%)
- OCR accuracy > 80% on static detected bibs (target: 90%+)
- Clear understanding of failure cases

---

### Milestone 1.3: Timing Line & Crossing Detection
**Deliverables:**
- [ ] Implement virtual timing line (configurable position)
- [ ] Implement basic centroid tracking
- [ ] Implement crossing detection logic
- [ ] Test with simulated video sequences

**Success Criteria:**
- Crossing detection working in controlled video
- Position ordering correct for multi-person crossings
- Basic pipeline end-to-end functional

---

### Milestone 1.4: CTP01 Protocol Prototype
**Deliverables:**
- [ ] Implement CTP01 message generation
- [ ] Test with ChronoTrack/scoring software (if available)
- [ ] Implement basic offline queue (in-memory)
- [ ] Document protocol compliance status

**Success Criteria:**
- Valid CTP01 messages generated
- Messages accepted by scoring software (or simulator)
- Sequence numbers working correctly

**Phase 1 Exit Gate:**
- [ ] All milestones complete
- [ ] Static image accuracy > 80%
- [ ] CTP01 integration proven
- [ ] Hardware requirements estimated
- [ ] **Decision: Proceed to Phase 2**

---

## Phase 2: Hardware Integration & Real-World Testing
**Goal**: Build working prototype, test in real conditions

### Milestone 2.1: Hardware Platform Setup
**Deliverables:**
- [ ] Select and procure edge computing platform (Jetson Orin Nano recommended)
- [ ] Select and procure camera module
- [ ] Select and procure GPS module
- [ ] Assemble development prototype
- [ ] Benchmark inference performance

**Success Criteria:**
- Real-time inference achieved (30+ fps)
- GPS time synchronization working
- Stable operation for 2+ hours continuous

---

### Milestone 2.2: Real-Time Pipeline
**Deliverables:**
- [ ] Port CV pipeline to edge device
- [ ] Optimize for real-time performance (TensorRT/ONNX)
- [ ] Implement frame capture and preprocessing
- [ ] Implement GPS timestamp integration
- [ ] Profile and optimize bottlenecks

**Success Criteria:**
- 30+ fps sustained processing
- GPS timestamps accurate to ±100ms
- Memory usage stable over time

---

### Milestone 2.3: Motion & Multi-Runner Testing
**Deliverables:**
- [ ] Test with video of moving runners
- [ ] Tune motion blur handling
- [ ] Test multi-person tracking (3+ simultaneous)
- [ ] Validate position ordering accuracy
- [ ] Collect failure case examples

**Success Criteria:**
- > 70% detection accuracy on moving subjects
- Correct position ordering in multi-person scenarios
- Documented failure modes and conditions

---

### Milestone 2.4: Communication Layer
**Deliverables:**
- [ ] Implement persistent offline queue (SQLite)
- [ ] Implement cellular modem integration
- [ ] Implement CTP01 client with reconnection
- [ ] Test offline → online sync
- [ ] Implement connection status monitoring

**Success Criteria:**
- Zero data loss during connectivity gaps
- Automatic reconnection and replay
- Sequence numbers maintained correctly

---

### Milestone 2.5: Controlled Field Test
**Deliverables:**
- [ ] Conduct controlled test (20-50 people)
- [ ] Compare results with manual timing
- [ ] Measure real-world detection accuracy
- [ ] Gather setup/operation feedback
- [ ] Document lessons learned

**Success Criteria:**
- > 70% detection accuracy in field
- Correct position ordering
- Setup time < 20 minutes
- Identified improvement areas

**Phase 2 Exit Gate:**
- [ ] All milestones complete
- [ ] Working hardware prototype
- [ ] Field-validated accuracy metrics
- [ ] Offline queue proven reliable
- [ ] **Decision: Proceed to Phase 3**

---

## Phase 3: Operator Interface & Small Race Deployment
**Goal**: Deploy as backup system in real races

### Milestone 3.1: Web UI Implementation
**Deliverables:**
- [ ] Implement FastAPI backend
- [ ] Build dashboard page (status, recent crossings)
- [ ] Build setup page (timing line calibration)
- [ ] Build evidence viewer
- [ ] Build settings page
- [ ] Test on mobile devices (phone/tablet)

**Success Criteria:**
- Operator can configure unit via phone browser
- Real-time status visible
- Timing line calibration working
- Evidence photos viewable

---

### Milestone 3.2: Evidence Capture System
**Deliverables:**
- [ ] Implement configurable evidence capture
- [ ] Implement storage management
- [ ] Implement evidence retrieval API
- [ ] Implement auto-cleanup policy
- [ ] Test storage usage over long duration

**Success Criteria:**
- Evidence capture working in all modes
- Storage usage predictable
- Evidence retrievable for review

---

### Milestone 3.3: System Integration Testing
**Deliverables:**
- [ ] Full integration test with scoring software
- [ ] Test all failure scenarios
- [ ] Test GPS signal loss handling
- [ ] Test cellular dropout handling
- [ ] Test power loss recovery
- [ ] Document operational procedures

**Success Criteria:**
- System recovers from all failure modes
- Data integrity maintained
- Operational runbook drafted

---

### Milestone 3.4: Small Race Deployment (Beta 1)
**Deliverables:**
- [ ] Deploy as backup at small race (< 500 people)
- [ ] Monitor performance vs RFID
- [ ] Capture and analyze misses
- [ ] Gather operator feedback
- [ ] Document real-world issues

**Success Criteria:**
- Zero interference with RFID
- > 75% detection accuracy
- Operator finds system usable
- Issues documented for resolution

---

### Milestone 3.5: Iteration & Refinement
**Deliverables:**
- [ ] Address issues from Beta 1
- [ ] Improve accuracy based on failure analysis
- [ ] Enhance UI based on feedback
- [ ] Second small race deployment (Beta 2)
- [ ] Validate improvements

**Success Criteria:**
- > 85% detection accuracy
- Key issues resolved
- Operator confidence increased

**Phase 3 Exit Gate:**
- [ ] All milestones complete
- [ ] Deployed in 2+ small races successfully
- [ ] > 85% detection accuracy
- [ ] Operator feedback positive
- [ ] **Decision: Proceed to Phase 4**

---

## Phase 4: Production Hardening
**Goal**: Prepare for commercial release

### Milestone 4.1: Enclosure & Weatherproofing
**Deliverables:**
- [ ] Design weatherproof enclosure (IP54)
- [ ] Test in rain/wet conditions
- [ ] Test temperature range (0-40°C)
- [ ] Integrate power management
- [ ] Validate battery life (6+ hours)

**Success Criteria:**
- IP54 rating achieved or equivalent
- Operates in specified temperature range
- 6+ hour battery life confirmed

---

### Milestone 4.2: OTA Update System
**Deliverables:**
- [ ] Implement firmware update mechanism
- [ ] Implement remote configuration
- [ ] Implement remote log retrieval
- [ ] Test update rollback capability
- [ ] Document update procedures

**Success Criteria:**
- Updates deployable remotely
- Rollback works if update fails
- Configuration changeable remotely

---

### Milestone 4.3: Medium Race Deployment
**Deliverables:**
- [ ] Deploy at medium race (500-1500 people)
- [ ] Validate scale performance
- [ ] Stress test peak crossing density
- [ ] Full-day operation test
- [ ] Document scalability limits

**Success Criteria:**
- Maintains accuracy at scale
- Handles peak loads
- Stable throughout race
- No data loss

---

### Milestone 4.4: Production Documentation
**Deliverables:**
- [ ] Complete operator manual
- [ ] Quick start guide
- [ ] Troubleshooting guide
- [ ] Field setup best practices
- [ ] Integration guide (for software vendors)

**Success Criteria:**
- Documentation sufficient for trained operators
- Common issues addressed
- Setup procedures clear

**Phase 4 Exit Gate:**
- [ ] All milestones complete
- [ ] Weather-resistant unit
- [ ] OTA updates working
- [ ] Documentation complete
- [ ] **Decision: Proceed to Phase 5**

---

## Phase 5: Commercial Launch
**Goal**: Release product to market

### Milestone 5.1: Beta Customer Program
**Deliverables:**
- [ ] Identify 3-5 beta customers (timing companies)
- [ ] Provide units for extended evaluation
- [ ] Gather feedback over 2-3 race season events
- [ ] Iterate based on feedback
- [ ] Document case studies

**Success Criteria:**
- Beta customers successfully using system
- Feedback incorporated into product
- Case studies document success

---

### Milestone 5.2: Large Race Validation
**Deliverables:**
- [ ] Deploy at large race (1500-3000 people)
- [ ] Full production conditions
- [ ] Third-party validation of accuracy
- [ ] Document as reference deployment

**Success Criteria:**
- > 85% detection accuracy maintained
- Successful backup operation
- Reference deployment documented

---

### Milestone 5.3: Production Manufacturing
**Deliverables:**
- [ ] Finalize hardware BOM
- [ ] Establish manufacturing/assembly process
- [ ] Quality control procedures
- [ ] Packaging and shipping
- [ ] Initial production run (10-20 units)

**Success Criteria:**
- Repeatable production process
- QC procedures defined
- Units ready for shipment

---

### Milestone 5.4: Commercial Release
**Deliverables:**
- [ ] Pricing finalized
- [ ] Sales process defined
- [ ] Support process defined
- [ ] Marketing materials
- [ ] First commercial sales

**Success Criteria:**
- Product available for purchase
- Support infrastructure in place
- First paying customers

**Phase 5 Exit Gate:**
- [ ] All milestones complete
- [ ] Product commercially available
- [ ] Support processes working
- [ ] **Product launched!**

---

## Future Phases (Post-Launch)

### Phase 6: Feature Expansion
- Lap counting for multi-lap races
- Multi-camera per timing point
- Primary timing mode (higher accuracy)
- Cloud backup/sync option

### Phase 7: Platform Expansion
- Triathlon/multisport support
- Trail/ultra event support
- Different race types

---

## Risk Management

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OCR accuracy insufficient | Medium | High | Early testing with real bibs, fallback to RFID |
| Edge device too slow | Low | High | Hardware options, model optimization |
| Motion blur problematic | Medium | Medium | Camera selection, preprocessing |
| CTP01 integration issues | Low | Medium | Early testing, protocol well-documented |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Complex setup deters adoption | Medium | High | Focus on UX, trained operators |
| Weather damage | Medium | Medium | IP54 enclosure, shade hood |
| Cellular coverage gaps | Medium | Low | Offline queue handles gracefully |

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Market not ready | Low | High | Start as backup (lower barrier) |
| Competition | Medium | Medium | Focus on integration, reliability |
| Pricing challenges | Medium | Medium | Beta customer feedback on value |

---

## Success Metrics

| Phase | Key Metric | Target |
|-------|------------|--------|
| Phase 1 | Static OCR accuracy | > 80% |
| Phase 2 | Motion OCR accuracy | > 70% |
| Phase 3 | Field detection rate | > 85% |
| Phase 4 | Reliability | 99% uptime in race |
| Phase 5 | Customer satisfaction | > 4/5 rating |

---

## Dependencies & Prerequisites

### Phase 1 Prerequisites
- [x] COCO dataset available (1000 images)
- [x] Development environment setup
- [x] OCR training data collected (10,853 verified crops)
- [ ] Access to scoring software for integration testing

### Phase 2 Prerequisites
- [ ] Hardware procurement budget approved
- [ ] Test video footage of runners

### Phase 3 Prerequisites
- [ ] Access to small race for testing
- [ ] Relationship with timing company (internal or partner)

### Phase 4 Prerequisites
- [ ] Enclosure design/manufacturing capability
- [ ] Access to medium race

### Phase 5 Prerequisites
- [ ] Beta customer relationships
- [ ] Manufacturing/assembly capability
- [ ] Sales and support infrastructure
