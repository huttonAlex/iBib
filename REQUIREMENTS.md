# Requirements & Specifications

## Product Overview

**PointCam** is a computer vision-based race timing system designed as a backup/supplement to RFID timing systems. It uses cameras to visually detect race bib numbers and timing line crossings.

**Target Market**: Race timing companies using MYLAPS/ChronoTrack systems
**Business Model**: Product for sale/license to timing companies
**Primary Use Case**: Backup finish line timing or intermediate checkpoint timing

---

## Functional Requirements

### FR-1: Bib Number Detection

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | System MUST detect race bibs in camera frame | Must |
| FR-1.2 | System MUST read numbers from detected bibs | Must |
| FR-1.3 | System MUST handle various bib fonts, sizes, and colors | Must |
| FR-1.4 | System MUST provide confidence scores (0-1) for each detection | Must |
| FR-1.5 | System MUST handle partially obscured bibs (best effort) | Should |
| FR-1.6 | System SHOULD handle motion blur at typical running speeds | Should |

### FR-2: Timing Line Recognition

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | System MUST identify the timing line/point in camera view | Must |
| FR-2.2 | System MUST detect when a participant crosses the timing line | Must |
| FR-2.3 | System MUST associate crossings with correct bib numbers | Must |
| FR-2.4 | System MUST handle up to 10 simultaneous crossings | Must |
| FR-2.5 | System MUST support configurable timing line position | Must |
| FR-2.6 | Typical scenario: 1-3 simultaneous crossings | Info |

### FR-3: Timing & Position Accuracy

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | System MUST maintain correct finishing order/position | Must |
| FR-3.2 | System MUST achieve ±1 second timestamp accuracy | Must |
| FR-3.3 | System MUST prioritize position accuracy over absolute time | Must |
| FR-3.4 | System MUST timestamp all crossing events with GPS-synchronized time | Must |
| FR-3.5 | Position accuracy is PRIMARY, absolute time is SECONDARY | Info |

### FR-4: Data Output & Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | System MUST output: bib number, timestamp, confidence score | Must |
| FR-4.2 | System MUST implement CTP01 protocol (ChronoTrack Socket Protocol) | Must |
| FR-4.3 | System MUST support real-time push mode streaming | Must |
| FR-4.4 | System MUST support sequence numbers for connection recovery | Must |
| FR-4.5 | System MUST queue data locally during connectivity loss | Must |
| FR-4.6 | System MUST sync queued data when connectivity restored | Must |
| FR-4.7 | System SHOULD support CSV export for offline analysis | Should |

### FR-5: Unknown/Low-Confidence Handling

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | System MUST log crossing with timestamp even if bib unreadable | Must |
| FR-5.2 | System MUST mark unreadable bibs as "UNKNOWN" in data stream | Must |
| FR-5.3 | System MUST capture evidence frame for unknown detections | Must |
| FR-5.4 | System SHOULD alert operator on low-confidence detections | Should |

### FR-6: Evidence Capture

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | System MUST support configurable evidence frame capture | Must |
| FR-6.2 | Capture modes: ALL, LOW_CONFIDENCE_ONLY, NONE | Must |
| FR-6.3 | System MUST store frames with crossing event association | Must |
| FR-6.4 | System MUST provide evidence retrieval for dispute resolution | Must |
| FR-6.5 | System SHOULD support configurable retention period | Should |

### FR-7: Configuration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | System MUST support per-race configuration | Must |
| FR-7.2 | Configurable: confidence threshold, evidence mode, location name | Must |
| FR-7.3 | System MUST persist configuration across restarts | Must |
| FR-7.4 | System SHOULD support configuration via web interface | Should |

---

## Performance Requirements

### PR-1: Processing Speed

| ID | Requirement | Priority |
|----|-------------|----------|
| PR-1.1 | Real-time processing on-board (no server dependency) | Must |
| PR-1.2 | Minimum 30 fps effective detection rate | Must |
| PR-1.3 | Target 60 fps camera capture | Should |
| PR-1.4 | End-to-end latency < 500ms from crossing to data output | Should |

### PR-2: Scale

| ID | Requirement | Priority |
|----|-------------|----------|
| PR-2.1 | Handle races up to 3,000 participants | Must |
| PR-2.2 | Track up to 10 simultaneous timing line crossings | Must |
| PR-2.3 | Continuous operation for 8+ hours | Must |
| PR-2.4 | Typical: 1-3 simultaneous crossings | Info |

### PR-3: Reliability

| ID | Requirement | Priority |
|----|-------------|----------|
| PR-3.1 | Target >85% detection accuracy in production | Must |
| PR-3.2 | Zero data loss during connectivity interruptions | Must |
| PR-3.3 | Graceful degradation under processing overload | Should |
| PR-3.4 | Automatic recovery from transient failures | Should |

---

## Non-Functional Requirements

### NFR-1: Hardware & Form Factor

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-1.1 | Small, portable form factor | Must |
| NFR-1.2 | Standard tripod mountable | Must |
| NFR-1.3 | Camera positioned at bib height | Must |
| NFR-1.4 | Weather resistant enclosure (IP54 minimum) | Should |
| NFR-1.5 | GPS module for time synchronization | Must |
| NFR-1.6 | Cellular modem for data transmission | Must |

### NFR-2: Power

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-2.1 | Battery operated for 6+ hours minimum | Must |
| NFR-2.2 | Support external power connection | Should |
| NFR-2.3 | Graceful shutdown on low battery | Must |
| NFR-2.4 | Battery level indicator/reporting | Should |

### NFR-3: Connectivity

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-3.1 | Primary: Cellular (4G/LTE) data connection | Must |
| NFR-3.2 | Offline-first architecture | Must |
| NFR-3.3 | Automatic reconnection on connectivity loss | Must |
| NFR-3.4 | Connection status indication | Must |

### NFR-4: Environmental

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-4.1 | Operating temperature: 0°C to 40°C | Must |
| NFR-4.2 | Operate in various lighting (dawn, midday, dusk, overcast) | Must |
| NFR-4.3 | Handle direct sunlight (with shade hood) | Should |
| NFR-4.4 | Light rain operation (IP54) | Should |

### NFR-5: Usability

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-5.1 | Setup time under 15 minutes by trained operator | Must |
| NFR-5.2 | Simple calibration process | Must |
| NFR-5.3 | Web-based operator interface (accessible from phone/tablet) | Must |
| NFR-5.4 | Visual status indicators (LEDs or display) | Must |
| NFR-5.5 | Alert system for issues (audio/visual) | Should |

### NFR-6: Time Synchronization

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-6.1 | GPS-based time synchronization | Must |
| NFR-6.2 | Time accuracy: ±100ms absolute | Should |
| NFR-6.3 | Continue operation during GPS signal loss | Must |
| NFR-6.4 | Clock drift compensation during GPS outage | Should |

---

## Integration Requirements

### IR-1: CTP01 Protocol Compliance

| ID | Requirement | Priority |
|----|-------------|----------|
| IR-1.1 | Implement CTP01 as TCP client | Must |
| IR-1.2 | Default port: 61611 | Must |
| IR-1.3 | Support push mode streaming | Must |
| IR-1.4 | Support sequence numbers for stream recovery | Must |
| IR-1.5 | Support connection-id for reconnection handling | Must |
| IR-1.6 | Support configurable time format (normal, iso, unix, msecs) | Should |
| IR-1.7 | Support location naming for multi-point identification | Must |

### IR-2: Data Format

| ID | Requirement | Priority |
|----|-------------|----------|
| IR-2.1 | Tag event format compatible with CT01_33 | Must |
| IR-2.2 | Field separator: tilde (~) | Must |
| IR-2.3 | Line terminator: CRLF (\r\n) | Must |
| IR-2.4 | Include bib number in tagcode field position | Must |

### IR-3: Scoring Software Compatibility

| ID | Requirement | Priority |
|----|-------------|----------|
| IR-3.1 | Compatible with ChronoTrack-based scoring software | Must |
| IR-3.2 | Compatible with MYLAPS-based systems | Should |
| IR-3.3 | CSV export for manual import workflows | Should |

---

## Commercial Requirements

### CR-1: Product Packaging

| ID | Requirement | Priority |
|----|-------------|----------|
| CR-1.1 | Complete hardware unit ready to deploy | Must |
| CR-1.2 | Operator manual (printed and digital) | Must |
| CR-1.3 | Quick start guide | Must |
| CR-1.4 | Carrying case/protection | Should |

### CR-2: Support & Maintenance

| ID | Requirement | Priority |
|----|-------------|----------|
| CR-2.1 | Remote firmware update capability (OTA) | Must |
| CR-2.2 | Remote diagnostics/log retrieval | Should |
| CR-2.3 | Unit identification for fleet management | Must |
| CR-2.4 | Simple naming system for unit identification | Must |

### CR-3: Documentation

| ID | Requirement | Priority |
|----|-------------|----------|
| CR-3.1 | Operator manual for timing staff | Must |
| CR-3.2 | Integration guide for software vendors | Should |
| CR-3.3 | Troubleshooting guide | Must |
| CR-3.4 | Field setup best practices | Must |

---

## System Constraints

### SC-1: Deployment Role

- Primary deployment: BACKUP to RFID systems
- Must not interfere with primary RFID operations
- Future: May evolve to primary timing system

### SC-2: Operator Profile

- Trained timing company staff
- Familiar with race timing concepts
- Technical but not software developers

### SC-3: Race Types (Initial Focus)

- Road running events (5K, 10K, Half Marathon, Marathon)
- Finish line and intermediate checkpoint timing
- Participant count: up to 3,000

---

## Future Considerations (Out of Scope for v1)

| Feature | Notes |
|---------|-------|
| Lap counting | Nice to have for multi-lap races |
| Multi-camera per timing point | Different angles for redundancy |
| Photo finish (sub-second) | Would require different camera/approach |
| Facial recognition | Privacy concerns, not planned |
| Primary timing mode | Depends on accuracy achievements |
| Triathlon/multisport | Different bib visibility challenges |

---

## Acceptance Criteria

### Phase 1: Proof of Concept

- [ ] Successfully read bib numbers from existing 1000-image dataset (>80% accuracy)
- [ ] Detect timing line in controlled conditions
- [ ] Record single runner crossing with GPS-synced timestamp
- [ ] Basic CTP01 message generation

### Phase 2: Field Testing

- [ ] Successfully read bibs from moving runners (>70% accuracy)
- [ ] Handle multiple simultaneous crossings (3+)
- [ ] Maintain correct position order
- [ ] Real-time processing on selected hardware
- [ ] Offline queue functioning

### Phase 3: Backup Deployment

- [ ] Successfully operate as backup in small race (<500 people)
- [ ] CTP01 integration with scoring software working
- [ ] >85% detection accuracy in field conditions
- [ ] Zero data loss during connectivity gaps
- [ ] Operator web interface functional

### Phase 4: Production Ready

- [ ] Operate reliably in races up to 3,000 people
- [ ] Weather resistant operation
- [ ] 6+ hour battery life
- [ ] OTA update capability
- [ ] Production documentation complete
- [ ] Beta customer feedback incorporated
