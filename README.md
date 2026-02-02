# PointCam - Photo Timing Point System

A computer vision-based race timing backup system that uses cameras to detect race bib numbers and timing line crossings as a supplement/backup to RFID timing systems.

## Overview

**Problem**: RFID race timing systems are reliable but require backup solutions that aren't susceptible to the same failure modes (antenna issues, tag failures, reader malfunctions).

**Solution**: Small form-factor camera system that:
- Visually detects and reads race bib numbers
- Recognizes timing point markers
- Records when participants cross timing lines
- Streams data to scoring software via CTP01 protocol
- Provides position-accurate timing data as RFID backup

## Key Features

- **Real-time on-board processing** - No server dependency
- **CTP01 protocol integration** - Compatible with ChronoTrack/MYLAPS systems
- **Offline-first architecture** - Handles cellular connectivity gaps
- **GPS time synchronization** - Accurate timestamps without network
- **Web-based operator interface** - Configure via phone/tablet
- **Configurable evidence capture** - Photos for dispute resolution
- **±1 second timing accuracy** with high position accuracy

## Target Use Cases

- Finish line backup timing
- Intermediate checkpoint timing
- Road running events (5K, 10K, Half Marathon, Marathon)
- Races up to 3,000 participants

## Current Status

**Phase**: 1 - Proof of Concept
**Status**: Documentation complete, ready for development

### Available Assets
- 1000 tagged bib images (COCO format) for training
- CTP01 protocol documentation

## Documentation

### Core Documents
- [REQUIREMENTS.md](REQUIREMENTS.md) - Detailed requirements and specifications
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and technical design
- [ROADMAP.md](ROADMAP.md) - Development phases and milestones

### Technical Documents
- [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) - Technical decision log (17 decisions)
- [docs/INTEGRATION_PROTOCOL.md](docs/INTEGRATION_PROTOCOL.md) - CTP01 protocol implementation
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) - All configuration options
- [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) - Developer handbook
- [docs/TECH_NOTES.md](docs/TECH_NOTES.md) - Technical findings and experiments

## Hardware (Planned)

| Component | Recommendation | Est. Cost |
|-----------|---------------|-----------|
| Compute | NVIDIA Jetson Orin Nano (8GB) | $500 |
| Camera | High frame rate (60fps), good low-light | $50-200 |
| GPS | u-blox NEO-M9N with PPS | $40 |
| Cellular | 4G/LTE modem | $50 |
| Battery | 20000mAh USB-C PD | $50 |
| Enclosure | IP54 weatherproof | $50-100 |
| **Total** | | **$800-1000** |

## Software Stack

- **Language**: Python 3.10+
- **Object Detection**: YOLOv8 (nano)
- **OCR**: PaddleOCR
- **ML Inference**: TensorRT (Jetson) / ONNX Runtime
- **Web UI**: FastAPI
- **Database**: SQLite
- **Protocol**: CTP01 (ChronoTrack Socket Protocol)

## Quick Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PointCam Unit                        │
├─────────────────────────────────────────────────────────┤
│  Camera → CV Pipeline → Timing Engine → CTP01 Client   │
│              ↑              ↓                ↓          │
│          GPS Time      Data Store    Offline Queue     │
│              ↑              ↓                ↓          │
│           Web UI ←→ Configuration    Cellular Modem    │
└─────────────────────────────────────────────────────────┘
                              ↓
                    Scoring Software (CTP01 Server)
```

## Development Roadmap Summary

| Phase | Goal | Key Deliverables |
|-------|------|------------------|
| **Phase 1** | Proof of Concept | Model training, OCR validation, CTP01 prototype |
| **Phase 2** | Hardware Integration | Real-time processing, field testing |
| **Phase 3** | Small Race Deployment | Web UI, small race beta deployments |
| **Phase 4** | Production Hardening | Weatherproofing, OTA updates, documentation |
| **Phase 5** | Commercial Launch | Beta customers, manufacturing, sales |

See [ROADMAP.md](ROADMAP.md) for detailed milestones and deliverables.

## Getting Started (Development)

*Setup instructions to be added in Phase 1*

### Prerequisites (Expected)
- Python 3.10+
- CUDA-capable GPU (for training) or Jetson device
- 1000-image COCO dataset (available)

### Quick Start
```bash
# Clone repository
git clone <repo-url>
cd pointCam

# Setup environment (TBD)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt  # TBD

# Run tests (TBD)
pytest tests/
```

## Business Model

PointCam is developed as a **product for sale/license** to race timing companies.

**Target Market**: Race timing companies using MYLAPS/ChronoTrack systems who need reliable RFID backup.

## Contributing

This project is in early development. See [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) for contribution guidelines.

## License

*To be determined*

## Contact

*To be added*
