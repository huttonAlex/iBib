# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PointCam is a computer vision-based race timing backup system that detects bib numbers and timing line crossings using cameras. It integrates with race timing software via the CTP01 protocol (ChronoTrack/MYLAPS compatible) and runs entirely on-device (no server dependency).

**Current Status**: Phase 1 (Proof of Concept) - Documentation complete, implementation beginning.

## Build and Development Commands

```bash
# Installation
pip install -e .[dev]                       # Install in dev mode with dev dependencies
pip install -e .[gpu]                       # GPU variant (paddlepaddle-gpu)

# Testing
pytest tests/                               # Run all tests
pytest tests/test_file.py::test_name        # Run single test

# Code Quality
black src/ tests/                           # Format code (line-length: 100)
ruff check src/ tests/                      # Lint
ruff check --fix src/ tests/                # Auto-fix lint issues

# Dataset Validation
python scripts/validate_dataset.py --images data/images --annotations data/annotations/instances.json
```

## Architecture

### Processing Pipeline
```
Frame (60fps) → Preprocess → Bib Detect (YOLOv8) → OCR (PaddleOCR) → Track & Cross-Detect → Timing Engine → CTP01 Queue → Network
```

### Core Components
1. **CV Pipeline** - YOLOv8 Nano for bib detection, PaddleOCR for number recognition
2. **Timing Engine** - Manages crossing events, generates CTP01 messages
3. **GPS Time Module** - GPS-synchronized timestamps with drift compensation
4. **Communication Layer** - CTP01 client with offline queuing
5. **Data Store** - SQLite for events, logs, configuration
6. **Evidence Store** - JPEG capture with configurable retention

### Target Hardware
- **Primary**: NVIDIA Jetson Orin Nano (8GB) with TensorRT optimization
- **Alternative**: Raspberry Pi 5 (8GB) with ONNX Runtime

## Key Technical Patterns

### CTP01 Protocol
- **Format**: `CT01_33~<seq>~<location>~<bib>~<time>~<flags>~<reader_id>~<lap>\r\n`
- **Default Port**: 61611
- **Example**: `CT01_33~1~finish~1234~14:02:15.31~0~POINTCAM~0\r\n`

### Data Structures
```python
@dataclass
class CrossingEvent:
    sequence_number: int           # Unique, monotonic
    bib_number: str               # "1234" or "UNKNOWN"
    timestamp: datetime           # GPS-synchronized
    confidence: float             # 0.0 - 1.0
    position_in_group: int        # For simultaneous crossings
    frame_number: int             # Source frame reference
    evidence_path: Optional[str]  # Path to captured frame
```

### Database Tables
- `crossings` - Detected crossing events with confidence and evidence
- `queue` - CTP01 messages pending transmission
- `logs` - System logs by component
- `config` - Persistent key-value configuration

### Evidence Capture
- **Modes**: ALL, LOW_CONFIDENCE_ONLY, NONE
- **Path Pattern**: `evidence/{race_date}/{hour}/{sequence}_{bib}_{timestamp}.jpg`

## Performance Requirements

- **Detection Rate**: ≥30 fps real-time processing
- **End-to-End Latency**: <500ms crossing to data output
- **Bib Recall**: ≥95% (correct bibs identified / ground truth finishers)
- **Precision**: ≥95% (correct bibs / all bibs emitted)
- **Crossing Recall**: ≥95% (crossings detected / ground truth finishers)
- **Position Accuracy**: PRIMARY priority (correct finish order critical)
- **Time Accuracy**: ±1 second (RFID provides primary times)
- **Scale**: Up to 3,000 participants, 10 simultaneous crossings

## Development Guidelines

### Commit Messages
Use format: `[Component] Brief description`
- `[OCR] Add PaddleOCR evaluation script`
- `[Docs] Update Phase 1.2 completion status`

### Code Standards
- Python 3.10+ with type hints
- 100 character line length (Black)
- Ruff linting (E, F, W, I, N rules)

### Documentation Updates
- Major technical decisions go in `docs/DESIGN_DECISIONS.md`
- Experimental results go in `docs/TECH_NOTES.md`

## Key Documentation

- `ARCHITECTURE.md` - Full system design and component details
- `REQUIREMENTS.md` - Functional and non-functional requirements
- `docs/RECOGNITION_SYSTEM.md` - Bib recognition system details
- `docs/INTEGRATION_PROTOCOL.md` - CTP01 protocol specification
- `docs/CONFIGURATION.md` - All configuration options
