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
Frame → [prefetch] → Bib Detect (YOLOv8 TRT) → Track → Crop+Filter → OCR (CRNN TRT) → Vote → Classify
                          ↓ (parallel thread)                                                      ↓
                     Pose Detect (YOLOv8-pose TRT) ──────────────────────────────→ Person-Bib Assoc → Crossing
```

### Core Components
1. **CV Pipeline** - YOLOv8 Nano for bib detection, CRNN (TensorRT) for number recognition
2. **Timing Engine** - Manages crossing events, generates CTP01 messages
3. **GPS Time Module** - GPS-synchronized timestamps with drift compensation
4. **Communication Layer** - CTP01 client with offline queuing
5. **Data Store** - SQLite for events, logs, configuration
6. **Evidence Store** - JPEG capture with configurable retention

### Target Hardware
- **Primary**: NVIDIA Jetson Orin Nano (8GB) with TensorRT optimization
- **Alternative**: Raspberry Pi 5 (8GB) with ONNX Runtime

### Jetson Access
- **SSH (Tailscale)**: `ssh alex@100.103.210.41` (works from anywhere)
- **SSH (LAN)**: `ssh alex@100.103.210.41` (home network only)
- **Repo path**: `/home/alex/pointcam`
- **Python**: `venv/bin/python` (always use the venv, not system Python)

### Deploying to Jetson
Push from the dev machine then pull on the Jetson. This keeps both repos in sync via git instead of ad-hoc scp.

```bash
# On dev machine: commit and push
git push origin master

# On Jetson: pull and rebuild frontend static (if changed)
ssh alex@100.103.210.41 "cd /home/alex/pointcam && git pull"
```

If the web UI server is running, it will auto-reload with `--dev`. Otherwise restart it:
```bash
ssh alex@100.103.210.41 "cd /home/alex/pointcam && nohup venv/bin/python -m pointcam.web --dev > /tmp/webui.log 2>&1 &"
```

### Jetson Model Files
```
models/bib_detector_v3.pt        # YOLOv8n bib detector (PyTorch)
models/bib_detector_v3.engine    # YOLOv8n bib detector (TensorRT FP16) ← use this
models/ocr_crnn.onnx             # CRNN OCR (ONNX)
models/ocr_crnn.engine           # CRNN OCR (TensorRT, built by ORT on first run)
models/ocr_parseq.onnx           # PARSeq OCR (ONNX — broken on Jetson, Where(9) op)
yolov8n-pose.pt                  # YOLOv8n-pose (PyTorch)
yolov8n-pose.engine              # YOLOv8n-pose (TensorRT FP16) ← use this
runs/ocr_finetune/parseq_gpu_v1/best.pt  # PARSeq PyTorch checkpoint (fallback)
```

### Running Benchmarks on the Jetson
Long-running pipeline benchmarks (~2 hours) should be launched via `nohup` on the Jetson, then monitored periodically — never run directly as a blocking SSH command or background bash task (SSH will timeout and lose output).

```bash
# Launch on Jetson (recommended: CRNN TRT + TRT detector)
ssh alex@100.103.210.41 "cd /home/alex/pointcam && nohup venv/bin/python scripts/test_video_pipeline.py REC-0006-A.mp4 --no-video --bib-set gt_bibs.txt --crossing-mode zone --placement right --ocr crnn --ocr-backend tensorrt --detector models/bib_detector_v3.engine > /tmp/runN.log 2>&1 &"

# Monitor progress periodically
ssh alex@100.103.210.41 "wc -l /home/alex/pointcam/runs/pipeline_test/REC-0006-A_crossings.csv; ps aux | grep test_video | grep -v grep | wc -l"

# Check if finished (process count = 0), then get results
ssh alex@100.103.210.41 "tail -50 /tmp/runN.log"

# Copy results locally for scoring
scp alex@100.103.210.41:/home/alex/pointcam/runs/pipeline_test/REC-0006-A_crossings.csv pipeline_crossings_runN.csv
```

### TensorRT Engine Export (one-time, run on Jetson)
```bash
# Bib detector
venv/bin/python -c "from ultralytics import YOLO; YOLO('models/bib_detector_v3.pt').export(format='engine', half=True, imgsz=640)"

# Pose model
venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='engine', half=True, imgsz=640)"

# CRNN TRT engine is built automatically by ORT TensorRT EP on first run (~3-4 min warmup)
```

### Scoring Against Ground Truth
Ground truth: `C:\Users\alex\Downloads\5k-run-walk-overall-results-20260214163650-0500.csv` (UTF-16, tab-separated, 2,622 finishers). Estimated ~933 visible finishers in REC-0006-A.mp4.

### Current Best Results
**CRNN TRT + TRT detector**: 795 TP, 1 FP, 99.9% precision, 85.2% visible recall, ~21 fps (stride=2).

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
