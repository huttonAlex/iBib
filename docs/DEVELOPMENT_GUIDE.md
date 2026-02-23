# Development Guide

## Purpose

This guide provides practical information for developers working on the PointCam project.

## Project Structure

```
pointCam/
├── README.md                 # Project overview
├── REQUIREMENTS.md           # Detailed requirements
├── ARCHITECTURE.md           # System architecture
├── ROADMAP.md               # Development phases and milestones
├── docs/
│   ├── DESIGN_DECISIONS.md  # Technical decision log
│   ├── DEVELOPMENT_GUIDE.md # This file
│   ├── MODEL_ASSETS.md       # Pinned model assets & verification
│   └── TECH_NOTES.md        # Technical findings and notes
├── src/                     # Source code (pipeline + modules)
├── tests/                   # Unit + integration tests
├── data/                    # Datasets and sample data
├── models/                  # Trained model assets (pinned)
└── scripts/                 # Utility scripts & CLI tools
```

## Development Phases

See [ROADMAP.md](../ROADMAP.md) for detailed phase breakdown. Current phase information:

**Current Phase**: Phase 1 - Proof of Concept
**Current Milestone**: 1.2 OCR fine-tuning pipeline built; 1.3 timing-line + crossing integration in progress

## Setting Up Development Environment

### Prerequisites

**Currently Required:**
- Python 3.10+
- pip for package management
- Git for version control
- ONNX Runtime (for OCR inference)

### Installation
```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

# Verify model assets
python scripts/verify_model_assets.py
# If needed:
# python scripts/fetch_model_assets.py --base-url <MODEL_URL>
```

### Hardware Setup

*To be populated after Phase 2.1*

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints where practical
- Docstrings for all public functions/classes
- Maximum line length: 100 characters

### Documentation
- Update relevant .md files when making architectural decisions
- Log significant decisions in DESIGN_DECISIONS.md
- Document technical findings in TECH_NOTES.md
- Keep README.md current with project status

### Version Control

**Branch Strategy** (to be formalized):
- `main` - Stable releases
- `develop` - Integration branch
- `feature/*` - Feature development
- `phase-X/*` - Phase-specific work

**Commit Messages**:
- Use clear, descriptive messages
- Reference issues/milestones where applicable
- Format: `[Component] Brief description`
  - Example: `[OCR] Add PaddleOCR evaluation script`
  - Example: `[Docs] Update Phase 1.2 completion status`

## Testing Strategy

### Phase 1: Manual Testing
- Collect metrics on sample datasets
- Document results in TECH_NOTES.md

### Phase 2+: Automated Testing
- Unit tests for core components
- Pipeline integration test (synthetic frames)
- Integration tests for pipeline
- Performance benchmarks
- Field test protocols

*Detailed testing framework to be established in Phase 1.4*

## Data Management

### Dataset Organization

```
data/
├── raw/                    # Original images/videos
│   ├── bibs/              # Bib images for training/testing
│   ├── test_videos/       # Test footage
│   └── field_tests/       # Real race footage
├── processed/             # Preprocessed data
├── annotations/           # Labels, ground truth
└── results/               # Test outputs
```

### Data Collection Guidelines

**Bib Images**:
- Variety of fonts, sizes, colors
- Different lighting conditions
- Various angles and distances
- Motion blur samples
- Partial occlusions

**Video Samples**:
- Different runner speeds
- Multiple simultaneous runners
- Various time-of-day lighting
- Weather conditions (when available)

**Privacy**:
- No facial close-ups
- Focus on bibs only
- Obtain permissions for race footage
- Anonymize data when sharing externally

## Performance Benchmarking

### Key Metrics to Track

**Accuracy**:
- Bib detection rate (%)
- OCR accuracy (%)
- Position ordering accuracy (%)
- False positive rate
- False negative rate

**Performance**:
- Frames per second (fps)
- Processing latency (ms per frame)
- Memory usage
- CPU/GPU utilization

**System**:
- Setup time
- Continuous operation duration
- Power consumption
- Failure modes encountered

### Benchmark Protocol

*To be established in Phase 1.4*

## Documentation Updates

### When to Update Docs

**REQUIREMENTS.md**:
- Requirements change or are clarified
- New constraints identified
- Acceptance criteria modified

**ARCHITECTURE.md**:
- Component design changes
- Technology selection made
- Integration approach modified

**ROADMAP.md**:
- Milestones completed (check off items)
- Timeline adjustments
- New phases/milestones added
- Risk updates

**DESIGN_DECISIONS.md**:
- Any significant technical decision
- Technology selections
- Approach changes

**TECH_NOTES.md**:
- Experimental results
- Library evaluations
- Performance findings
- Lessons learned

## Communication

### Project Updates

*To be determined - could be*:
- Weekly progress summaries
- Milestone completion reports
- Blocker identification
- Demo videos of progress

### Issue Tracking

*To be set up - GitHub Issues or similar*

## Phase-Specific Guidelines

### Phase 1: Proof of Concept

**Focus**: Rapid experimentation and learning
**Priority**: Document findings > perfect code
**Output**: Technology recommendations, baseline metrics

**Guidelines**:
- Prioritize breadth over depth
- Test multiple approaches quickly
- Document everything in TECH_NOTES.md
- Don't over-engineer
- Notebook-style development is acceptable

### Phase 2: Real-World Testing

**Focus**: Robust implementation
**Priority**: Reliability > features
**Output**: Working prototype, field test data

**Guidelines**:
- Code quality matters
- Error handling required
- Performance monitoring
- Field test protocols

### Phase 3+: Production

**Focus**: Production readiness
**Priority**: Quality > speed
**Output**: Deployable system

**Guidelines**:
- Full test coverage
- Production-grade error handling
- Documentation for operators
- Deployment automation

## Troubleshooting Common Issues

*To be populated as issues are encountered*

## Useful Resources

### Computer Vision
- OpenCV Documentation: https://docs.opencv.org/
- PyTorch Vision: https://pytorch.org/vision/
- TensorFlow Lite: https://www.tensorflow.org/lite

### OCR
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- Tesseract: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR

### Edge Computing
- Raspberry Pi: https://www.raspberrypi.org/documentation/
- NVIDIA Jetson: https://developer.nvidia.com/embedded/jetson
- Coral: https://coral.ai/docs/

### Race Timing
- *To be added: race timing standards, RFID integration docs*

## Contributing

*To be formalized - but generally*:
- Read relevant documentation first
- Discuss major changes before implementing
- Update docs with code changes
- Test thoroughly
- Seek review for significant changes

## Questions?

*To be updated with contact/discussion forum information*
