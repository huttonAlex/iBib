# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PointCam Unit                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Camera     │───>│  CV Pipeline │───>│   Timing     │───>│  Data    │ │
│  │   Module     │    │              │    │   Engine     │    │  Store   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│                                                 │                    │     │
│  ┌──────────────┐                              │                    │     │
│  │   GPS Time   │──────────────────────────────┘                    │     │
│  │   Module     │                                                   │     │
│  └──────────────┘                                                   │     │
│                                                                     │     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │     │
│  │   Config     │<──>│   Web UI     │<──>│   Evidence   │<─────────┘     │
│  │   Manager    │    │   Server     │    │   Store      │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
│                             ^                                             │
│                             │ HTTP (Local)                               │
│  ┌──────────────────────────┼────────────────────────────────────────┐   │
│  │              Communication Layer                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │   CTP01     │  │   Offline   │  │   Cellular  │               │   │
│  │  │   Client    │  │   Queue     │  │   Modem     │               │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                          │                                │
└──────────────────────────────────────────┼────────────────────────────────┘
                                           │ TCP/Cellular
                                           v
                              ┌──────────────────────┐
                              │   Scoring Software   │
                              │   (CTP01 Server)     │
                              └──────────────────────┘
```

---

## Component Breakdown

### 1. Camera Module

**Responsibilities:**
- Capture high frame rate video (60 fps target)
- Provide consistent image quality across lighting conditions
- Hardware mounting interface (tripod)

**Hardware Options (2025+):**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| Raspberry Pi Camera Module 3 | Good quality, native Pi support | Limited to Pi ecosystem | Development |
| Arducam 64MP | High resolution, Pi compatible | May be overkill | Alternative |
| USB3 Industrial Camera | High frame rate, global shutter | Higher cost, complex | Production |
| IMX462 (Sony) | Excellent low-light | Requires custom integration | Consider |

**Outputs:**
- Raw frame data to CV Pipeline
- Frame metadata (timestamp, frame number)

**Key Specifications:**
- Resolution: 1920x1080 minimum
- Frame rate: 60 fps target
- Interface: CSI or USB3
- Low-light sensitivity: Important for dawn/dusk races

---

### 2. Computer Vision Pipeline

The core processing component running on edge device.

```
┌─────────────────────────────────────────────────────────────┐
│                    CV Pipeline Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frame ──> Preprocess ──> Bib Detect ──> OCR ──> Track     │
│    │           │              │           │         │       │
│    v           v              v           v         v       │
│  60fps     Resize/        YOLOv8      PaddleOCR  Position  │
│           Enhance         Nano         Lite      Tracking   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1 Frame Preprocessing

```python
# Conceptual pipeline
Raw Frame → Resize (inference size) → Color normalize → Enhanced Frame
```

**Functions:**
- Resize to model input size (640x640 typical for YOLO)
- Lighting normalization (histogram equalization or learned)
- Region of interest cropping (if timing line area known)

**Performance Target:** <5ms per frame

#### 2.2 Bib Detection Module

```
Enhanced Frame → Object Detection Model → Bib Bounding Boxes
```

**Technology Selection:**

| Model | Speed (fps) | Accuracy | Edge Support | Recommendation |
|-------|-------------|----------|--------------|----------------|
| YOLOv8n | 60+ | Good | Excellent | **Primary choice** |
| YOLOv8s | 40+ | Better | Good | If accuracy needed |
| RT-DETR | 30+ | Excellent | Moderate | Future option |
| MobileNet-SSD | 100+ | Moderate | Excellent | Fallback |

**Training:**
- Use existing 1000-image COCO dataset
- Augmentation: rotation, brightness, blur
- Target mAP: >0.85 on test set

**Outputs:**
- Bounding boxes around detected bibs
- Confidence scores (0-1)
- Frame coordinates

#### 2.3 Number Recognition (OCR)

```
Bib Region → OCR Processing → Bib Number + Confidence
```

**Technology Selection:**

| Library | Speed | Accuracy | Font Flexibility | Recommendation |
|---------|-------|----------|-----------------|----------------|
| PaddleOCR | Fast | Excellent | High | **Primary choice** |
| EasyOCR | Medium | Good | High | Alternative |
| Tesseract | Fast | Moderate | Needs training | Fallback |
| Custom CNN | Varies | Customizable | Maximum | Future option |

**Processing:**
- Crop bib region from frame
- Preprocess (grayscale, contrast enhance, deskew)
- Run OCR inference
- Post-process (filter non-numeric, validate format)

**Outputs:**
- Detected number (string)
- Character-level confidence
- Overall confidence score

#### 2.4 Timing Line Detection

```
Frame → Line Position → Virtual Timing Line
```

**Approach: Virtual Line (Recommended)**

- Operator calibrates line position during setup via Web UI
- Line stored as percentage of frame width/height
- No physical markers required in race environment

**Alternative: ArUco Markers**
- Place markers at timing line edges
- Auto-detect line position
- More robust to camera movement
- Requires marker placement

**Outputs:**
- Line coordinates in frame (x1, y1, x2, y2)
- Valid/invalid status

#### 2.5 Crossing Detection & Tracking

```
Bib Position + Timing Line → Crossing Events
```

**Tracking Algorithm:**
1. Track bib positions across frames using centroid tracking or SORT
2. Detect when bib centroid crosses timing line
3. Record crossing frame and interpolated timestamp
4. Handle multiple simultaneous crossings with position ordering

**Multi-Object Tracking Options:**

| Algorithm | Speed | Accuracy | Complexity |
|-----------|-------|----------|------------|
| Centroid Tracking | Very Fast | Good for sparse | Simple |
| SORT | Fast | Good | Moderate |
| DeepSORT | Moderate | Excellent | Complex |
| ByteTrack | Fast | Excellent | Moderate |

**Recommendation:** Start with Centroid Tracking, upgrade to ByteTrack if needed.

**Position Ordering:**
- When multiple bibs cross in same frame window (±100ms)
- Order by: crossing timestamp, then horizontal position in frame
- Critical for maintaining correct finish order

---

### 3. GPS Time Module

**Responsibilities:**
- Provide accurate UTC time
- Synchronize system clock
- Handle GPS signal loss gracefully

**Hardware:**
- GPS module with PPS (Pulse Per Second) output
- Example: u-blox NEO-M8N or NEO-M9N
- Cost: $30-50

**Time Accuracy:**
- With GPS lock: ±10ms
- Without GPS (drift): ±1 second per hour (typical)

**Implementation:**
```python
class GPSTimeSync:
    def get_timestamp(self) -> datetime:
        """Return GPS-synchronized timestamp"""
        if self.has_gps_lock:
            return self.gps_time + self.pps_offset
        else:
            return self.system_time + self.last_known_offset
```

**Failure Mode:**
- If GPS lock lost, continue using system clock with last known offset
- Log GPS status for post-race analysis
- Alert operator if GPS unavailable for >5 minutes

---

### 4. Timing Engine

**Responsibilities:**
- Manage crossing events
- Generate CTP01-compatible messages
- Maintain sequence numbers
- Order events by position

**Data Model:**

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

    def to_ctp01(self, location: str, time_format: str = "normal") -> str:
        """Generate CTP01 tag event line"""
        # CT01_33~<seq>~<location>~<bib>~<time>~<flags>~<reader_id>~<lap>
        time_str = self._format_time(time_format)
        return f"CT01_33~{self.sequence_number}~{location}~{self.bib_number}~{time_str}~0~POINTCAM~0\r\n"
```

**Functions:**
- Receive crossing events from CV pipeline
- Assign sequence numbers
- Calculate position within crossing groups
- Generate CTP01 messages
- Send to Communication Layer

---

### 5. Communication Layer

The communication layer handles all data transmission with offline-first architecture.

```
┌────────────────────────────────────────────────────────────┐
│                 Communication Layer                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │
│  │   CTP01     │<───>│   Offline   │<───>│  Cellular   │ │
│  │   Client    │     │   Queue     │     │  Manager    │ │
│  └─────────────┘     └─────────────┘     └─────────────┘ │
│        │                   │                    │         │
│        v                   v                    v         │
│  Protocol Logic      SQLite Queue       Connection Mgmt  │
│  Handshake           Persistence        Auto-reconnect   │
│  Push Mode           Ordering           Status Monitor   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 5.1 CTP01 Client

**Responsibilities:**
- Implement ChronoTrack Socket Protocol 1
- Manage TCP connection to scoring software
- Handle push mode streaming
- Process acknowledgments

**Connection Flow:**
```
1. TCP Connect to server (default port 61611)
2. Receive greeting: <program>~<version>~CTP01
3. Send response: PointCam~<version>~<num_requests>
4. Send init requests: stream-mode=push, location=single, etc.
5. (Optional) Authenticate
6. Begin push mode: start~<location>
7. Stream crossing events
```

**Example Message:**
```
CT01_33~1~finish~1234~14:02:15.31~0~POINTCAM~0\r\n
```

#### 5.2 Offline Queue

**Responsibilities:**
- Buffer events during connectivity loss
- Persist to survive power loss
- Maintain ordering for replay

**Implementation:**
- SQLite database for persistence
- FIFO queue with sequence numbers
- Mark events as "sent" vs "pending"
- Replay pending events on reconnection

**Schema:**
```sql
CREATE TABLE crossing_queue (
    id INTEGER PRIMARY KEY,
    sequence_number INTEGER UNIQUE,
    ctp01_message TEXT,
    timestamp DATETIME,
    sent BOOLEAN DEFAULT FALSE,
    sent_at DATETIME
);
```

#### 5.3 Cellular Manager

**Responsibilities:**
- Monitor cellular connection status
- Handle modem control
- Report signal strength
- Manage data usage (optional)

**Hardware:**
- 4G/LTE USB modem or HAT
- SIM card with data plan
- Example: Quectel EC25, SIM7600

---

### 6. Data Store

**Local Storage (SQLite):**

```sql
-- Crossing events
CREATE TABLE crossings (
    id INTEGER PRIMARY KEY,
    sequence_number INTEGER UNIQUE,
    bib_number TEXT,
    timestamp DATETIME,
    confidence REAL,
    position_in_group INTEGER,
    frame_number INTEGER,
    evidence_path TEXT,
    sent BOOLEAN DEFAULT FALSE
);

-- System logs
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    level TEXT,
    component TEXT,
    message TEXT
);

-- Configuration
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME
);
```

---

### 7. Evidence Store

**Responsibilities:**
- Capture and store crossing evidence frames
- Associate frames with crossing events
- Manage storage space
- Provide retrieval interface

**Storage Strategy:**
- Save as JPEG (quality 85) for space efficiency
- Filename: `{sequence_number}_{bib}_{timestamp}.jpg`
- Directory structure: `evidence/{race_date}/{hour}/`
- Auto-cleanup based on retention policy

**Capture Modes:**
1. **ALL**: Capture frame for every crossing
2. **LOW_CONFIDENCE**: Only when confidence < threshold
3. **NONE**: No evidence capture (save storage)

---

### 8. Configuration Manager

**Responsibilities:**
- Store and retrieve configuration
- Provide defaults
- Validate settings
- Persist across restarts

**Configuration Categories:**

```yaml
# Example configuration structure
unit:
  name: "Finish-1"
  id: "PC-001"

camera:
  resolution: [1920, 1080]
  framerate: 60
  exposure_mode: "auto"

detection:
  confidence_threshold: 0.7
  timing_line_position: [0.5, 0.0, 0.5, 1.0]  # Vertical line at center

evidence:
  capture_mode: "LOW_CONFIDENCE"  # ALL, LOW_CONFIDENCE, NONE
  retention_days: 7
  quality: 85

communication:
  server_host: "192.168.1.100"
  server_port: 61611
  location_name: "finish"
  time_format: "normal"

gps:
  enabled: true
  port: "/dev/ttyUSB0"
  baudrate: 9600
```

---

### 9. Web UI Server

**Responsibilities:**
- Serve operator interface
- Handle configuration changes
- Display real-time status
- Provide evidence viewer

**Technology:**
- FastAPI or Flask backend
- Simple HTML/JS frontend (no heavy framework needed)
- Serve on local network (e.g., `http://pointcam.local:8080`)

**Screens:**

1. **Dashboard**
   - Connection status (cellular, GPS, scoring server)
   - Detection count / rate
   - Recent crossings list
   - Alerts

2. **Setup**
   - Timing line calibration (click on video feed)
   - Location name configuration
   - Server connection settings

3. **Evidence**
   - Browse captured frames
   - Filter by confidence, time, bib
   - Export capability

4. **Settings**
   - All configuration options
   - Unit name/ID
   - Capture mode selection

**Access:**
- Via phone/tablet browser over local WiFi
- Or hotspot created by PointCam unit

---

## Hardware Architecture

### Recommended Production Platform

**Primary: NVIDIA Jetson Orin Nano (8GB)**

| Component | Specification |
|-----------|---------------|
| Compute | 6-core ARM Cortex-A78AE, 1024-core GPU |
| Memory | 8GB LPDDR5 |
| AI Performance | 40 TOPS |
| Power | 7-15W |
| Price | ~$500 |

**Why Jetson Orin Nano:**
- GPU acceleration for YOLO inference
- Sufficient for 60fps processing
- Good power efficiency for battery operation
- Strong ecosystem (JetPack, TensorRT)
- NVIDIA support for ML workloads

**Alternative: Raspberry Pi 5 (8GB)**

| Component | Specification |
|-----------|---------------|
| Compute | Quad-core Cortex-A76 @ 2.4GHz |
| Memory | 8GB LPDDR4X |
| AI Performance | Limited (no GPU acceleration) |
| Power | 5-12W |
| Price | ~$80 |

**Use for:** Development, lower-accuracy acceptable, budget constraints

### Hardware BOM (Bill of Materials)

| Component | Example Part | Est. Cost |
|-----------|--------------|-----------|
| Compute Module | Jetson Orin Nano | $500 |
| Camera | IMX477 or industrial | $50-200 |
| GPS Module | u-blox NEO-M9N | $40 |
| Cellular Modem | Quectel EC25 | $50 |
| Battery | 20000mAh USB-C PD | $50 |
| Enclosure | Custom/IP54 rated | $50-100 |
| Misc (cables, mount) | Various | $50 |
| **Total** | | **$800-1000** |

---

## Software Stack

### Operating System
- Linux-based: JetPack (Jetson) or Raspberry Pi OS
- Real-time kernel patches if needed for timing precision

### Programming Language
- **Primary**: Python 3.10+
- **Performance critical**: C++ with Python bindings (optional)

### Key Libraries

| Purpose | Library | Version |
|---------|---------|---------|
| Computer Vision | OpenCV | 4.8+ |
| Object Detection | Ultralytics YOLOv8 | Latest |
| OCR | PaddleOCR | 2.7+ |
| ML Inference | ONNX Runtime or TensorRT | Latest |
| Web Server | FastAPI | 0.100+ |
| Database | SQLite | 3.x |
| GPS | gpsd / pyserial | Latest |
| Async I/O | asyncio | Built-in |

### Inference Optimization

**For Jetson:**
- Export models to TensorRT for 2-3x speedup
- Use FP16 precision for speed/accuracy tradeoff
- Batch inference if multiple cameras

**For Raspberry Pi:**
- Use ONNX Runtime with XNNPACK
- Consider quantized INT8 models
- May need to reduce frame rate

---

## Processing Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Main Processing Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Frame Capture ─────────────────────────────────────┐      │
│      │ Camera grabs frame @ 60fps                       │      │
│      │ GPS timestamp attached                           │      │
│      v                                                  │      │
│   2. Preprocessing ─────────────────────────────────────┤      │
│      │ Resize to 640x640                               │      │
│      │ Normalize colors                                 │      │
│      v                                                  │      │
│   3. Bib Detection ─────────────────────────────────────┤      │
│      │ YOLOv8 inference                                │      │
│      │ Returns: [bbox, confidence] for each bib        │      │
│      v                                                  │      │
│   4. OCR ───────────────────────────────────────────────┤      │
│      │ For each detected bib:                          │      │
│      │   Crop, preprocess, run PaddleOCR               │      │
│      │   Returns: bib_number, confidence               │      │
│      v                                                  │      │
│   5. Tracking & Crossing Detection ─────────────────────┤      │
│      │ Update tracker with bib positions               │      │
│      │ Check for timing line crossings                 │      │
│      │ If crossing: create CrossingEvent               │      │
│      v                                                  │      │
│   6. Event Processing ──────────────────────────────────┤      │
│      │ Assign sequence number                          │      │
│      │ Capture evidence frame (if configured)          │      │
│      │ Store to database                               │      │
│      v                                                  │      │
│   7. Communication ─────────────────────────────────────┘      │
│      │ Generate CTP01 message                                  │
│      │ Send to queue (offline-first)                          │
│      │ Queue sends to server when connected                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Failure Modes & Handling

### Camera Failure
- **Detection**: No frames received for >1 second
- **Response**: Alert operator, log error
- **Recovery**: Automatic retry, manual restart if needed

### GPS Signal Loss
- **Detection**: No GPS fix for >30 seconds
- **Response**: Continue with system clock + last offset, alert operator
- **Recovery**: Auto-resume when signal returns

### Cellular Connectivity Loss
- **Detection**: TCP connection drops or no response
- **Response**: Queue events locally, retry connection
- **Recovery**: Replay queued events when reconnected

### Processing Overload
- **Detection**: Frame processing taking >frame interval
- **Response**: Skip frames (with logging), alert if sustained
- **Recovery**: Reduce processing load or frame rate

### Low Storage
- **Detection**: Storage <500MB free
- **Response**: Alert operator, auto-delete old evidence
- **Recovery**: Manual cleanup or larger storage

### Low Battery
- **Detection**: Battery <20%
- **Response**: Alert operator
- **Detection**: Battery <10%
- **Response**: Graceful shutdown, save state

---

## Security & Privacy

### Data Handling
- No facial recognition or personal identification beyond bib numbers
- Local processing only (no cloud dependency)
- Evidence frames focus on bibs, not faces
- Data retention policies enforced automatically

### Network Security
- CTP01 supports authentication (optional)
- Web UI accessible only on local network
- No external ports exposed by default
- HTTPS for Web UI (self-signed cert acceptable)

### Physical Security
- Unit should be attended or secured during races
- No sensitive data stored beyond race timing info

---

## Scalability Considerations

### Current Scope (v1)
- Single camera per PointCam unit
- Single timing point per unit
- Up to 10 simultaneous crossings
- Races up to 3,000 participants

### Future Expansion
- **Multi-camera per point**: Different angles for redundancy/accuracy
- **Multi-unit coordination**: Central management, shared data
- **Cloud backup**: Optional sync to cloud for redundancy
- **Fleet management**: Remote monitoring of multiple units

---

## Integration Points

### CTP01 Protocol
- Primary integration with scoring software
- See [INTEGRATION_PROTOCOL.md](docs/INTEGRATION_PROTOCOL.md) for details

### Export Formats
- CSV for manual import
- JSON for API integration
- CTP01 format for direct RFID system replacement

### Future Integrations
- MYLAPS direct protocol (if different from CTP01)
- Results publishing APIs
- Live tracking systems
