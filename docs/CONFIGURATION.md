# Configuration Reference

This document describes all configurable options for PointCam.

---

## Configuration File

Configuration is stored in YAML format at `/etc/pointcam/config.yaml` (or equivalent on the device).

---

## Configuration Sections

### Unit Identification

```yaml
unit:
  # Human-readable name for this unit (used in CTP01 location field)
  # Operator can change this per-race via Web UI
  name: "Finish-1"

  # Unique device identifier (auto-generated on first boot)
  # Used for OTA updates and fleet management
  id: "PC-A1B2C3D4"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `unit.name` | string | "PointCam" | Location name, sent in CTP01 protocol |
| `unit.id` | string | auto | Unique device ID |

---

### Camera Settings

```yaml
camera:
  # Resolution (width x height)
  resolution: [1920, 1080]

  # Target frame rate
  framerate: 60

  # Exposure mode: auto, manual
  exposure_mode: "auto"

  # Manual exposure value (only if exposure_mode=manual)
  exposure_value: 10000

  # White balance mode: auto, manual
  white_balance_mode: "auto"

  # Flip image: none, horizontal, vertical, both
  flip: "none"
```

| Setting | Type | Default | Range | Description |
|---------|------|---------|-------|-------------|
| `camera.resolution` | [int, int] | [1920, 1080] | Hardware dependent | Capture resolution |
| `camera.framerate` | int | 60 | 30-120 | Target FPS |
| `camera.exposure_mode` | string | "auto" | auto, manual | Exposure control |
| `camera.flip` | string | "none" | none, horizontal, vertical, both | Image orientation |

---

### Detection Settings

```yaml
detection:
  # Minimum confidence to report a bib number (0.0-1.0)
  # Below this threshold, bib is marked as "UNKNOWN"
  confidence_threshold: 0.7

  # Timing line position in frame
  # Format: [x1, y1, x2, y2] as fractions (0.0-1.0)
  # Default is vertical line at center of frame
  timing_line: [0.5, 0.0, 0.5, 1.0]

  # Crossing detection direction
  # "left_to_right", "right_to_left", "any"
  crossing_direction: "any"

  # Minimum time between duplicate detections of same bib (seconds)
  # Prevents double-counting if bib lingers near line
  debounce_time: 2.0

  # Maximum bibs to track simultaneously
  max_tracked_bibs: 20
```

| Setting | Type | Default | Range | Description |
|---------|------|---------|-------|-------------|
| `detection.confidence_threshold` | float | 0.7 | 0.0-1.0 | Minimum confidence for valid bib |
| `detection.timing_line` | [float x4] | [0.5,0,0.5,1] | 0.0-1.0 each | Line position in frame |
| `detection.crossing_direction` | string | "any" | left_to_right, right_to_left, any | Valid crossing direction |
| `detection.debounce_time` | float | 2.0 | 0.5-10.0 | Duplicate prevention window |
| `detection.max_tracked_bibs` | int | 20 | 5-50 | Max simultaneous tracking |

---

### Evidence Capture

```yaml
evidence:
  # Capture mode:
  # - "all": Capture frame for every crossing
  # - "low_confidence": Only when confidence < threshold
  # - "none": No evidence capture
  capture_mode: "low_confidence"

  # JPEG quality (1-100)
  quality: 85

  # Retention period in days (auto-delete older evidence)
  retention_days: 7

  # Maximum storage usage in MB (0 = unlimited)
  max_storage_mb: 5000

  # Include timestamp overlay on evidence images
  timestamp_overlay: true
```

| Setting | Type | Default | Range | Description |
|---------|------|---------|-------|-------------|
| `evidence.capture_mode` | string | "low_confidence" | all, low_confidence, none | When to capture frames |
| `evidence.quality` | int | 85 | 1-100 | JPEG quality |
| `evidence.retention_days` | int | 7 | 1-365 | Auto-delete after N days |
| `evidence.max_storage_mb` | int | 5000 | 0-50000 | Storage limit (0=unlimited) |
| `evidence.timestamp_overlay` | bool | true | true/false | Add timestamp to image |

---

### Communication Settings

```yaml
communication:
  # Scoring software server address
  server_host: "192.168.1.100"

  # Server port (CTP01 default is 61611)
  server_port: 61611

  # Time format for CTP01 messages
  # "normal" (HH:MM:SS.hh), "iso", "unix", "msecs"
  time_format: "normal"

  # Connection timeout (seconds)
  connect_timeout: 10

  # Reconnection attempts before alerting operator
  max_reconnect_attempts: 10

  # Reconnection backoff maximum (seconds)
  reconnect_max_backoff: 60

  # Enable CTP01 authentication (if server requires)
  authentication:
    enabled: false
    user_id: ""
    password: ""
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `communication.server_host` | string | "" | Scoring software IP/hostname |
| `communication.server_port` | int | 61611 | TCP port |
| `communication.time_format` | string | "normal" | CTP01 time format |
| `communication.connect_timeout` | int | 10 | Connection timeout (s) |
| `communication.max_reconnect_attempts` | int | 10 | Before operator alert |
| `communication.authentication.enabled` | bool | false | Use CTP01 auth |

---

### GPS Settings

```yaml
gps:
  # Enable GPS time synchronization
  enabled: true

  # Serial port for GPS module
  port: "/dev/ttyUSB0"

  # Baud rate
  baudrate: 9600

  # Timeout for GPS fix (seconds) before alerting
  fix_timeout: 120

  # PPS (Pulse Per Second) GPIO pin (-1 to disable)
  pps_pin: 18
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `gps.enabled` | bool | true | Enable GPS sync |
| `gps.port` | string | "/dev/ttyUSB0" | GPS serial port |
| `gps.baudrate` | int | 9600 | Serial baud rate |
| `gps.fix_timeout` | int | 120 | Alert if no fix after N seconds |
| `gps.pps_pin` | int | 18 | GPIO pin for PPS (-1=disabled) |

---

### Cellular Settings

```yaml
cellular:
  # Enable cellular modem
  enabled: true

  # APN settings (carrier-specific)
  apn: "internet"
  apn_user: ""
  apn_password: ""

  # Signal strength warning threshold (dBm)
  signal_warning_threshold: -100

  # Interface name
  interface: "wwan0"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `cellular.enabled` | bool | true | Enable cellular |
| `cellular.apn` | string | "internet" | Carrier APN |
| `cellular.signal_warning_threshold` | int | -100 | Alert below this dBm |

---

### Web UI Settings

```yaml
webui:
  # Enable web interface
  enabled: true

  # Listen address (0.0.0.0 for all interfaces)
  host: "0.0.0.0"

  # Web UI port
  port: 8080

  # Enable HTTPS (requires cert/key)
  https: false
  cert_file: "/etc/pointcam/cert.pem"
  key_file: "/etc/pointcam/key.pem"

  # Session timeout (minutes)
  session_timeout: 60
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `webui.enabled` | bool | true | Enable web interface |
| `webui.host` | string | "0.0.0.0" | Listen address |
| `webui.port` | int | 8080 | HTTP port |
| `webui.https` | bool | false | Enable HTTPS |
| `webui.session_timeout` | int | 60 | Session timeout (min) |

---

### Logging Settings

```yaml
logging:
  # Log level: debug, info, warning, error
  level: "info"

  # Log to file
  file_enabled: true
  file_path: "/var/log/pointcam/pointcam.log"

  # Max log file size (MB) before rotation
  max_file_size_mb: 50

  # Number of rotated files to keep
  max_files: 5

  # Log to console (for debugging)
  console_enabled: false
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `logging.level` | string | "info" | Log verbosity |
| `logging.file_enabled` | bool | true | Write to log file |
| `logging.max_file_size_mb` | int | 50 | Log rotation size |
| `logging.max_files` | int | 5 | Rotated files to keep |

---

### System Settings

```yaml
system:
  # Device hostname
  hostname: "pointcam"

  # Enable WiFi hotspot for setup
  wifi_hotspot:
    enabled: true
    ssid: "PointCam-Setup"
    password: "pointcam123"

  # Power management
  power:
    low_battery_warning: 20  # Percent
    critical_battery: 10     # Percent, triggers shutdown
    shutdown_on_critical: true

  # Automatic updates
  updates:
    auto_check: true
    auto_install: false  # Require manual approval
```

---

## Default Configuration File

```yaml
# PointCam Configuration
# Edit this file or use the Web UI to configure settings

unit:
  name: "PointCam"
  id: ""  # Auto-generated

camera:
  resolution: [1920, 1080]
  framerate: 60
  exposure_mode: "auto"
  flip: "none"

detection:
  confidence_threshold: 0.7
  timing_line: [0.5, 0.0, 0.5, 1.0]
  crossing_direction: "any"
  debounce_time: 2.0
  max_tracked_bibs: 20

evidence:
  capture_mode: "low_confidence"
  quality: 85
  retention_days: 7
  max_storage_mb: 5000
  timestamp_overlay: true

communication:
  server_host: ""
  server_port: 61611
  time_format: "normal"
  connect_timeout: 10
  max_reconnect_attempts: 10
  reconnect_max_backoff: 60
  authentication:
    enabled: false
    user_id: ""
    password: ""

gps:
  enabled: true
  port: "/dev/ttyUSB0"
  baudrate: 9600
  fix_timeout: 120
  pps_pin: 18

cellular:
  enabled: true
  apn: "internet"
  apn_user: ""
  apn_password: ""
  signal_warning_threshold: -100
  interface: "wwan0"

webui:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  https: false
  session_timeout: 60

logging:
  level: "info"
  file_enabled: true
  file_path: "/var/log/pointcam/pointcam.log"
  max_file_size_mb: 50
  max_files: 5
  console_enabled: false

system:
  hostname: "pointcam"
  wifi_hotspot:
    enabled: true
    ssid: "PointCam-Setup"
    password: "pointcam123"
  power:
    low_battery_warning: 20
    critical_battery: 10
    shutdown_on_critical: true
  updates:
    auto_check: true
    auto_install: false
```

---

## Environment Variables

Some settings can be overridden via environment variables:

| Variable | Config Equivalent | Description |
|----------|-------------------|-------------|
| `POINTCAM_SERVER_HOST` | `communication.server_host` | Server address |
| `POINTCAM_SERVER_PORT` | `communication.server_port` | Server port |
| `POINTCAM_UNIT_NAME` | `unit.name` | Unit name |
| `POINTCAM_LOG_LEVEL` | `logging.level` | Log level |
| `POINTCAM_CONFIG_FILE` | - | Path to config file |

Environment variables take precedence over config file values.

---

## Configuration via Web UI

Most settings can be changed via the Web UI at `http://pointcam.local:8080/settings`:

1. Navigate to Settings page
2. Modify desired values
3. Click Save
4. Some changes require restart (indicated in UI)

**Settings requiring restart:**
- Camera resolution/framerate
- GPS port
- Web UI port
- Cellular settings

**Settings applied immediately:**
- Unit name
- Confidence threshold
- Evidence capture mode
- Server host/port

---

## Configuration Validation

On startup, PointCam validates configuration:

1. Check all required fields present
2. Validate value ranges
3. Test hardware access (camera, GPS, cellular)
4. Log warnings for invalid/missing values
5. Use defaults for missing optional values

**Validation errors are logged** and may prevent startup if critical settings are invalid.
