# Integration Protocol Specification

This document describes how PointCam integrates with race timing/scoring software using the ChronoTrack Socket Protocol 1 (CTP01).

---

## Protocol Overview

**Protocol**: ChronoTrack Socket Protocol 1 (CTP01)
**Transport**: TCP
**Default Port**: 61611
**Field Separator**: Tilde (`~`)
**Line Terminator**: CRLF (`\r\n`)

### PointCam Role

PointCam acts as a **CTP01 client**, connecting to scoring software which acts as the **CTP01 server**. This is the same role that ChronoTrack RFID hardware plays.

```
┌────────────┐                      ┌────────────────────┐
│  PointCam  │ ─── TCP/61611 ───>  │  Scoring Software  │
│  (Client)  │                      │  (Server)          │
└────────────┘                      └────────────────────┘
```

---

## Connection Flow

### 1. Initial Connection

```
1. PointCam connects to scoring software (TCP port 61611)
2. Server sends greeting
3. PointCam responds with identification
4. PointCam sends initialization requests
5. Connection ready for data streaming
```

### 2. Greeting Exchange

**Server sends:**
```
<program-name>~<program-version>~CTP01\r\n
```

**Example:**
```
RunScore~Version 6.5~CTP01\r\n
```

**PointCam responds:**
```
PointCam~<version>~<number-of-requests>\r\n
```

**Example:**
```
PointCam~1.0.0~4\r\n
```

### 3. Initialization Requests

After the greeting, PointCam sends initialization requests:

```
stream-mode=push\r\n
location=single\r\n
time-format=normal\r\n
pushmode-ack=true\r\n
```

**Supported Options:**

| Request | Value | Description |
|---------|-------|-------------|
| `stream-mode` | `push` | Real-time event streaming (required) |
| `location` | `single` | Single location per connection |
| `time-format` | `normal`, `iso`, `unix`, `msecs` | Timestamp format |
| `pushmode-ack` | `true` | Request ack on start/stop commands |
| `tagevent-format` | `CT01_33` | Tag event message format |

### 4. Start Streaming

**PointCam sends:**
```
start~<location-name>\r\n
```

**Example:**
```
start~finish\r\n
```

**Server acknowledges (if pushmode-ack=true):**
```
ack~start~finish\r\n
```

---

## Data Messages

### Tag Observation (Crossing Event)

When a participant crosses the timing line, PointCam sends a tag observation message.

**Format (CT01_33):**
```
CT01_33~<seq>~<location>~<bib>~<time>~<flags>~<reader-id>~<lap>\r\n
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `CT01_33` | Literal | Format identifier |
| `seq` | Integer | Sequence number (monotonic, unique) |
| `location` | String | Location name (e.g., "finish", "split-1") |
| `bib` | String | Bib number or "UNKNOWN" |
| `time` | String | Timestamp (format depends on time-format) |
| `flags` | Integer | Reserved, always 0 |
| `reader-id` | String | Device identifier ("POINTCAM") |
| `lap` | Integer | Lap count (0 for single-pass) |

**Examples:**

```
# Normal time format (HH:MM:SS.hh)
CT01_33~1~finish~1234~14:02:15.31~0~POINTCAM~0\r\n
CT01_33~2~finish~567~14:02:15.45~0~POINTCAM~0\r\n
CT01_33~3~finish~UNKNOWN~14:02:16.02~0~POINTCAM~0\r\n

# ISO time format
CT01_33~1~finish~1234~2026-02-01T14:02:15.31~0~POINTCAM~0\r\n

# Unix timestamp format
CT01_33~1~finish~1234~1738418535.31~0~POINTCAM~0\r\n

# Milliseconds since midnight
CT01_33~1~finish~1234~50535310~0~POINTCAM~0\r\n
```

### Time Format Options

| Format | Example | Description |
|--------|---------|-------------|
| `normal` | `14:02:15.31` | HH:MM:SS.hh (hundredths) |
| `iso` | `2026-02-01T14:02:15.31` | ISO 8601 with hundredths |
| `unix` | `1738418535.31` | Unix epoch with hundredths |
| `msecs` | `50535310` | Milliseconds since midnight |

### Sequence Numbers

- Sequence numbers are unique and monotonically increasing
- Start from 1 for each connection
- Used for stream recovery on reconnection
- Server should track last received sequence per location

**Resuming a stream:**
```
start~finish~4530\r\n
```
This resumes from sequence number 4530 (inclusive).

---

## Connection Recovery

### Detecting Disconnection

- TCP connection drops
- No response to `ping` command
- Socket error

### Recovery Procedure

1. PointCam detects disconnection
2. Queue events locally with sequence numbers
3. Attempt reconnection (with backoff)
4. On reconnect: complete greeting/init
5. Resume stream from last acknowledged sequence
6. Replay queued events

### Connection ID

PointCam can request connection ID to verify reconnection:

**Request:**
```
getconnectionid\r\n
```

**Response:**
```
ack~getconnectionid~<unique-id>\r\n
```

**Example:**
```
getconnectionid\r\n
ack~getconnectionid~b617318655057264e28bc0b6fb378c8ef146be00\r\n
```

---

## Commands

### Generic Commands

**Ping (connection check):**
```
ping\r\n
ack~ping\r\n
```

**Get Event Info:**
```
geteventinfo\r\n
ack~geteventinfo~<event-name>~<event-id>~<event-description>\r\n
```

**Get Locations:**
```
getlocations\r\n
ack~getlocations~finish~split-1~split-2\r\n
```

### Push Mode Commands

**Start all locations:**
```
start\r\n
```

**Start specific location:**
```
start~finish\r\n
```

**Start from sequence number:**
```
start~finish~4530\r\n
```

**Stop streaming:**
```
stop\r\n
stop~finish\r\n
```

### Error Responses

```
err~<command>~<error-message>\r\n
```

**Examples:**
```
err~~unknown command\r\n
err~start~invalid location\r\n
```

---

## PointCam Implementation Details

### Message Generation

```python
def generate_ctp01_message(crossing: CrossingEvent, location: str,
                           time_format: str = "normal") -> str:
    """Generate CTP01-compliant tag observation message."""

    time_str = format_time(crossing.timestamp, time_format)

    # Handle unknown bibs
    bib = crossing.bib_number if crossing.bib_number else "UNKNOWN"

    return (f"CT01_33~{crossing.sequence_number}~{location}~"
            f"{bib}~{time_str}~0~POINTCAM~0\r\n")
```

### Offline Queue Schema

```sql
CREATE TABLE message_queue (
    id INTEGER PRIMARY KEY,
    sequence_number INTEGER UNIQUE NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    sent BOOLEAN DEFAULT FALSE,
    sent_at DATETIME,
    ack_received BOOLEAN DEFAULT FALSE
);

-- Index for efficient unsent message retrieval
CREATE INDEX idx_unsent ON message_queue(sent, sequence_number);
```

### Reconnection Logic

```python
class CTP01Client:
    def __init__(self, host: str, port: int = 61611):
        self.host = host
        self.port = port
        self.location = "finish"
        self.last_sent_seq = 0
        self.connection_id = None

    async def connect_with_retry(self, max_attempts: int = 10):
        """Connect with exponential backoff."""
        for attempt in range(max_attempts):
            try:
                await self._connect()
                await self._handshake()
                await self._resume_stream()
                return True
            except ConnectionError:
                wait = min(2 ** attempt, 60)  # Max 60 seconds
                await asyncio.sleep(wait)
        return False

    async def _resume_stream(self):
        """Resume stream from last sent sequence."""
        if self.last_sent_seq > 0:
            await self.send(f"start~{self.location}~{self.last_sent_seq + 1}")
        else:
            await self.send(f"start~{self.location}")
```

---

## Testing & Validation

### Protocol Compliance Checklist

- [ ] Greeting exchange works with scoring software
- [ ] Initialization requests accepted
- [ ] Push mode streaming functional
- [ ] Sequence numbers monotonic and unique
- [ ] Time format correct for selected option
- [ ] Reconnection with sequence resume works
- [ ] UNKNOWN bibs handled correctly
- [ ] Connection ID retrieval works

### Test Scenarios

1. **Normal operation**: Connect, stream events, verify received
2. **Connection drop**: Simulate disconnect, verify queue, verify resume
3. **Sequence gaps**: Verify server handles gaps gracefully
4. **High volume**: Stream 100+ events rapidly
5. **Long duration**: Stream for 4+ hours continuous

### Simulator

For development, a simple CTP01 server simulator:

```python
# Simple CTP01 server for testing
import asyncio

async def handle_client(reader, writer):
    # Send greeting
    writer.write(b"TestServer~1.0~CTP01\r\n")
    await writer.drain()

    # Read client greeting
    data = await reader.readline()
    print(f"Client: {data.decode().strip()}")

    # Read init requests
    data = await reader.readline()
    num_requests = int(data.decode().strip().split('~')[2])
    for _ in range(num_requests):
        req = await reader.readline()
        print(f"Init: {req.decode().strip()}")

    # Handle commands
    while True:
        data = await reader.readline()
        if not data:
            break
        cmd = data.decode().strip()
        print(f"Received: {cmd}")

        if cmd.startswith("start"):
            writer.write(b"ack~start~finish\r\n")
        elif cmd == "ping":
            writer.write(b"ack~ping\r\n")
        await writer.drain()

async def main():
    server = await asyncio.start_server(handle_client, '0.0.0.0', 61611)
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

---

## Compatibility Notes

### Scoring Software Compatibility

| Software | Status | Notes |
|----------|--------|-------|
| RunScore | Expected | CTP01 documented support |
| Race Director | Expected | CTP01 documented support |
| ChronoTrack Live | Expected | Native CTP01 |
| Other CTP01-compatible | Should work | Standard protocol |

### Known Limitations

1. **UNKNOWN bibs**: Scoring software must handle non-numeric bib values
2. **Reader ID**: "POINTCAM" used instead of RFID reader ID
3. **Lap counting**: Always 0 in v1 (future: lap counting support)
4. **Gun time**: Not supported in v1 (camera doesn't detect start gun)

### Future Enhancements

- Gun time support via external trigger
- Lap counting for multi-lap races
- Multi-location per connection (location=multi)
- Authentication support for remote connections

---

## Reference: CTP01 Protocol Source

Full CTP01 specification from ChronoTrack documentation:
- Location: `C:/Users/alex/Projects/project88-scoring-worktree/ChronoTrack_Socket_Protocol/`
- Document: `CTP01 - Socket Protocol 01 - Updated 10_13_2015.pdf`
- Revision: 8 (2015-10-13)
