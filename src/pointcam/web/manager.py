"""PipelineManager — state machine wrapping camera + models + process_frames().

State machine:  IDLE → CONFIGURED → STARTING → RUNNING → STOPPING → IDLE
                                                           ERROR → IDLE

Singleton: the Jetson has one camera and one GPU, so only one pipeline runs
at a time.
"""

from __future__ import annotations

import enum
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pointcam.web.frame_buffer import FrameBuffer


class PipelineState(str, enum.Enum):
    IDLE = "idle"
    CONFIGURED = "configured"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class PipelineManagerConfig:
    """Validated configuration stored by the manager."""

    source: str = "csi"  # csi | usb | file
    video_path: Optional[str] = None
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    detector_path: str = "models/bib_detector_v3.pt"
    ocr_model: str = "parseq"  # parseq | crnn
    ocr_backend: str = "pytorch"  # pytorch | onnx | tensorrt
    conf_threshold: float = 0.25
    ocr_conf_threshold: float = 0.5
    placement: str = "center"  # left | center | right
    crossing_mode: str = "zone"  # line | zone
    crossing_direction: str = "any"
    timing_line: Optional[str] = None  # "x1,y1,x2,y2"
    debounce_time: float = 2.0
    pose_model: str = "yolov8n-pose.pt"
    stride: int = 1
    bib_set_path: Optional[str] = None
    bib_range: Optional[str] = None
    record: bool = False
    run_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineManagerConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class RunStats:
    """Live stats updated during a run."""

    frame_idx: int = 0
    elapsed_sec: float = 0.0
    total_crossings: int = 0
    unknown_crossings: int = 0
    fps: float = 0.0
    total_detections: int = 0


class PipelineManager:
    """Singleton state machine controlling the pipeline lifecycle."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._project_root = project_root or Path.cwd()
        self._state = PipelineState.IDLE
        self._config = PipelineManagerConfig()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._camera = None
        self._last_error: Optional[str] = None
        self._run_stats = RunStats()
        self._run_output_dir: Optional[Path] = None
        self.frame_buffer = FrameBuffer()

        # Callbacks for WebSocket broadcast (set by app.py)
        self._crossing_callbacks: List[Callable] = []
        self._progress_callbacks: List[Callable] = []

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def config(self) -> PipelineManagerConfig:
        return self._config

    @property
    def stats(self) -> RunStats:
        return self._run_stats

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def run_output_dir(self) -> Optional[Path]:
        return self._run_output_dir

    # -- Configuration -------------------------------------------------------

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Validate and store configuration. Transitions IDLE → CONFIGURED."""
        with self._lock:
            if self._state not in (PipelineState.IDLE, PipelineState.CONFIGURED, PipelineState.ERROR):
                raise RuntimeError(f"Cannot configure in state {self._state}")
            self._config = PipelineManagerConfig.from_dict(config_dict)
            self._state = PipelineState.CONFIGURED
            self._last_error = None

    # -- Start / Stop --------------------------------------------------------

    def start(self) -> None:
        """Start the pipeline in a background thread."""
        with self._lock:
            if self._state not in (PipelineState.CONFIGURED, PipelineState.ERROR):
                raise RuntimeError(f"Cannot start in state {self._state}")
            self._state = PipelineState.STARTING
            self._last_error = None
            self._run_stats = RunStats()

        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline")
        self._thread.start()

    def stop(self) -> None:
        """Stop the running pipeline."""
        with self._lock:
            if self._state not in (PipelineState.RUNNING, PipelineState.STARTING):
                return
            self._state = PipelineState.STOPPING

        if self._camera is not None:
            self._camera.stop()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.frame_buffer.clear()

        with self._lock:
            self._state = PipelineState.IDLE

    def reset(self) -> None:
        """Reset from ERROR state back to IDLE."""
        with self._lock:
            if self._state == PipelineState.ERROR:
                self._state = PipelineState.IDLE

    # -- Crossing / progress callback registration ---------------------------

    def add_crossing_callback(self, cb: Callable) -> None:
        self._crossing_callbacks.append(cb)

    def remove_crossing_callback(self, cb: Callable) -> None:
        try:
            self._crossing_callbacks.remove(cb)
        except ValueError:
            pass

    def add_progress_callback(self, cb: Callable) -> None:
        self._progress_callbacks.append(cb)

    def remove_progress_callback(self, cb: Callable) -> None:
        try:
            self._progress_callbacks.remove(cb)
        except ValueError:
            pass

    # -- Status snapshot for API ---------------------------------------------

    def status_dict(self) -> Dict[str, Any]:
        s = self._run_stats
        return {
            "state": self._state.value,
            "frame_idx": s.frame_idx,
            "elapsed_sec": round(s.elapsed_sec, 1),
            "total_crossings": s.total_crossings,
            "unknown_crossings": s.unknown_crossings,
            "fps": round(s.fps, 1),
            "total_detections": s.total_detections,
            "last_error": self._last_error,
            "run_output_dir": str(self._run_output_dir) if self._run_output_dir else None,
        }

    # -- Internal: pipeline thread -------------------------------------------

    def _run(self) -> None:
        """Pipeline thread entry point. Mirrors run_live.py logic."""
        try:
            self._run_inner()
        except Exception as e:
            with self._lock:
                self._state = PipelineState.ERROR
                self._last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            self._camera = None
            self.frame_buffer.clear()

    def _run_inner(self) -> None:
        cfg = self._config
        root = self._project_root

        # -- Resolve paths ---------------------------------------------------
        def _resolve(p: str) -> str:
            return p if Path(p).is_absolute() else str(root / p)

        detector_path = _resolve(cfg.detector_path)
        pose_model = cfg.pose_model

        # -- Open camera -----------------------------------------------------
        from pointcam.camera import CameraSource

        if cfg.source == "csi":
            camera = CameraSource.csi(cfg.width, cfg.height, cfg.fps, cfg.device_index)
        elif cfg.source == "usb":
            camera = CameraSource.usb(cfg.device_index, cfg.width, cfg.height, cfg.fps)
        elif cfg.source == "file":
            if not cfg.video_path:
                raise ValueError("video_path required for file source")
            camera = CameraSource.file(_resolve(cfg.video_path))
        else:
            raise ValueError(f"Unknown source: {cfg.source}")

        self._camera = camera

        # -- Output directory ------------------------------------------------
        run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / "runs" / "live" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_output_dir = output_dir

        # -- Load models -----------------------------------------------------
        from pointcam.pipeline import (
            PipelineConfig,
            ProgressInfo,
            UltralyticsBibDetector,
            _resolve_device,
            process_frames,
        )
        from pointcam.crossing import CrossingEvent, PoseDetector
        from pointcam.recognition import BibSetValidator

        device = _resolve_device()
        detector = UltralyticsBibDetector(detector_path, device=device)

        # OCR model
        if cfg.ocr_backend == "pytorch":
            from pointcam.inference import PARSeqOCR

            ckpt = root / "runs/ocr_finetune/parseq_gpu_v1/best.pt"
            ocr_model = PARSeqOCR(str(ckpt), device=device)
        elif cfg.ocr_backend == "tensorrt":
            if cfg.ocr_model == "parseq":
                from pointcam.inference import OnnxTensorRTParseqOCR

                ocr_model = OnnxTensorRTParseqOCR(str(root / "models/ocr_parseq.onnx"))
            else:
                from pointcam.inference import TensorRTCrnnOCR

                ocr_model = TensorRTCrnnOCR(str(root / "models/ocr_crnn.onnx"))
        else:
            if cfg.ocr_model == "parseq":
                from pointcam.inference import OnnxParseqOCR

                ocr_model = OnnxParseqOCR(str(root / "models/ocr_parseq.onnx"))
            else:
                from pointcam.inference import OnnxCrnnOCR

                ocr_model = OnnxCrnnOCR(str(root / "models/ocr_crnn.onnx"))

        # Pose detector
        pose_detector = None
        if cfg.crossing_mode == "zone" or cfg.timing_line:
            pose_detector = PoseDetector(model_path=pose_model, conf=0.5, device=device)

        # Bib validator
        bib_validator = None
        if cfg.bib_set_path:
            bib_validator = BibSetValidator.from_file(_resolve(cfg.bib_set_path))
        elif cfg.bib_range:
            start, end = map(int, cfg.bib_range.split("-"))
            bib_validator = BibSetValidator.from_range(start, end)

        # Timing line
        timing_line_coords = None
        if cfg.timing_line:
            parts = [float(x) for x in cfg.timing_line.split(",")]
            if len(parts) == 4:
                timing_line_coords = tuple(parts)

        # -- Pipeline config -------------------------------------------------
        pipeline_config = PipelineConfig(
            conf_threshold=cfg.conf_threshold,
            ocr_conf_threshold=cfg.ocr_conf_threshold,
            enable_quality_filter=True,
            write_video=False,
            write_raw_video=cfg.record,
            placement=cfg.placement,
            timing_line_coords=timing_line_coords,
            crossing_direction=cfg.crossing_direction,
            debounce_time=cfg.debounce_time,
            enable_person_detect=True,
            pose_model_path=pose_model,
            stride=cfg.stride,
            start_time=0.0,
            enable_ocr_skip=True,
            crossing_mode=cfg.crossing_mode,
        )

        # -- Callbacks -------------------------------------------------------
        def on_crossing(event: CrossingEvent) -> None:
            s = self._run_stats
            s.total_crossings += 1
            if event.bib_number == "UNKNOWN":
                s.unknown_crossings += 1
            for cb in self._crossing_callbacks:
                try:
                    cb(event)
                except Exception:
                    pass

        def on_progress(info: ProgressInfo) -> None:
            s = self._run_stats
            s.frame_idx = info.frame_idx
            s.elapsed_sec = info.elapsed_sec
            s.fps = info.fps
            s.total_detections = info.total_detections
            s.total_crossings = info.total_crossings
            s.unknown_crossings = info.unknown_crossings
            for cb in self._progress_callbacks:
                try:
                    cb(info)
                except Exception:
                    pass

        def on_frame(frame: np.ndarray) -> None:
            self.frame_buffer.update(frame)

        # -- Transition to RUNNING -------------------------------------------
        with self._lock:
            if self._state != PipelineState.STARTING:
                return  # stop() was called before we finished loading
            self._state = PipelineState.RUNNING

        # -- Run pipeline ----------------------------------------------------
        total = camera.total_frames if cfg.source == "file" else None

        with camera:
            process_frames(
                frames=camera,
                fps=camera.fps,
                detector=detector,
                ocr_model=ocr_model,
                output_dir=output_dir,
                output_stem=run_name,
                config=pipeline_config,
                bib_validator=bib_validator,
                pose_detector=pose_detector,
                show=False,
                total_frames=total,
                frame_size=(camera.width, camera.height),
                on_crossing=on_crossing,
                on_progress=on_progress,
                on_frame=on_frame,
                print_summary=False,
            )

        # Normal completion
        with self._lock:
            if self._state == PipelineState.RUNNING:
                self._state = PipelineState.IDLE
