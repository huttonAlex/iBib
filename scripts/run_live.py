#!/usr/bin/env python3
"""Live camera pipeline for race-day bib detection and crossing timing.

Captures frames from a camera (CSI, USB, or video file) and runs the full
bib detection + OCR + crossing pipeline in real-time. Crossings are printed
to the terminal as they happen and written to CSV.

Usage:
    # Jetson CSI camera (default)
    python scripts/run_live.py --crossing-mode zone --placement right --bib-set bibs.txt

    # USB webcam
    python scripts/run_live.py --source usb --crossing-mode zone --placement right

    # Video file (for testing)
    python scripts/run_live.py --source file --video REC-0006-A.mp4 --crossing-mode zone

    # With live preview
    python scripts/run_live.py --source csi --show --crossing-mode zone --placement right
"""

import argparse
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pointcam.camera import CameraSource
from pointcam.crossing import CrossingEvent, PoseDetector
from pointcam.inference import (
    OnnxCrnnOCR,
    OnnxParseqOCR,
    OnnxTensorRTParseqOCR,
    PARSeqOCR,
    TensorRTCrnnOCR,
)
from pointcam.pipeline import PipelineConfig, ProgressInfo, UltralyticsBibDetector, process_frames
from pointcam.recognition import BibSetValidator


def _resolve_device(preferred=None):
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_parseq_tokenizer(parseq_onnx: Path) -> Path:
    tokenizer_path = parseq_onnx.with_suffix(".tokenizer.json")
    if not tokenizer_path.exists():
        print("ERROR: PARSeq tokenizer file not found:")
        print(f"  {tokenizer_path}")
        sys.exit(1)
    return tokenizer_path


def main():
    parser = argparse.ArgumentParser(
        description="Live camera pipeline for bib detection and crossing timing"
    )

    # Source selection
    parser.add_argument(
        "--source",
        type=str,
        default="csi",
        choices=["csi", "usb", "file"],
        help="Frame source (default: csi)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file path (when --source file)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index: sensor_id for CSI, /dev/videoN for USB (default: 0)",
    )

    # Resolution
    parser.add_argument("--width", type=int, default=1920, help="Frame width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Frame height (default: 1080)")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30)")

    # Pipeline config
    parser.add_argument(
        "--detector",
        type=str,
        default="models/bib_detector_v3.pt",
        help="Path to YOLO detector weights",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default="parseq",
        choices=["parseq", "crnn"],
        help="OCR model (default: parseq)",
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="pytorch",
        choices=["onnx", "tensorrt", "pytorch"],
        help="OCR inference backend (default: pytorch)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.5, help="OCR confidence threshold")
    parser.add_argument("--bib-set", type=str, default=None, help="Path to bib set file")
    parser.add_argument("--bib-range", type=str, default=None, help="Bib range (e.g. '1-3000')")
    parser.add_argument(
        "--placement",
        type=str,
        default="center",
        choices=["left", "right", "center"],
        help="Camera placement relative to finish line (default: center)",
    )
    parser.add_argument(
        "--crossing-mode",
        type=str,
        default="zone",
        choices=["line", "zone"],
        help="Crossing detection mode (default: zone)",
    )
    parser.add_argument(
        "--timing-line",
        type=str,
        default=None,
        help="Timing line as x1,y1,x2,y2 (normalized coords)",
    )
    parser.add_argument(
        "--crossing-direction",
        type=str,
        default="any",
        choices=["left_to_right", "right_to_left", "any"],
        help="Valid crossing direction (default: any)",
    )
    parser.add_argument(
        "--debounce-time",
        type=float,
        default=2.0,
        help="Min seconds between crossings for same track (default: 2.0)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolov8n-pose.pt",
        help="Path to YOLOv8-pose weights",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--no-ocr-skip",
        action="store_true",
        help="Disable OCR skip optimization for stable tracks",
    )

    # Output
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output subdirectory name (default: timestamp)",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Manual clock offset in seconds for displayed timestamps",
    )
    parser.add_argument("--show", action="store_true", help="Live video preview")
    parser.add_argument("--record", action="store_true", help="Record video to output directory")
    parser.add_argument(
        "--tui", action="store_true", help="Rich terminal dashboard (requires pip install pointcam[tui])"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # -- Open camera source --------------------------------------------------
    if args.source == "csi":
        camera = CameraSource.csi(args.width, args.height, args.fps, args.device)
    elif args.source == "usb":
        camera = CameraSource.usb(args.device, args.width, args.height, args.fps)
    elif args.source == "file":
        if not args.video:
            print("ERROR: --video PATH required when --source file")
            sys.exit(1)
        video_path = args.video if Path(args.video).is_absolute() else str(project_root / args.video)
        camera = CameraSource.file(video_path)
    else:
        print(f"ERROR: Unknown source: {args.source}")
        sys.exit(1)

    # -- Ctrl+C handling: stop camera gracefully -----------------------------
    def _signal_handler(sig, frame):
        if not args.tui:
            print("\n  Ctrl+C received — stopping camera...")
        camera.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    # -- Output directory ----------------------------------------------------
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "live" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Resolve paths -------------------------------------------------------
    detector_path = (
        args.detector if Path(args.detector).is_absolute() else str(project_root / args.detector)
    )

    # -- Load models ---------------------------------------------------------
    device = _resolve_device()

    if not args.tui:
        print("=" * 70)
        print("PointCam Live Pipeline")
        print("=" * 70)
        print(f"Source:    {camera.name}")
        print(f"Detector: {detector_path}")
        print(f"OCR:      {args.ocr} (backend: {args.ocr_backend})")
        print(f"Placement: {args.placement}")
        print(f"Crossing:  {args.crossing_mode}")
        print(f"Output:   {output_dir}")

    # Bib validator
    bib_validator = None
    if args.bib_set:
        bib_set_path = (
            args.bib_set if Path(args.bib_set).is_absolute() else str(project_root / args.bib_set)
        )
        bib_validator = BibSetValidator.from_file(bib_set_path)
        print(f"Bib set:  {bib_set_path} ({len(bib_validator.bib_set)} bibs)")
    elif args.bib_range:
        try:
            start, end = map(int, args.bib_range.split("-"))
            bib_validator = BibSetValidator.from_range(start, end)
            print(f"Bib range: {start}-{end}")
        except ValueError:
            print(f"ERROR: Invalid bib range: {args.bib_range}")
            sys.exit(1)
    else:
        print("Bib set:  None (validation disabled)")

    print()
    print("Loading models...")
    detector = UltralyticsBibDetector(detector_path, device=device)
    print(f"  Detector loaded (device: {device})")

    # OCR model
    if args.ocr_backend == "pytorch":
        if args.ocr == "parseq":
            ckpt = project_root / "runs/ocr_finetune/parseq_gpu_v1/best.pt"
            ocr_model = PARSeqOCR(str(ckpt), device=device)
            print(f"  PARSeq PyTorch loaded (device: {device})")
        else:
            print("ERROR: PyTorch backend only supports PARSeq.")
            sys.exit(1)
    elif args.ocr_backend == "tensorrt":
        if args.ocr == "parseq":
            parseq_onnx = project_root / "models/ocr_parseq.onnx"
            _ensure_parseq_tokenizer(parseq_onnx)
            ocr_model = OnnxTensorRTParseqOCR(str(parseq_onnx))
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = TensorRTCrnnOCR(str(crnn_onnx))
    else:
        if args.ocr == "parseq":
            parseq_onnx = project_root / "models/ocr_parseq.onnx"
            _ensure_parseq_tokenizer(parseq_onnx)
            ocr_model = OnnxParseqOCR(str(parseq_onnx))
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = OnnxCrnnOCR(str(crnn_onnx))

    # Pose detector
    pose_detector = None
    if args.crossing_mode == "zone" or args.timing_line:
        pose_detector = PoseDetector(
            model_path=args.pose_model,
            conf=0.5,
            device=device,
        )
        print(f"  Pose detector loaded (device: {device})")

    # -- Timing line parsing -------------------------------------------------
    timing_line_coords = None
    if args.timing_line:
        try:
            parts = [float(x) for x in args.timing_line.split(",")]
            if len(parts) != 4:
                raise ValueError("need 4 values")
            timing_line_coords = tuple(parts)
        except ValueError:
            print(f"ERROR: Invalid timing line: {args.timing_line}")
            sys.exit(1)

    # -- Pipeline config -----------------------------------------------------
    config = PipelineConfig(
        conf_threshold=args.conf,
        ocr_conf_threshold=args.ocr_conf,
        enable_quality_filter=True,
        write_video=args.record,
        placement=args.placement,
        timing_line_coords=timing_line_coords,
        crossing_direction=args.crossing_direction,
        debounce_time=args.debounce_time,
        enable_person_detect=True,
        pose_model_path=args.pose_model,
        stride=args.stride,
        start_time=0.0,
        enable_ocr_skip=not args.no_ocr_skip,
        crossing_mode=args.crossing_mode,
    )

    # -- TUI dashboard (optional) ---------------------------------------------
    dashboard = None
    if args.tui:
        try:
            from pointcam.dashboard import LiveDashboard
        except ImportError:
            print("ERROR: --tui requires the 'rich' package.")
            print("  Install with: pip install pointcam[tui]")
            sys.exit(1)
        dashboard = LiveDashboard(
            camera_name=camera.name,
            ocr_model=f"{args.ocr} ({args.ocr_backend})",
            crossing_mode=args.crossing_mode,
            placement=args.placement,
        )

    # -- Crossing callback (print to terminal) -------------------------------
    time_offset = timedelta(seconds=args.time_offset)
    crossing_count = [0]  # mutable for closure

    def on_crossing_plain(event: CrossingEvent):
        crossing_count[0] += 1
        wall_time = datetime.now() + time_offset
        ts = wall_time.strftime("%H:%M:%S")
        bib = event.bib_number
        conf = event.confidence
        src = event.source
        print(f"  >> CROSSING #{crossing_count[0]:>4d}  bib={bib:<6s}  conf={conf:.2f}  [{src}]  {ts}")

    def on_crossing_tui(event: CrossingEvent):
        crossing_count[0] += 1
        dashboard.on_crossing(event)

    crossing_cb = on_crossing_tui if dashboard else on_crossing_plain
    progress_cb = dashboard.on_progress if dashboard else None

    # -- Run pipeline --------------------------------------------------------
    if not dashboard:
        print()
        print("=" * 70)
        print("LIVE — press Ctrl+C to stop")
        print("=" * 70)
        print()

    t_start = time.time()

    tui_ctx = dashboard if dashboard else nullcontext()

    with camera, tui_ctx:
        # For file sources, pass total_frames so progress % works
        total = camera.total_frames if args.source == "file" else None
        fps = camera.fps

        result = process_frames(
            frames=camera,
            fps=fps,
            detector=detector,
            ocr_model=ocr_model,
            output_dir=output_dir,
            output_stem=run_name,
            config=config,
            bib_validator=bib_validator,
            pose_detector=pose_detector,
            show=args.show,
            total_frames=total,
            frame_size=(camera.width, camera.height),
            on_crossing=crossing_cb,
            on_progress=progress_cb,
        )

    elapsed = time.time() - t_start

    # -- Print final summary -------------------------------------------------
    print()
    print("=" * 70)
    print("LIVE SESSION COMPLETE")
    print("=" * 70)
    print(f"Duration:    {elapsed:.1f}s")
    print(f"Frames:      {result.stats.frames_processed}")
    print(f"Crossings:   {result.stats.total_crossings}")
    print(f"  Known bib: {result.stats.total_crossings - result.stats.unknown_crossings}")
    print(f"  Unknown:   {result.stats.unknown_crossings}")
    print(f"Output dir:  {output_dir}")
    if result.outputs.crossing_log_path:
        print(f"Crossings:   {result.outputs.crossing_log_path}")
    if result.outputs.detection_log_path:
        print(f"Detections:  {result.outputs.detection_log_path}")


if __name__ == "__main__":
    main()
