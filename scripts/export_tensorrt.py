#!/usr/bin/env python3
"""Export models to TensorRT-optimized formats for Jetson deployment.

Run this script **on the Jetson** (TensorRT builds device-specific engines).

Exports:
1. YOLOv8 bib detector  → .engine via ultralytics (trivial)
2. YOLOv8n-pose          → .engine via ultralytics (trivial)
3. PARSeq OCR            → stays .onnx, accelerated via ONNX Runtime TensorRT EP at runtime
4. CRNN OCR              → .engine via trtexec

Usage:
    # Export all models (default paths)
    python scripts/export_tensorrt.py

    # Export specific model(s)
    python scripts/export_tensorrt.py --models bib pose crnn

    # Custom paths
    python scripts/export_tensorrt.py --bib-weights runs/detect/bib_detector/weights/best.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


def export_yolo_engine(weights_path: str, imgsz: int = 640, half: bool = True) -> Path:
    """Export a YOLO model to TensorRT .engine via ultralytics.

    Args:
        weights_path: Path to .pt weights.
        imgsz: Input image size.
        half: Use FP16 quantization.

    Returns:
        Path to the exported .engine file.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    engine_path = model.export(format="engine", half=half, imgsz=imgsz)
    print(f"  Exported: {engine_path}")
    return Path(engine_path)


def export_crnn_trtexec(onnx_path: str, engine_path: str, fp16: bool = True) -> Path:
    """Export CRNN ONNX model to TensorRT engine via trtexec.

    CRNN has a simple CTC architecture with no dynamic control flow,
    so direct trtexec conversion works well.

    Args:
        onnx_path: Path to CRNN .onnx file.
        engine_path: Output .engine path.
        fp16: Use FP16 precision.

    Returns:
        Path to the exported .engine file.
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  trtexec FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        sys.exit(1)

    print(f"  Exported: {engine_path}")
    return Path(engine_path)


def check_parseq_onnx(onnx_path: str) -> None:
    """Verify PARSeq ONNX exists and print TensorRT EP usage note.

    PARSeq uses an autoregressive decoder with dynamic control flow
    that cannot be fully converted to a static TensorRT engine.
    Instead, ONNX Runtime with the TensorRT Execution Provider compiles
    compatible subgraphs to TensorRT and falls back to CUDA EP for the rest.
    The TRT engine cache is built on first run and reused thereafter.

    Args:
        onnx_path: Path to PARSeq .onnx file.
    """
    path = Path(onnx_path)
    if not path.exists():
        print(f"  WARNING: PARSeq ONNX not found at {onnx_path}")
        print("  Export it first: python scripts/export_ocr_model.py --model parseq")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  PARSeq ONNX: {onnx_path} ({size_mb:.0f} MB)")
    print("  No pre-export needed — ONNX Runtime + TensorRT EP builds and caches")
    print("  the TRT engine on first inference run. Subsequent runs reuse the cache.")
    print("  Use: OnnxTensorRTParseqOCR from pointcam.inference")


def main():
    parser = argparse.ArgumentParser(
        description="Export models to TensorRT for Jetson deployment"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bib", "pose", "parseq", "crnn"],
        choices=["bib", "pose", "parseq", "crnn"],
        help="Which models to export (default: all)",
    )
    parser.add_argument(
        "--bib-weights",
        default="runs/detect/bib_detector/weights/best.pt",
        help="Path to YOLOv8 bib detector weights",
    )
    parser.add_argument(
        "--pose-weights",
        default="yolov8n-pose.pt",
        help="Path to YOLOv8n-pose weights",
    )
    parser.add_argument(
        "--parseq-onnx",
        default="models/ocr_parseq.onnx",
        help="Path to PARSeq ONNX model",
    )
    parser.add_argument(
        "--crnn-onnx",
        default="models/ocr_crnn.onnx",
        help="Path to CRNN ONNX model",
    )
    parser.add_argument(
        "--crnn-engine",
        default="models/ocr_crnn.engine",
        help="Output path for CRNN TensorRT engine",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO input image size (default: 640)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 (use FP32 instead)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    half = not args.no_half

    def resolve(p: str) -> str:
        return str(project_root / p) if not Path(p).is_absolute() else p

    print("=" * 60)
    print("TensorRT Model Export")
    print("=" * 60)

    if "bib" in args.models:
        print(f"\n[1/4] YOLOv8 Bib Detector → TensorRT engine")
        bib_path = resolve(args.bib_weights)
        if not Path(bib_path).exists():
            print(f"  SKIP: Weights not found at {bib_path}")
        else:
            export_yolo_engine(bib_path, imgsz=args.imgsz, half=half)

    if "pose" in args.models:
        print(f"\n[2/4] YOLOv8n-Pose → TensorRT engine")
        pose_path = resolve(args.pose_weights)
        if not Path(pose_path).exists():
            print(f"  SKIP: Weights not found at {pose_path}")
            print("  Run the pipeline once with --pose-model yolov8n-pose.pt to auto-download")
        else:
            export_yolo_engine(pose_path, imgsz=args.imgsz, half=half)

    if "parseq" in args.models:
        print(f"\n[3/4] PARSeq OCR → ONNX Runtime + TensorRT EP (no pre-export)")
        check_parseq_onnx(resolve(args.parseq_onnx))

    if "crnn" in args.models:
        print(f"\n[4/4] CRNN OCR → TensorRT engine via trtexec")
        crnn_onnx = resolve(args.crnn_onnx)
        crnn_engine = resolve(args.crnn_engine)
        if not Path(crnn_onnx).exists():
            print(f"  SKIP: ONNX not found at {crnn_onnx}")
        else:
            export_crnn_trtexec(crnn_onnx, crnn_engine, fp16=half)

    print("\n" + "=" * 60)
    print("Export complete.")
    print()
    print("Next steps:")
    print("  1. Pass .engine paths to the pipeline instead of .pt paths")
    print("     (ultralytics auto-detects format from file extension)")
    print("  2. Use --ocr-backend tensorrt for TensorRT-accelerated OCR")
    print("  3. First PARSeq inference will be slow (TRT EP cache build)")
    print("=" * 60)


if __name__ == "__main__":
    main()
