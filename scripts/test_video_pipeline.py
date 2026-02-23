#!/usr/bin/env python3
"""Test full bib detection + OCR pipeline on video with Tier 1 improvements.

Runs YOLOv8 bib detector + PARSeq/CRNN OCR on each frame with:
- Bib set validation (reject impossible numbers)
- Multi-frame voting (temporal consistency)
- Confidence thresholding (flag uncertain reads)
- Person detection via YOLOv8n-pose (chest keypoint crossing)
- Persistent person-bib association with temporal voting
- Bib-level dedup to prevent double-counting

Outputs annotated video, detection log, crossing log, and review queue.

Usage:
    python scripts/test_video_pipeline.py path/to/video.mp4
    python scripts/test_video_pipeline.py path/to/video.mp4 --ocr crnn
    python scripts/test_video_pipeline.py path/to/video.mp4 --bib-set bibs.txt
    python scripts/test_video_pipeline.py path/to/video.mp4 --placement right --no-video
    python scripts/test_video_pipeline.py path/to/video.mp4 --timing-line 0.5,0.0,0.5,1.0
    python scripts/test_video_pipeline.py path/to/video.mp4 --show
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pointcam.inference import (
    OnnxCrnnOCR,
    OnnxParseqOCR,
    OnnxTensorRTParseqOCR,
    TensorRTCrnnOCR,
)
from pointcam.pipeline import PipelineConfig, process_video
from pointcam.recognition import BibSetValidator


class PARSeqOCR:
    """PARSeq OCR from fine-tuned PyTorch checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        from PIL import Image  # noqa: F811

        self._torch = torch
        self._Image = Image
        self.device = device
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval().to(device)

    def predict(self, crop_bgr):
        import numpy as np
        import cv2

        torch = self._torch
        Image = self._Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img = pil_img.resize((128, 32), Image.BICUBIC)
        t = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        tensor = ((t - 0.5) / 0.5).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(-1)
            preds, probs_out = self.model.tokenizer.decode(probs)

        text = preds[0]
        p = probs_out[0]
        if p.numel() == 0:
            confidence = 0.5
        elif p.dim() == 1:
            confidence = p.cumprod(-1)[-1].item()
        else:
            confidence = p.cumprod(-1)[:, -1].item()

        digits = "".join(c for c in text if c.isdigit())
        return digits, confidence

    def predict_batch(self, crops_bgr):
        import numpy as np
        import cv2

        torch = self._torch
        if not crops_bgr:
            return []

        preprocessed = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 32), interpolation=cv2.INTER_CUBIC)
            t = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
            preprocessed.append((t - 0.5) / 0.5)

        batch = torch.from_numpy(np.stack(preprocessed)).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = logits.softmax(-1)
            preds, probs_out = self.model.tokenizer.decode(probs)

        results = []
        for i in range(len(crops_bgr)):
            text = preds[i]
            p = probs_out[i]
            if p.numel() == 0:
                confidence = 0.5
            elif p.dim() == 1:
                confidence = p.cumprod(-1)[-1].item()
            else:
                confidence = p.cumprod(-1)[:, -1].item()
            digits = "".join(c for c in text if c.isdigit())
            results.append((digits, confidence))
        return results


def _ensure_parseq_tokenizer(parseq_onnx: Path) -> Path:
    tokenizer_path = parseq_onnx.with_suffix(".tokenizer.json")
    if not tokenizer_path.exists():
        print("ERROR: PARSeq tokenizer file not found:")
        print(f"  {tokenizer_path}")
        print("\nGenerate or fetch model assets first:")
        print("  python scripts/fetch_model_assets.py --base-url <URL>")
        print("  OR")
        print("  python scripts/export_ocr_model.py --model parseq --checkpoint <ckpt>")
        sys.exit(1)
    return tokenizer_path


def main():
    parser = argparse.ArgumentParser(
        description="Test bib detection + OCR pipeline on video with Tier 1 improvements"
    )
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument(
        "--ocr",
        type=str,
        default="parseq",
        choices=["parseq", "crnn"],
        help="OCR model to use (default: parseq)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="models/bib_detector_v2.pt",
        help="Path to YOLO detector weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/pipeline_test",
        help="Output directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.5,
        help="OCR confidence threshold for 'good' reads",
    )
    parser.add_argument(
        "--bib-set",
        type=str,
        default=None,
        help="Path to bib set file (one number per line) for validation",
    )
    parser.add_argument(
        "--bib-range",
        type=str,
        default=None,
        help="Bib number range for validation (e.g., '1-3000' or '1000-2000')",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show video while processing",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable crop quality filtering (not recommended)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated output video (much faster)",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="center",
        choices=["left", "right", "center"],
        help="Camera placement relative to finish line (affects crop padding). "
             "See docs/CAMERA_PLACEMENT.md for guidance. (default: center)",
    )
    parser.add_argument(
        "--timing-line",
        type=str,
        default=None,
        help="Timing line as normalized coords: x1,y1,x2,y2 (e.g., '0.5,0.0,0.5,1.0'). "
             "Enables crossing detection.",
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
        help="Minimum seconds between crossings for the same track (default: 2.0)",
    )
    parser.add_argument(
        "--crossing-mode",
        type=str,
        default="line",
        choices=["line", "zone"],
        help="Crossing detection mode: 'line' = timing line crossing (requires --timing-line), "
             "'zone' = emit once per person track after minimum age (for head-on cameras). "
             "(default: line)",
    )
    parser.add_argument(
        "--no-person-detect",
        action="store_true",
        help="Disable person detection (use bib tracker for crossings instead)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolov8n-pose.pt",
        help="Path to YOLOv8-pose weights (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, no skipping)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Skip to this time in seconds before processing (default: 0.0)",
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt", "pytorch"],
        help="OCR inference backend. 'pytorch' uses torch.hub PARSeq (GPU-capable).",
    )
    parser.add_argument(
        "--no-ocr-skip",
        action="store_true",
        help="Disable OCR skip optimization for stable tracks (always run OCR)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    video_path = args.video if Path(args.video).is_absolute() else str(project_root / args.video)
    detector_path = (
        args.detector if Path(args.detector).is_absolute() else str(project_root / args.detector)
    )
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Bib Detection + OCR Pipeline Test")
    print("=" * 70)
    print(f"Video:    {video_path}")
    print(f"Detector: {detector_path}")
    print(f"OCR:      {args.ocr} (backend: {args.ocr_backend})")
    print(f"Placement: {args.placement}")
    print(f"OCR skip: {'disabled' if args.no_ocr_skip else 'enabled'}")
    print(f"Output:   {output_dir}")

    bib_validator = None
    if args.bib_set:
        bib_set_path = (
            args.bib_set if Path(args.bib_set).is_absolute() else str(project_root / args.bib_set)
        )
        bib_validator = BibSetValidator.from_file(bib_set_path)
        print(f"Bib set:  {bib_set_path} ({len(bib_validator.bib_set)} numbers)")
    elif args.bib_range:
        try:
            start, end = map(int, args.bib_range.split("-"))
            bib_validator = BibSetValidator.from_range(start, end)
            print(f"Bib range: {start}-{end} ({len(bib_validator.bib_set)} numbers)")
        except ValueError:
            print(f"ERROR: Invalid bib range format: {args.bib_range}")
            print("  Expected format: START-END (e.g., '1-3000')")
            sys.exit(1)
    else:
        print("Bib set:  None (validation disabled)")

    print()
    print("Loading OCR model...")
    if args.ocr_backend == "pytorch":
        if args.ocr == "parseq":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = project_root / "runs/ocr_finetune/parseq_gpu_v1/best.pt"
            ocr_model = PARSeqOCR(str(ckpt), device=device)
            print(f"  PARSeq PyTorch loaded from {ckpt} (device: {device})")
        else:
            print("ERROR: PyTorch backend only supports PARSeq, not CRNN.")
            sys.exit(1)
    elif args.ocr_backend == "tensorrt":
        if args.ocr == "parseq":
            parseq_onnx = project_root / "models/ocr_parseq.onnx"
            _ensure_parseq_tokenizer(parseq_onnx)
            ocr_model = OnnxTensorRTParseqOCR(str(parseq_onnx))
            print(f"  PARSeq TensorRT loaded from {parseq_onnx}")
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = TensorRTCrnnOCR(str(crnn_onnx))
            print(f"  CRNN TensorRT loaded from {crnn_onnx}")
    else:
        if args.ocr == "parseq":
            parseq_onnx = project_root / "models/ocr_parseq.onnx"
            _ensure_parseq_tokenizer(parseq_onnx)
            ocr_model = OnnxParseqOCR(str(parseq_onnx))
            print(f"  PARSeq ONNX loaded from {parseq_onnx}")
        else:
            crnn_onnx = project_root / "models/ocr_crnn.onnx"
            ocr_model = OnnxCrnnOCR(str(crnn_onnx))
            print(f"  CRNN ONNX loaded from {crnn_onnx}")

    timing_line_coords = None
    if args.timing_line:
        try:
            parts = [float(x) for x in args.timing_line.split(",")]
            if len(parts) != 4:
                raise ValueError("need 4 values")
            timing_line_coords = tuple(parts)
        except ValueError:
            print(f"ERROR: Invalid timing line format: {args.timing_line}")
            print("  Expected format: x1,y1,x2,y2 (e.g., '0.5,0.0,0.5,1.0')")
            sys.exit(1)

    config = PipelineConfig(
        conf_threshold=args.conf,
        ocr_conf_threshold=args.ocr_conf,
        enable_quality_filter=not args.no_quality_filter,
        write_video=not args.no_video,
        placement=args.placement,
        timing_line_coords=timing_line_coords,
        crossing_direction=args.crossing_direction,
        debounce_time=args.debounce_time,
        enable_person_detect=not args.no_person_detect,
        pose_model_path=args.pose_model,
        stride=args.stride,
        start_time=args.start_time,
        enable_ocr_skip=not args.no_ocr_skip,
        crossing_mode=args.crossing_mode,
    )

    process_video(
        video_path=video_path,
        detector_path=detector_path,
        ocr_model=ocr_model,
        output_dir=output_dir,
        config=config,
        bib_validator=bib_validator,
        show=args.show,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
