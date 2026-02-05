#!/usr/bin/env python3
"""Export fine-tuned OCR model to ONNX for edge deployment.

Exports PyTorch checkpoint to ONNX (opset 17), verifies accuracy parity,
and measures ONNX Runtime inference speed.

Usage:
    python scripts/export_ocr_model.py --model crnn --checkpoint runs/ocr_finetune/crnn_v1/best.pt
    python scripts/export_ocr_model.py --model parseq --checkpoint runs/ocr_finetune/parseq_v1/best.pt
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Allow importing from scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_crnn(checkpoint_path: str, output_path: str):
    """Export CRNN model to ONNX."""
    from finetune_ocr import BibNumberCRNN

    model = BibNumberCRNN()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # Dummy input: (B, 1, 32, 128) grayscale
    dummy = torch.randn(1, 1, 32, 128)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"  CRNN exported to {output_path}")
    return model, dummy


def export_parseq(checkpoint_path: str, output_path: str):
    """Export PARSeq model to ONNX.

    PARSeq uses an autoregressive decoder with data-dependent control flow,
    which is incompatible with torch.export. We use the legacy JIT tracing
    mode via dynamo=False.
    """
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # PARSeq input: (B, 3, 32, 128) RGB
    dummy = torch.randn(1, 3, 32, 128)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"  PARSeq exported to {output_path}")
    return model, dummy


def verify_onnx(onnx_path: str):
    """Verify ONNX model is valid."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ONNX model verified: {onnx_path}")

    # Print model size
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")


def verify_accuracy(
    pytorch_model,
    onnx_path: str,
    test_dir: Path,
    model_type: str,
    limit: int = 200,
) -> tuple[float, float]:
    """Compare PyTorch vs ONNX accuracy on test set.

    Returns (pytorch_accuracy, onnx_accuracy).
    """
    from finetune_ocr import decode_ctc

    tsv_path = test_dir / "labels.tsv"
    images_dir = test_dir / "images"

    if not tsv_path.exists():
        print(f"  WARNING: {tsv_path} not found, skipping accuracy verification")
        return 0.0, 0.0

    samples = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append({
                "img_path": str(images_dir / row["filename"]),
                "label": row["label"],
            })

    if limit > 0:
        samples = samples[:limit]

    # ONNX session
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    pytorch_correct = 0
    onnx_correct = 0
    mismatches = 0
    total = 0

    pytorch_model.eval()

    for sample in samples:
        img = cv2.imread(sample["img_path"])
        if img is None:
            continue

        gt = sample["label"]

        if model_type == "crnn":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            resized = cv2.resize(gray, (128, 32))
            tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

            # PyTorch prediction
            with torch.no_grad():
                pt_output = pytorch_model(tensor)
            pt_pred = decode_ctc(pt_output[0])

            # ONNX prediction
            onnx_input = tensor.numpy()
            onnx_output = session.run(None, {input_name: onnx_input})[0]
            onnx_pred = decode_ctc(torch.from_numpy(onnx_output[0]))

        elif model_type == "parseq":
            from torchvision import transforms
            from PIL import Image

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            transform = transforms.Compose([
                transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
            tensor = transform(pil_img).unsqueeze(0)

            # PyTorch prediction
            with torch.no_grad():
                pt_logits = pytorch_model(tensor)
                pt_probs = pt_logits.softmax(-1)
                pt_preds, _ = pytorch_model.tokenizer.decode(pt_probs)
            pt_pred = "".join(c for c in pt_preds[0] if c.isdigit())

            # ONNX prediction
            onnx_input = tensor.numpy()
            onnx_output = session.run(None, {input_name: onnx_input})[0]
            onnx_logits = torch.from_numpy(onnx_output)
            onnx_probs = onnx_logits.softmax(-1)
            onnx_preds, _ = pytorch_model.tokenizer.decode(onnx_probs)
            onnx_pred = "".join(c for c in onnx_preds[0] if c.isdigit())

        else:
            continue

        if pt_pred == gt:
            pytorch_correct += 1
        if onnx_pred == gt:
            onnx_correct += 1
        if pt_pred != onnx_pred:
            mismatches += 1
        total += 1

    pt_acc = pytorch_correct / max(total, 1)
    onnx_acc = onnx_correct / max(total, 1)

    print(f"\n  Accuracy verification ({total} samples):")
    print(f"    PyTorch:    {pt_acc:.1%}")
    print(f"    ONNX:       {onnx_acc:.1%}")
    print(f"    Mismatches: {mismatches}/{total} ({mismatches/max(total,1):.1%})")
    print(f"    Difference: {abs(pt_acc - onnx_acc):.1%}")

    if abs(pt_acc - onnx_acc) > 0.01:
        print("  WARNING: Accuracy difference exceeds 1% threshold!")
    else:
        print("  OK: Accuracy within 1% tolerance")

    return pt_acc, onnx_acc


def measure_inference_speed(onnx_path: str, model_type: str, n_runs: int = 500):
    """Measure ONNX Runtime inference speed."""
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    if model_type == "crnn":
        dummy = np.random.randn(1, 1, 32, 128).astype(np.float32)
    elif model_type == "parseq":
        dummy = np.random.randn(1, 3, 32, 128).astype(np.float32)
    else:
        dummy = np.random.randn(1, 3, 32, 128).astype(np.float32)

    # Warmup
    for _ in range(50):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    print(f"\n  ONNX Runtime inference speed ({n_runs} runs):")
    print(f"    Mean:   {times.mean():.2f} ms")
    print(f"    Median: {np.median(times):.2f} ms")
    print(f"    P95:    {np.percentile(times, 95):.2f} ms")
    print(f"    P99:    {np.percentile(times, 99):.2f} ms")
    print(f"    Min:    {times.min():.2f} ms")
    print(f"    Max:    {times.max():.2f} ms")

    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="Export OCR model to ONNX")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["crnn", "parseq"],
        help="Model type to export (TrOCR uses HF export, not supported here)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to fine-tuned checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output ONNX path (default: models/ocr_{model}.onnx)",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/ocr_dataset",
        help="Path to dataset for accuracy verification (default: data/ocr_dataset)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip accuracy verification",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.output is None:
        args.output = f"models/ocr_{args.model}.onnx"

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        args.checkpoint
        if Path(args.checkpoint).is_absolute()
        else str(project_root / args.checkpoint)
    )

    print("=" * 70)
    print(f"ONNX Export: {args.model}")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_path}")

    # Export
    print("\n[1/4] Exporting to ONNX...")
    if args.model == "crnn":
        pytorch_model, dummy = export_crnn(checkpoint_path, str(output_path))
    elif args.model == "parseq":
        pytorch_model, dummy = export_parseq(checkpoint_path, str(output_path))

    # Verify ONNX structure
    print("\n[2/4] Verifying ONNX model...")
    verify_onnx(str(output_path))

    # Accuracy verification
    if not args.skip_verify:
        print("\n[3/4] Verifying accuracy parity...")
        test_dir = project_root / args.dataset / "test"
        verify_accuracy(pytorch_model, str(output_path), test_dir, args.model)
    else:
        print("\n[3/4] Skipping accuracy verification")

    # Inference speed
    print("\n[4/4] Measuring ONNX Runtime inference speed...")
    speed = measure_inference_speed(str(output_path), args.model)

    print(f"\nExport complete: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
