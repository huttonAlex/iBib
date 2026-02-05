#!/usr/bin/env python3
"""Evaluate pretrained and fine-tuned OCR models on the bib crop test set.

Evaluates TrOCR-small, PARSeq, and CRNN (VGG-BiLSTM-CTC) with metrics:
- Exact match accuracy (whole bib number correct)
- Character-level accuracy (Levenshtein-based)
- Accuracy by digit count (1/2/3/4-digit breakdown)
- Accuracy by source event (generalization test)
- Inference speed (ms/image on CPU)

Usage:
    python scripts/evaluate_ocr_models.py [--dataset data/ocr_dataset] [--models all]
    python scripts/evaluate_ocr_models.py --models parseq --checkpoint runs/ocr_finetune/parseq_v1/best.pt
"""

import argparse
import csv
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Common model interface
# ---------------------------------------------------------------------------

class OCRModel(ABC):
    """Common interface for all OCR models."""

    name: str = "base"

    @abstractmethod
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Predict bib number from a crop image.

        Args:
            image: BGR numpy array (H, W, 3)

        Returns:
            (predicted_text, confidence) where text is digits-only
        """
        ...

    def warmup(self, n: int = 5):
        """Run dummy predictions to warm up the model."""
        dummy = np.zeros((64, 128, 3), dtype=np.uint8)
        for _ in range(n):
            self.predict(dummy)


# ---------------------------------------------------------------------------
# TrOCR-small wrapper
# ---------------------------------------------------------------------------

class TrOCRModel(OCRModel):
    name = "trocr-small"

    def __init__(self, checkpoint: str | None = None):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        model_id = checkpoint or "microsoft/trocr-small-printed"
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.eval()

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        # Convert BGR to RGB PIL
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=10,
                num_beams=3,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Extract confidence from sequence scores
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            confidence = torch.exp(outputs.sequences_scores).item()
        else:
            confidence = 0.5

        # Filter to digits only
        digits = "".join(c for c in text if c.isdigit())
        return digits, confidence


# ---------------------------------------------------------------------------
# PARSeq wrapper
# ---------------------------------------------------------------------------

class PARSeqModel(OCRModel):
    name = "parseq"

    def __init__(self, checkpoint: str | None = None):
        if checkpoint and Path(checkpoint).suffix == ".pt":
            # Load fine-tuned checkpoint
            self._load_finetuned(checkpoint)
        else:
            self._load_pretrained()

    def _load_pretrained(self):
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        self.model.eval()
        self.transform = self._get_transform()

    def _load_finetuned(self, checkpoint_path: str):
        # Load base model first, then override weights
        self.model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.transform = self._get_transform()

    def _get_transform(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        tensor = self.transform(pil_img).unsqueeze(0)

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


# ---------------------------------------------------------------------------
# CRNN wrapper
# ---------------------------------------------------------------------------

class CRNNModel(OCRModel):
    name = "crnn"

    def __init__(self, checkpoint: str | None = None):
        if checkpoint is None:
            print("  CRNN: No pretrained model available, requires fine-tuned checkpoint")
            self.model = None
            return

        self.model = self._build_model()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def _build_model(self):
        """Build CRNN architecture matching the fine-tuning script."""
        import torch.nn as nn

        class BibNumberCRNN(nn.Module):
            def __init__(self, img_height=32, num_classes=11):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(True), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                    nn.ReLU(True), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256),
                    nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
                    nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, (2, 1)), nn.BatchNorm2d(512),
                    nn.ReLU(True),
                )
                self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
                self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(512, num_classes)

            def forward(self, x):
                conv = self.cnn(x)
                b, c, h, w = conv.size()
                conv = conv.squeeze(2).permute(0, 2, 1)
                rnn_out, _ = self.rnn1(conv)
                rnn_out, _ = self.rnn2(rnn_out)
                return self.fc(rnn_out)

        return BibNumberCRNN()

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            return "", 0.0

        # Preprocess: grayscale, resize to 32x128
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            log_probs = output.log_softmax(2)
            probs = output.softmax(2)

        # Greedy CTC decode
        indices = log_probs[0].argmax(dim=-1)
        max_probs = probs[0].max(dim=-1).values

        result = []
        conf_values = []
        prev_idx = -1
        for t, idx in enumerate(indices.tolist()):
            if idx != 0 and idx != prev_idx:  # 0 = CTC blank
                result.append(str(idx - 1))  # 1-10 -> '0'-'9'
                conf_values.append(max_probs[t].item())
            prev_idx = idx

        text = "".join(result)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return text, confidence


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[tuple[str, float]],
    ground_truths: list[str],
    events: list[str],
    digit_counts: list[int],
) -> dict:
    """Compute all evaluation metrics."""
    n = len(predictions)
    assert n == len(ground_truths) == len(events) == len(digit_counts)

    # Exact match
    exact_matches = sum(1 for (p, _), gt in zip(predictions, ground_truths) if p == gt)
    exact_acc = exact_matches / n if n > 0 else 0.0

    # Character-level accuracy (1 - normalized edit distance)
    char_accs = []
    for (pred, _), gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            char_accs.append(1.0 if len(pred) == 0 else 0.0)
        else:
            ed = editdistance.eval(pred, gt)
            char_accs.append(1.0 - ed / max(len(pred), len(gt)))
    char_acc = float(np.mean(char_accs)) if char_accs else 0.0

    # Per digit-count accuracy
    dc_metrics = {}
    for dc in sorted(set(digit_counts)):
        dc_mask = [i for i, d in enumerate(digit_counts) if d == dc]
        dc_correct = sum(1 for i in dc_mask if predictions[i][0] == ground_truths[i])
        dc_metrics[dc] = {
            "count": len(dc_mask),
            "correct": dc_correct,
            "accuracy": dc_correct / len(dc_mask) if dc_mask else 0.0,
        }

    # Per event accuracy
    event_metrics = {}
    for event in sorted(set(events)):
        ev_mask = [i for i, e in enumerate(events) if e == event]
        ev_correct = sum(1 for i in ev_mask if predictions[i][0] == ground_truths[i])
        event_metrics[event] = {
            "count": len(ev_mask),
            "correct": ev_correct,
            "accuracy": ev_correct / len(ev_mask) if ev_mask else 0.0,
        }

    # Confidence stats
    confidences = [c for _, c in predictions]
    correct_conf = [c for (p, c), gt in zip(predictions, ground_truths) if p == gt]
    wrong_conf = [c for (p, c), gt in zip(predictions, ground_truths) if p != gt]

    return {
        "total": n,
        "exact_match_accuracy": exact_acc,
        "exact_matches": exact_matches,
        "char_level_accuracy": char_acc,
        "digit_count_metrics": dc_metrics,
        "event_metrics": event_metrics,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_correct_confidence": float(np.mean(correct_conf)) if correct_conf else 0.0,
        "avg_wrong_confidence": float(np.mean(wrong_conf)) if wrong_conf else 0.0,
    }


def load_test_set(dataset_dir: Path) -> list[dict]:
    """Load test set from Common format."""
    test_dir = dataset_dir / "test"
    tsv_path = test_dir / "labels.tsv"

    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run prepare_ocr_dataset.py first.")
        sys.exit(1)

    samples = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            img_path = test_dir / "images" / row["filename"]
            samples.append({
                "img_path": str(img_path),
                "label": row["label"],
                "event": row["event"],
                "digit_count": int(row["digit_count"]),
            })

    return samples


def evaluate_model(
    model: OCRModel,
    samples: list[dict],
) -> tuple[dict, list[dict]]:
    """Run evaluation on all samples and return metrics + error details."""
    print(f"\n  Warming up {model.name}...")
    model.warmup()

    predictions = []
    ground_truths = []
    events = []
    digit_counts = []
    errors = []
    times = []

    print(f"  Evaluating {len(samples)} samples...")
    for i, sample in enumerate(samples):
        img = cv2.imread(sample["img_path"])
        if img is None:
            predictions.append(("", 0.0))
            ground_truths.append(sample["label"])
            events.append(sample["event"])
            digit_counts.append(sample["digit_count"])
            continue

        t0 = time.perf_counter()
        pred, conf = model.predict(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

        predictions.append((pred, conf))
        ground_truths.append(sample["label"])
        events.append(sample["event"])
        digit_counts.append(sample["digit_count"])

        if pred != sample["label"]:
            errors.append({
                "img_path": sample["img_path"],
                "ground_truth": sample["label"],
                "predicted": pred,
                "confidence": conf,
                "event": sample["event"],
                "digit_count": sample["digit_count"],
            })

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(samples)}...")

    metrics = compute_metrics(predictions, ground_truths, events, digit_counts)
    metrics["avg_inference_ms"] = float(np.mean(times)) if times else 0.0
    metrics["median_inference_ms"] = float(np.median(times)) if times else 0.0
    metrics["p95_inference_ms"] = float(np.percentile(times, 95)) if times else 0.0

    return metrics, errors


def format_report(model_name: str, metrics: dict) -> str:
    """Format metrics as a markdown report section."""
    lines = [f"## {model_name}\n"]
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Exact Match Accuracy | {metrics['exact_match_accuracy']:.1%} ({metrics['exact_matches']}/{metrics['total']}) |")
    lines.append(f"| Character-Level Accuracy | {metrics['char_level_accuracy']:.1%} |")
    lines.append(f"| Avg Confidence (correct) | {metrics['avg_correct_confidence']:.3f} |")
    lines.append(f"| Avg Confidence (wrong) | {metrics['avg_wrong_confidence']:.3f} |")
    lines.append(f"| Avg Inference (ms) | {metrics['avg_inference_ms']:.1f} |")
    lines.append(f"| Median Inference (ms) | {metrics['median_inference_ms']:.1f} |")
    lines.append(f"| P95 Inference (ms) | {metrics['p95_inference_ms']:.1f} |")

    lines.append(f"\n### Accuracy by Digit Count\n")
    lines.append(f"| Digits | Count | Correct | Accuracy |")
    lines.append(f"|--------|-------|---------|----------|")
    for dc, m in sorted(metrics["digit_count_metrics"].items()):
        lines.append(f"| {dc} | {m['count']} | {m['correct']} | {m['accuracy']:.1%} |")

    lines.append(f"\n### Accuracy by Event\n")
    lines.append(f"| Event | Count | Correct | Accuracy |")
    lines.append(f"|-------|-------|---------|----------|")
    for event, m in sorted(metrics["event_metrics"].items()):
        lines.append(f"| {event} | {m['count']} | {m['correct']} | {m['accuracy']:.1%} |")

    return "\n".join(lines)


def write_error_analysis(errors: list[dict], output_path: Path, model_name: str):
    """Write per-model error analysis CSV."""
    if not errors:
        return

    csv_path = output_path / f"{model_name}_errors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "img_path", "ground_truth", "predicted", "confidence", "event", "digit_count",
        ])
        writer.writeheader()
        writer.writerows(errors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR models on bib test set")
    parser.add_argument(
        "--dataset", type=str, default="data/ocr_dataset",
        help="Path to prepared dataset (default: data/ocr_dataset)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["trocr", "parseq", "crnn"],
        choices=["trocr", "parseq", "crnn", "all"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to fine-tuned checkpoint (applies to all selected models)",
    )
    parser.add_argument(
        "--output", type=str, default="runs/ocr_eval",
        help="Output directory for results (default: runs/ocr_eval)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of test samples (0=all, default: 0)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / args.dataset
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in args.models:
        args.models = ["trocr", "parseq", "crnn"]

    print("=" * 70)
    print("OCR Model Evaluation")
    print("=" * 70)

    # Load test set
    print(f"\nLoading test set from {dataset_dir}...")
    samples = load_test_set(dataset_dir)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"  {len(samples)} test samples")

    # Evaluate each model
    all_reports = ["# OCR Model Comparison\n"]
    all_metrics = {}

    model_classes = {
        "trocr": TrOCRModel,
        "parseq": PARSeqModel,
        "crnn": CRNNModel,
    }

    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        model_cls = model_classes[model_name]

        try:
            model = model_cls(checkpoint=args.checkpoint)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            all_reports.append(f"\n## {model_name}\n\nFailed to load: {e}\n")
            continue

        if hasattr(model, "model") and model.model is None:
            print(f"  Skipping {model_name} (no model loaded)")
            all_reports.append(f"\n## {model_name}\n\nSkipped: requires fine-tuned checkpoint\n")
            continue

        metrics, errors = evaluate_model(model, samples)
        all_metrics[model_name] = metrics

        # Print summary
        print(f"\n  Results for {model_name}:")
        print(f"    Exact match: {metrics['exact_match_accuracy']:.1%}")
        print(f"    Char-level:  {metrics['char_level_accuracy']:.1%}")
        print(f"    Avg speed:   {metrics['avg_inference_ms']:.1f} ms/image")

        # Generate report
        report = format_report(model_name, metrics)
        all_reports.append(f"\n{report}\n")

        # Write error analysis
        write_error_analysis(errors, output_dir, model_name)
        if errors:
            print(f"    Errors written: {len(errors)} -> {output_dir}/{model_name}_errors.csv")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary comparison table
    if len(all_metrics) > 1:
        all_reports.append("\n## Summary Comparison\n")
        all_reports.append("| Model | Exact Match | Char Acc | Avg ms/img |")
        all_reports.append("|-------|-------------|----------|------------|")
        for name, m in all_metrics.items():
            all_reports.append(
                f"| {name} | {m['exact_match_accuracy']:.1%} "
                f"| {m['char_level_accuracy']:.1%} "
                f"| {m['avg_inference_ms']:.1f} |"
            )

    # Write comparison report
    report_path = output_dir / "comparison.md"
    with open(report_path, "w") as f:
        f.write("\n".join(all_reports))
    print(f"\nComparison report: {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
