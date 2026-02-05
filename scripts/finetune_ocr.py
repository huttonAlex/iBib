#!/usr/bin/env python3
"""Fine-tune OCR models on the bib crop dataset.

Supports CRNN, PARSeq, and TrOCR-small. Auto-detects GPU if available.

Usage:
    python scripts/finetune_ocr.py --model crnn
    python scripts/finetune_ocr.py --model parseq --batch-size 128
    python scripts/finetune_ocr.py --model trocr --batch-size 32
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Augmentation pipeline (matches prepare_ocr_dataset.py)
# ---------------------------------------------------------------------------

def get_augmentation_pipeline():
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class BibCropDataset(Dataset):
    """Dataset for bib crop images with labels."""

    def __init__(
        self,
        tsv_path: str,
        images_dir: str,
        transform=None,
        augment=None,
        img_height: int = 32,
        img_width: int = 128,
        grayscale: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.augment = augment
        self.img_height = img_height
        self.img_width = img_width
        self.grayscale = grayscale
        self.samples = []

        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "label": row["label"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.images_dir / sample["filename"]

        img = cv2.imread(str(img_path))
        if img is None:
            # Return black image as fallback
            if self.grayscale:
                img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            else:
                img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            # Online augmentation
            if self.augment is not None:
                img = self.augment(image=img)["image"]

            if self.grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, (self.img_width, self.img_height))

        # Normalize to [0, 1]
        tensor = torch.from_numpy(img).float() / 255.0

        if self.grayscale:
            tensor = tensor.unsqueeze(0)  # (1, H, W)
        else:
            tensor = tensor.permute(2, 0, 1)  # (3, H, W)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, sample["label"]


class TrOCRDataset(Dataset):
    """Dataset for TrOCR (uses HuggingFace processor)."""

    def __init__(self, tsv_path: str, images_dir: str, processor, augment=None):
        from PIL import Image

        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment
        self.samples = []

        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "label": row["label"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        sample = self.samples[idx]
        img_path = self.images_dir / sample["filename"]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((64, 128, 3), dtype=np.uint8)

        if self.augment is not None:
            img = self.augment(image=img)["image"]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize label
        labels = self.processor.tokenizer(
            sample["label"],
            padding="max_length",
            max_length=10,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return pixel_values, labels


# ---------------------------------------------------------------------------
# CRNN Model
# ---------------------------------------------------------------------------

class BibNumberCRNN(nn.Module):
    """CRNN for bib number recognition with CTC loss."""

    def __init__(self, img_height=32, num_classes=11):
        """num_classes=11: 0=blank, 1-10=digits '0'-'9'"""
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


def encode_label_ctc(label: str) -> torch.Tensor:
    """Encode digit string to CTC target. '0'->'1', ..., '9'->'10'."""
    return torch.tensor([int(c) + 1 for c in label], dtype=torch.long)


def decode_ctc(output: torch.Tensor) -> str:
    """Greedy CTC decode: output shape (T, num_classes)."""
    indices = output.argmax(dim=-1)
    result = []
    prev = -1
    for idx in indices.tolist():
        if idx != 0 and idx != prev:
            result.append(str(idx - 1))
        prev = idx
    return "".join(result)


def crnn_collate_fn(batch):
    """Collate for CTC training: variable length labels."""
    images = torch.stack([b[0] for b in batch])
    labels_raw = [b[1] for b in batch]

    encoded = [encode_label_ctc(l) for l in labels_raw]
    label_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    labels = torch.cat(encoded)

    return images, labels, label_lengths


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_crnn(args, train_loader, val_loader, output_dir: Path, device=None):
    """Train CRNN with CTC loss."""
    if device is None:
        device = torch.device("cpu")
    model = BibNumberCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    best_acc = 0.0
    patience_counter = 0
    history = []

    print(f"\n  Training CRNN for up to {args.epochs} epochs...")
    print(f"  LR={args.lr}, batch={args.batch_size}, patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, labels, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            optimizer.zero_grad()
            output = model(images)  # (B, T, C)
            log_probs = output.permute(1, 0, 2).log_softmax(2)  # (T, B, C)
            input_lengths = torch.full(
                (output.size(0),), output.size(1), dtype=torch.long, device=device
            )

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_acc = evaluate_crnn(model, val_loader, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, "
            f"val_acc={val_acc:.1%}, lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "lr": current_lr,
        })

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "device": str(device),
            }, output_dir / "best.pt")
            print(f"    -> New best: {val_acc:.1%}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best validation accuracy: {best_acc:.1%}")
    return best_acc


def evaluate_crnn(model, loader, device=None):
    """Evaluate CRNN exact match accuracy."""
    return _evaluate_model_on_loader(model, loader, "crnn", device)


def _evaluate_model_on_loader(model, loader, model_type, device=None):
    """Evaluate any model on a DataLoader, returning exact match accuracy."""
    if device is None:
        device = torch.device("cpu")
    model.eval()
    correct = 0
    total = 0

    # We need raw labels, so iterate dataset directly
    dataset = loader.dataset
    batch_size = loader.batch_size or 1

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch_items = [dataset[j] for j in batch_indices]

            if model_type == "crnn":
                images = torch.stack([item[0] for item in batch_items]).to(device)
                labels = [item[1] for item in batch_items]

                output = model(images)
                for j in range(len(batch_items)):
                    pred = decode_ctc(output[j])
                    if pred == labels[j]:
                        correct += 1
                    total += 1

            elif model_type == "parseq":
                images = torch.stack([item[0] for item in batch_items]).to(device)
                labels = [item[1] for item in batch_items]

                logits = model(images)
                probs = logits.softmax(-1)
                preds, _ = model.tokenizer.decode(probs)

                for j in range(len(batch_items)):
                    pred_digits = "".join(c for c in preds[j] if c.isdigit())
                    if pred_digits == labels[j]:
                        correct += 1
                    total += 1

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# PARSeq fine-tuning
# ---------------------------------------------------------------------------

def train_parseq(args, train_loader, val_loader, output_dir: Path, device=None):
    """Fine-tune PARSeq."""
    if device is None:
        device = torch.device("cpu")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    best_acc = 0.0
    patience_counter = 0
    history = []

    print(f"\n  Training PARSeq for up to {args.epochs} epochs...")
    print(f"  LR={args.lr}, batch={args.batch_size}, patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, labels_raw in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            # Use PARSeq's built-in loss computation
            logits, loss, _ = model.forward_logits_loss(images, labels_raw)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_acc = _evaluate_model_on_loader(model, val_loader, "parseq", device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, "
            f"val_acc={val_acc:.1%}, lr={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "lr": current_lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "device": str(device),
            }, output_dir / "best.pt")
            print(f"    -> New best: {val_acc:.1%}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best validation accuracy: {best_acc:.1%}")
    return best_acc


def parseq_collate_fn(batch):
    """Collate for PARSeq: images + raw string labels."""
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


# ---------------------------------------------------------------------------
# TrOCR fine-tuning
# ---------------------------------------------------------------------------

def train_trocr(args, train_dataset, val_dataset, output_dir: Path, device=None):
    """Fine-tune TrOCR-small with encoder freezing strategy."""
    if device is None:
        device = torch.device("cpu")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = "microsoft/trocr-small-printed"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Configure for digit-only generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 10

    model = model.to(device)

    # Phase 1: Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    use_gpu = device.type == "cuda"
    num_workers = 4 if use_gpu else 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu,
    )

    best_acc = 0.0
    patience_counter = 0
    history = []
    unfreeze_epoch = 5
    unfrozen = False

    print(f"\n  Training TrOCR for up to {args.epochs} epochs...")
    print(f"  LR={args.lr}, batch={args.batch_size}, patience={args.patience}")
    print(f"  Encoder frozen for first {unfreeze_epoch} epochs")

    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder after specified epochs
        if epoch == unfreeze_epoch + 1 and not unfrozen:
            print("  Unfreezing encoder with 10x lower LR...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": args.lr / 10},
                {"params": model.decoder.parameters(), "lr": args.lr},
            ])
            unfrozen = True

        model.train()
        train_loss = 0.0
        n_batches = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_acc = _evaluate_trocr(model, processor, val_loader, device)

        print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, val_acc={val_acc:.1%}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(str(output_dir / "best"))
            processor.save_pretrained(str(output_dir / "best"))
            print(f"    -> New best: {val_acc:.1%}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best validation accuracy: {best_acc:.1%}")
    return best_acc


def _evaluate_trocr(model, processor, loader, device=None):
    """Evaluate TrOCR exact match accuracy."""
    if device is None:
        device = torch.device("cpu")
    model.eval()
    correct = 0
    total = 0

    dataset = loader.dataset

    with torch.no_grad():
        for i in range(len(dataset)):
            pixel_values, labels_tensor = dataset[i]
            pixel_values = pixel_values.unsqueeze(0).to(device)

            outputs = model.generate(pixel_values, max_length=10)
            pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            pred_digits = "".join(c for c in pred if c.isdigit())

            # Decode ground truth
            gt = processor.tokenizer.decode(labels_tensor, skip_special_tokens=True)
            gt_digits = "".join(c for c in gt if c.isdigit())

            if pred_digits == gt_digits:
                correct += 1
            total += 1

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OCR model on bib crops")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["crnn", "parseq", "trocr"],
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/ocr_dataset",
        help="Path to prepared dataset (default: data/ocr_dataset)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: runs/ocr_finetune/{model}_v1)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--no-augment", action="store_true", help="Disable online augmentation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / args.dataset

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("  No GPU detected, using CPU")

    # Set model-specific defaults (GPU gets larger batch sizes)
    use_gpu = device.type == "cuda"
    defaults = {
        "crnn": {"batch_size": 512 if use_gpu else 128, "lr": 1e-3, "patience": 15},
        "parseq": {"batch_size": 128 if use_gpu else 32, "lr": 1e-4, "patience": 10},
        "trocr": {"batch_size": 32 if use_gpu else 8, "lr": 5e-5, "patience": 10},
    }
    d = defaults[args.model]
    if args.batch_size is None:
        args.batch_size = d["batch_size"]
    if args.lr is None:
        args.lr = d["lr"]
    if args.patience is None:
        args.patience = d["patience"]

    if args.output is None:
        args.output = f"runs/ocr_finetune/{args.model}_v1"

    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    augment = get_augmentation_pipeline() if not args.no_augment else None

    print("=" * 70)
    print(f"Fine-tuning: {args.model}")
    print("=" * 70)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  LR:      {args.lr}")
    print(f"  Patience: {args.patience}")
    print(f"  Augment: {not args.no_augment}")
    print(f"  Device:  {device}")

    train_tsv = dataset_dir / "train" / "labels.tsv"
    val_tsv = dataset_dir / "val" / "labels.tsv"
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "val" / "images"

    for p in [train_tsv, val_tsv, train_images, val_images]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run prepare_ocr_dataset.py first.")
            sys.exit(1)

    num_workers = 4 if use_gpu else 0

    if args.model == "crnn":
        train_ds = BibCropDataset(
            str(train_tsv), str(train_images),
            augment=augment, img_height=32, img_width=128, grayscale=True,
        )
        val_ds = BibCropDataset(
            str(val_tsv), str(val_images),
            augment=None, img_height=32, img_width=128, grayscale=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=crnn_collate_fn, pin_memory=use_gpu,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=crnn_collate_fn, pin_memory=use_gpu,
        )

        train_crnn(args, train_loader, val_loader, output_dir, device)

    elif args.model == "parseq":
        from torchvision import transforms

        parseq_transform = transforms.Compose([
            transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(0.5, 0.5),
        ])

        train_ds = BibCropDataset(
            str(train_tsv), str(train_images),
            transform=parseq_transform, augment=augment,
            img_height=32, img_width=128, grayscale=False,
        )
        val_ds = BibCropDataset(
            str(val_tsv), str(val_images),
            transform=parseq_transform, augment=None,
            img_height=32, img_width=128, grayscale=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=parseq_collate_fn, pin_memory=use_gpu,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=parseq_collate_fn, pin_memory=use_gpu,
        )

        train_parseq(args, train_loader, val_loader, output_dir, device)

    elif args.model == "trocr":
        from transformers import TrOCRProcessor

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

        train_ds = TrOCRDataset(
            str(train_tsv), str(train_images),
            processor=processor, augment=augment,
        )
        val_ds = TrOCRDataset(
            str(val_tsv), str(val_images),
            processor=processor, augment=None,
        )

        train_trocr(args, train_ds, val_ds, output_dir, device)

    # Save args for reproducibility
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
