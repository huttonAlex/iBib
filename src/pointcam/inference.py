"""OCR inference classes for Jetson/edge deployment (ONNX Runtime + TensorRT).

Provides drop-in replacements for the PyTorch OCR models in
``test_video_pipeline.py`` with identical ``predict()`` / ``predict_batch()``
interfaces, but backed by ONNX Runtime + TensorRT Execution Provider.

Classes:
    OnnxParseqOCR          — PARSeq via ONNX Runtime (CUDA/CPU)
    OnnxTensorRTParseqOCR  — PARSeq via ONNX Runtime with TRT EP + CUDA EP fallback
    OnnxCrnnOCR            — CRNN via ONNX Runtime (CUDA/CPU)
    TensorRTCrnnOCR        — CRNN via ONNX Runtime with TRT EP (or .engine via trtexec)

Usage:
    from pointcam.inference import OnnxParseqOCR, OnnxTensorRTParseqOCR, OnnxCrnnOCR

    ocr = OnnxParseqOCR("models/ocr_parseq.onnx")
    text, conf = ocr.predict(crop_bgr)
    results = ocr.predict_batch([crop1, crop2, crop3])
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import json
import cv2
import numpy as np


@dataclass
class DigitDetail:
    """Per-digit OCR confidence detail."""

    digit: str
    prob: float
    runner_up_digit: str
    runner_up_prob: float


@dataclass
class DetailedOCRResult:
    """OCR result with per-digit confidence breakdown."""

    text: str
    confidence: float
    per_digit: List[DigitDetail] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PARSeq Tokenizer (offline JSON)
# ---------------------------------------------------------------------------


class ParseqTokenizer:
    """Lightweight PARSeq tokenizer for offline decoding."""

    def __init__(
        self,
        vocab: List[str],
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
    ) -> None:
        self.vocab = vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    @classmethod
    def from_json(cls, path: str) -> "ParseqTokenizer":
        with open(path, "r") as f:
            data = json.load(f)
        vocab = data.get("vocab")
        if not vocab:
            raise ValueError(f"Tokenizer file missing vocab: {path}")
        return cls(
            vocab=vocab,
            bos_id=data.get("bos_id"),
            eos_id=data.get("eos_id"),
            pad_id=data.get("pad_id"),
        )

    def decode(self, probs: np.ndarray) -> List[Tuple[str, float]]:
        """Decode probability tensor to (digits, confidence) per sample."""
        if probs.ndim != 3:
            raise ValueError(f"Expected (B, T, V) probs, got shape {probs.shape}")
        vocab_size = probs.shape[-1]
        if vocab_size != len(self.vocab):
            raise ValueError(
                f"Tokenizer vocab size {len(self.vocab)} != model vocab {vocab_size}"
            )

        results: List[Tuple[str, float]] = []
        for sample in probs:
            indices = sample.argmax(axis=-1)
            token_probs = sample[np.arange(sample.shape[0]), indices]
            chars: List[str] = []
            confs: List[float] = []
            for idx, prob in zip(indices, token_probs):
                if self.eos_id is not None and idx == self.eos_id:
                    break
                if self.bos_id is not None and idx == self.bos_id:
                    continue
                if self.pad_id is not None and idx == self.pad_id:
                    continue
                if idx < 0 or idx >= len(self.vocab):
                    continue
                ch = self.vocab[idx]
                if ch.isdigit():
                    chars.append(ch)
                    confs.append(float(prob))
            text = "".join(chars)
            confidence = float(np.prod(confs)) if confs else 0.5
            results.append((text, confidence))
        return results

    def decode_detailed(self, probs: np.ndarray) -> List[DetailedOCRResult]:
        """Decode probability tensor to DetailedOCRResult with per-digit info."""
        if probs.ndim != 3:
            raise ValueError(f"Expected (B, T, V) probs, got shape {probs.shape}")
        vocab_size = probs.shape[-1]
        if vocab_size != len(self.vocab):
            raise ValueError(
                f"Tokenizer vocab size {len(self.vocab)} != model vocab {vocab_size}"
            )

        # Build set of vocab indices that are digits
        digit_indices = [i for i, ch in enumerate(self.vocab) if ch.isdigit()]

        results: List[DetailedOCRResult] = []
        for sample in probs:
            indices = sample.argmax(axis=-1)
            token_probs = sample[np.arange(sample.shape[0]), indices]
            chars: List[str] = []
            confs: List[float] = []
            digits: List[DigitDetail] = []
            for t, (idx, prob) in enumerate(zip(indices, token_probs)):
                if self.eos_id is not None and idx == self.eos_id:
                    break
                if self.bos_id is not None and idx == self.bos_id:
                    continue
                if self.pad_id is not None and idx == self.pad_id:
                    continue
                if idx < 0 or idx >= len(self.vocab):
                    continue
                ch = self.vocab[idx]
                if ch.isdigit():
                    chars.append(ch)
                    confs.append(float(prob))
                    # Find runner-up among digit tokens
                    digit_probs = [(di, float(sample[t, di])) for di in digit_indices if di != idx]
                    if digit_probs:
                        ru_idx, ru_prob = max(digit_probs, key=lambda x: x[1])
                        ru_digit = self.vocab[ru_idx]
                    else:
                        ru_digit = ch
                        ru_prob = 0.0
                    digits.append(DigitDetail(ch, float(prob), ru_digit, ru_prob))
            text = "".join(chars)
            confidence = float(np.prod(confs)) if confs else 0.5
            results.append(DetailedOCRResult(text, confidence, digits))
        return results


# ---------------------------------------------------------------------------
# PARSeq — ONNX Runtime (CUDA/CPU)
# ---------------------------------------------------------------------------


class OnnxParseqOCR:
    """PARSeq OCR via ONNX Runtime (CUDA/CPU providers).

    Args:
        onnx_path: Path to PARSeq ``.onnx`` model.
        tokenizer_path: Path to tokenizer JSON. Defaults to ``<onnx>.tokenizer.json``.
    """

    INPUT_W = 128
    INPUT_H = 32

    def __init__(self, onnx_path: str, tokenizer_path: Optional[str] = None):
        import onnxruntime as ort
        from pathlib import Path

        onnx_p = Path(onnx_path)
        if tokenizer_path is None:
            tokenizer_path = str(onnx_p.with_suffix(".tokenizer.json"))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"  WARNING: CUDA EP failed, falling back to CPU for PARSeq")
        self.input_name = self.session.get_inputs()[0].name
        self._tokenizer = ParseqTokenizer.from_json(tokenizer_path)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def _decode_output(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return self._tokenizer.decode(probs)

    def _decode_output_detailed(self, logits: np.ndarray) -> List[DetailedOCRResult]:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return self._tokenizer.decode_detailed(probs)

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        tensor = self._preprocess(crop_bgr)[np.newaxis]
        logits = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode_output(logits)[0]

    def predict_detailed(self, crop_bgr: np.ndarray) -> DetailedOCRResult:
        tensor = self._preprocess(crop_bgr)[np.newaxis]
        logits = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode_output_detailed(logits)[0]

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops_bgr:
            return []
        batch = np.stack([self._preprocess(c) for c in crops_bgr])
        logits = self.session.run(None, {self.input_name: batch})[0]
        return self._decode_output(logits)

    def predict_batch_detailed(self, crops_bgr: List[np.ndarray]) -> List[DetailedOCRResult]:
        if not crops_bgr:
            return []
        batch = np.stack([self._preprocess(c) for c in crops_bgr])
        logits = self.session.run(None, {self.input_name: batch})[0]
        return self._decode_output_detailed(logits)


# ---------------------------------------------------------------------------
# PARSeq — ONNX Runtime with TensorRT EP
# ---------------------------------------------------------------------------


class OnnxTensorRTParseqOCR:
    """PARSeq OCR accelerated via ONNX Runtime + TensorRT Execution Provider.

    The TensorRT EP compiles compatible ONNX subgraphs into TensorRT engines
    (cached on disk after first run) and falls back to CUDA EP for ops that
    TRT cannot handle (e.g. autoregressive decoder control flow).

    Args:
        onnx_path: Path to PARSeq ``.onnx`` model.
        tokenizer_path: Path to PARSeq tokenizer JSON (offline).
            Defaults to ``<onnx>.tokenizer.json``.
        trt_cache_dir: Directory for TRT engine cache (created if needed).
            Defaults to a ``trt_cache`` folder next to the ONNX file.
        max_batch_size: Maximum batch size for TRT optimization.
    """

    # PARSeq input size (width, height)
    INPUT_W = 128
    INPUT_H = 32

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: Optional[str] = None,
        trt_cache_dir: Optional[str] = None,
        max_batch_size: int = 16,
    ):
        import onnxruntime as ort
        from pathlib import Path

        onnx_p = Path(onnx_path)
        if tokenizer_path is None:
            tokenizer_path = str(onnx_p.with_suffix(".tokenizer.json"))
        if trt_cache_dir is None:
            trt_cache_dir = str(onnx_p.parent / "trt_cache")
        Path(trt_cache_dir).mkdir(parents=True, exist_ok=True)

        # TensorRT EP options — cache engines for fast subsequent loads
        trt_options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_dir,
            "trt_max_workspace_size": str(1 << 30),  # 1 GB
        }

        providers = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"  Loading PARSeq ONNX with TRT EP (first run builds cache)...")
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

        # Report which EP is active
        active_providers = self.session.get_providers()
        print(f"  Active providers: {active_providers}")

        self._tokenizer = ParseqTokenizer.from_json(tokenizer_path)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a single BGR crop to model input format.

        Returns (1, 3, 32, 128) float32 array normalized to [-1, 1].
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_CUBIC)
        # HWC -> CHW, float32, normalize to [-1, 1]
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def _decode_output(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        """Decode ONNX output logits using PARSeq tokenizer.

        Args:
            logits: (batch, seq_len, vocab_size) float32 array.

        Returns:
            List of (digits_string, confidence) tuples.
        """
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return self._tokenizer.decode(probs)

    def _decode_output_detailed(self, logits: np.ndarray) -> List[DetailedOCRResult]:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return self._tokenizer.decode_detailed(probs)

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Predict bib number from a single BGR crop.

        Args:
            crop_bgr: BGR image of cropped bib region.

        Returns:
            (digit_string, confidence) tuple.
        """
        tensor = self._preprocess(crop_bgr)[np.newaxis]  # (1, 3, 32, 128)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode_output(logits)[0]

    def predict_detailed(self, crop_bgr: np.ndarray) -> DetailedOCRResult:
        tensor = self._preprocess(crop_bgr)[np.newaxis]
        logits = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode_output_detailed(logits)[0]

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Predict bib numbers from a batch of BGR crops.

        Args:
            crops_bgr: List of BGR crop images.

        Returns:
            List of (digit_string, confidence) tuples.
        """
        if not crops_bgr:
            return []

        batch = np.stack([self._preprocess(c) for c in crops_bgr])  # (N, 3, 32, 128)
        logits = self.session.run(None, {self.input_name: batch})[0]
        return self._decode_output(logits)

    def predict_batch_detailed(self, crops_bgr: List[np.ndarray]) -> List[DetailedOCRResult]:
        if not crops_bgr:
            return []
        batch = np.stack([self._preprocess(c) for c in crops_bgr])
        logits = self.session.run(None, {self.input_name: batch})[0]
        return self._decode_output_detailed(logits)


# ---------------------------------------------------------------------------
# PARSeq — PyTorch (torch.hub)
# ---------------------------------------------------------------------------


class PARSeqOCR:
    """PARSeq OCR from fine-tuned PyTorch checkpoint.

    Uses ``torch.hub`` to load the PARSeq architecture and applies a
    fine-tuned state dict.  Heavier than the ONNX variants but required
    when ONNX Runtime cannot handle certain ops (e.g. on Jetson).

    Args:
        checkpoint_path: Path to ``.pt`` checkpoint (state dict or full).
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch

        self._torch = torch
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

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        import torch

        torch = self._torch
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 32), interpolation=cv2.INTER_CUBIC)
        t = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy((t - 0.5) / 0.5).unsqueeze(0).to(self.device)

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

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
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

    def predict_batch_detailed(self, crops_bgr: List[np.ndarray]) -> List[DetailedOCRResult]:
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

        # Build digit-index mapping from model's tokenizer vocab
        itos = self.model.tokenizer._itos
        digit_indices = [i for i, ch in enumerate(itos) if ch.isdigit()]

        probs_np = probs.cpu().numpy()

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

            digits_text = ""
            digit_details = []
            pos = 1  # skip BOS at position 0
            for ch in text:
                if pos >= probs_np.shape[1]:
                    break
                if ch.isdigit():
                    digits_text += ch
                    top_idx = int(probs_np[i, pos].argmax())
                    top_prob = float(probs_np[i, pos, top_idx])
                    best_ru_idx, best_ru_prob = top_idx, 0.0
                    for di in digit_indices:
                        if di != top_idx and probs_np[i, pos, di] > best_ru_prob:
                            best_ru_idx = di
                            best_ru_prob = float(probs_np[i, pos, di])
                    digit_details.append(
                        DigitDetail(
                            digit=ch,
                            prob=top_prob,
                            runner_up_digit=itos[best_ru_idx] if best_ru_idx != top_idx else ch,
                            runner_up_prob=best_ru_prob,
                        )
                    )
                pos += 1

            results.append(
                DetailedOCRResult(
                    text=digits_text,
                    confidence=confidence,
                    per_digit=digit_details,
                )
            )
        return results


# ---------------------------------------------------------------------------
# CRNN — ONNX Runtime (CUDA/CPU)
# ---------------------------------------------------------------------------


class OnnxCrnnOCR:
    """CRNN OCR via ONNX Runtime (CUDA/CPU providers)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, np.newaxis, :, :]

        output = self.session.run(None, {self.input_name: tensor})[0]
        logits = output[0]

        indices = logits.argmax(axis=-1)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        max_probs = probs.max(axis=-1)

        result = []
        conf_values = []
        prev_idx = -1
        for t, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:
                result.append(str(idx - 1))
                conf_values.append(max_probs[t])
            prev_idx = idx

        text = "".join(result)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return text, confidence

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops_bgr:
            return []

        batch = []
        for crop in crops_bgr:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_LINEAR)
            batch.append(resized.astype(np.float32) / 255.0)
        batch = np.stack(batch)[:, np.newaxis, :, :]

        output = self.session.run(None, {self.input_name: batch})[0]

        results = []
        for i in range(len(crops_bgr)):
            results.append(self._ctc_decode(output[i]))
        return results

    @staticmethod
    def _ctc_decode(logits: np.ndarray) -> Tuple[str, float]:
        indices = logits.argmax(axis=-1)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        max_probs = probs.max(axis=-1)

        result = []
        conf_values = []
        prev_idx = -1
        for t, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:
                result.append(str(idx - 1))
                conf_values.append(max_probs[t])
            prev_idx = idx

        text = "".join(result)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return text, confidence


# ---------------------------------------------------------------------------
# CRNN — ONNX Runtime with TensorRT EP
# ---------------------------------------------------------------------------


class TensorRTCrnnOCR:
    """CRNN OCR accelerated via ONNX Runtime + TensorRT Execution Provider.

    The CRNN architecture is a simple CNN+RNN+CTC pipeline with no dynamic
    control flow, so TRT EP can compile the entire graph efficiently.

    Supports both ``.onnx`` (via TRT EP) and pre-built ``.engine`` files
    (via trtexec, loaded through ORT TRT EP).

    Args:
        model_path: Path to CRNN ``.onnx`` or ``.engine`` file.
        trt_cache_dir: Directory for TRT engine cache.
    """

    def __init__(
        self,
        model_path: str,
        trt_cache_dir: Optional[str] = None,
    ):
        import onnxruntime as ort
        from pathlib import Path

        model_p = Path(model_path)

        if model_p.suffix == ".engine":
            # For pre-built .engine files, use the original ONNX via TRT EP
            # The .engine is built by trtexec; at runtime we still load .onnx
            # and let TRT EP use its cache (which matches the trtexec output).
            # If you have the .onnx, prefer passing that.
            onnx_path = str(model_p.with_suffix(".onnx"))
            print(f"  Note: .engine passed; loading {onnx_path} with TRT EP instead")
            model_path = onnx_path

        if trt_cache_dir is None:
            trt_cache_dir = str(model_p.parent / "trt_cache")
        Path(trt_cache_dir).mkdir(parents=True, exist_ok=True)

        trt_options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_dir,
            "trt_max_workspace_size": str(1 << 30),
        }

        providers = [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"  Loading CRNN with TRT EP...")
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

        active_providers = self.session.get_providers()
        print(f"  Active providers: {active_providers}")

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a single BGR crop to CRNN input format.

        Returns (1, 1, 32, 128) float32 array normalized to [0, 1].
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        return tensor[np.newaxis, np.newaxis, :, :]  # (1, 1, 32, 128)

    @staticmethod
    def _ctc_decode(logits: np.ndarray) -> Tuple[str, float]:
        """Greedy CTC decode on a single sequence of logits.

        Args:
            logits: (T, num_classes) float32 array.

        Returns:
            (text, confidence) tuple.
        """
        indices = logits.argmax(axis=-1)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        max_probs = probs.max(axis=-1)

        result = []
        conf_values = []
        prev_idx = -1
        for t, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:  # 0 = CTC blank
                result.append(str(idx - 1))  # 1-10 -> '0'-'9'
                conf_values.append(max_probs[t])
            prev_idx = idx

        text = "".join(result)
        confidence = float(np.mean(conf_values)) if conf_values else 0.0
        return text, confidence

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Predict bib number from a single BGR crop.

        Args:
            crop_bgr: BGR image of cropped bib region.

        Returns:
            (digit_string, confidence) tuple.
        """
        tensor = self._preprocess(crop_bgr)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._ctc_decode(output[0])

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Predict bib numbers from a batch of BGR crops.

        Batches all crops into a single ONNX Runtime call.

        Args:
            crops_bgr: List of BGR crop images.

        Returns:
            List of (digit_string, confidence) tuples.
        """
        if not crops_bgr:
            return []

        # Stack into (N, 1, 32, 128)
        batch = np.concatenate(
            [self._preprocess(c) for c in crops_bgr], axis=0
        )
        output = self.session.run(None, {self.input_name: batch})[0]

        results = []
        for i in range(len(crops_bgr)):
            results.append(self._ctc_decode(output[i]))
        return results
