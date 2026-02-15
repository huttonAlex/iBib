"""TensorRT-accelerated OCR inference classes for Jetson deployment.

Provides drop-in replacements for the PyTorch OCR models in
``test_video_pipeline.py`` with identical ``predict()`` / ``predict_batch()``
interfaces, but backed by ONNX Runtime + TensorRT Execution Provider.

Classes:
    OnnxTensorRTParseqOCR  — PARSeq via ONNX Runtime with TRT EP + CUDA EP fallback
    TensorRTCrnnOCR        — CRNN via ONNX Runtime with TRT EP (or .engine via trtexec)

Usage:
    from pointcam.inference import OnnxTensorRTParseqOCR, TensorRTCrnnOCR

    ocr = OnnxTensorRTParseqOCR("models/ocr_parseq.onnx")
    text, conf = ocr.predict(crop_bgr)
    results = ocr.predict_batch([crop1, crop2, crop3])
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# PARSeq — ONNX Runtime with TensorRT EP
# ---------------------------------------------------------------------------


class OnnxTensorRTParseqOCR:
    """PARSeq OCR accelerated via ONNX Runtime + TensorRT Execution Provider.

    The TensorRT EP compiles compatible ONNX subgraphs into TensorRT engines
    (cached on disk after first run) and falls back to CUDA EP for ops that
    TRT cannot handle (e.g. autoregressive decoder control flow).

    Tokenizer decoding uses the PARSeq ``torch.hub`` model's tokenizer —
    this is pure Python string manipulation after the initial load and does
    not use GPU.

    Args:
        onnx_path: Path to PARSeq ``.onnx`` model.
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
        trt_cache_dir: Optional[str] = None,
        max_batch_size: int = 16,
    ):
        import onnxruntime as ort
        from pathlib import Path

        onnx_p = Path(onnx_path)
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

        # Load tokenizer from PARSeq torch hub (pure Python decode)
        self._tokenizer = self._load_tokenizer()

    @staticmethod
    def _load_tokenizer():
        """Load PARSeq tokenizer for decoding model output logits.

        This grabs the tokenizer from the torch hub PARSeq model.
        The tokenizer is used only for string decoding (no GPU ops).
        """
        import torch

        model = torch.hub.load(
            "baudm/parseq", "parseq", pretrained=True, trust_repo=True
        )
        return model.tokenizer

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
        import torch

        probs = torch.from_numpy(logits).softmax(-1)
        preds, probs_out = self._tokenizer.decode(probs)

        results = []
        for i in range(len(preds)):
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
