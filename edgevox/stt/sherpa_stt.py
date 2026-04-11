"""Sherpa-ONNX Zipformer STT backend for Vietnamese.

30M params (int8), RTF ~0.01 on CPU — native ONNX, Apache 2.0.
Uses transducer (encoder + decoder + joiner) architecture.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from edgevox.stt import BaseSTT

log = logging.getLogger(__name__)

# Primary: consolidated repo; Fallback: original upstream
_MODELS_REPO = "nrl-ai/edgevox-models"
_MODELS_SUBFOLDER = "stt/sherpa-zipformer-vi-30M-int8"
_FALLBACK_REPO = "csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
_MODEL_FILES = ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]


def _ensure_model() -> Path:
    """Download the model if needed, trying consolidated repo first."""
    # Try consolidated repo
    try:
        first = hf_hub_download(_MODELS_REPO, _MODEL_FILES[0], subfolder=_MODELS_SUBFOLDER)
        model_dir = Path(first).parent
        for f in _MODEL_FILES[1:]:
            hf_hub_download(_MODELS_REPO, f, subfolder=_MODELS_SUBFOLDER)
        return model_dir
    except Exception:
        log.warning(f"Failed to download from {_MODELS_REPO}, trying fallback...")

    model_dir = Path(snapshot_download(_FALLBACK_REPO, allow_patterns=_MODEL_FILES))
    return model_dir


def _pick_provider() -> str:
    from edgevox.core.gpu import has_cuda

    return "cuda" if has_cuda() else "cpu"


class SherpaSTT(BaseSTT):
    """Sherpa-ONNX Zipformer transducer for Vietnamese."""

    def __init__(self, device: str | None = None):
        import sherpa_onnx

        provider = device or _pick_provider()
        log.info(f"Loading Sherpa-ONNX Zipformer-vi-30M (int8) on {provider}...")

        model_dir = _ensure_model()

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )
        self._model_size = "zipformer-vi-30M-int8"
        self._device = provider
        self._warmed_up = False
        log.info("Sherpa-ONNX loaded (30M params, int8).")

    def transcribe(self, audio: np.ndarray, language: str = "vi") -> str:
        t0 = time.perf_counter()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio.astype(np.float32))
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip().capitalize()

        elapsed = time.perf_counter() - t0
        if not self._warmed_up:
            self._warmed_up = True
            log.info(f'STT Sherpa-ONNX (warmup): {elapsed:.2f}s -> "{text}"')
        else:
            log.info(f'STT Sherpa-ONNX: {elapsed:.2f}s -> "{text}"')
        return text
