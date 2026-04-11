"""Faster-whisper STT backend. Works for all languages."""

from __future__ import annotations

import logging
import time

import numpy as np

from edgevox.stt import BaseSTT

log = logging.getLogger(__name__)


def _pick_whisper_model_and_device() -> tuple[str, str, str]:
    """Return (model_size, device, compute_type) based on hardware."""
    from edgevox.core.gpu import get_nvidia_vram_gb, get_ram_gb

    vram_gb = get_nvidia_vram_gb()
    if vram_gb is not None:
        if vram_gb >= 8:
            return "large-v3-turbo", "cuda", "float16"
        return "small", "cuda", "float16"

    ram_gb = get_ram_gb()
    if ram_gb >= 32:
        return "large-v3-turbo", "cpu", "int8"
    if ram_gb >= 16:
        return "medium", "cpu", "int8"
    return "small", "cpu", "int8"


class WhisperSTT(BaseSTT):
    """faster-whisper based speech-to-text."""

    def __init__(self, model_size: str | None = None, device: str | None = None):
        from faster_whisper import WhisperModel

        self._backend_name = "Whisper"
        if model_size and device:
            self._model_size = model_size
            self._device = device
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            self._model_size, self._device, compute_type = _pick_whisper_model_and_device()

        log.info(f"Loading Whisper: {self._model_size} on {self._device} ({compute_type})")
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=compute_type,
        )
        log.info("Whisper loaded.")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.perf_counter() - t0
        log.info(f'STT Whisper ({self._model_size}): {elapsed:.2f}s -> "{text}"')
        return text
