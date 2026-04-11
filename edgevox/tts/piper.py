"""Piper TTS backend for Vietnamese."""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)


class PiperTTS(BaseTTS):
    """Piper ONNX TTS for Vietnamese."""

    sample_rate = 22_050

    _HF_VOICES: ClassVar[dict[str, str]] = {
        "vi-female": "speaches-ai/piper-vi_VN-vais1000-medium",
        "vi-male": "speaches-ai/piper-vi_VN-25hours_single-low",
    }

    def __init__(self, voice: str = "vi-female"):
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        repo_id = self._HF_VOICES.get(voice)
        if not repo_id:
            raise ValueError(f"Unknown Piper voice: {voice}. Available: {list(self._HF_VOICES)}")

        log.info(f"Loading Piper TTS ({voice}: {repo_id})...")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self.sample_rate = self._voice.config.sample_rate
        log.info(f"Piper TTS loaded (sample_rate={self.sample_rate}).")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        audio_chunks = []
        for chunk in self._voice.synthesize(text):
            audio_chunks.append(chunk.audio_float_array)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_chunks).astype(np.float32)
        pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        audio = np.concatenate([audio, pad])

        elapsed = time.perf_counter() - t0
        duration = len(audio) / self.sample_rate
        log.info(f"Piper TTS: {elapsed:.2f}s -> {duration:.1f}s audio")
        return audio
