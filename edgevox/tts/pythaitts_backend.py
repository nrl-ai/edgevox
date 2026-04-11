"""PyThaiTTS backend — Apache-2.0 licensed Thai TTS using Lunarlist ONNX."""

from __future__ import annotations

import logging
import time

import numpy as np
import soundfile as sf

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)


class PyThaiTTSBackend(BaseTTS):
    """Thai TTS via PyThaiTTS lunarlist_onnx engine (Tacotron2 ONNX, Apache 2.0)."""

    sample_rate = 22_050

    def __init__(self, voice: str = "th-default"):
        from pythaitts import TTS

        log.info("Loading PyThaiTTS (lunarlist_onnx)...")
        self._tts = TTS(pretrained="lunarlist_onnx")
        # Warm up to cache model load
        self._tts.tts("ทดสอบ")
        log.info("PyThaiTTS loaded (sample_rate=22050).")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        wav_path = self._tts.tts(text)
        audio, sr = sf.read(wav_path)
        audio = audio.astype(np.float32)
        self.sample_rate = sr

        pad = np.zeros(int(sr * 0.2), dtype=np.float32)
        audio = np.concatenate([audio, pad])

        elapsed = time.perf_counter() - t0
        duration = len(audio) / sr
        log.info(f"PyThaiTTS: {elapsed:.2f}s -> {duration:.1f}s audio (RTF={elapsed / duration:.3f})")
        return audio
