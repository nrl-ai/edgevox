"""Supertonic TTS backend — lightweight ONNX model for Korean and other languages.

Models served from nrl-ai/edgevox-models with fallback to Supertone/supertonic-2.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)

_MODELS_REPO = "nrl-ai/edgevox-models"
_SUPERTONIC_SUBFOLDER = "supertonic"
_FALLBACK_REPO = "Supertone/supertonic-2"

_SUPERTONIC_FILES = [
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/duration_predictor.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
    "config.json",
]


def _ensure_supertonic_model() -> Path:
    """Download supertonic model, trying consolidated repo first."""

    cache_dir = Path(os.path.expanduser("~")) / ".cache" / "edgevox" / "supertonic2"

    # Try consolidated repo
    try:
        first = hf_hub_download(_MODELS_REPO, _SUPERTONIC_FILES[0], subfolder=_SUPERTONIC_SUBFOLDER)
        src_dir = Path(first).parent.parent  # go up from onnx/ subfolder
        for f in _SUPERTONIC_FILES[1:]:
            hf_hub_download(_MODELS_REPO, f, subfolder=_SUPERTONIC_SUBFOLDER)

        # Copy to cache dir for supertonic TTS() to find
        if not (cache_dir / "onnx" / "tts.json").exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            for f in _SUPERTONIC_FILES:
                dest = cache_dir / f
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_dir / f, dest)
        return cache_dir
    except Exception:
        log.warning(f"Failed to download Supertonic from {_MODELS_REPO}, using auto_download fallback...")
        return cache_dir  # TTS(auto_download=True) will handle it


# voice_name -> (display description)
SUPERTONIC_VOICES: dict[str, str] = {
    "ko-F1": "Korean Female (calm, steady)",
    "ko-F2": "Korean Female (bright, cheerful)",
    "ko-F3": "Korean Female (clear, professional)",
    "ko-F4": "Korean Female (crisp, confident)",
    "ko-F5": "Korean Female (kind, gentle)",
    "ko-M1": "Korean Male (lively, upbeat)",
    "ko-M2": "Korean Male (deep, calm)",
    "ko-M3": "Korean Male (polished, authoritative)",
    "ko-M4": "Korean Male (soft, approachable)",
    "ko-M5": "Korean Male (warm, soothing)",
}


class SupertonicTTS(BaseTTS):
    """Supertonic ONNX TTS — real-time on CPU, supports Korean natively."""

    sample_rate = 44_100

    def __init__(self, voice: str = "ko-F1", lang: str = "ko"):
        from supertonic import TTS

        self._lang = lang
        # voice format: "ko-F1" -> extract "F1"
        voice_name = voice.split("-", 1)[1] if "-" in voice else voice
        if voice_name not in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
            raise ValueError(f"Unknown Supertonic voice: {voice}. Use ko-F1..F5 or ko-M1..M5")

        log.info(f"Loading Supertonic TTS (voice={voice_name}, lang={lang})...")
        model_dir = _ensure_supertonic_model()
        self._tts = TTS(model_dir=model_dir, auto_download=True)
        self._style = self._tts.get_voice_style(voice_name=voice_name)
        self.sample_rate = self._tts.sample_rate
        log.info(f"Supertonic TTS loaded (sample_rate={self.sample_rate}).")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        wav, duration = self._tts.synthesize(
            text=text,
            voice_style=self._style,
            lang=self._lang,
            total_steps=5,
            speed=1.05,
            verbose=False,
        )

        audio = wav[0, : int(self.sample_rate * duration[0].item())].astype(np.float32)
        pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        audio = np.concatenate([audio, pad])

        elapsed = time.perf_counter() - t0
        audio_duration = len(audio) / self.sample_rate
        log.info(f"Supertonic TTS: {elapsed:.2f}s -> {audio_duration:.1f}s audio (RTF={elapsed / audio_duration:.3f})")
        return audio
