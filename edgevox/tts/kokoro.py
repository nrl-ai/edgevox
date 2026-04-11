"""Kokoro-82M TTS backend via kokoro-onnx — no PyTorch required."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.request

import numpy as np
from huggingface_hub import hf_hub_download

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)

# Primary: consolidated repo; Fallback: GitHub releases
_MODELS_REPO = "nrl-ai/edgevox-models"
_KOKORO_ONNX_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_MODEL_FILENAME = "kokoro-v1.0.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"

# Kokoro lang_code -> kokoro-onnx lang string
_LANG_MAP = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


def _ensure_kokoro_model() -> tuple[str, str]:
    """Download kokoro model, trying consolidated repo first then GitHub releases."""

    # Try consolidated repo
    try:
        model_path = hf_hub_download(_MODELS_REPO, _MODEL_FILENAME, subfolder="kokoro")
        voices_path = hf_hub_download(_MODELS_REPO, _VOICES_FILENAME, subfolder="kokoro")
        return model_path, voices_path
    except Exception:
        log.warning(f"Failed to download Kokoro from {_MODELS_REPO}, trying GitHub releases...")

    # Fallback: GitHub releases
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "edgevox", "kokoro")
    os.makedirs(cache_dir, exist_ok=True)

    model_path = os.path.join(cache_dir, _MODEL_FILENAME)
    voices_path = os.path.join(cache_dir, _VOICES_FILENAME)

    for filename, path in [(_MODEL_FILENAME, model_path), (_VOICES_FILENAME, voices_path)]:
        if not os.path.exists(path):
            url = f"{_KOKORO_ONNX_RELEASE}/{filename}"
            log.info(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
            log.info(f"Saved to {path}")

    return model_path, voices_path


class KokoroTTS(BaseTTS):
    """Kokoro-82M TTS via ONNX — no PyTorch required."""

    sample_rate = 24_000

    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        from kokoro_onnx import Kokoro

        log.info(f"Loading Kokoro-ONNX TTS (voice={voice}, lang={lang_code})...")
        model_path, voices_path = _ensure_kokoro_model()
        self._kokoro = Kokoro(model_path, voices_path)
        self._voice = voice
        self._lang = _LANG_MAP.get(lang_code, "en-us")
        log.info("Kokoro-ONNX TTS loaded.")

    def set_language(self, lang_code: str, voice: str | None = None) -> None:
        """Switch language and optionally voice without reloading the model."""
        self._lang = _LANG_MAP.get(lang_code, "en-us")
        if voice:
            self._voice = voice
        log.info(f"Kokoro TTS switched to lang={self._lang}, voice={self._voice}")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        audio, sr = self._kokoro.create(text, voice=self._voice, speed=1.0, lang=self._lang)
        elapsed = time.perf_counter() - t0
        duration = len(audio) / sr
        log.info(f"Kokoro TTS: {elapsed:.2f}s -> {duration:.1f}s audio")
        return audio

    def synthesize_stream(self, text: str):
        async def _collect():
            chunks = []
            async for audio, _sr in self._kokoro.create_stream(
                text,
                voice=self._voice,
                speed=1.0,
                lang=self._lang,
            ):
                chunks.append(audio)
            return chunks

        # Run async generator synchronously
        loop = asyncio.new_event_loop()
        try:
            chunks = loop.run_until_complete(_collect())
        finally:
            loop.close()
        yield from chunks
