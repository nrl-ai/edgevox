"""Kokoro-82M TTS backend via kokoro-onnx — no PyTorch required."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)

# kokoro-onnx model files from GitHub releases
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
    """Download kokoro-onnx model files if not cached, return (model_path, voices_path)."""
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
        filename="onnx/model_quantized.onnx",
    )
    voices_path = hf_hub_download(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
        filename="voices.npy",
    )
    return model_path, voices_path


def _ensure_kokoro_model_from_github() -> tuple[str, str]:
    """Download from kokoro-onnx GitHub releases. Caches in ~/.cache/edgevox/."""
    import urllib.request

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
        model_path, voices_path = _ensure_kokoro_model_from_github()
        self._kokoro = Kokoro(model_path, voices_path)
        self._voice = voice
        self._lang = _LANG_MAP.get(lang_code, "en-us")
        log.info("Kokoro-ONNX TTS loaded.")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        audio, sr = self._kokoro.create(text, voice=self._voice, speed=1.0, lang=self._lang)
        elapsed = time.perf_counter() - t0
        duration = len(audio) / sr
        log.info(f"Kokoro TTS: {elapsed:.2f}s -> {duration:.1f}s audio")
        return audio

    def synthesize_stream(self, text: str):
        import asyncio

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
