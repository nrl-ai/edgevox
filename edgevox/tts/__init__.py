"""Text-to-Speech backends with multi-language support."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def get_piper_voices() -> dict[str, str]:
    """Return available Piper voice IDs, lazily imported to avoid circular deps."""
    from edgevox.tts.piper import PiperTTS

    return dict(PiperTTS._VOICES)


class BaseTTS:
    """Base TTS interface. All backends implement this."""

    sample_rate: int = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


def create_tts(language: str = "en", voice: str | None = None, backend: str | None = None) -> BaseTTS:
    """Factory: create the right TTS backend for the language.

    Args:
        language: ISO 639-1 language code.
        voice: TTS voice name (backend-specific).
        backend: Force a TTS backend ("kokoro", "piper", "supertonic", or "pythaitts"). None = auto from language config.
    """
    from edgevox.core.config import get_lang

    cfg = get_lang(language)
    tts_backend = backend or cfg.tts_backend
    if tts_backend == "piper":
        from edgevox.tts.piper import PiperTTS

        return PiperTTS(voice=voice or cfg.default_voice)
    elif tts_backend == "supertonic":
        from edgevox.tts.supertonic import SupertonicTTS

        return SupertonicTTS(voice=voice or cfg.default_voice, lang=language)
    elif tts_backend == "pythaitts":
        from edgevox.tts.pythaitts_backend import PyThaiTTSBackend

        return PyThaiTTSBackend(voice=voice or cfg.default_voice)
    else:
        from edgevox.tts.kokoro import KokoroTTS

        return KokoroTTS(voice=voice or cfg.default_voice, lang_code=cfg.kokoro_lang)


# Backward compat
TTS = BaseTTS
SAMPLE_RATE = 24_000
