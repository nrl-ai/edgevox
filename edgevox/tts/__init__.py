"""Text-to-Speech backends with multi-language support."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

PIPER_VI_VOICES = {
    "vi-female": "vi_VN-vais1000-medium",
    "vi-male": "vi_VN-25hours_single-low",
}


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
        backend: Force a TTS backend ("kokoro" or "piper"). None = auto from language config.
    """
    from edgevox.core.config import get_lang

    cfg = get_lang(language)
    tts_backend = backend or cfg.tts_backend
    if tts_backend == "piper":
        from edgevox.tts.piper import PiperTTS

        return PiperTTS(voice=voice or cfg.default_voice)
    else:
        from edgevox.tts.kokoro import KokoroTTS

        return KokoroTTS(voice=voice or cfg.default_voice, lang_code=cfg.kokoro_lang)


# Backward compat
TTS = BaseTTS
SAMPLE_RATE = 24_000
