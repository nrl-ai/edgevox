"""Speech-to-Text backends with language-aware model selection."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class BaseSTT:
    """Base STT interface. All backends implement this."""

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        raise NotImplementedError


def create_stt(language: str = "en", model_size: str | None = None, device: str | None = None) -> BaseSTT:
    """Factory: create the best STT backend for the language."""
    from edgevox.core.config import get_lang

    cfg = get_lang(language)
    use_chunkformer = model_size == "chunkformer" or (cfg.stt_backend == "chunkformer" and model_size is None)

    if use_chunkformer:
        try:
            from edgevox.stt.chunkformer import ChunkFormerSTT

            return ChunkFormerSTT(device=device)
        except Exception as e:
            log.warning(f"ChunkFormer failed to load ({e}), falling back to Whisper")

    from edgevox.stt.whisper import WhisperSTT

    return WhisperSTT(model_size=model_size, device=device)


# Backward compat
STT = BaseSTT
