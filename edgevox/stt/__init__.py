"""Speech-to-Text backends with language-aware model selection."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class BaseSTT:
    """Base STT interface. All backends implement this."""

    _model_size: str = ""
    _device: str = ""
    _backend_name: str = ""

    @property
    def display_name(self) -> str:
        """Human-readable name for the model info panel."""
        parts = [p for p in (self._backend_name, self._model_size) if p]
        return " ".join(parts) if parts else type(self).__name__

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        raise NotImplementedError


def create_stt(language: str = "en", model_size: str | None = None, device: str | None = None) -> BaseSTT:
    """Factory: create the best STT backend for the language."""
    from edgevox.core.config import get_lang

    cfg = get_lang(language)

    # Sherpa-ONNX Zipformer: fast Vietnamese transducer (default for vi)
    use_sherpa = model_size == "sherpa" or (cfg.stt_backend == "sherpa" and model_size is None)
    if use_sherpa:
        try:
            from edgevox.stt.sherpa_stt import SherpaSTT

            return SherpaSTT(device=device)
        except Exception as e:
            log.warning(f"Sherpa-ONNX failed to load ({e}), falling back to Whisper")

    # ChunkFormer: legacy Vietnamese backend
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
