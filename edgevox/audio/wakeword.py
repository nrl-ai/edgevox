"""Wake word detection using pymicro-wakeword.

Lightweight TFLite-based wake word detection (~5MB, no torch/onnx).
Supports: hey_jarvis, alexa, hey_mycroft, okay_nabu.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Map user-friendly names to pymicro-wakeword Model enum values
_WAKE_WORD_MAP = {
    "hey jarvis": "HEY_JARVIS",
    "hey_jarvis": "HEY_JARVIS",
    "alexa": "ALEXA",
    "hey mycroft": "HEY_MYCROFT",
    "hey_mycroft": "HEY_MYCROFT",
    "okay nabu": "OKAY_NABU",
    "okay_nabu": "OKAY_NABU",
}

AVAILABLE_WAKE_WORDS = ["hey jarvis", "alexa", "hey mycroft", "okay nabu"]


class WakeWordDetector:
    """Lightweight always-on wake word detector."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        threshold: float = 0.5,
    ):
        from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

        self._threshold = threshold
        self._wake_words = wake_words or ["hey jarvis"]

        # Resolve wake word name to Model enum
        ww_name = self._wake_words[0].lower().strip()
        enum_name = _WAKE_WORD_MAP.get(ww_name)
        if not enum_name:
            raise ValueError(f"Unknown wake word: '{ww_name}'. Available: {AVAILABLE_WAKE_WORDS}")

        model_enum = getattr(Model, enum_name)
        self._model = MicroWakeWord.from_builtin(model_enum)
        self._features = MicroWakeWordFeatures()
        self._detected_name = ww_name
        log.info(f"WakeWord loaded: {ww_name} (pymicro-wakeword)")

    def detect(self, audio_chunk: np.ndarray) -> str | None:
        """Process a 16kHz float32 audio chunk. Returns wake word name or None."""
        # Convert float32 [-1, 1] to int16 bytes
        if audio_chunk.dtype == np.float32:
            audio_int16 = (np.clip(audio_chunk, -1, 1) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)

        audio_bytes = audio_int16.tobytes()

        for feat in self._features.process_streaming(audio_bytes):
            result = self._model.process_streaming(feat)
            if result:
                log.info(f"Wake word detected: {self._detected_name}")
                return self._detected_name

        return None

    def reset(self):
        self._features = type(self._features)()
