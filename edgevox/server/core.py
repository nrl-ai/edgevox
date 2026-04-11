"""Process-wide model singletons for the WebSocket server.

Loading STT/LLM/TTS is expensive (seconds + GB of VRAM), so all sessions share
one ``ServerCore`` instance. Inference is serialized through ``inference_lock``
because the underlying llama-cpp / faster-whisper / kokoro models are not
thread-safe with concurrent generation requests.

Per-session conversation history is stored on each ``SessionState`` and swapped
into ``llm._history`` while the lock is held — see ``ws.py`` for the swap.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from edgevox.core.config import LANGUAGES, get_lang
from edgevox.llm import LLM
from edgevox.stt import create_stt
from edgevox.tts import create_tts, get_piper_voices

if TYPE_CHECKING:
    from edgevox.server.session import SessionState
    from edgevox.stt import BaseSTT
    from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)


class ServerCore:
    """Holds shared models and the inference lock for the FastAPI app."""

    def __init__(
        self,
        language: str = "en",
        stt_model: str | None = None,
        stt_device: str | None = None,
        llm_model: str | None = None,
        tts_backend: str | None = None,
        voice: str | None = None,
    ):
        self.language = language
        cfg = get_lang(language)

        log.info("Loading STT/LLM/TTS for server (language=%s)…", language)
        t0 = time.perf_counter()
        self.stt: BaseSTT = create_stt(language=language, model_size=stt_model, device=stt_device)
        self.llm = LLM(model_path=llm_model, language=language)
        self.tts: BaseTTS = create_tts(language=language, voice=voice, backend=tts_backend)
        log.info("Models loaded in %.1fs", time.perf_counter() - t0)

        # Warm up the TTS so the first user request doesn't pay engine init cost.
        try:
            _ = self.tts.synthesize(cfg.test_phrase)
        except Exception:
            log.exception("TTS warm-up failed (non-fatal)")

        self.voice = voice or cfg.default_voice
        self.inference_lock = asyncio.Lock()
        self.sessions: dict[str, SessionState] = {}

        # Snapshot the empty (system-prompt-only) history so new sessions start fresh.
        self._base_history: list[dict] = list(self.llm._history)

    def fresh_history(self) -> list[dict]:
        """Return a copy of the system-prompt-only history for new sessions."""
        return [dict(m) for m in self._base_history]

    def voices_for_language(self, language: str) -> list[str]:
        """Return available voice IDs for the given language."""
        cfg = get_lang(language)
        if cfg.tts_backend == "kokoro":
            all_kokoro = [
                "af_heart",
                "af_bella",
                "af_nicole",
                "af_sarah",
                "af_sky",
                "am_adam",
                "am_michael",
                "bf_emma",
                "bf_isabella",
                "bm_george",
                "bm_lewis",
                "ef_dora",
                "em_alex",
                "ff_siwis",
                "hf_alpha",
                "hm_omega",
                "if_sara",
                "im_nicola",
                "jf_alpha",
                "jm_beta",
                "pf_dora",
                "pm_alex",
                "zf_xiaobei",
                "zf_xiaoni",
                "zm_yunjian",
            ]
            prefix = cfg.kokoro_lang
            matching = [v for v in all_kokoro if v.startswith(prefix)]
            others = [v for v in all_kokoro if not v.startswith(prefix)]
            return matching + others
        elif cfg.tts_backend == "supertonic":
            from edgevox.tts.supertonic import SUPERTONIC_VOICES

            return list(SUPERTONIC_VOICES.keys())
        else:
            prefix = cfg.code + "-"
            piper_voices = get_piper_voices()
            matching = [v for v in piper_voices if v.startswith(prefix)]
            others = [v for v in piper_voices if not v.startswith(prefix)]
            return matching + others

    def info(self) -> dict:
        return {
            "language": self.language,
            "languages": sorted(LANGUAGES.keys()),
            "voice": self.voice,
            "voices": self.voices_for_language(self.language),
            "stt": type(self.stt).__name__,
            "tts": type(self.tts).__name__,
            "tts_sample_rate": getattr(self.tts, "sample_rate", 24_000),
            "active_sessions": len(self.sessions),
        }
