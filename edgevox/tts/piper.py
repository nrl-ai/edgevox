"""Piper ONNX TTS backend — lightweight VITS models for multiple languages.

Models are served from nrl-ai/edgevox-models on HuggingFace with fallback
to the original upstream repos.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np
from huggingface_hub import hf_hub_download

from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)

MODELS_REPO = "nrl-ai/edgevox-models"


class PiperTTS(BaseTTS):
    """Piper ONNX TTS supporting Vietnamese, German, Russian, Arabic, and Indonesian."""

    sample_rate = 22_050

    # voice_id -> subfolder inside nrl-ai/edgevox-models
    _VOICES: ClassVar[dict[str, str]] = {
        # Vietnamese (3 voices)
        "vi-vais1000": "piper/vi-vais1000",
        "vi-25hours": "piper/vi-25hours",
        "vi-vivos": "piper/vi-vivos",
        # German (10 voices)
        "de-thorsten-high": "piper/de-thorsten-high",
        "de-thorsten": "piper/de-thorsten",
        "de-thorsten-low": "piper/de-thorsten-low",
        "de-thorsten-emotional": "piper/de-thorsten-emotional",
        "de-kerstin": "piper/de-kerstin",
        "de-ramona": "piper/de-ramona",
        "de-eva": "piper/de-eva",
        "de-karlsson": "piper/de-karlsson",
        "de-pavoque": "piper/de-pavoque",
        "de-mls": "piper/de-mls",
        # Russian (4 voices)
        "ru-irina": "piper/ru-irina",
        "ru-dmitri": "piper/ru-dmitri",
        "ru-denis": "piper/ru-denis",
        "ru-ruslan": "piper/ru-ruslan",
        # Arabic (2 voices)
        "ar-kareem": "piper/ar-kareem",
        "ar-kareem-low": "piper/ar-kareem-low",
        # Indonesian (1 voice)
        "id-news": "piper/id-news",
    }

    # Fallback: original upstream repos (used if nrl-ai/edgevox-models is unavailable)
    _FALLBACK_REPOS: ClassVar[dict[str, str]] = {
        "vi-vais1000": "speaches-ai/piper-vi_VN-vais1000-medium",
        "vi-25hours": "speaches-ai/piper-vi_VN-25hours_single-low",
        "vi-vivos": "speaches-ai/piper-vi_VN-vivos-x_low",
        "de-thorsten-high": "speaches-ai/piper-de_DE-thorsten-high",
        "de-thorsten": "speaches-ai/piper-de_DE-thorsten-medium",
        "de-thorsten-low": "speaches-ai/piper-de_DE-thorsten-low",
        "de-thorsten-emotional": "speaches-ai/piper-de_DE-thorsten_emotional-medium",
        "de-kerstin": "speaches-ai/piper-de_DE-kerstin-low",
        "de-ramona": "speaches-ai/piper-de_DE-ramona-low",
        "de-eva": "speaches-ai/piper-de_DE-eva_k-x_low",
        "de-karlsson": "speaches-ai/piper-de_DE-karlsson-low",
        "de-pavoque": "speaches-ai/piper-de_DE-pavoque-low",
        "de-mls": "speaches-ai/piper-de_DE-mls-medium",
        "ru-irina": "speaches-ai/piper-ru_RU-irina-medium",
        "ru-dmitri": "speaches-ai/piper-ru_RU-dmitri-medium",
        "ru-denis": "speaches-ai/piper-ru_RU-denis-medium",
        "ru-ruslan": "speaches-ai/piper-ru_RU-ruslan-medium",
        "ar-kareem": "speaches-ai/piper-ar_JO-kareem-medium",
        "ar-kareem-low": "speaches-ai/piper-ar_JO-kareem-low",
        "id-news": "giganticlab/piper-id_ID-news_tts-medium",
    }

    @staticmethod
    def _download_voice(voice: str) -> tuple[str, str]:
        """Download model.onnx + config.json, trying consolidated repo first."""
        subfolder = PiperTTS._VOICES.get(voice)
        if subfolder:
            try:
                model_path = hf_hub_download(MODELS_REPO, "model.onnx", subfolder=subfolder)
                config_path = hf_hub_download(MODELS_REPO, "config.json", subfolder=subfolder)
                return model_path, config_path
            except Exception:
                log.warning(f"Failed to download {voice} from {MODELS_REPO}, trying fallback...")

        fallback_repo = PiperTTS._FALLBACK_REPOS.get(voice)
        if not fallback_repo:
            raise ValueError(f"Unknown Piper voice: {voice}. Available: {list(PiperTTS._VOICES)}")

        model_path = hf_hub_download(fallback_repo, "model.onnx")
        config_path = hf_hub_download(fallback_repo, "config.json")
        return model_path, config_path

    def __init__(self, voice: str = "vi-vais1000"):
        from piper import PiperVoice

        if voice not in self._VOICES:
            raise ValueError(f"Unknown Piper voice: {voice}. Available: {list(self._VOICES)}")

        log.info(f"Loading Piper TTS ({voice})...")
        model_path, config_path = self._download_voice(voice)

        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self.sample_rate = self._voice.config.sample_rate
        log.info(f"Piper TTS loaded (sample_rate={self.sample_rate}).")

    def synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        audio_chunks = []
        for chunk in self._voice.synthesize(text):
            audio_chunks.append(chunk.audio_float_array)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_chunks).astype(np.float32)
        pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        audio = np.concatenate([audio, pad])

        elapsed = time.perf_counter() - t0
        duration = len(audio) / self.sample_rate
        log.info(f"Piper TTS: {elapsed:.2f}s -> {duration:.1f}s audio")
        return audio
