"""Centralized language configuration for EdgeVox.

Single source of truth for all language-specific settings:
STT backend, TTS backend, voice defaults, test phrases, display names.
Adding a new language = adding one entry to LANGUAGES.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for a single language."""

    code: str  # ISO 639-1 code
    name: str  # Display name
    stt_backend: str = "whisper"  # "whisper", "sherpa", or "chunkformer"
    tts_backend: str = "kokoro"  # "kokoro" or "piper"
    kokoro_lang: str = "a"  # Kokoro lang_code (ignored if tts_backend != "kokoro")
    default_voice: str = "af_heart"  # Default TTS voice
    test_phrase: str = "Ready."  # Warm-up / test phrase
    whisper_lang: str = ""  # Whisper language code override (empty = use .code)

    @property
    def whisper_code(self) -> str:
        return self.whisper_lang or self.code


# All supported languages
LANGUAGES: dict[str, LanguageConfig] = {}


def _reg(cfg: LanguageConfig):
    LANGUAGES[cfg.code] = cfg


# --- Kokoro-supported languages (9 lang codes: a b e f h i j p z) ---

_reg(
    LanguageConfig(
        code="en",
        name="English",
        kokoro_lang="a",
        default_voice="af_heart",
        test_phrase="Ready.",
    )
)
_reg(
    LanguageConfig(
        code="en-gb",
        name="English (British)",
        kokoro_lang="b",
        default_voice="bf_emma",
        test_phrase="Ready.",
        whisper_lang="en",
    )
)
_reg(
    LanguageConfig(
        code="fr",
        name="French",
        kokoro_lang="f",
        default_voice="ff_siwis",
        test_phrase="Bonjour.",
    )
)
_reg(
    LanguageConfig(
        code="es",
        name="Spanish",
        kokoro_lang="e",
        default_voice="ef_dora",
        test_phrase="Hola.",
    )
)
_reg(
    LanguageConfig(
        code="hi",
        name="Hindi",
        kokoro_lang="h",
        default_voice="hf_alpha",
        test_phrase="Namaste.",
    )
)
_reg(
    LanguageConfig(
        code="it",
        name="Italian",
        kokoro_lang="i",
        default_voice="if_sara",
        test_phrase="Ciao.",
    )
)
_reg(
    LanguageConfig(
        code="pt",
        name="Portuguese",
        kokoro_lang="p",
        default_voice="pf_dora",
        test_phrase="Ola.",
    )
)
_reg(
    LanguageConfig(
        code="ja",
        name="Japanese",
        kokoro_lang="j",
        default_voice="jf_alpha",
        test_phrase="Konnichiwa.",
    )
)
_reg(
    LanguageConfig(
        code="zh",
        name="Chinese",
        kokoro_lang="z",
        default_voice="zf_xiaobei",
        test_phrase="Ni hao.",
    )
)

# --- Vietnamese (custom STT + Piper TTS) ---

_reg(
    LanguageConfig(
        code="vi",
        name="Vietnamese",
        stt_backend="sherpa",
        tts_backend="piper",
        default_voice="vi-vais1000",
        test_phrase="San sang.",
    )
)

# --- Piper-supported languages (lightweight VITS ONNX models) ---

_reg(
    LanguageConfig(
        code="de",
        name="German",
        tts_backend="piper",
        default_voice="de-thorsten",
        test_phrase="Hallo.",
    )
)
_reg(
    LanguageConfig(
        code="ru",
        name="Russian",
        tts_backend="piper",
        default_voice="ru-irina",
        test_phrase="Privet.",
    )
)
_reg(
    LanguageConfig(
        code="ar",
        name="Arabic",
        tts_backend="piper",
        default_voice="ar-kareem",
        test_phrase="Marhaba.",
    )
)
_reg(
    LanguageConfig(
        code="id",
        name="Indonesian",
        tts_backend="piper",
        default_voice="id-news",
        test_phrase="Halo.",
    )
)

# --- Supertonic TTS (lightweight ONNX, real-time on CPU) ---

_reg(
    LanguageConfig(
        code="ko",
        name="Korean",
        tts_backend="supertonic",
        default_voice="ko-F1",
        test_phrase="안녕하세요.",
    )
)

# --- PyThaiTTS (Tacotron2 ONNX, Apache 2.0) ---

_reg(
    LanguageConfig(
        code="th",
        name="Thai",
        tts_backend="pythaitts",
        default_voice="th-default",
        test_phrase="สวัสดี.",
    )
)


def get_lang(code: str) -> LanguageConfig:
    """Get language config by code. Falls back to English for unknown codes."""
    return LANGUAGES.get(code, LANGUAGES["en"])


def lang_options() -> list[tuple[str, str]]:
    """Return (display_name, code) tuples for UI selectors, sorted by name."""
    # Put primary Kokoro-supported languages first, then others
    primary = []
    secondary = []
    for cfg in LANGUAGES.values():
        label = cfg.name
        if (cfg.tts_backend == "kokoro" and cfg.kokoro_lang != "a") or cfg.code == "en" or cfg.tts_backend != "kokoro":
            primary.append((label, cfg.code))
        else:
            label = f"{cfg.name} (STT only)"
            secondary.append((label, cfg.code))
    primary.sort(key=lambda x: x[0])
    secondary.sort(key=lambda x: x[0])
    return primary + secondary


def needs_stt_reload(old_code: str, new_code: str) -> bool:
    """Check if switching languages requires reloading the STT model."""
    old = get_lang(old_code)
    new = get_lang(new_code)
    return old.stt_backend != new.stt_backend
