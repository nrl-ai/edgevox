"""EdgeVox — Sub-second local voice AI for robots and edge devices."""

__version__ = "0.1.2"


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    if name == "create_stt":
        from edgevox.stt import create_stt

        return create_stt
    if name == "create_tts":
        from edgevox.tts import create_tts

        return create_tts
    if name == "LLM":
        from edgevox.llm import LLM

        return LLM
    if name == "AudioRecorder":
        from edgevox.audio import AudioRecorder

        return AudioRecorder
    if name == "get_lang":
        from edgevox.core.config import get_lang

        return get_lang
    if name == "LANGUAGES":
        from edgevox.core.config import LANGUAGES

        return LANGUAGES
    raise AttributeError(f"module 'edgevox' has no attribute {name!r}")
