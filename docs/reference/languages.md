# Language Configuration

All language settings are centralized in `edgevox/languages.py`.

## LanguageConfig

Each language is defined by a `LanguageConfig` dataclass:

```python
@dataclass(frozen=True)
class LanguageConfig:
    code: str           # ISO 639-1 code ("en", "vi", "fr")
    name: str           # Display name ("English", "Vietnamese")
    stt_backend: str    # "whisper" or "chunkformer"
    tts_backend: str    # "kokoro" or "piper"
    kokoro_lang: str    # Kokoro lang code ("a", "b", "e", "f", ...)
    default_voice: str  # Default TTS voice name
    test_phrase: str    # Warm-up phrase for TTS
    whisper_lang: str   # Override for Whisper language code
```

## API

### `get_lang(code: str) -> LanguageConfig`

Get config by language code. Returns English config for unknown codes.

```python
from edgevox.languages import get_lang

cfg = get_lang("ja")
cfg.name           # "Japanese"
cfg.kokoro_lang    # "j"
cfg.default_voice  # "jf_alpha"
cfg.test_phrase    # "Konnichiwa."
```

### `lang_options() -> list[tuple[str, str]]`

Returns `(display_name, code)` tuples for UI selectors.

```python
from edgevox.languages import lang_options

for name, code in lang_options():
    print(f"{code}: {name}")
```

### `needs_stt_reload(old_code, new_code) -> bool`

Check if switching languages requires a new STT model.

```python
from edgevox.languages import needs_stt_reload

needs_stt_reload("en", "fr")  # False (both use Whisper)
needs_stt_reload("en", "vi")  # True (Whisper → ChunkFormer)
```

### `LANGUAGES: dict[str, LanguageConfig]`

Dictionary of all registered languages.

## Adding a Language

```python
# In edgevox/languages.py

_reg(LanguageConfig(
    code="tr",              # ISO code
    name="Turkish",          # Display name
    stt_backend="whisper",   # STT backend
    tts_backend="kokoro",    # TTS backend
    kokoro_lang="a",         # Kokoro lang (or "a" for English fallback)
    default_voice="af_heart",# Default voice
    test_phrase="Merhaba.",  # Warmup text
))
```

The language is immediately available in the TUI, CLI, and all factories.
