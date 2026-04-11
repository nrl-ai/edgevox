# Languages

EdgeVox supports 15 languages with automatic STT/TTS backend selection.

## Fully Supported Languages

These languages have native TTS support via Kokoro-82M (near-commercial quality):

| Language | Code | STT | TTS | Default Voice |
|----------|------|-----|-----|---------------|
| English | `en` | Whisper | Kokoro (`a`) | `af_heart` |
| English (British) | `en-gb` | Whisper | Kokoro (`b`) | `bf_emma` |
| French | `fr` | Whisper | Kokoro (`f`) | `ff_siwis` |
| Spanish | `es` | Whisper | Kokoro (`e`) | `ef_dora` |
| Hindi | `hi` | Whisper | Kokoro (`h`) | `hf_alpha` |
| Italian | `it` | Whisper | Kokoro (`i`) | `if_sara` |
| Portuguese | `pt` | Whisper | Kokoro (`p`) | `pf_dora` |
| Japanese | `ja` | Whisper | Kokoro (`j`) | `jf_alpha` |
| Chinese | `zh` | Whisper | Kokoro (`z`) | `zf_xiaobei` |

## Vietnamese (Specialized)

Vietnamese uses dedicated models for best accuracy:

| Component | Model | Details |
|-----------|-------|---------|
| STT | ChunkFormer-CTC-Large-Vie | 110M params, 4.18% WER on VIVOS |
| TTS | Piper ONNX | `vi-female` (vais1000-medium) or `vi-male` |

ChunkFormer is 14x smaller than PhoWhisper-large with better accuracy. Falls back to Whisper if unavailable.

## STT-Only Languages

These languages work with Whisper STT but fall back to English Kokoro for TTS:

| Language | Code | Notes |
|----------|------|-------|
| Korean | `ko` | TTS responds in English |
| German | `de` | TTS responds in English |
| Thai | `th` | TTS responds in English |
| Russian | `ru` | TTS responds in English |
| Arabic | `ar` | TTS responds in English |
| Indonesian | `id` | TTS responds in English |

::: tip Adding native TTS
As Kokoro adds more languages, these will automatically get native TTS. Update the language config in `edgevox/languages.py`.
:::

## Switching Languages

### Via TUI

```
/lang fr          # Switch to French
/lang vi          # Switch to Vietnamese
/langs            # List all languages with backends
```

Or use the Language dropdown in the side panel.

### Via CLI

```bash
python -m edgevox tui --language fr
```

### Via Code

```python
from edgevox.languages import get_lang, LANGUAGES

cfg = get_lang("ja")
print(cfg.name)          # "Japanese"
print(cfg.stt_backend)   # "whisper"
print(cfg.tts_backend)   # "kokoro"
print(cfg.kokoro_lang)   # "j"
print(cfg.default_voice) # "jf_alpha"
```

## Adding a New Language

Add one entry to `edgevox/languages.py`:

```python
_reg(LanguageConfig(
    code="tr",
    name="Turkish",
    kokoro_lang="a",          # fallback to English TTS
    default_voice="af_heart",
    test_phrase="Merhaba.",
))
```

The language will automatically appear in:
- TUI language dropdown
- `/langs` command output
- CLI `--language` option
