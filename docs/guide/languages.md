# Languages

EdgeVox supports 16 languages with automatic STT/TTS backend selection.

## Kokoro Languages (Near-Commercial Quality)

These languages have native TTS support via Kokoro-82M:

| Language | Code | STT | TTS | Default Voice |
|----------|------|-----|-----|---------------|
| 🇺🇸 English | `en` | Whisper | Kokoro (`a`) | `af_heart` |
| 🇬🇧 English (British) | `en-gb` | Whisper | Kokoro (`b`) | `bf_emma` |
| 🇫🇷 French | `fr` | Whisper | Kokoro (`f`) | `ff_siwis` |
| 🇪🇸 Spanish | `es` | Whisper | Kokoro (`e`) | `ef_dora` |
| 🇮🇳 Hindi | `hi` | Whisper | Kokoro (`h`) | `hf_alpha` |
| 🇮🇹 Italian | `it` | Whisper | Kokoro (`i`) | `if_sara` |
| 🇧🇷 Portuguese | `pt` | Whisper | Kokoro (`p`) | `pf_dora` |
| 🇯🇵 Japanese | `ja` | Whisper | Kokoro (`j`) | `jf_alpha` |
| 🇨🇳 Chinese | `zh` | Whisper | Kokoro (`z`) | `zf_xiaobei` |

## 🇻🇳 Vietnamese (Specialized)

Vietnamese uses dedicated models for best accuracy:

| Component | Model | Details |
|-----------|-------|---------|
| STT | Sherpa-ONNX Zipformer (30M, int8) | RTF ~0.01 on CPU, Apache 2.0 |
| TTS | Piper ONNX | 3 voices: `vi-vais1000`, `vi-25hours`, `vi-vivos` |

Falls back to Whisper STT if Sherpa-ONNX is unavailable.

## Piper Languages (Lightweight ONNX)

These languages use Piper VITS models — lightweight and real-time on CPU:

| Language | Code | STT | Voices |
|----------|------|-----|--------|
| 🇩🇪 German | `de` | Whisper | 10 voices (`de-thorsten`, `de-kerstin`, `de-ramona`, ...) |
| 🇷🇺 Russian | `ru` | Whisper | 4 voices (`ru-irina`, `ru-dmitri`, `ru-denis`, `ru-ruslan`) |
| 🇸🇦 Arabic | `ar` | Whisper | 2 voices (`ar-kareem`, `ar-kareem-low`) |
| 🇮🇩 Indonesian | `id` | Whisper | 1 voice (`id-news`) |

## 🇰🇷 Korean (Supertonic)

Korean uses the Supertonic-2 ONNX model — real-time on CPU with 10 voice styles:

| Voice | Description |
|-------|-------------|
| `ko-F1` .. `ko-F5` | 5 female voices (calm, bright, clear, crisp, kind) |
| `ko-M1` .. `ko-M5` | 5 male voices (lively, deep, polished, soft, warm) |

## 🇹🇭 Thai (PyThaiTTS)

Thai uses PyThaiTTS with a Tacotron2 ONNX model (Apache 2.0):

| Voice | Model | Sample Rate |
|-------|-------|-------------|
| `th-default` | lunarlist_onnx | 22,050 Hz |

## Switching Languages

### Via TUI

```
/lang fr          # Switch to French
/lang vi          # Switch to Vietnamese
/lang ko          # Switch to Korean (Supertonic TTS)
/langs            # List all languages with backends
/voices           # List voices for current language
```

Or use the Language and Voice dropdowns in the side panel.

### Via Web UI

Use the language and voice selectors in the web interface, or type `/lang ko` in the text input.

### Via CLI

```bash
edgevox --language fr
edgevox --language ko --voice ko-M2
edgevox --web-ui --language de --voice de-thorsten
```

### Via Code

```python
from edgevox.core.config import get_lang, LANGUAGES

cfg = get_lang("ja")
print(cfg.name)          # "Japanese"
print(cfg.stt_backend)   # "whisper"
print(cfg.tts_backend)   # "kokoro"
print(cfg.kokoro_lang)   # "j"
print(cfg.default_voice) # "jf_alpha"

cfg = get_lang("ko")
print(cfg.tts_backend)   # "supertonic"
print(cfg.default_voice) # "ko-F1"
```

## Adding a New Language

Add one entry to `edgevox/core/config.py`:

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
- TUI language and voice dropdowns
- Web UI selectors
- `/langs` command output
- CLI `--language` option
