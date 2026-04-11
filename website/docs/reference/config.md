# Configuration

EdgeVox auto-detects hardware and selects optimal settings. Override with CLI flags or environment variables.

## Auto-Detection

### STT Model Selection

EdgeVox picks the Whisper model based on available resources:

```
CUDA GPU (>= 8GB VRAM) → large-v3-turbo (cuda, float16)
CUDA GPU (< 8GB VRAM)  → small (cuda, float16)
CPU (>= 32GB RAM)       → large-v3-turbo (cpu, int8)
CPU (>= 16GB RAM)       → medium (cpu, int8)
CPU (< 16GB RAM)        → small (cpu, int8)
```

Override with `--stt` and `--stt-device`.

Vietnamese defaults to Sherpa-ONNX Zipformer (30M int8) — falls back to Whisper automatically.

### TTS Selection

Determined by language config in `edgevox/core/config.py`:

- **Kokoro-82M**: English, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese
- **Piper ONNX**: Vietnamese, German, Russian, Arabic, Indonesian
- **Supertonic**: Korean
- **PyThaiTTS**: Thai

Override with `--tts` flag.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EDGEVOX_MODEL_PATH` | Path to LLM GGUF file |
| `CUDA_VISIBLE_DEVICES` | GPU selection for multi-GPU systems |

## Model Hosting

Models are auto-downloaded to HuggingFace cache (`~/.cache/huggingface/`). Most TTS/STT models are consolidated in the `nrl-ai/edgevox-models` repo with automatic fallback to upstream sources.

| Model | Source | Size |
|-------|--------|------|
| Whisper large-v3-turbo | `deepdml/faster-whisper-large-v3-turbo-ct2` | ~1.5GB |
| Sherpa Zipformer (vi) | `nrl-ai/edgevox-models` | ~30MB |
| Gemma 4 E2B IT | (local GGUF) | ~2.5GB |
| Kokoro-82M | `nrl-ai/edgevox-models` | ~338MB |
| Supertonic-2 | `nrl-ai/edgevox-models` | ~255MB |
| PyThaiTTS | `nrl-ai/edgevox-models` | ~163MB |
| Piper voices | `nrl-ai/edgevox-models` | ~50-100MB each |
| Silero VAD | `snakers4/silero-vad` | ~2MB |
