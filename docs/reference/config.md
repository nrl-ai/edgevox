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

Override with `--whisper-model` and `--whisper-device`.

### TTS Selection

Determined by language config in `edgevox/languages.py`:

- **Kokoro-82M**: English, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese
- **Piper ONNX**: Vietnamese
- **Kokoro (English fallback)**: Korean, German, Thai, Russian, Arabic, Indonesian

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EDGEVOX_MODEL_PATH` | Path to LLM GGUF file |
| `CUDA_VISIBLE_DEVICES` | GPU selection for multi-GPU systems |

## Model Paths

Models are auto-downloaded to HuggingFace cache (`~/.cache/huggingface/`).

| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Whisper large-v3-turbo | `deepdml/faster-whisper-large-v3-turbo-ct2` | ~1.5GB |
| Gemma 4 E2B IT | (local GGUF) | ~2.5GB |
| Kokoro-82M | `hexgrad/Kokoro-82M` | ~170MB |
| ChunkFormer-CTC-Large-Vie | `khanhld/chunkformer-ctc-large-vie` | ~440MB |
| Piper Vietnamese | `speaches-ai/piper-vi_VN-vais1000-medium` | ~75MB |
| Silero VAD | `snakers4/silero-vad` | ~2MB |
