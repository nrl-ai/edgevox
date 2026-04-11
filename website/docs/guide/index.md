# Introduction

EdgeVox is a **sub-second local voice AI** designed for robots, edge devices, and anyone who wants private voice interaction without cloud dependencies.

## What is EdgeVox?

EdgeVox is a streaming voice pipeline that chains together:

```
Microphone → VAD → STT → LLM → TTS → Speaker
```

Each component runs locally on your machine. The streaming architecture means the bot starts speaking before it finishes thinking — delivering first audio in **~0.8 seconds**.

## Key Design Principles

- **Portability first** — runs on an i9+RTX3080 desktop or an M1 MacBook Air
- **Language-aware** — automatically selects the best STT/TTS models per language
- **Interruptible** — speak over the bot at any time to cut it off
- **Developer-friendly** — TUI with slash commands, Web UI, and simple CLI modes

## Pipeline Components

| Component | Default Model | Purpose |
|-----------|--------------|---------|
| **VAD** | Silero VAD v6 | Voice activity detection (32ms chunks) |
| **STT** | Faster-Whisper | Speech-to-text (auto-sizes by VRAM) |
| **LLM** | Gemma 4 E2B IT Q4_K_M | Chat via llama-cpp-python |
| **TTS** | Kokoro-82M | Text-to-speech (24kHz, 9 native languages) |

## Multi-Language TTS/STT Backends

| Language | STT | TTS Backend |
|----------|-----|-------------|
| English, French, Spanish, etc. | Faster-Whisper | Kokoro-82M |
| Vietnamese | Sherpa-ONNX (Zipformer 30M) | Piper ONNX |
| German, Russian, Arabic, Indonesian | Faster-Whisper | Piper ONNX |
| Korean | Faster-Whisper | Supertonic |
| Thai | Faster-Whisper | PyThaiTTS |

Models are hosted on `nrl-ai/edgevox-models` (HuggingFace) with automatic fallback to upstream repos.

## Next Steps

- [Quick Start](/guide/quickstart) — install and run in 5 minutes
- [Architecture](/guide/architecture) — deep dive into the streaming pipeline
- [Languages](/guide/languages) — all supported languages and backends
