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
- **Developer-friendly** — TUI with slash commands for testing voices, models, and languages

## Pipeline Components

| Component | Default Model | Purpose |
|-----------|--------------|---------|
| **VAD** | Silero VAD v6 | Voice activity detection (32ms chunks) |
| **STT** | Faster-Whisper | Speech-to-text (auto-sizes by VRAM) |
| **LLM** | Gemma 4 E2B IT Q4_K_M | Chat via llama-cpp-python |
| **TTS** | Kokoro-82M | Text-to-speech (24kHz, 9 native languages) |

## Special Language Support

Vietnamese gets dedicated models for best accuracy:

| Component | Vietnamese Model | Why |
|-----------|-----------------|-----|
| STT | ChunkFormer-CTC-Large-Vie | 4.18% WER, 14x smaller than Whisper |
| TTS | Piper ONNX | Lightweight, good Vietnamese quality |

## Next Steps

- [Quick Start](/guide/quickstart) — install and run in 5 minutes
- [Architecture](/guide/architecture) — deep dive into the streaming pipeline
- [Languages](/guide/languages) — all 15 supported languages
