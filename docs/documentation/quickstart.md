# Quick Start

Get EdgeVox running in 5 minutes.

## Prerequisites

- Python 3.10+ (up to 3.13)
- CUDA GPU (recommended, 6GB+ VRAM) or Apple Silicon
- ~4GB disk for models (auto-downloaded on first run)

## Install

```bash
# Create conda environment
conda create -n voicebot python=3.12 -y
conda activate voicebot

# Install with CUDA support
pip install -e .
```

::: tip Why conda?
On some systems, `uv`/`pip` CUDA builds can break. Conda with prebuilt wheels is more reliable for PyTorch + CUDA.
:::

## Download Models

```bash
# Auto-download all required models
python -m edgevox.setup_models
```

This downloads:
- Whisper STT model (auto-sized by your VRAM)
- Sherpa-ONNX Zipformer Vietnamese STT
- Gemma 4 E2B LLM (Q4_K_M quantization)
- Kokoro-82M TTS
- Silero VAD

## Run

### TUI Mode (default)

```bash
edgevox
```

### Web UI

```bash
edgevox --web-ui
# Opens at http://127.0.0.1:8765
```

### Simple CLI

```bash
edgevox --simple-ui
edgevox --simple-ui --text-mode  # text-only, no mic
```

### With specific options

```bash
# Vietnamese with wake word
edgevox --language vi --wakeword "hey jarvis"

# Specify GPU device and Whisper model
edgevox --stt large-v3-turbo --stt-device cuda

# Korean with Supertonic TTS
edgevox --language ko

# Web UI on custom host/port
edgevox --web-ui --host 0.0.0.0 --port 9000

# With ROS2 bridge
edgevox --ros2
```

## First Run

On first launch, EdgeVox will:

1. Load Whisper STT (~2s on GPU)
2. Load Gemma 4 LLM (~1s)
3. Load Kokoro TTS + warmup (~1s)
4. Start listening

You'll see the TUI with a status bar showing "Listening... speak now". Just talk!

## Quick Test

Once running, try these slash commands:

```
/say Hello, this is a TTS test     # Preview TTS output
/mictest                            # Record 3s + playback
/langs                              # List all languages
/lang fr                            # Switch to French
/voice de-thorsten                  # Switch to German Piper voice
/model small                        # Switch to smaller Whisper model
```
