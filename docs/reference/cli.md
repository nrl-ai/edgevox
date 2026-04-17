# CLI Reference

## Usage

```bash
edgevox [mode] [options]
```

## Modes

EdgeVox supports three UI modes (mutually exclusive):

| Flag | Mode | Description |
|------|------|-------------|
| _(default)_ | TUI | Interactive terminal UI with waveform, slash commands |
| `--web-ui` | Web UI | FastAPI server with browser-based interface |
| `--simple-ui` | Simple CLI | Minimal terminal interface |

## Shared Options

These options work across all modes:

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | `en` | Language code (en, vi, fr, ko, de, th, ...) |
| `--voice` | auto | TTS voice name |
| `--stt` | auto | STT model size (tiny, base, small, medium, large-v3-turbo, sherpa) |
| `--stt-device` | auto | Device for STT (cuda, cpu) |
| `--llm` | auto | LLM GGUF: preset slug (e.g. `qwen3-1.7b`), local path, or `hf:repo:file.gguf` |
| `--tts` | auto | Force TTS backend (kokoro, piper, supertonic, pythaitts) |
| `-v, --verbose` | off | Enable debug logging |

## TUI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mic` | system default | Microphone device index |
| `--spk` | system default | Speaker device index |
| `--wakeword` | none | Wake word phrase (e.g., "hey jarvis") |
| `--session-timeout` | `30` | Seconds of silence before session ends (with wake word) |
| `--ros2` | `false` | Enable ROS2 bridge |
| `--ros2-namespace` | `/edgevox` | ROS2 namespace for the EdgeVox node |

## Web UI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8765` | Bind port |

## Simple CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text-mode` | `false` | Text-only mode, no microphone |

## Examples

```bash
# Default TUI
edgevox

# Web UI on all interfaces
edgevox --web-ui --host 0.0.0.0 --port 9000

# Vietnamese with Sherpa STT
edgevox --language vi

# Korean with Supertonic TTS
edgevox --language ko --voice ko-M2

# German with specific Piper voice
edgevox --language de --voice de-thorsten-high

# Text-only mode for testing LLM
edgevox --simple-ui --text-mode

# Wake word mode with 60s timeout
edgevox --wakeword "hey pilot" --session-timeout 60

# Specific audio devices
edgevox --mic 2 --spk 4

# Swap LLM for a preset — see edgevox.llm.models.PRESETS for the catalog
edgevox --llm qwen3-1.7b
edgevox --llm llama-3.2-3b
edgevox --llm robobrain-2.0-7b      # embodied / spatial reasoning

# Custom LLM model from HuggingFace (explicit repo:file)
edgevox --llm hf:bartowski/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf

# ROS2 bridge for robotics
edgevox --ros2

# ROS2 with custom namespace (multi-robot)
edgevox --ros2 --ros2-namespace /robot1/voice

# ROS2 + wake word for hands-free robot
edgevox --ros2 --wakeword "hey jarvis" --language vi
```
