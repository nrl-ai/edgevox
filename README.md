# EdgeVox

**Sub-second local voice AI for robots and edge devices.**

No cloud APIs. No internet after setup. Fully private. Powered by Gemma 4.

```
    ______    __         _    __
   / ____/___/ /___ ____| |  / /___  _  __
  / __/ / __  / __ `/ _ \ | / / __ \| |/_/
 / /___/ /_/ / /_/ /  __/ |/ / /_/ />  <
/_____/\__,_/\__, /\___/|___/\____/_/|_|
            /____/
```

**Stack:** Silero VAD -> faster-whisper (STT) -> Gemma 4 E2B IT via llama.cpp (LLM) -> Kokoro 82M (TTS)

**Tested latency:** **0.80s** end-to-end on RTX 3080 (STT 0.40s + LLM 0.33s + TTS 0.08s)

## Features

- **Streaming pipeline** — speaks first sentence while LLM generates the rest
- **Interrupt support** — speak while bot is talking to cut it off
- **Wake word detection** — "Hey Jarvis" / "Lily" (optional, via OpenWakeWord)
- **Beautiful TUI** — ASCII logo, sparkline waveform, latency history, GPU/RAM monitor, model info panel
- **ROS2 bridge** — publishes STT/TTS/state to ROS2 topics for robotics integration
- **Slash commands** — `/reset`, `/lang`, `/voice`, `/say`, `/mictest`, `/model` in the TUI
- **Chat export** — Ctrl+S to save conversation as markdown
- **15 languages** — English, Vietnamese, French, Spanish, Hindi, Italian, Portuguese, Japanese, Chinese, Korean, German, Thai, Russian, Arabic, Indonesian
- **Auto-detects hardware** — GPU layers, model size, STT model

## Hardware Requirements

| Device | RAM | GPU | Expected Latency |
|--------|-----|-----|-------------------|
| PC (i9 + RTX 3080 16GB) | 64GB | CUDA | **~0.8s** |
| Jetson Orin Nano | 8GB | CUDA | ~1.5-2s |
| MacBook Air M1 | 8GB | Metal | ~2-3s |
| Any modern laptop | 16GB+ | CPU only | ~2-4s |

## Quick Start

```bash
# 1. Create conda environment
conda create -n voicebot python=3.12 -y
conda activate voicebot

# 2. Install llama-cpp-python with CUDA (prebuilt wheels)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# For Apple Silicon (Metal):
# CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# For CPU only:
# pip install llama-cpp-python

# 3. Install EdgeVox
pip install -e .

# 4. Download all models (~3GB total)
edgevox-setup

# 5. Run!
edgevox
```

## Usage

```bash
# TUI mode (default, recommended)
edgevox

# With wake word
edgevox --wakeword "hey jarvis"

# With ROS2 bridge (for robotics)
edgevox --ros2

# CLI mode (simpler, no TUI)
edgevox-cli

# Text mode (no microphone)
edgevox-cli --text-mode

# Custom options
edgevox \
    --whisper-model large-v3-turbo \
    --voice am_adam \
    --language en
```

## TUI Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset conversation |
| `M` | Mute/Unmute mic |
| `/` | Open command input |
| `Ctrl+S` | Export chat to markdown |

### Slash Commands

| Command | Action |
|---------|--------|
| `/reset` | Reset conversation |
| `/lang XX` | Switch language (en, vi, fr, ko, ...) |
| `/langs` | List all supported languages |
| `/say TEXT` | TTS preview — speak text directly |
| `/mictest` | Record 3s + playback to test audio |
| `/model SIZE` | Switch Whisper model (small/medium/large-v3-turbo) |
| `/voice XX` | Switch TTS voice |
| `/voices` | List available voices |
| `/export` | Export chat to markdown |
| `/mute` | Mute microphone |
| `/unmute` | Unmute microphone |
| `/help` | Show all commands |

## ROS2 Integration

EdgeVox can publish voice pipeline events to ROS2 topics, making it easy to add voice interaction to any robot.

```bash
# Install with ROS2 support
pip install -e ".[ros2]"

# Run with ROS2 bridge
edgevox --ros2
```

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/edgevox/transcription` | `std_msgs/String` | User's speech (STT output) |
| `/edgevox/response` | `std_msgs/String` | Bot's response text |
| `/edgevox/state` | `std_msgs/String` | Pipeline state (listening, thinking, speaking) |
| `/edgevox/audio_level` | `std_msgs/Float32` | Mic level (0.0-1.0) |
| `/edgevox/metrics` | `std_msgs/String` | JSON latency metrics |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/edgevox/tts_request` | `std_msgs/String` | Send text for the bot to speak |
| `/edgevox/command` | `std_msgs/String` | Commands: reset, mute, unmute |

### Example: Robot Integration

```python
import rclpy
from std_msgs.msg import String

# Listen to what the user says
node.create_subscription(String, '/edgevox/transcription', on_user_speech, 10)

# Make the robot say something
pub = node.create_publisher(String, '/edgevox/tts_request', 10)
msg = String()
msg.data = "I detected an obstacle ahead."
pub.publish(msg)
```

## Architecture

```
                        EdgeVox Pipeline
 +-----------+     +------------+     +----------------+
 | Microphone|---->| Silero VAD |---->| faster-whisper  |
 |           |     | (32ms)     |     | (STT)          |
 +-----------+     +------------+     +--------+-------+
                                               |
                                               v
                                      +----------------+
                                      | Gemma 4 E2B IT |
                                      | (streaming)    |
                                      +--------+-------+
                                               | sentence by sentence
                                               v
 +-----------+     +------------+     +----------------+
 |  Speaker  |<----| Kokoro 82M |<----| Sentence       |
 |           |     | (TTS)      |     | Splitter       |
 +-----------+     +------------+     +----------------+
                         |
                         v (optional)
                   +------------+
                   | ROS2 Bridge|----> /edgevox/* topics
                   +------------+
```

## Model Sizes

| Component | Model | Size | RAM |
|-----------|-------|------|-----|
| VAD | Silero VAD v6 | ~2MB | ~10MB |
| STT | whisper-small | 500MB | ~600MB |
| STT | whisper-large-v3-turbo | 1.5GB | ~2GB |
| LLM | Gemma 4 E2B IT Q4_K_M | 1.8GB | ~2.5GB |
| TTS | Kokoro 82M | 200MB | ~300MB |
| Wake | OpenWakeWord | ~2MB | ~10MB |

**M1 Air (8GB):** whisper-small + Q4_K_M = **3.4GB**
**PC with GPU:** whisper-large-v3-turbo + Q4_K_M = **5.8GB**

## Documentation

Full docs: [EdgeVox Docs](https://edgevox-ai.github.io/edgevox/) (built with VitePress)

```bash
cd website && npm run dev
```

## License

MIT
