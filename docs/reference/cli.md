# CLI Reference

## Usage

```bash
python -m edgevox <command> [options]
```

## Commands

### `tui`

Launch the interactive terminal UI.

```bash
python -m edgevox tui [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | `en` | Language code (en, vi, fr, es, ...) |
| `--voice` | auto | TTS voice name |
| `--whisper-model` | auto | Whisper model size (tiny, base, small, medium, large-v3-turbo) |
| `--whisper-device` | auto | Device for Whisper (cuda, cpu) |
| `--model-path` | auto | Path to LLM GGUF file |
| `--mic-device` | system default | Microphone device index |
| `--spk-device` | system default | Speaker device index |
| `--wakeword` | none | Wake word phrase (e.g., "hey jarvis") |
| `--session-timeout` | `30` | Seconds of silence before session ends (with wake word) |
| `--ros2` | `false` | Enable ROS2 bridge |

### `text`

Text-only chat mode (no audio).

```bash
python -m edgevox text [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | auto | Path to LLM GGUF file |
| `--language` | `en` | Language for TTS output |

## Examples

```bash
# Basic English usage
python -m edgevox tui

# Vietnamese with female voice
python -m edgevox tui --language vi --voice vi-female

# French with specific Whisper model
python -m edgevox tui --language fr --whisper-model large-v3-turbo

# Wake word mode with 60s timeout
python -m edgevox tui --wakeword "hey pilot" --session-timeout 60

# Specific audio devices
python -m edgevox tui --mic-device 2 --spk-device 4

# Text mode for testing LLM
python -m edgevox text
```
