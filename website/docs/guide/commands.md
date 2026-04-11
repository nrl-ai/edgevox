# TUI Commands

EdgeVox's TUI provides slash commands for development and testing.

## Command Reference

Type `/` in the command bar (or press `/` key) to enter commands.

### General

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history and reset LLM context |
| `/export` | Export chat to `~/edgevox_chat_<timestamp>.md` |
| `/mute` | Mute the microphone |
| `/unmute` | Unmute the microphone |
| `/help` | Show all available commands |

### Voice Development

| Command | Description |
|---------|-------------|
| `/say TEXT` | Synthesize and play text directly (skip LLM) |
| `/mictest` | Record 3 seconds from mic, then play back |
| `/model SIZE` | Hot-swap Whisper STT model |
| `/voice NAME` | Change TTS voice (reloads TTS backend if needed) |
| `/voices` | List available TTS voices for current language |

### Language

| Command | Description |
|---------|-------------|
| `/lang CODE` | Switch language (reloads STT/TTS if needed) |
| `/langs` | List all supported languages with backends |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset chat |
| `M` | Toggle mute |
| `/` | Focus command bar |
| `Ctrl+S` | Export chat to file |

## Examples

### Testing TTS quality

```
/say The quick brown fox jumps over the lazy dog.
/lang fr
/say Bonjour, comment allez-vous aujourd'hui?
/lang ko
/say 안녕하세요, 오늘 날씨가 좋습니다.
```

### Comparing voices across backends

```
/lang de
/voices                    # See 10 German Piper voices
/voice de-thorsten-high    # High-quality Thorsten
/say Hallo, wie geht es Ihnen?
/voice de-kerstin          # Switch to Kerstin
/say Hallo, wie geht es Ihnen?

/lang ko
/voices                    # See 10 Supertonic voices
/voice ko-F2               # Bright female
/say 안녕하세요!
/voice ko-M2               # Deep male
/say 안녕하세요!
```

### Comparing Whisper models

```
/model small
# speak a test phrase...
/model large-v3-turbo
# speak the same phrase, compare accuracy in chat log
```

### Testing mic/speaker setup

```
/mictest
# Records 3 seconds, shows peak/RMS levels, plays back
# Useful for verifying the right mic/speaker is selected
```
