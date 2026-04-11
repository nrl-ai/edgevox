# EdgeVox — Claude Code Rules

Sub-second local voice AI for robots and edge devices. Pure-Python package, no cloud dependencies, runs on CPU/CUDA/Metal.

## Project layout

```
edgevox/
├── edgevox/            # Source package
│   ├── audio/          # VAD, mic capture, playback
│   ├── stt/            # STT backends (faster-whisper, sherpa-onnx)
│   ├── llm/            # llama.cpp / Gemma integration
│   ├── tts/            # TTS backends (Kokoro, Piper, Supertonic, PyThaiTTS)
│   ├── core/           # Pipeline orchestration
│   ├── cli/            # CLI entrypoints
│   ├── ui/             # TUI widgets
│   ├── integrations/   # ROS2 bridge, etc.
│   ├── tui.py          # Main TUI app
│   └── setup_models.py # Model downloader
│   ├── server/         # FastAPI web UI + WebSocket server
├── webui/              # React frontend (Vite + Tailwind)
├── scripts/            # Utility scripts (model upload, etc.)
├── voices/             # Voice config files
├── docs/               # Project docs
├── website/            # VitePress site
└── pyproject.toml
```

Entrypoints (see `pyproject.toml`):
- `edgevox` → `edgevox.tui:main` (TUI default, `--web-ui` for web, `--simple-ui` for CLI)
- `edgevox-cli` → `edgevox.cli.main:main`
- `edgevox-setup` → `edgevox.setup_models:main`

## Supported languages & backends

| Language | STT | TTS |
|----------|-----|-----|
| English, French, Spanish, etc. | faster-whisper | Kokoro |
| Vietnamese | sherpa-onnx (zipformer) | Piper |
| German, Russian, Arabic, Indonesian | faster-whisper | Piper |
| Korean | faster-whisper | Supertonic |
| Thai | faster-whisper | PyThaiTTS |

Models are hosted on `nrl-ai/edgevox-models` (HuggingFace) with fallback to upstream repos.

## Coding rules

- **Python ≥ 3.10.** Use modern syntax (`X | Y` unions, `match`, `dict[str, int]`).
- **Format and lint with ruff.** Line length 120. Run `ruff format` then `ruff check --fix`.
- **No trailing summaries in code comments.** Comment the *why*, not the *what*.
- **Type hints on public functions.** Internal helpers may skip them when obvious.
- **No prints in library code.** Use `rich`/`textual` for user-facing output, `logging` for diagnostics.
- **No new top-level dependencies without reason.** Prefer the stdlib. If you must add one, update `pyproject.toml`.
- **Hardware-aware code paths must degrade gracefully** — CUDA/Metal/CPU fallbacks, never crash on missing accelerator.
- **Never commit model files** (`.gguf`, `.onnx`, `.bin`, weights). They live under `models/` which is gitignored.

## Audio / model conventions

- Sample rate: **16 kHz mono int16** for capture and STT input.
- TTS output: resample to device rate via `sounddevice`.
- VAD frame size: **32 ms** (512 samples @ 16 kHz).
- Latency budget: STT < 0.5 s, LLM first token < 0.4 s, TTS first chunk < 0.1 s on RTX 3080.
- Treat the streaming pipeline as the contract: do not introduce blocking calls that hold the event loop.

## Tooling

- **pre-commit** runs ruff (lint + format), gitleaks, and standard hygiene hooks. Install once with `pre-commit install`.
- **gitleaks** scans for secrets on every commit. If a finding is a false positive, allowlist it in `.gitleaks.toml` with a comment explaining why — do not delete the finding.
- **pytest** for tests (`pytest`, asyncio mode = auto). Tests live under `tests/`.

## Workflow expectations

- Read files before editing them. Don't propose changes to code you haven't looked at.
- Run `ruff format` + `ruff check --fix` before declaring a task done.
- Don't bypass hooks (`--no-verify`) — fix the underlying issue.
- Don't add yourself or Claude as a commit author / co-author.
- Prefer editing existing files over creating new ones; don't create README/docs files unless asked.
- If a change touches the streaming pipeline, manually note the latency impact in the PR description.

## What NOT to do

- Don't add cloud API calls or telemetry. EdgeVox is offline-first.
- Don't introduce GPL-licensed dependencies (project is MIT).
- Don't commit `dist/`, `build/`, `*.egg-info/`, model weights, or recordings.
- Don't add speculative abstractions or "future-proofing" beyond what the task requires.
