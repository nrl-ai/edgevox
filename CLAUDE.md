# EdgeVox — Claude Code Rules

Offline voice agent framework for robots. Pure-Python package, no cloud dependencies, runs on CPU/CUDA/Metal.

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

## Architecture principles

- **Plug-and-play, customizable by default.** Every component — STT backend, TTS backend, LLM, VAD, agent loop behavior, pipeline stage, tool, skill, hook — must be swappable without editing core code. Prefer Protocols, registries, and decorators over hard-coded paths. New behavior lands as a new plugin/hook/backend, not as a patch to an existing module. If you find yourself adding a conditional to core for a specific use case, step back and extract it into an injection point instead.

## Agent harness architecture

The agent harness (`edgevox/agents/` + `edgevox/llm/hooks_slm.py` + `edgevox/llm/tool_parsers/`) is fully documented under `docs/guide/`:

- [`agent-loop.md`](docs/guide/agent-loop.md) — the six-fire-point loop, parallel dispatch, handoff short-circuit.
- [`hooks.md`](docs/guide/hooks.md) — hook authoring contract, built-ins, ordering rules.
- [`memory.md`](docs/guide/memory.md) — `MemoryStore` / `SessionStore` / `NotesFile` / `Compactor`.
- [`interrupt.md`](docs/guide/interrupt.md) — barge-in signals + cancel-token plumbing.
- [`multiagent.md`](docs/guide/multiagent.md) — Blackboard, BackgroundAgent, AgentPool.
- [`tool-calling.md`](docs/guide/tool-calling.md) — parser chain + grammar-constrained decoding roadmap.

Structural decisions with long-term consequences are captured under `docs/adr/`. Add a new ADR (numbered sequentially, short template: Context / Decision / Alternatives / Consequences / Verification) when a change locks in public API shape, a thread-safety contract, or a new required dep.

### Harness rules

- **Typed `AgentContext` fields** (`ctx.tool_registry`, `ctx.llm`, `ctx.interrupt`, `ctx.memory`, `ctx.artifacts`, `ctx.blackboard`) are the public plumbing surface. `ctx.state` is user-only scratch — framework code must not write magic keys there.
- **Hook-owned state** lives under `ctx.hook_state[id(self)]`. Keying by `id(self)` is what guarantees two instances of the same hook class don't share state. See [ADR-002](docs/adr/002-typed-ctx-hook-state.md).
- **Barge-in is enforceable, not advisory.** Every `LLM.complete` call threads `ctx.interrupt.cancel_token` via `stop_event=…` so llama-cpp's `stopping_criteria` actually halts generation within one decode step. See [ADR-001](docs/adr/001-cancel-token-plumbing.md).
- **Tokenizer-exact token counts.** `estimate_tokens(messages, llm)` and `LLM.count_tokens` replace the `chars // 4` heuristic when an LLM is available. Required for correct context-window decisions on CJK / Vietnamese / Thai.
- **Tool-call parsing runs raw-first.** `parse_tool_calls_from_content` tries detectors against the raw content before stripping `<think>` blocks — Qwen3 emits tool calls inside reasoning blocks (see [llama.cpp#20837](https://github.com/ggml-org/llama.cpp/issues/20837)).
- **Preset parsers are validated at load.** `resolve_preset(slug)` asserts every name in `tool_call_parsers=(...)` is a registered detector; a typo fails loudly rather than silently disabling detection.
- **Model-emitted tool-call ids round-trip.** Mistral's `[TOOL_CALLS]` format carries a 9-char id that the follow-up `role="tool"` message must reuse. `ToolCallItem.id` plumbs this through the parser chain and the agent loop.

### Preferred import surfaces

- Agent framework: `from edgevox.agents import LLMAgent, AgentContext, Session, Handoff, ...`
- Built-in hooks: `from edgevox.agents.hooks_builtin import MemoryInjectionHook, TokenBudgetHook, ...`
- SLM hardening: `from edgevox.llm.hooks_slm import default_slm_hooks`
- Memory: `from edgevox.agents.memory import JSONMemoryStore, NotesFile, Compactor, estimate_tokens`
- Multi-agent: `from edgevox.agents.multiagent import Blackboard, BackgroundAgent, AgentPool`
- Interrupt: `from edgevox.agents.interrupt import InterruptController, InterruptPolicy, EnergyBargeInWatcher`

Avoid reaching into private modules or `_agent_harness.py` directly.

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

- **uv** for package management. Use `uv pip install` / `uv venv` instead of bare `pip` / `python -m venv`. See https://docs.astral.sh/uv/.
- **pre-commit** runs ruff (lint + format), gitleaks, and standard hygiene hooks. Install once with `pre-commit install`.
- **gitleaks** scans for secrets on every commit. If a finding is a false positive, allowlist it in `.gitleaks.toml` with a comment explaining why — do not delete the finding.
- **pytest** for tests (`pytest`, asyncio mode = auto). Tests live under `tests/`.

## Workflow expectations

- Read files before editing them. Don't propose changes to code you haven't looked at.
- Run `ruff format` + `ruff check --fix` before declaring a task done.
- Don't bypass hooks (`--no-verify`) — fix the underlying issue.
- Don't add yourself or Claude as a commit author / co-author. Specifically: no `Co-Authored-By: Claude …` trailer, no `🤖 Generated with Claude Code` footer — commit messages end after the body, nothing else.
- Prefer editing existing files over creating new ones; don't create README/docs files unless asked.
- If a change touches the streaming pipeline, manually note the latency impact in the PR description.
- **Prefer Mermaid diagrams over ASCII art** in any markdown doc (`docs/`, `website/`, `README.md`, PR descriptions). GitHub and VitePress render ```mermaid``` fenced blocks natively; hand-drawn box-and-line ASCII is harder to read, impossible to edit cleanly, and breaks under monospace-font changes. The only acceptable ASCII diagrams are directory trees (`├──` / `└──`) — those stay as-is.

## What NOT to do

- Don't add cloud API calls or telemetry. EdgeVox is offline-first.
- Don't introduce GPL-licensed dependencies (project is MIT).
- **License verification is mandatory before adding any new dependency.** Check the package's actual license (PyPI/GitHub/documentation — not just one guess) and record the result inline in ``pyproject.toml`` next to the dep (e.g. ``# MIT`` / ``# LGPL-3 — dynamic-linked, compatible``). Copyleft licenses to refuse: GPL-2/3, AGPL, SSPL. LGPL is acceptable only for pure dynamic-link libraries (PySide6, Qt Multimedia, rlottie). CC-BY-SA is acceptable for *assets* (SVG piece sets etc.), never for source-level dependencies. When in doubt, pick a permissive alternative or flag for discussion.
- Don't commit `dist/`, `build/`, `*.egg-info/`, model weights, or recordings.
- Don't add speculative abstractions or "future-proofing" beyond what the task requires.
