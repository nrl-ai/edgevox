# EdgeVox — Claude Code Rules

Offline voice agent framework for robots and desktop apps. Pure-Python package, no cloud dependencies, runs on CPU/CUDA/Metal.

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

The agent harness (`edgevox/agents/` + `edgevox/llm/hooks_slm.py` + `edgevox/llm/tool_parsers/`) is fully documented under `docs/documentation/`:

- [`agent-loop.md`](docs/documentation/agent-loop.md) — the six-fire-point loop, parallel dispatch, handoff short-circuit.
- [`hooks.md`](docs/documentation/hooks.md) — hook authoring contract, built-ins, ordering rules.
- [`memory.md`](docs/documentation/memory.md) — `MemoryStore` / `SessionStore` / `NotesFile` / `Compactor`.
- [`interrupt.md`](docs/documentation/interrupt.md) — barge-in signals + cancel-token plumbing.
- [`multiagent.md`](docs/documentation/multiagent.md) — Blackboard, BackgroundAgent, AgentPool.
- [`tool-calling.md`](docs/documentation/tool-calling.md) — parser chain + grammar-constrained decoding roadmap.

### Harness rules

- **Typed `AgentContext` fields** (`ctx.tool_registry`, `ctx.llm`, `ctx.interrupt`, `ctx.memory`, `ctx.artifacts`, `ctx.blackboard`) are the public plumbing surface. `ctx.state` is user-only scratch — framework code must not write magic keys there.
- **Hook-owned state** lives under `ctx.hook_state[id(self)]`. Keying by `id(self)` is what guarantees two instances of the same hook class don't share state.
- **Barge-in is enforceable, not advisory.** Every `LLM.complete` call threads `ctx.interrupt.cancel_token` via `stop_event=…` so llama-cpp's `stopping_criteria` actually halts generation within one decode step.
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
- **Imports go at the top of the file.** Only push an import inside a function when something concrete forces it — circular-import break, optional/heavy dependency behind a capability check, or lazy-load to shave CLI startup latency. Convenience or "it's only used in one place" is not a reason; move it up.
- **No new top-level dependencies without reason.** Prefer the stdlib. If you must add one, update `pyproject.toml`.
- **Hardware-aware code paths must degrade gracefully** — CUDA/Metal/CPU fallbacks, never crash on missing accelerator.
- **Never commit model files** (`.gguf`, `.onnx`, `.bin`, weights). They live under `models/` which is gitignored.

## Audio / model conventions

- Sample rate: **16 kHz mono int16** for capture and STT input.
- TTS output: resample to device rate via `sounddevice`.
- VAD frame size: **32 ms** (512 samples @ 16 kHz).
- Latency budget: STT < 0.5 s, LLM first token < 0.4 s, TTS first chunk < 0.1 s on RTX 3080.
- Treat the streaming pipeline as the contract: do not introduce blocking calls that hold the event loop.

## Benchmark discipline — verified numbers only

Every metric that appears in a user-facing surface — README, model card,
website, blog post, talk slide, PR description — must come from a script
in [`benchmarks/`](benchmarks/) that runs from a clean clone, OR a cited
public source with a working URL. Disclaimers like "preliminary,"
"internal estimate," or "indicative" do **not** rescue fabricated metrics
— they create the appearance of data that doesn't exist. When numbers
aren't available, leave the cell empty, write "TBD," or omit the table
entirely. Empty is honest; fake is not.

We have shipped fabricated numbers on this project once (the "~800 ms TTFT
on Jetson Orin Nano" claim that was stripped April 2026 because no one
had ever benched on Jetson). The rules below exist so that does not
recur.

### Methodology (the script existing isn't enough — the protocol must be sound)

- **Throughput / latency:** warm up before timing (3+ calls), report
  best-of-N (N≥3), state the protocol in the result. Cold-start
  artefacts inflate ratios (a 135× claim that was actually ~21× because
  the comparison target lazy-loaded a model is real, caught on a sister
  project).
- **Comparison targets:** pin and report the version of the third-party
  tool (e.g. `faster-whisper==1.0.3`) and the bench date. Re-run when
  bumping versions.
- **Hardware fingerprint:** every result JSON records platform, Python
  version, CPU, GPU (if any). Latency numbers without hardware context
  are meaningless.
- **Single-run results don't count.** They're noisy. Best-of-N or
  median.
- **Real corpora, not toy fixtures.** Quality benches use the actual
  model output the user will encounter. Toy fixtures are for harness
  validation only.

### Sanity-check before claiming a number

Three checks before any bench result goes into a doc, README, or commit
message:

- **Implausible-metric check.** 0 % or 100 % accuracy on a real model
  is almost always a bench bug. Sub-second TTFT on a CPU-only laptop
  with `whisper-large-v3-turbo` is suspect — investigate before claiming.
- **Cross-reference upstream numbers.** If our number disagrees
  materially with the model card / paper of the underlying component
  (>10 % relative on the same metric), pause and investigate.
- **Dump 5 raw samples.** If predictions look obviously wrong but the
  metric looks fine, the metric is broken. Two minutes of `print(input,
  output, gold)` saves a week of fallout from a wrong README number.

## Docs stay in sync with results — every commit, not later

When a result lands (new bench number, new shipped module, new gotcha
caught), update the relevant docs in the **same commit** (or the next
one) — never accumulate a backlog of "docs to refresh." A README with
the previous-quarter status is a regression in the user-facing surface
even when the code is fine.

Minimum propagation matrix:

| Trigger | Update in same commit |
|---|---|
| New bench number | `benchmarks/README.md` row, `README.md` table cell (or remove the TBD), `docs/documentation/<area>.md` if claimed there |
| New supported language / TTS / STT backend | `README.md` languages table, `docs/documentation/index.md`, `CHANGELOG.md` |
| New tool-call detector | `CLAUDE.md` § "Preset parsers", `docs/documentation/tool-calling.md` |
| New ROS2 topic / service / action | `docs/documentation/ros2.md`, README ROS2 section |
| Version bump | `pyproject.toml`, `CHANGELOG.md`, README status badge |

Hard rule: **never claim a number in a doc that doesn't exist yet.**
Order is always (a) measure, (b) update doc with the cited number;
never the other way around. Speculative or "TBD" numbers in docs are
the worst kind of documentation debt.

## Citation block on every published artifact

Every release note, blog post, talk slide, paper, or external write-up
about EdgeVox lists **Viet-Anh Nguyen (`vietanh@nrl.ai`)** in its
citation block, alongside any organisational attribution. The
canonical form lives in `README.md` and is:

```bibtex
@software{edgevox2026,
  author = {Nguyen, Viet-Anh and {Neural Research Lab}},
  title  = {EdgeVox: on-device voice agents for robotics},
  year   = {2026},
  url    = {https://github.com/nrl-ai/edgevox},
  note   = {MIT License}
}
```

Format adapts per medium (paper authors list, slide footer, blog
byline, etc.). Apply retroactively when patching old artifacts.

## Model-file trust ladder

EdgeVox bundles model weights from multiple sources (faster-whisper /
CT2, Kokoro / Piper / Supertonic / PyThaiTTS for TTS, llama.cpp GGUF
for LLM). Before pinning a new model in `setup_models.py` or any
recipe, audit the weight format:

| Format | Status | Why |
|---|---|---|
| `safetensors` | ✅ **always preferred** | Deterministic, zero code execution on load. |
| GGUF (llama.cpp) | ✅ acceptable | Custom binary format, deterministic, no pickle path. The default LLM format. |
| ONNX (Silero VAD, sherpa-onnx) | ✅ acceptable | Public spec, deterministic. |
| HF `.bin` / `pytorch_model.bin` | ⚠️ acceptable from a major lab when no safetensors variant exists | Pickled, but audited at scale and downloaded with HF SHA256 from a known-trusted host. Prefer the safetensors revision when both exist. Document the choice in the wrapper docstring. |
| `.pkl` / `.pickle` | ❌ **auto-reject regardless of source** | Same RCE surface as `.bin`, without HF checksum infrastructure or publisher accountability. |
| Opaque native binaries (CRFsuite, custom) | ⚠️ acceptable when license + format are documented and the format spec is public | Deterministic but opaque. Prefer in-tree reimplementation for small models. |

The license check (existing rule above) is necessary but not
sufficient — file format is checked too.

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

## Writing docs (`docs/`, `README.md`, `website/`)

- **Always quote Mermaid labels that contain `(`, `)`, `@`, `:`, `,`, `&`, `<br/>`, or punctuation other than letters, digits, `_`, and `-`.** Use `["foo(args)"]` for nodes and `-->|"@tool call"|` for edge labels. Unquoted parentheses inside `[...]` or `|...|` blow up the flowchart parser silently — the page still loads but the diagram disappears. After editing any mermaid block, open the page locally (`npm run docs:dev`) and confirm zero `Parse error on line N` entries in the browser console. The breakage is invisible in source review.
- **Test docs in the dev server before declaring done.** `npm run docs:dev` (VitePress on `:5173`) catches mermaid parse errors, broken links, and dead anchors that the markdown source hides. For visual changes (new diagrams, layout, hero, sidebar), capture a Playwright screenshot of the affected page so the reviewer can spot regressions without spinning up the server.
- **Cross-page links are root-absolute, no extension** — `/documentation/hooks`, never `./hooks.md` or `hooks` (the latter resolves wrong under `cleanUrls: true`). Mermaid node-click targets follow the same convention.
- **Sidebar registration is mandatory.** A new page under `docs/documentation/` is invisible until it's added to the `sidebar` block in `docs/.vitepress/config.ts`. Update both at the same time.

## What NOT to do

- Don't add cloud API calls or telemetry. EdgeVox is offline-first.
- **Don't reference this file in user-facing surfaces.** README, `docs/`, blog posts, model cards, talks, PR descriptions to external repos, CHANGELOG entries — none of these may cite "per CLAUDE.md §X" or link here. This file is an AI-operator instruction document; leaking it into user-facing surfaces leaks the instruction layer to readers who came for the software. When you need to invoke a policy from this file in a user-facing doc, restate the underlying policy in self-contained terms ("per our verified-benchmarks rule" instead of "per CLAUDE.md §benchmark-discipline").
- Don't introduce GPL-licensed dependencies (project is MIT).
- **License verification is mandatory before adding any new dependency.** Check the package's actual license (PyPI/GitHub/documentation — not just one guess) and record the result inline in ``pyproject.toml`` next to the dep (e.g. ``# MIT`` / ``# LGPL-3 — dynamic-linked, compatible``). Copyleft licenses to refuse: GPL-2/3, AGPL, SSPL. LGPL is acceptable only for pure dynamic-link libraries (PySide6, Qt Multimedia, rlottie). CC-BY-SA is acceptable for *assets* (SVG piece sets etc.), never for source-level dependencies. When in doubt, pick a permissive alternative or flag for discussion.
- Don't commit `dist/`, `build/`, `*.egg-info/`, model weights, or recordings.
- Don't add speculative abstractions or "future-proofing" beyond what the task requires.
