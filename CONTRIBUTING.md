# Contributing to EdgeVox

Thanks for considering a contribution. EdgeVox is a small, opinionated framework — we accept fixes, features, and docs PRs that fit the project's goals: offline, plug-and-play, streaming, hardware-aware.

## Setup

```bash
git clone https://github.com/nrl-ai/edgevox
cd edgevox
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

System deps for the audio path:
- **Linux:** `apt-get install libsndfile1 portaudio19-dev`
- **macOS:** `brew install portaudio`

Models are downloaded on first use, but you can pre-fetch them:
```bash
edgevox-setup
```

## Run tests

```bash
pytest -m "not integration"           # fast suite, no model downloads
pytest                                # full suite (integration tests fetch models)
ruff check . && ruff format --check . # lint + format gates
pre-commit run --all-files            # everything CI runs
```

CI runs the matrix on Python 3.10–3.13. To target a specific version locally:
```bash
uv venv --python 3.10
```

## Branch and PR conventions

- Branch off `main`. Direct pushes to `main` are blocked for non-owners.
- Conventional commit prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `release:`. Match the existing log style.
- Open a draft PR early if the change is non-trivial — saves rework on architecture mismatches.
- **Do not** add `Co-Authored-By: Claude` or `🤖 Generated with Claude Code` trailers/footers to commits.

## Architecture rules (these gate review)

- **Plug-and-play.** Every layer — STT, TTS, LLM, VAD, hooks, skills, tools, parsers — swaps via Protocols and registries. New behaviour lands as a plugin, not a conditional in core. If you find yourself adding `if model == "something"` to a core path, extract an injection point instead.
- **Offline-first.** No cloud APIs, no telemetry. Whisper / Gemma / Kokoro / Piper / Supertonic / PyThaiTTS run locally — period.
- **Streaming budget.** STT < 0.5 s, LLM first token < 0.4 s, TTS first chunk < 0.1 s on an RTX 3080. PRs that touch the audio path must note the latency impact in the description.
- **Hardware-aware.** CUDA / Metal / CPU all degrade gracefully. Missing accelerator is a config decision, not a crash.
- **Permissive licenses only.** New dependencies are checked at add-time and recorded inline in `pyproject.toml` next to the dep (e.g. `# MIT`, `# LGPL-3 — dynamic-linked, compatible`). GPL / AGPL / SSPL are refused. LGPL is acceptable only for pure dynamic-link libraries (PySide6, Qt Multimedia, rlottie). When in doubt, pick a permissive alternative or open an issue first.

`CLAUDE.md` in the repo root has the full ruleset — read that for anything not covered here.

## Documentation

Docs live in `docs/` (VitePress + Mermaid). Preview your changes:
```bash
npm install               # one-time
npm run docs:dev          # http://localhost:5173
npm run docs:build        # produces docs/.vitepress/dist
```

Two doc-specific rules:
- **Mermaid labels** with `(`, `)`, `@`, `:`, `,`, `&`, or `<br/>` must be quoted (`["foo(args)"]`, `-->|"@tool call"|`). Unquoted parentheses inside `[...]` blow up the parser silently — the page loads but the diagram disappears. Verify zero parse errors in the browser console after editing.
- **New pages** under `docs/documentation/` must be registered in the sidebar in `docs/.vitepress/config.ts`. The page is invisible until that registration lands.

## Running EdgeVox locally

```bash
edgevox-setup             # download models (one-time, ~5 GB)
edgevox                   # voice pipeline TUI
edgevox-agent robot-panda --text-mode
edgevox-chess-robot       # RookApp — PySide6 desktop chess partner
```

## What we won't merge

- Cloud API integrations or telemetry of any kind.
- GPL / AGPL / SSPL dependencies.
- Speculative abstractions for hypothetical future requirements.
- Model files (`.gguf` / `.onnx` / `.bin` / weights) — they live on HuggingFace at [`nrl-ai/edgevox-models`](https://huggingface.co/nrl-ai/edgevox-models).
- Files under `dist/`, `build/`, `*.egg-info/`, recordings, or local cache directories.

## License

By contributing you agree your work is released under the project's [MIT license](LICENSE).

## Questions?

Start in [Discussions](https://github.com/nrl-ai/edgevox/discussions) before opening an issue. For design-level questions, drop a thread there and tag a maintainer.
