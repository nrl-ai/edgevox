<!-- Thanks for contributing to EdgeVox! Fill the sections below; delete what doesn't apply. -->

## Summary

<!-- One-paragraph description of what this PR changes and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature (non-breaking)
- [ ] Breaking change (API removed / signature changed)
- [ ] Docs / website only
- [ ] Refactor / chore (no behaviour change)
- [ ] Release / version bump

## Streaming pipeline impact

<!--
Required if this PR touches the audio / STT / LLM / TTS path. Quote
the measurement you took. Baselines on RTX 3080:
  - STT first chunk      < 0.5 s
  - LLM first token       < 0.4 s
  - TTS first audio chunk < 0.1 s
Otherwise: N/A.
-->

## Checklist

- [ ] `pytest -m "not integration"` passes locally
- [ ] `ruff check . && ruff format --check .` passes
- [ ] `pre-commit run --all-files` passes
- [ ] Docs updated if user-visible behaviour changed (`npm run docs:dev` to preview)
- [ ] Mermaid diagrams render without parse errors in the browser console (if any were touched)
- [ ] If a new dependency was added: license verified and recorded inline in `pyproject.toml`
- [ ] No `Co-Authored-By: Claude` / `🤖 Generated` trailers in commits
- [ ] No model files (`.gguf` / `.onnx` / `.bin` / weights) or recordings committed

## Tested on

<!-- Hardware + OS + Python version. Example:
"RTX 3080 / Ubuntu 24.04 / Python 3.12.4" -->

## Linked issues

<!-- Closes #123 — or N/A. -->
