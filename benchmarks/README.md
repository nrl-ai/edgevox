# Benchmarks

Reproducible measurements for EdgeVox. Every number that appears in a
user-facing surface — README, model card, blog post, talk slide — must
come from a script in this tree that runs from a clean clone.

```
benchmarks/
├── perf/        throughput / latency of pure-Python hot paths
├── accuracy/    quality on real corpora (TBD — voice pipeline + skill quality)
└── results/     baseline JSON snapshots (committed) + ad-hoc runs (gitignored)
```

## Discipline

These rules are non-negotiable. They exist because we have already shipped
fabricated numbers once on this project (see commit history) and we are
not doing it again.

1. **No projected numbers.** A cell in a doc that doesn't have a measured
   value is empty, "TBD", or omitted — never an estimate dressed up with
   a tilde or "approximately."
2. **Warmup + best-of-N.** Throughput / latency benches run the function
   3+ times and report the best run. Single-run numbers are noise. Cold-
   start results inflate ratios — caught a 135× → 21× speedup claim that
   was a lazy-load artifact.
3. **Pin comparison targets.** When benchmarking against a third-party
   library, record the version (`underthesea==9.4.0`) and the bench date.
   Re-run when bumping versions.
4. **Document hardware fingerprint.** Each result JSON records platform,
   Python version, CPU model, GPU (if any), RAM. Latency numbers without
   hardware context are meaningless.
5. **Real corpora, not toy fixtures.** Quality benches use the actual
   model output the user will encounter. Toy fixtures are for harness
   validation only.
6. **Honest empties beat fake numbers.** Quality cells in any doc table
   without a committed-and-runnable bench script must be left empty / TBD.
   Disclaimers like "preliminary" or "internal estimate" do not rescue
   fabricated metrics.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'

# Performance — ~10 seconds, no model downloads needed
python benchmarks/perf/bench_tool_parsing.py
python benchmarks/perf/bench_tool_parsing.py --json benchmarks/results/run.json
```

Compare against the committed baseline:

```bash
python benchmarks/perf/bench_tool_parsing.py --json /tmp/run.json
diff <(jq -S '.results' benchmarks/results/baseline_tool_parsing.json) \
     <(jq -S '.results' /tmp/run.json)
```

## What lives here today

| Bench | What it measures | Inputs | Status |
|---|---|---|---|
| `perf/bench_tool_parsing.py` | `parse_tool_calls()` throughput across all 7 detector formats (Hermes, Mistral, Qwen2.5, Llama-3.2, Granite, Pythonic, xLAM) | Synthetic but format-faithful tool-call payloads, 10 calls each | shipped |

## What's coming

| Bench | What it will measure | Why |
|---|---|---|
| `perf/bench_voice_ttft.py` | end-to-end first-audio latency on the device that runs it | The headline number EdgeVox is built around — must be measured per-hardware, not projected |
| `perf/bench_safety_preempt.py` | wall-clock from stop-word frame to skill `cancel()` resolved | The "halt before LLM" architectural claim needs a number |
| `accuracy/bench_stt_quality.py` | WER per language on a public corpus (CommonVoice slices) | Voice pipeline quality vs whisper baseline |
| `accuracy/bench_tool_dispatch.py` | tool-name + arg-binding correctness on a held-out prompt set | SLM hardening claim ("no malformed JSON, no fabricated tool names") needs a number |

Empty cells in `README.md` and `docs/` are filled when these land — not before.
