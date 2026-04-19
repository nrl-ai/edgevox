"""Iterate candidate personas against the robot tool-calling scenarios.

Given a list of candidate persona strings for one agent (inline via
``--persona`` flags, or a YAML file with a ``personas:`` list), run each
variant through the scenario set of :mod:`scripts.bench_robot_tool_calling`
against a single reference LLM and report which persona scored best.

This is the deliberate counterpart to the full benchmark:

- **Benchmark** measures every model against one fixed persona.
- **Optimizer** measures every persona against one fixed model.

Running a small, fast model as the optimizer target (e.g.
``qwen2.5-1.5b``) is intentional — if a prompt change helps a weak SLM
pick the right tool, it will generally help stronger ones too, and the
iteration loop stays under a minute per candidate.

Usage::

    python scripts/optimize_robot_prompts.py \\
        --agent humanoid \\
        --model qwen2.5-1.5b \\
        --persona "You are G1..." \\
        --persona "You are a Unitree humanoid..."

    python scripts/optimize_robot_prompts.py \\
        --agent panda \\
        --personas-file personas/panda_candidates.yaml

The winning persona is printed to stdout at the end along with a
per-scenario diff against the baseline (the APP's current persona).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.llm import LLM
from scripts.bench_robot_tool_calling import (  # type: ignore[import-not-found]
    Scenario,
    _openai_schemas_for,
    grade,
    scenarios,
)


@dataclass
class PersonaRun:
    label: str
    persona: str
    avg_score: float
    per_scenario: dict[str, int]
    tool_call_rate: float
    total_s: float


def _load_candidates(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return (label, persona) pairs. Always includes the APP's baseline
    persona first so every candidate is graded relative to it."""
    _, baseline_persona, _, _ = _openai_schemas_for(args.agent)
    out: list[tuple[str, str]] = [("baseline", baseline_persona)]
    if args.persona:
        for i, p in enumerate(args.persona, start=1):
            out.append((f"candidate_{i}", p))
    if args.personas_file:
        text = Path(args.personas_file).read_text()
        if args.personas_file.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:
                raise SystemExit("PyYAML required to load .yaml personas file — pip install pyyaml") from e
            data = yaml.safe_load(text)
            for i, item in enumerate(data.get("personas", []), start=1):
                label = item.get("label", f"yaml_{i}") if isinstance(item, dict) else f"yaml_{i}"
                persona = item.get("persona") if isinstance(item, dict) else item
                out.append((label, str(persona)))
        else:
            # Plain-text file: each persona separated by a line of at least three "-".
            for i, chunk in enumerate(text.split("\n---\n"), start=1):
                chunk = chunk.strip()
                if chunk:
                    out.append((f"file_{i}", chunk))
    return out


def _run_persona(
    llm: LLM,
    persona: str,
    scns: list[Scenario],
    schemas: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> PersonaRun:
    parsers = tuple(llm._tool_call_parsers)
    scores: dict[str, int] = {}
    call_count = 0
    total = 0
    t0 = time.perf_counter()
    for scn in scns:
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": scn.user},
        ]
        try:
            raw = llm.complete(messages, tools=schemas, temperature=temperature, max_tokens=max_tokens, stream=False)
            gr = grade(scn, raw, parsers, schemas)
        except Exception as e:
            gr = type(
                "G",
                (),
                {
                    "score": 0,
                    "predicted_tool": None,
                    "flags": [f"exception: {e}"],
                    "predicted_args": {},
                    "reply_text": "",
                },
            )()
        scores[scn.name] = gr.score
        if scn.expected_tool is not None:
            total += 1
            if gr.predicted_tool is not None:
                call_count += 1
    elapsed = time.perf_counter() - t0
    avg = sum(scores.values()) / max(1, len(scores))
    rate = call_count / max(1, total)
    return PersonaRun("?", persona, avg, scores, rate, elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, choices=["scout", "irsim", "panda", "humanoid"])
    parser.add_argument("--model", default="qwen2.5-1.5b", help="Reference LLM slug (default: qwen2.5-1.5b).")
    parser.add_argument(
        "--persona", action="append", default=[], help="Inline candidate persona (repeat for multiple)."
    )
    parser.add_argument("--personas-file", help="Path to a YAML or plain-text file of personas.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args()

    candidates = _load_candidates(args)
    if len(candidates) < 2:
        print("Add at least one candidate via --persona or --personas-file.", file=sys.stderr)
        return 1

    _, _, schemas, _ = _openai_schemas_for(args.agent)
    scns = [s for s in scenarios() if s.agent == args.agent]

    print(f"Loading {args.model!r} ...", flush=True)
    llm = LLM(model_path=f"preset:{args.model}", n_ctx=args.n_ctx)

    runs: list[PersonaRun] = []
    for label, persona in candidates:
        print(f"\n── {label} ──", flush=True)
        r = _run_persona(
            llm,
            persona,
            scns,
            schemas,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        r.label = label
        print(f"  avg={r.avg_score:.1f}  tool-rate={r.tool_call_rate * 100:.0f}%  ({r.total_s:.1f}s)")
        for s in scns:
            print(f"    [{s.name}] {r.per_scenario.get(s.name, 0)}")
        runs.append(r)

    print("\n=== Summary ===")
    print(f"{'label':20s}  {'avg':>6s}  {'rate':>6s}  {'time':>6s}  Δ vs baseline")
    baseline = runs[0]
    for r in runs:
        delta = f"{r.avg_score - baseline.avg_score:+.1f}" if r.label != "baseline" else "—"
        print(f"{r.label:20s}  {r.avg_score:6.1f}  {r.tool_call_rate * 100:5.0f}%  {r.total_s:5.1f}s  {delta}")

    winner = max(runs, key=lambda r: r.avg_score)
    print(f"\nBest: {winner.label!r} (avg {winner.avg_score:.1f})")
    if winner.label != "baseline":
        print("\nPer-scenario delta (winner - baseline):")
        for s in scns:
            d = winner.per_scenario.get(s.name, 0) - baseline.per_scenario.get(s.name, 0)
            if d != 0:
                print(f"  {s.name:30s}  {d:+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
