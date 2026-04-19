"""Prompt-ablation sweep for the chess commentary briefing.

Drops each briefing component one at a time and rescores the same
scenario corpus so we can see which signals actually pull weight vs
which are tokens we can strip to save context / latency.

Runs against a single model (default Gemma 4 E2B — the production
champion). For each ablation variant:

* rebuild every scenario's directive with that section suppressed
  (via the new ``ablate`` kwarg on ``_build_ground_truth``);
* call the LLM with the merged system + briefing;
* grade with the same heuristic grader used in
  ``scripts/eval_llm_commentary.py``;
* record score + latency so we can spot cheap wins (section
  suppressed, score unchanged → drop it in production).

Writes ``docs/documentation/reports/data/prompt-ablation.json`` and
a companion ``prompt-ablation.md`` ranked table. Prod code stays
unchanged on the None-default path.

Usage::

    python scripts/bench_prompt_ablation.py                   # gemma-4-e2b, all 3 personas
    python scripts/bench_prompt_ablation.py --model qwen3-1.7b
    python scripts/bench_prompt_ablation.py --personas casual --repeats 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgevox.examples.agents.chess_robot.commentary_gate import (
    _build_ground_truth,
    _record_turn_history,
)
from edgevox.examples.agents.chess_robot.prompts import ROOK_TOOL_GUIDANCE as _ROOK_TOOL_GUIDANCE
from edgevox.llm import LLM
from scripts.eval_llm_commentary import (
    _PERSONA_PROMPTS,
    Scenario,
    _extract_text,
    _FakeEnv,
    _FakeState,
    grade,
    recompute_with_stockfish,
    scenarios,
)

# ---------------------------------------------------------------------------
# Ablation matrix.
# ---------------------------------------------------------------------------
#
# Each variant declares which briefing components to suppress. Keys:
#
# * ``briefing_ablate`` — components skipped by ``_build_ground_truth``.
# * ``drop_tool_guidance`` — omit ``ROOK_TOOL_GUIDANCE`` from the system.
# * ``drop_persona`` — use a minimal bland system prompt instead of
#   the persona description.
#
# ``baseline`` is the production directive shape; everything else
# measures marginal cost of a single component.

ABLATIONS: dict[str, dict] = {
    "baseline": {"briefing_ablate": frozenset()},
    # ``role_header`` and ``footer`` are no-ops — after the first
    # ablation sweep confirmed neither component carried quality
    # signal, both were dropped from the default directive. Kept in
    # ABLATIONS for historical comparison but they produce the same
    # directive shape as ``baseline`` now.
    "no_role_header": {"briefing_ablate": frozenset({"role_header"})},
    "no_footer": {"briefing_ablate": frozenset({"footer"})},
    "no_move_desc": {"briefing_ablate": frozenset({"move_desc"})},
    "no_classification": {"briefing_ablate": frozenset({"classification"})},
    "no_material": {"briefing_ablate": frozenset({"material"})},
    "no_score": {"briefing_ablate": frozenset({"score"})},
    "no_situation": {"briefing_ablate": frozenset({"situation"})},
    # Strip everything except bare facts.
    "facts_only": {
        "briefing_ablate": frozenset(
            {"situation"},
        ),
    },
    # Test whether ``ROOK_TOOL_GUIDANCE`` — the 500-token pronoun /
    # grounding block in the system prompt — is still pulling weight
    # now that the briefing is slimmer.
    "no_tool_guidance": {
        "briefing_ablate": frozenset(),
        "drop_tool_guidance": True,
    },
    # Persona prompt stripped — tests whether the persona header
    # matters given the SITUATION line carries the tone.
    "no_persona": {
        "briefing_ablate": frozenset(),
        "drop_persona": True,
    },
}


# ---------------------------------------------------------------------------
# Directive + message helpers.
# ---------------------------------------------------------------------------


def _directive_for_ablated(scn: Scenario, ablate: frozenset[str]) -> str | None:
    """Same logic as ``eval_llm_commentary._directive_for`` but threads
    ``ablate`` into the builder. Returns ``None`` when the gate would
    stay silent — caller skips the scenario in that variant."""
    state = _FakeState(
        san_history=list(scn.san_history),
        last_move_san=scn.san_history[-1] if scn.san_history else None,
        last_move_classification=scn.classification,
        eval_cp=scn.eval_cp,
        is_game_over=scn.is_game_over,
        game_over_reason=scn.game_over_reason,
        winner=scn.winner,
    )
    env = _FakeEnv(user_plays=scn.user_plays)
    env.set_state(state)
    session_state: dict = {"greeted": scn.greeted_before}
    _record_turn_history(state, env, session_state)
    return _build_ground_truth(state, env, session_state, ablate=ablate)


def _build_messages(scn: Scenario, directive: str, variant: dict) -> list[dict]:
    persona_prompt = (
        "You are a brief chess commentator. Keep replies short and natural."
        if variant.get("drop_persona")
        else _PERSONA_PROMPTS.get(scn.persona, _PERSONA_PROMPTS["casual"])
    )
    parts: list[str] = []
    if not variant.get("drop_tool_guidance"):
        parts.append(_ROOK_TOOL_GUIDANCE)
    parts.append(persona_prompt)
    system = "\n\n---\n\n".join(parts)
    briefing = f"[CHESS BRIEFING — internal context, do not read aloud verbatim]\n{directive}\n[END BRIEFING]"
    return [
        {"role": "system", "content": f"{system}\n\n---\n\n{briefing}"},
        {"role": "user", "content": scn.user_task},
    ]


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    name: str
    score_avg: float
    score_min: int
    score_max: int
    latency_avg_s: float
    latency_p95_s: float
    flag_counts: dict[str, int]
    num_runs: int
    per_scenario: list[dict] = field(default_factory=list)


def run_variant(
    variant_name: str,
    variant: dict,
    llm: LLM,
    scns: list[Scenario],
    personas: list[str],
    *,
    temperature: float,
    max_tokens: int,
    repeats: int,
) -> VariantResult:
    scores: list[int] = []
    latencies: list[float] = []
    flag_counts: dict[str, int] = {}
    per_scenario: list[dict] = []
    ablate = variant.get("briefing_ablate", frozenset())

    for persona in personas:
        for scn in scns:
            scn.persona = persona
            directive = _directive_for_ablated(scn, ablate)
            if directive is None:
                continue  # Gate silent — no LLM call to ablate.
            messages = _build_messages(scn, directive, variant)
            for rep in range(repeats):
                t0 = time.perf_counter()
                raw = llm.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                elapsed = time.perf_counter() - t0
                reply = _extract_text(raw)
                g = grade(scn, directive, reply)
                scores.append(g.score)
                latencies.append(elapsed)
                for f in g.flags:
                    key = f.split(":", 1)[0].split(" ", 1)[0]  # coarse bucket
                    flag_counts[key] = flag_counts.get(key, 0) + 1
                per_scenario.append(
                    {
                        "variant": variant_name,
                        "scenario": scn.name,
                        "persona": persona,
                        "repeat": rep,
                        "score": g.score,
                        "latency_s": round(elapsed, 3),
                        "flags": g.flags,
                        "reply": reply,
                    }
                )

    latencies_sorted = sorted(latencies)
    p95_idx = max(0, int(0.95 * len(latencies_sorted)) - 1) if latencies_sorted else 0
    return VariantResult(
        name=variant_name,
        score_avg=sum(scores) / len(scores) if scores else 0.0,
        score_min=min(scores) if scores else 0,
        score_max=max(scores) if scores else 0,
        latency_avg_s=sum(latencies) / len(latencies) if latencies else 0.0,
        latency_p95_s=latencies_sorted[p95_idx] if latencies_sorted else 0.0,
        flag_counts=flag_counts,
        num_runs=len(scores),
        per_scenario=per_scenario,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-ablation sweep.")
    parser.add_argument("--model", default="gemma-4-e2b")
    parser.add_argument("--personas", default="casual,grandmaster,trash_talker")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument(
        "--variants",
        default=",".join(ABLATIONS.keys()),
        help="Comma-separated variant names to run (default: all).",
    )
    parser.add_argument(
        "--output-json",
        default="docs/documentation/reports/data/prompt-ablation.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/documentation/reports/prompt-ablation.md",
    )
    parser.add_argument(
        "--skip-stockfish",
        action="store_true",
        help="Use hand-set eval_cp / classification from the scenarios (no stockfish recomputation).",
    )
    parser.add_argument(
        "--stockfish-path",
        default=None,
        help="Path to the stockfish binary. Defaults to $STOCKFISH_PATH or 'stockfish' on $PATH.",
    )
    args = parser.parse_args()

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ABLATIONS:
            print(f"error: unknown variant {v!r}; known: {sorted(ABLATIONS)}", file=sys.stderr)
            return 2

    scns = scenarios()
    if not args.skip_stockfish:
        import os

        sf_path = args.stockfish_path or os.environ.get("STOCKFISH_PATH", "stockfish")
        print(f"recomputing scenario eval_cp with stockfish ({sf_path})…")
        scns = recompute_with_stockfish(scns, stockfish_path=sf_path)

    print(f"loading {args.model}…")
    t0 = time.perf_counter()
    llm = LLM(model_path=args.model, n_ctx=args.n_ctx)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    results: list[VariantResult] = []
    for name in variants:
        print(f"\n=== variant: {name} ===")
        t0 = time.perf_counter()
        result = run_variant(
            name,
            ABLATIONS[name],
            llm,
            scns,
            personas,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        )
        print(
            f"  score_avg={result.score_avg:5.1f}  "
            f"lat_avg={result.latency_avg_s:.2f}s  "
            f"lat_p95={result.latency_p95_s:.2f}s  "
            f"runs={result.num_runs}  "
            f"total={time.perf_counter() - t0:.1f}s"
        )
        results.append(result)

    # --- Output ---
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "personas": personas,
        "repeats": args.repeats,
        "variants": [
            {
                "name": r.name,
                "score_avg": round(r.score_avg, 2),
                "score_min": r.score_min,
                "score_max": r.score_max,
                "latency_avg_s": round(r.latency_avg_s, 3),
                "latency_p95_s": round(r.latency_p95_s, 3),
                "num_runs": r.num_runs,
                "flag_counts": r.flag_counts,
                "per_scenario": r.per_scenario,
            }
            for r in results
        ],
    }
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")

    out_md = Path(args.output_md)
    _write_md(out_md, data, results)
    print(f"wrote {out_md}")
    return 0


def _write_md(path: Path, data: dict, results: list[VariantResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort: baseline first, then by score_avg descending.
    def _key(r: VariantResult) -> tuple[int, float]:
        return (0 if r.name == "baseline" else 1, -r.score_avg)

    sorted_res = sorted(results, key=_key)
    lines: list[str] = []
    lines.append("# Prompt-ablation sweep — chess commentary\n")
    lines.append(
        f"Model: **{data['model']}** · temp: {data['temperature']} · "
        f"max_tokens: {data['max_tokens']} · personas: {', '.join(data['personas'])} · "
        f"repeats: {data['repeats']}\n"
    )
    lines.append(
        "Each variant suppresses a single component of the commentary briefing (see "
        "`_build_ground_truth` in `commentary_gate.py`). `baseline` is the production "
        "directive shape; a variant that matches baseline quality while shrinking the "
        "prompt is a straightforward latency / context win.\n"
    )
    lines.append("## Ranked scoreboard\n")
    lines.append("| Variant | Score (avg) | Δ vs baseline | Latency avg | p95 | Runs |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    base = next((r for r in results if r.name == "baseline"), None)
    for r in sorted_res:
        delta = "" if base is None or r.name == "baseline" else f"{r.score_avg - base.score_avg:+.1f}"
        lines.append(
            f"| `{r.name}` | {r.score_avg:.1f} | {delta} | "
            f"{r.latency_avg_s:.2f}s | {r.latency_p95_s:.2f}s | {r.num_runs} |"
        )
    lines.append("\n## Flag buckets per variant\n")
    lines.append("| Variant | Top flags |")
    lines.append("|---|---|")
    for r in sorted_res:
        top = sorted(r.flag_counts.items(), key=lambda kv: -kv[1])[:5]
        flags = ", ".join(f"{k}:{v}" for k, v in top) or "—"
        lines.append(f"| `{r.name}` | {flags} |")
    lines.append("\n## Notes\n")
    lines.append(
        "* Δ score close to zero = candidate for removal. Large negative Δ = component is "
        "load-bearing; worth keeping even at the prompt-size cost.\n"
        "* Scores are the heuristic grader's 0-100 score aggregated across scenarios x personas.\n"
        "* Full per-scenario replies and flags live in `data/prompt-ablation.json`.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
