"""Throughput bench for ``parse_tool_calls`` across all 7 detector formats.

The agent loop calls ``parse_tool_calls()`` on every assistant message that
might contain a tool call. This bench measures the per-format hot-path cost
on synthetic-but-format-faithful inputs (no model load required), so the
number is reproducible from a clean clone in <10 seconds.

Run:

    python benchmarks/perf/bench_tool_parsing.py
    python benchmarks/perf/bench_tool_parsing.py --json benchmarks/results/run.json
    python benchmarks/perf/bench_tool_parsing.py --runs 5

The committed baseline lives at
``benchmarks/results/baseline_tool_parsing.json`` and is the canonical
reference for regression checking.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from edgevox.llm.tool_parsers import parse_tool_calls

# Tool schema reused across all formats so detectors can validate names.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_light",
            "description": "Turn a room's light on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "on": {"type": "boolean"},
                },
                "required": ["room", "on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Drive the robot to a named room.",
            "parameters": {
                "type": "object",
                "properties": {"room": {"type": "string"}},
                "required": ["room"],
            },
        },
    },
]

# Format-faithful payloads. Each is a real-shape assistant output that the
# named detector must parse successfully. Multi-call payloads stress the
# parser's repeated-match path.
_PAYLOADS: dict[str, str] = {
    "hermes": (
        '<tool_call>{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}</tool_call>\n'
        '<tool_call>{"name": "navigate_to", "arguments": {"room": "bedroom"}}</tool_call>'
    ),
    "qwen25": (
        "<tool_call>\n"
        '{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "navigate_to", "arguments": {"room": "bedroom"}}\n'
        "</tool_call>"
    ),
    "llama32": (
        '<|python_tag|>{"name": "set_light", "parameters": {"room": "kitchen", "on": true}}'
        ';{"name": "navigate_to", "parameters": {"room": "bedroom"}}'
    ),
    "mistral": (
        '[TOOL_CALLS] [{"name": "set_light", "arguments": {"room": "kitchen", "on": true}, '
        '"id": "abc123def"}, {"name": "navigate_to", "arguments": {"room": "bedroom"}, '
        '"id": "xyz789ghi"}]'
    ),
    "pythonic": ('[set_light(room="kitchen", on=True), navigate_to(room="bedroom")]'),
    "xlam": (
        '[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}, '
        '{"name": "navigate_to", "arguments": {"room": "bedroom"}}]'
    ),
    "granite": (
        '<tool_call>[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}, '
        '{"name": "navigate_to", "arguments": {"room": "bedroom"}}]</tool_call>'
    ),
}


def _hardware_fingerprint() -> dict[str, str]:
    """Capture enough hardware context that latency numbers stay interpretable."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def _bench_one(name: str, payload: str, runs: int, ops_per_run: int) -> dict[str, float]:
    """Time ``parse_tool_calls`` against one detector ``runs`` times, return best.

    Each run dispatches ``ops_per_run`` parses to make the timer reading
    statistically stable on fast machines (sub-millisecond per parse).
    """
    # Warmup — JIT, import caches, dispatch tables.
    for _ in range(3):
        parse_tool_calls(payload, _TOOLS, detectors=[name])

    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(ops_per_run):
            result = parse_tool_calls(payload, _TOOLS, detectors=[name])
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best = elapsed

    if result is None or len(result) == 0:
        raise RuntimeError(f"detector '{name}' returned no calls — payload mismatch?")

    return {
        "best_total_s": best,
        "ops_per_s": ops_per_run / best,
        "us_per_op": (best / ops_per_run) * 1_000_000,
        "calls_returned": len(result),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Best-of-N runs (default 5).")
    parser.add_argument(
        "--ops-per-run",
        type=int,
        default=1000,
        help="Parses per timed run (default 1000).",
    )
    parser.add_argument("--json", type=Path, help="Optional path to write machine-readable JSON.")
    args = parser.parse_args(argv)

    print(f"Tool-parsing throughput (best of {args.runs} runs, {args.ops_per_run} ops/run)")
    print("=" * 78)
    print(f"{'detector':>10}  {'best ms':>10}  {'ops/sec':>14}  {'µs/op':>10}  {'calls':>6}")
    print("-" * 78)

    results: dict[str, dict[str, float]] = {}
    for name in _PAYLOADS:
        m = _bench_one(name, _PAYLOADS[name], runs=args.runs, ops_per_run=args.ops_per_run)
        results[name] = m
        print(
            f"{name:>10}  {m['best_total_s'] * 1000:>10.2f}  "
            f"{m['ops_per_s']:>14,.0f}  {m['us_per_op']:>10.2f}  {int(m['calls_returned']):>6}"
        )

    if args.json:
        out = {
            "bench": "tool_parsing",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runs": args.runs,
            "ops_per_run": args.ops_per_run,
            "hardware": _hardware_fingerprint(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
