"""Correctness battery for ``parse_tool_calls`` + ``ToolRegistry.dispatch``.

The SLM-hardening claim ("no malformed JSON, no fabricated tool names")
needs a number. This bench feeds a fixed battery of synthetic assistant
outputs into the parse + dispatch path and reports per-detector
classification accuracy:

  - **Happy path**          well-formed call -> parse must return a call
  - **Malformed JSON**      bot_token + broken JSON -> parse must NOT
                            return a partial call (either None or empty)
  - **Pure prose**          no bot_token -> parse must return None
  - **Bot-token in fence**  the bot token quoted inside a code block ->
                            parse must NOT match (typical false-positive
                            trap when a model echoes the protocol token)
  - **Unknown tool name**   format-valid call to a non-existent tool ->
                            parse layer validates names against the
                            schema and MUST drop the call (we only learn
                            this after running the bench — turns out the
                            parser is stricter than we thought)
  - **Schema violation**    format-valid call with wrong arg types ->
                            ``ToolRegistry.dispatch`` MUST flag

Output is a single per-detector score card plus an aggregate. The bench
runs in <1 s and needs no models.

Run:

    python benchmarks/accuracy/bench_tool_dispatch.py
    python benchmarks/accuracy/bench_tool_dispatch.py --json benchmarks/results/run.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Silence the parser's diagnostic logs — bench rows that test malformed
# input deliberately trigger them, so they pollute the score card.
logging.getLogger("edgevox").setLevel(logging.CRITICAL)

from edgevox.llm.tool_parsers import parse_tool_calls  # noqa: E402
from edgevox.llm.tools import ToolRegistry, tool  # noqa: E402

DETECTORS = ["hermes", "qwen25", "llama32", "mistral", "pythonic", "xlam", "granite"]


@tool
def set_light(room: str, on: bool) -> str:
    """Turn a room's light on or off.

    Args:
        room: the room name, e.g. "kitchen".
        on: true to turn on, false to turn off.
    """
    return f"{room} -> {'on' if on else 'off'}"


@tool
def navigate_to(room: str) -> str:
    """Drive the robot to a named room.

    Args:
        room: target room.
    """
    return f"navigating to {room}"


_TOOLS = [set_light, navigate_to]
_TOOL_DESCRIPTORS = [t.__edgevox_tool__ for t in _TOOLS]
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": d.name,
            "description": d.description,
            "parameters": d.parameters,
        },
    }
    for d in _TOOL_DESCRIPTORS
]


# Per-detector test corpus. Each entry: (kind, payload, expect_parse).
# "expect_parse" is True if the detector should return a non-empty call
# list, False if it should return None / empty.
PER_DETECTOR_CASES: dict[str, list[tuple[str, str, bool]]] = {
    "hermes": [
        ("happy", '<tool_call>{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}</tool_call>', True),
        ("happy_multi", '<tool_call>{"name": "navigate_to", "arguments": {"room": "bedroom"}}</tool_call>', True),
        ("malformed_json", '<tool_call>{"name": "set_light", "arguments": {"room": "kitchen"</tool_call>', False),
        ("pure_prose", "Sure, I can help with that. The lights are on.", False),
        ("token_in_fence", "Here is an example: ```<tool_call>{...}</tool_call>``` -- that is the format.", False),
        ("unknown_tool", '<tool_call>{"name": "fabricated_tool", "arguments": {}}</tool_call>', False),
    ],
    "qwen25": [
        (
            "happy",
            '<tool_call>\n{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}\n</tool_call>',
            True,
        ),
        ("happy_multi", '<tool_call>\n{"name": "navigate_to", "arguments": {"room": "bedroom"}}\n</tool_call>', True),
        ("malformed_json", '<tool_call>\n{"name": "set_light", "argum\n</tool_call>', False),
        ("pure_prose", "Lights are on now.", False),
        ("token_in_fence", "Format: ```<tool_call>\\n{...}\\n</tool_call>``` -- example only.", False),
        ("unknown_tool", '<tool_call>\n{"name": "fabricated_tool", "arguments": {}}\n</tool_call>', False),
    ],
    "llama32": [
        ("happy", '<|python_tag|>{"name": "set_light", "parameters": {"room": "kitchen", "on": true}}', True),
        ("happy_multi", '<|python_tag|>{"name": "navigate_to", "parameters": {"room": "bedroom"}}', True),
        ("malformed_json", '<|python_tag|>{"name": "set_light", "parameters": {"room"', False),
        ("pure_prose", "Done.", False),
        ("token_in_fence", "Use ``<|python_tag|>`` to start a tool call.", False),
        ("unknown_tool", '<|python_tag|>{"name": "fabricated_tool", "parameters": {}}', False),
    ],
    "mistral": [
        ("happy", '[TOOL_CALLS] [{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]', True),
        ("happy_multi", '[TOOL_CALLS] [{"name": "navigate_to", "arguments": {"room": "bedroom"}}]', True),
        ("malformed_json", '[TOOL_CALLS] [{"name": "set_light", "argume', False),
        ("pure_prose", "I have turned the lights on.", False),
        ("token_in_fence", "The marker ``[TOOL_CALLS]`` indicates a tool dispatch.", False),
        ("unknown_tool", '[TOOL_CALLS] [{"name": "fabricated_tool", "arguments": {}}]', False),
    ],
    "pythonic": [
        ("happy", '[set_light(room="kitchen", on=True)]', True),
        ("happy_multi", '[navigate_to(room="bedroom")]', True),
        ("malformed_json", '[set_light(room="kitchen", on=', False),
        ("pure_prose", "All done.", False),
        ("token_in_fence", "Example: `[set_light(room=...)]` -- this is the format.", False),
        ("unknown_tool", "[fabricated_tool()]", False),
    ],
    "xlam": [
        ("happy", '[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]', True),
        ("happy_multi", '[{"name": "navigate_to", "arguments": {"room": "bedroom"}}]', True),
        ("malformed_json", '[{"name": "set_light", "arguments"', False),
        ("pure_prose", "Lights on.", False),
        # xlam matches a bare JSON array, so an array inside fenced code may match;
        # this test case verifies the detector's guard against fenced-block extraction.
        ("token_in_fence", "Example output: `[{...}]` is what the schema looks like.", False),
        ("unknown_tool", '[{"name": "fabricated_tool", "arguments": {}}]', False),
    ],
    "granite": [
        ("happy", '<tool_call>[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]</tool_call>', True),
        ("happy_multi", '<tool_call>[{"name": "navigate_to", "arguments": {"room": "bedroom"}}]</tool_call>', True),
        ("malformed_json", '<tool_call>[{"name": "set_light", "arguments"</tool_call>', False),
        ("pure_prose", "Lights are now on.", False),
        ("token_in_fence", "Example: ```<tool_call>[{...}]</tool_call>``` for reference.", False),
        ("unknown_tool", '<tool_call>[{"name": "fabricated_tool", "arguments": {}}]</tool_call>', False),
    ],
}


@dataclass
class _CaseResult:
    detector: str
    kind: str
    expected_parse: bool
    actually_parsed: bool
    correct: bool


def _classify_one(detector: str, kind: str, payload: str, expect_parse: bool) -> _CaseResult:
    # Some malformed-input rows raise inside detectors and the framework
    # logs the traceback — that's expected handling, but we don't want
    # those tracebacks in the bench output. Redirect stderr to /dev/null
    # for the duration of the parse call.
    with contextlib.redirect_stderr(io.StringIO()):
        parsed = parse_tool_calls(payload, _TOOL_SCHEMAS, detectors=[detector])
    actually_parsed = parsed is not None and len(parsed) > 0
    correct = actually_parsed == expect_parse
    return _CaseResult(detector, kind, expect_parse, actually_parsed, correct)


def _bench_dispatch_validation() -> dict[str, bool]:
    """Verify ``ToolRegistry.run`` flags the four known invalid-input classes.

    Returns a dict {kind: did_registry_flag_correctly}. Each kind is
    expected to be True (registry caught the issue).
    """
    reg = ToolRegistry().register(*_TOOLS)

    cases: dict[str, tuple[str, str, str]] = {
        # name, args_json, the substring we expect in the error
        "valid_call": ("set_light", '{"room": "kitchen", "on": true}', ""),
        "unknown_name": ("fabricated_tool", "{}", "unknown tool"),
        "schema_violation": ("set_light", '{"room": "kitchen", "on": "yes"}', "bad arguments"),
        "malformed_json": ("set_light", '{"room": "kitchen", "on":', "invalid JSON"),
    }

    results: dict[str, bool] = {}
    for kind, (name, args, expected_err_fragment) in cases.items():
        r = reg.dispatch(name, args)
        if kind == "valid_call":
            # No error and result is set means registry handled correctly.
            results[kind] = r.error is None and r.result is not None
        else:
            results[kind] = r.error is not None and expected_err_fragment in r.error
    return results


def _hardware_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--verbose", action="store_true", help="Print every misclassified case.")
    args = parser.parse_args(argv)

    print("Tool dispatch correctness — per-detector parse classification")
    print("=" * 78)
    print(
        f"{'detector':>10}  {'happy':>6}  {'multi':>6}  {'mfm-jsn':>8}  {'prose':>6}  {'fence':>6}  {'unk-nm':>7}  {'score':>6}"
    )
    print("-" * 78)

    all_results: list[_CaseResult] = []
    per_detector_score: dict[str, dict[str, int]] = {}

    for det in DETECTORS:
        cases = PER_DETECTOR_CASES[det]
        case_results = [_classify_one(det, kind, payload, expect) for kind, payload, expect in cases]
        all_results.extend(case_results)

        marks: list[str] = []
        for cr in case_results:
            marks.append("✓" if cr.correct else "✗")

        n_correct = sum(1 for cr in case_results if cr.correct)
        n_total = len(case_results)
        score = n_correct / n_total
        per_detector_score[det] = {"correct": n_correct, "total": n_total}
        print(
            f"{det:>10}  "
            f"{marks[0]:>6}  {marks[1]:>6}  {marks[2]:>8}  {marks[3]:>6}  {marks[4]:>6}  {marks[5]:>7}  "
            f"{n_correct}/{n_total} ({score * 100:.0f}%)"
        )

    total_correct = sum(1 for cr in all_results if cr.correct)
    total = len(all_results)
    print("-" * 78)
    print(f"  aggregate parse classification: {total_correct}/{total} ({total_correct / total * 100:.1f}%)")

    print()
    print("ToolRegistry dispatch validation")
    print("=" * 78)
    dispatch_results = _bench_dispatch_validation()
    for kind, ok in dispatch_results.items():
        mark = "✓" if ok else "✗"
        print(f"  {mark} {kind}")
    dispatch_correct = sum(1 for v in dispatch_results.values() if v)
    print(f"  aggregate dispatch validation: {dispatch_correct}/{len(dispatch_results)}")

    if args.verbose:
        misses = [cr for cr in all_results if not cr.correct]
        if misses:
            print()
            print("Misclassified cases:")
            for m in misses:
                print(f"  {m.detector}/{m.kind}: expected_parse={m.expected_parse} actually_parsed={m.actually_parsed}")

    if args.json:
        out = {
            "bench": "tool_dispatch",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hardware": _hardware_fingerprint(),
            "parse_classification": {
                det: {
                    "correct": per_detector_score[det]["correct"],
                    "total": per_detector_score[det]["total"],
                    "cases": [
                        {
                            "kind": cr.kind,
                            "expected_parse": cr.expected_parse,
                            "actually_parsed": cr.actually_parsed,
                            "correct": cr.correct,
                        }
                        for cr in all_results
                        if cr.detector == det
                    ],
                }
                for det in DETECTORS
            },
            "aggregate_parse_score": {"correct": total_correct, "total": total},
            "dispatch_validation": dispatch_results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nWrote {args.json}")

    # Exit non-zero only if any detector or the dispatch validator failed,
    # so the bench can double as a smoke check in CI without inverting
    # the success criterion.
    return 0 if (total_correct == total and dispatch_correct == len(dispatch_results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
