"""Smoke tests for the bench scripts under benchmarks/.

These ensure each bench can import + run end-to-end with tiny inputs in
under a couple of seconds. They don't validate the *numbers* — that's
what the committed baseline JSONs are for. They guard against the bench
itself silently breaking (e.g. an API drift in a parser the bench
exercises) so a future PR that reshapes a public surface fails CI here
instead of producing a meaningless score card the next time someone
runs a bench by hand.

Run only these:

    pytest tests/test_benchmarks_smoke.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BENCH_PERF = REPO / "benchmarks" / "perf"
BENCH_ACC = REPO / "benchmarks" / "accuracy"


def _run_bench(script: Path, *args: str) -> dict:
    """Run a bench script with a tmp JSON output and return the parsed dict."""
    tmp = REPO / "benchmarks" / "results" / f"smoke_{script.stem}.json"
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--json", str(tmp), *args],
            cwd=REPO,
            env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Bench may exit non-zero only when it detects a regression
        # (currently only bench_tool_dispatch does this) — for the
        # smoke test on the committed baseline that should never fire.
        assert result.returncode == 0, (
            f"{script.name} exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert tmp.exists(), f"{script.name} did not write {tmp}"
        return json.loads(tmp.read_text())
    finally:
        tmp.unlink(missing_ok=True)


def test_bench_tool_parsing_smoke() -> None:
    """`bench_tool_parsing.py` produces a result for every detector."""
    out = _run_bench(BENCH_PERF / "bench_tool_parsing.py", "--runs", "2", "--ops-per-run", "100")
    assert out["bench"] == "tool_parsing"
    assert "results" in out
    # All 7 detectors must produce a row, each must report 2 calls per
    # parse (the bench payloads contain two tool calls each).
    expected_detectors = {"hermes", "qwen25", "llama32", "mistral", "pythonic", "xlam", "granite"}
    assert set(out["results"]) == expected_detectors
    for det, row in out["results"].items():
        assert row["calls_returned"] == 2, f"{det}: expected 2 calls, got {row['calls_returned']}"
        assert row["ops_per_s"] > 0, f"{det}: zero ops/s — payload mismatch?"


def test_bench_safety_preempt_smoke() -> None:
    """`bench_safety_preempt.py` reports plausible cancellation latencies."""
    out = _run_bench(BENCH_PERF / "bench_safety_preempt.py", "--runs", "5")
    assert out["bench"] == "safety_preempt"
    assert len(out["results"]) == 5  # 0 / 10 / 25 / 50 / 100 ms quanta

    # Tightest row (0 ms quantum) is the framework-overhead floor; on
    # any modern CPU it has to be sub-millisecond. If this regresses
    # past 5 ms something has gone badly wrong with threading.Event.
    floor = out["results"][0]
    assert floor["poll_interval_ms"] == 0
    assert floor["median_ms"] < 5.0, f"framework-overhead floor regressed: {floor['median_ms']:.2f} ms"


def test_bench_tool_dispatch_smoke() -> None:
    """`bench_tool_dispatch.py` hits 100% on the committed correctness battery."""
    out = _run_bench(BENCH_ACC / "bench_tool_dispatch.py")
    assert out["bench"] == "tool_dispatch"
    parse = out["aggregate_parse_score"]
    # Hard expectation: parse-classification battery is fully passing on
    # the committed baseline. Any future regression here means a parser
    # silently changed behaviour on one of the 6 input categories.
    assert parse["correct"] == parse["total"], f"parse classification regressed: {parse['correct']}/{parse['total']}"
    # All 4 dispatch validation rows must remain True.
    for kind, ok in out["dispatch_validation"].items():
        assert ok is True, f"dispatch validation regressed on {kind}"


@pytest.mark.parametrize(
    "baseline_name",
    [
        "baseline_tool_parsing.json",
        "baseline_safety_preempt.json",
        "baseline_tool_dispatch.json",
    ],
)
def test_committed_baseline_is_well_formed(baseline_name: str) -> None:
    """Each committed baseline JSON parses + has the expected top-level keys.

    Catches a class of regression where someone re-runs a bench, the
    bench writes a file, but the file is malformed because of a tweak
    upstream. The smoke test of the baseline *file* is fast and runs
    even if the bench itself can't run in CI.
    """
    path = REPO / "benchmarks" / "results" / baseline_name
    assert path.exists(), f"missing committed baseline: {path}"
    data = json.loads(path.read_text())
    assert "bench" in data
    assert "generated" in data
    assert "hardware" in data
    assert "results" in data or "parse_classification" in data
