"""Wall-clock latency of skill cancellation: ``handle.cancel()`` -> CANCELLED.

The architectural claim "the LLM never enters the stop path" needs a number.
This bench measures the bounded latency from ``handle.cancel()`` (what fires
when the SafetyMonitor sees a stop-word frame) to the worker thread observing
``should_cancel()`` and the handle reaching CANCELLED. The LLM round-trip is
explicitly absent from this path — that is the point.

Two components stack on top of the number reported here:
  - VAD frame size (32 ms config, not measured here)
  - Skill body's poll cadence (caller-controlled — we sweep typical values)

Run:

    python benchmarks/perf/bench_safety_preempt.py
    python benchmarks/perf/bench_safety_preempt.py --json benchmarks/results/run.json
    python benchmarks/perf/bench_safety_preempt.py --runs 20

The committed baseline at ``benchmarks/results/baseline_safety_preempt.json``
is the canonical reference for regression checking.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

from edgevox.agents.skills import GoalHandle, GoalStatus


def _run_skill_body(handle: GoalHandle, poll_interval_s: float) -> None:
    """Realistic skill body: polls ``should_cancel`` every ``poll_interval_s``.

    A poll_interval of 0 means "tight loop, framework-overhead floor."
    Anything >0 simulates a skill doing real work between cancel checks
    (sensor read, motor step, tokenizer call, etc.).
    """
    while not handle.should_cancel():
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
    handle.mark_cancelled()


def _bench_one_quantum(poll_interval_s: float, runs: int) -> dict[str, float]:
    """Measure cancel-to-resolved latency for a given skill poll quantum.

    Returns the median + p95 of ``runs`` independent cancellations. Median
    matters more than best-of-N here because the cancellation latency is
    bounded by *the next* poll-interval expiry, which is uniform on
    ``[0, poll_interval]`` — so any single run is an arbitrary draw from
    that distribution and the median converges on its true mean.
    """
    samples_ms: list[float] = []
    for _ in range(runs):
        handle = GoalHandle()
        worker = threading.Thread(
            target=_run_skill_body,
            args=(handle, poll_interval_s),
            daemon=True,
        )
        worker.start()
        # Let the worker actually enter its loop before we cancel; if we
        # cancel before .start() returns the latency is dominated by
        # thread spin-up, not by the cancel path.
        time.sleep(max(poll_interval_s * 2, 0.005))

        cancel_at = time.perf_counter()
        handle.cancel()
        # _done_event is set inside mark_cancelled — that's the moment
        # the handle is observably terminal.
        handle._done_event.wait(timeout=5.0)
        resolved_at = time.perf_counter()

        if handle.status != GoalStatus.CANCELLED:
            raise RuntimeError(f"handle did not reach CANCELLED: {handle.status}")

        samples_ms.append((resolved_at - cancel_at) * 1000.0)
        worker.join(timeout=1.0)

    return {
        "poll_interval_ms": poll_interval_s * 1000.0,
        "runs": runs,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": statistics.quantiles(samples_ms, n=20)[-1] if len(samples_ms) >= 20 else max(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _hardware_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="Cancellations per quantum (default 30).")
    parser.add_argument("--json", type=Path, help="Optional path to write machine-readable JSON.")
    args = parser.parse_args(argv)

    # Sweep representative skill body poll cadences. 0 ms is the framework
    # overhead floor (tight loop, no real work between checks). 10/25/50 ms
    # cover typical sensor-driven skills. 100 ms is what you get when the
    # skill body is doing one tokenizer call per loop.
    quanta_ms = [0.0, 10.0, 25.0, 50.0, 100.0]

    print(f"Skill cancellation latency (median of {args.runs} runs per row)")
    print("=" * 78)
    print(f"{'skill poll':>12}  {'median':>10}  {'p95':>10}  {'min':>10}  {'max':>10}")
    print(f"{'(ms)':>12}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}")
    print("-" * 78)

    results: list[dict[str, float]] = []
    for q_ms in quanta_ms:
        r = _bench_one_quantum(q_ms / 1000.0, runs=args.runs)
        results.append(r)
        print(
            f"{r['poll_interval_ms']:>12.1f}  "
            f"{r['median_ms']:>10.2f}  "
            f"{r['p95_ms']:>10.2f}  "
            f"{r['min_ms']:>10.2f}  "
            f"{r['max_ms']:>10.2f}"
        )

    print()
    print("Read: cancellation lands within ~one poll quantum after handle.cancel().")
    print("The 0 ms row is the framework-overhead floor (no LLM, no sleep, just the")
    print("threading.Event signal + worker observing should_cancel()).")

    if args.json:
        out = {
            "bench": "safety_preempt",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runs_per_quantum": args.runs,
            "hardware": _hardware_fingerprint(),
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
