#!/usr/bin/env python3
"""Smoke-test every GGUF preset in :mod:`edgevox.llm.models`.

For each preset the script downloads (or uses cached) GGUF, instantiates
an :class:`edgevox.llm.LLM`, runs a short chat turn and a one-shot
tool-calling turn, and reports per-model latency + correctness.

Usage::

    uv run python scripts/smoke_test_llm_presets.py              # all presets
    uv run python scripts/smoke_test_llm_presets.py qwen3-1.7b   # just one
    uv run python scripts/smoke_test_llm_presets.py --list       # list presets
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from edgevox.llm import LLM, PRESETS, list_presets, tool
from edgevox.llm.tools import Tool

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
console = Console()


CHAT_PROMPT = "Say hello in one short sentence."
TOOL_PROMPT = "What time is it in Tokyo? Use the tool."


@tool
def get_time(timezone: str = "UTC") -> dict:
    """Return the current time in a given IANA timezone.

    Args:
        timezone: IANA timezone, e.g. ``America/Los_Angeles``, ``Asia/Tokyo``, ``UTC``.
    """
    return {"timezone": timezone, "iso": "2026-04-17T12:00:00"}


@dataclass
class PresetResult:
    slug: str
    family: str
    size_gb: float
    loaded: bool = False
    load_s: float | None = None
    chat_ok: bool = False
    chat_s: float | None = None
    chat_tokens: int | None = None
    chat_reply: str = ""
    tool_ok: bool = False
    tool_s: float | None = None
    tool_called_get_time: bool = False
    tool_reply: str = ""
    error: str = ""


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _run_chat_turn(llm: LLM, prompt: str) -> tuple[str, float, int]:
    t0 = time.perf_counter()
    reply = llm.chat(prompt)
    elapsed = time.perf_counter() - t0
    return reply, elapsed, _token_count(reply)


def _test_preset(slug: str) -> PresetResult:
    preset = PRESETS[slug]
    result = PresetResult(slug=slug, family=preset.family, size_gb=preset.size_gb)

    # --- Load -----------------------------------------------------------
    console.rule(f"[bold cyan]{slug}[/] ({preset.family}, ~{preset.size_gb:.1f} GB)")
    console.print(f"[dim]{preset.repo}/{preset.filename}[/dim]")

    try:
        t0 = time.perf_counter()
        llm = LLM(model_path=f"preset:{slug}", n_ctx=2048)
        result.load_s = time.perf_counter() - t0
        result.loaded = True
        console.print(f"[green]✓[/] loaded in {result.load_s:.2f}s")
    except Exception as exc:
        result.error = f"load: {exc!r}"
        console.print(f"[red]✗ load failed:[/] {exc}")
        traceback.print_exc()
        return result

    # --- Chat turn ------------------------------------------------------
    try:
        reply, elapsed, tokens = _run_chat_turn(llm, CHAT_PROMPT)
        result.chat_s = elapsed
        result.chat_tokens = tokens
        result.chat_reply = reply
        result.chat_ok = bool(reply.strip())
        status = "[green]✓[/]" if result.chat_ok else "[yellow]∅[/]"
        console.print(f"{status} chat: {elapsed:.2f}s, ~{tokens} tok → {reply[:80]!r}")
    except Exception as exc:
        result.error = f"chat: {exc!r}"
        console.print(f"[red]✗ chat failed:[/] {exc}")
        traceback.print_exc()
        return _cleanup(llm, result)

    # --- Tool-calling turn ---------------------------------------------
    try:
        # Fresh LLM with the tool registered so registry-based prompt kicks in.
        del llm
        gc.collect()
        llm = LLM(model_path=f"preset:{slug}", n_ctx=2048, tools=[get_time])

        t0 = time.perf_counter()
        reply = llm.chat(TOOL_PROMPT)
        result.tool_s = time.perf_counter() - t0
        result.tool_reply = reply

        called = [m for m in llm._history if m.get("role") == "tool" and m.get("name") == "get_time"]
        fallback_called = [
            m
            for m in llm._history
            if m.get("role") == "user" and "tool results" in m.get("content", "") and "get_time" in m["content"]
        ]
        result.tool_called_get_time = bool(called or fallback_called)
        result.tool_ok = bool(reply.strip())

        tag = "[green]✓[/]" if result.tool_called_get_time else "[yellow]⚠ no tool call[/]"
        console.print(f"{tag} tool: {result.tool_s:.2f}s → {reply[:80]!r}")
    except Exception as exc:
        result.error = f"tool: {exc!r}"
        console.print(f"[red]✗ tool test failed:[/] {exc}")
        traceback.print_exc()

    return _cleanup(llm, result)


def _cleanup(llm: LLM, result: PresetResult) -> PresetResult:
    import contextlib

    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    return result


def _render_summary(results: list[PresetResult]) -> None:
    table = Table(title="GGUF preset smoke-test summary", show_lines=False)
    table.add_column("preset", style="cyan", no_wrap=True)
    table.add_column("family", style="magenta")
    table.add_column("size GB", justify="right")
    table.add_column("load s", justify="right")
    table.add_column("chat s", justify="right")
    table.add_column("chat tok", justify="right")
    table.add_column("tool s", justify="right")
    table.add_column("tool called?", justify="center")
    table.add_column("status")

    for r in results:
        if not r.loaded:
            status = f"[red]load failed[/]: {r.error}"
        elif not r.chat_ok:
            status = f"[red]chat failed[/]: {r.error}"
        elif r.tool_called_get_time:
            status = "[green]ok + tool[/]"
        elif r.tool_ok:
            status = "[yellow]chat ok, no tool[/]"
        else:
            status = f"[red]{r.error or 'unknown'}[/]"

        table.add_row(
            r.slug,
            r.family,
            f"{r.size_gb:.1f}",
            f"{r.load_s:.2f}" if r.load_s else "—",
            f"{r.chat_s:.2f}" if r.chat_s else "—",
            str(r.chat_tokens) if r.chat_tokens else "—",
            f"{r.tool_s:.2f}" if r.tool_s else "—",
            "yes" if r.tool_called_get_time else "no",
            status,
        )
    console.print(table)


def _to_jsonable(results: list[PresetResult]) -> list[dict[str, Any]]:
    return [r.__dict__ for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test every EdgeVox LLM preset.")
    parser.add_argument("slugs", nargs="*", help="Preset slugs to test (defaults to all).")
    parser.add_argument("--list", action="store_true", help="List known presets and exit.")
    parser.add_argument("--json", metavar="PATH", help="Write full results as JSON to PATH.")
    args = parser.parse_args()

    if args.list:
        for p in list_presets():
            marker = " [embodied]" if p.embodied else ""
            console.print(f"  {p.slug:<22} {p.family:<10} ~{p.size_gb:.1f} GB{marker}  {p.description}")
        return 0

    if args.slugs:
        unknown = [s for s in args.slugs if s not in PRESETS]
        if unknown:
            console.print(f"[red]Unknown preset(s): {', '.join(unknown)}[/]")
            return 2
        slugs = list(args.slugs)
    else:
        slugs = [p.slug for p in list_presets()]

    results: list[PresetResult] = []
    for slug in slugs:
        results.append(_test_preset(slug))

    console.print()
    _render_summary(results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_to_jsonable(results), fh, indent=2, default=str)
        console.print(f"\n[dim]Full results written to {args.json}[/]")

    # Exit nonzero if any load failed or chat broke.
    any_broken = any((not r.loaded) or (not r.chat_ok) for r in results)
    return 1 if any_broken else 0


# Silence ruff on unused tool import — the decorator does the work.
_ = Tool

if __name__ == "__main__":
    sys.exit(main())
