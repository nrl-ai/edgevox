"""Reusable terminal trace hooks for live agent visibility.

When you want to *see* what the LLM is doing during an interactive
session -- which tool it's calling, what it said, what the tool returned
-- attach these hooks. They print compact one-liners to stdout / stderr
so the trace is readable alongside a sim viewer or voice REPL.

Two hooks ship:

  ``after_llm_trace``   prints the LLM's text reply (truncated) and
                        any tool calls it emitted.
  ``after_tool_trace``  prints each tool's return value (or error)
                        as the dispatcher receives it.

Use :func:`terminal_trace_hooks` to grab both at once::

    from edgevox.agents.trace_hooks import terminal_trace_hooks
    agent = LLMAgent(..., hooks=terminal_trace_hooks())

The hooks are idempotent across hops, safe with parallel tool dispatch
(each call prints on the calling thread), and have no impact on the
agent's actual behaviour -- they read payloads, never modify them.
"""

from __future__ import annotations

import sys
from typing import Any

from edgevox.agents.hooks import AFTER_LLM, AFTER_TOOL, hook

_LINE_PREFIX = "  ◆"
_LLM_TRUNC = 200
_RESULT_TRUNC = 160


def _emit(line: str) -> None:
    """Write to stderr so the trace doesn't tangle with structured stdout
    (e.g. JSON the agent might emit)."""
    print(line, file=sys.stderr, flush=True)


@hook(AFTER_LLM)
def after_llm_trace(point: str, ctx: Any, payload: dict) -> None:
    """Print the LLM's text reply + each tool call it dispatched."""
    content = (payload.get("content") or "").strip()
    if content:
        line = content.replace("\n", " ")
        if len(line) > _LLM_TRUNC:
            line = line[: _LLM_TRUNC - 1] + "…"
        _emit(f"{_LINE_PREFIX} llm: {line}")
    for tc in payload.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or getattr(tc, "name", "?")
        args = fn.get("arguments") or getattr(tc, "arguments", "")
        _emit(f"{_LINE_PREFIX} call: {name}({args})")
    return None


@hook(AFTER_TOOL)
def after_tool_trace(point: str, ctx: Any, payload: Any) -> None:
    """Print each tool's result or error as it returns to the loop.

    The after_tool payload may be a dict (later hops) or a
    ``ToolCallResult`` object (first hop) depending on which path
    inside ``_drive`` reaches us. Handle both shapes so the hook
    never raises.
    """
    if hasattr(payload, "result"):
        name = getattr(payload, "name", "?")
        result = getattr(payload, "result", None)
        error = getattr(payload, "error", None)
    else:
        try:
            name = payload.get("name", "?")
            result = payload.get("result")
            error = payload.get("error")
        except AttributeError:
            return None

    if error:
        _emit(f"{_LINE_PREFIX} err:  {name} -> {error}")
    elif result is not None:
        text = str(result).replace("\n", " ")
        if len(text) > _RESULT_TRUNC:
            text = text[: _RESULT_TRUNC - 1] + "…"
        _emit(f"{_LINE_PREFIX} ret:  {name} -> {text}")
    return None


def terminal_trace_hooks() -> list:
    """Return both trace hooks in dispatch-priority order.

    Pass directly to :class:`~edgevox.agents.base.LLMAgent`'s ``hooks=``
    or :class:`~edgevox.examples.agents.framework.AgentApp`'s ``hooks=``.
    """
    return [after_llm_trace, after_tool_trace]


__all__ = [
    "after_llm_trace",
    "after_tool_trace",
    "terminal_trace_hooks",
]
