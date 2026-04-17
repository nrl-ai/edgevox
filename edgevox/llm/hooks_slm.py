"""SLM-hardening hooks for the agent loop.

Mirrors the inline mitigations inside :func:`edgevox.llm.LLM._run_agent`
(loop detection, echoed-payload substitution, schema-retry enrichment)
as standalone :class:`~edgevox.agents.hooks.Hook` subclasses so they
can be enabled on :class:`edgevox.agents.LLMAgent` without touching
its :meth:`_drive` loop.

Why keep them as hooks instead of baking into the loop?

1. **Different models need different mitigations.** A 4B Gemma doesn't
   loop; a 1B Qwen does. Composing hooks lets us ship default presets
   per model family.
2. **Researchers tweak them.** Wanting 3 identical calls to hint
   instead of 2, or enabling the hint only for specific tools, is a
   configuration — not a fork.
3. **Testability.** Each mitigation is a 20-line class with tight tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_TOOL,
    ON_RUN_START,
    Hook,
    HookResult,
    ToolCallRequest,
)
from edgevox.llm._agent_harness import (
    FALLBACK_ECHOED_PAYLOAD,
    FALLBACK_LOOP_BREAK,
    LOOP_BREAK_AFTER,
    LOOP_HINT_AFTER,
    MAX_SCHEMA_RETRIES,
    build_loop_hint_payload,
    build_schema_retry_hint,
    fingerprint_call,
    is_argument_shape_error,
    looks_like_echoed_payload,
)

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.llm.tools import ToolCallResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoopDetectorHook
# ---------------------------------------------------------------------------


class LoopDetectorHook:
    """Fingerprint identical ``(tool, args)`` calls and short-circuit them.

    On the second identical call, substitutes a "you already called this"
    hint instead of dispatching. On the third, terminates the turn with
    a polite fallback reply. Resets per-turn counts at
    :data:`~edgevox.agents.hooks.ON_RUN_START`.
    """

    points = frozenset({ON_RUN_START, BEFORE_TOOL})

    def __init__(
        self,
        *,
        hint_after: int = LOOP_HINT_AFTER,
        break_after: int = LOOP_BREAK_AFTER,
        break_reply: str = FALLBACK_LOOP_BREAK,
    ) -> None:
        self.hint_after = hint_after
        self.break_after = break_after
        self.break_reply = break_reply

    def _counts(self, ctx: AgentContext) -> dict[str, int]:
        state = ctx.session.state.setdefault("__loop_detector__", {"counts": {}})
        return state["counts"]  # type: ignore[return-value]

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if point == ON_RUN_START:
            # Reset per-turn fingerprints.
            ctx.session.state["__loop_detector__"] = {"counts": {}}
            return None

        # BEFORE_TOOL
        req: ToolCallRequest = payload
        fp = fingerprint_call(req.name, req.arguments)
        counts = self._counts(ctx)
        counts[fp] = counts.get(fp, 0) + 1
        seen = counts[fp]

        if seen > self.break_after:
            # End the turn immediately — the model would just re-emit.
            log.warning("LoopDetectorHook: loop on %s after %d calls; ending turn", req.name, seen)
            return HookResult.end(self.break_reply, reason=f"loop: {req.name} x{seen}")

        if seen > self.hint_after:
            req.skip_dispatch = True
            req.synthetic_result = build_loop_hint_payload(req.name)
            req.skip_reason = "loop_hint"
            log.info("LoopDetectorHook: duplicate call to %s; hinting", req.name)
            return HookResult.replace(req, reason=f"duplicate: {req.name}")

        return None


# ---------------------------------------------------------------------------
# EchoedPayloadHook
# ---------------------------------------------------------------------------


class EchoedPayloadHook:
    """Substitute a fallback reply when the LLM echoes tool-result JSON.

    Small models sometimes parrot back the tool-result payload verbatim
    as their final answer. This hook catches that at ``after_llm`` (final
    hop, no tool calls) and replaces the content so TTS doesn't read
    raw JSON aloud.
    """

    points = frozenset({AFTER_LLM})

    def __init__(self, *, fallback: str = FALLBACK_ECHOED_PAYLOAD) -> None:
        self.fallback = fallback

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        tool_calls = payload.get("tool_calls") or []
        if tool_calls:
            # Only intervene when this is the final reply.
            return None
        if not looks_like_echoed_payload(content):
            return None
        log.info("EchoedPayloadHook: substituting fallback for echoed content")
        new_payload = dict(payload)
        new_payload["content"] = self.fallback
        return HookResult.replace(new_payload, reason="echoed payload substituted")


# ---------------------------------------------------------------------------
# SchemaRetryHook
# ---------------------------------------------------------------------------


class SchemaRetryHook:
    """Enrich bad-argument tool errors with the real parameter schema.

    When a tool raises a ``TypeError``-shaped argument error, injects a
    human-readable hint listing the correct parameters. Budget is one
    enrichment per tool per turn (matches the in-loop behavior).
    """

    points = frozenset({ON_RUN_START, AFTER_TOOL})

    def __init__(self, *, max_retries_per_tool: int = MAX_SCHEMA_RETRIES) -> None:
        self.max_retries_per_tool = max_retries_per_tool

    def _budget(self, ctx: AgentContext) -> dict[str, int]:
        return ctx.session.state.setdefault("__schema_retry__", {})

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if point == ON_RUN_START:
            ctx.session.state["__schema_retry__"] = {}
            return None

        outcome: ToolCallResult = payload
        if outcome.ok or not is_argument_shape_error(outcome.error):
            return None

        budget = self._budget(ctx)
        used = budget.get(outcome.name, 0)
        if used >= self.max_retries_per_tool:
            return None

        # We need the tool's parameter schema. The registry is on the
        # agent, not ctx directly. Caller attaches it via ctx.state.
        tools_registry = ctx.state.get("__tool_registry__") if hasattr(ctx, "state") else None
        parameters: dict | None = None
        if tools_registry is not None:
            tool = tools_registry.tools.get(outcome.name)
            if tool is not None:
                parameters = tool.parameters

        hint = build_schema_retry_hint(outcome.name, outcome.error or "", parameters)
        budget[outcome.name] = used + 1

        # Replace the result with a structured retry hint.
        outcome.result = {"retry_hint": hint}
        outcome.error = None  # turn it from error → actionable feedback
        return HookResult.replace(outcome, reason="schema retry hint attached")


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------


def default_slm_hooks() -> list[Hook]:
    """Recommended hook bundle for SLMs (1-8B) doing tool calls.

    Enable on any :class:`LLMAgent` backed by a small model::

        agent = LLMAgent(..., hooks=default_slm_hooks())

    Order matters — loop detection runs before schema retry at before_tool.
    """
    return [
        LoopDetectorHook(),
        EchoedPayloadHook(),
        SchemaRetryHook(),
    ]


def combine(*bundles: Iterable[Hook]) -> list[Hook]:
    """Convenience: flatten several hook-bundle factories into one list."""
    out: list[Hook] = []
    for b in bundles:
        out.extend(b)
    return out


__all__ = [
    "EchoedPayloadHook",
    "LoopDetectorHook",
    "SchemaRetryHook",
    "combine",
    "default_slm_hooks",
]
