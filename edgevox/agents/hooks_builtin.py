"""Battery-included hooks that ship with EdgeVox.

None of these are loaded by default — compose exactly the set you want:

.. code-block:: python

    from edgevox.agents.hooks_builtin import (
        SafetyGuardrailHook, AuditLogHook, PlanModeHook,
        TokenBudgetHook, ToolOutputTruncatorHook,
        MemoryInjectionHook, EpisodeLoggerHook, PersistSessionHook,
        ContextCompactionHook, NotesInjectorHook,
    )

    agent = LLMAgent(
        ...,
        hooks=[
            SafetyGuardrailHook(blocklist=["rm -rf", "disable safety"]),
            MemoryInjectionHook(memory_store=store),
            TokenBudgetHook(max_context_tokens=3000),
            ContextCompactionHook(compactor=Compactor()),
            ToolOutputTruncatorHook(max_chars=1500),
            PlanModeHook(confirm=["grasp", "place"], approver=console_approver),
            AuditLogHook(path="audit.jsonl"),
            EpisodeLoggerHook(memory_store=store),
            PersistSessionHook(session_store=store, session_id="default"),
        ],
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_TOOL,
    ON_RUN_END,
    ON_RUN_START,
    HookResult,
    ToolCallRequest,
)
from edgevox.agents.memory import Compactor, MemoryStore, NotesFile, SessionStore, estimate_tokens

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext, AgentResult
    from edgevox.llm.llamacpp import LLM
    from edgevox.llm.tools import ToolCallResult

log = logging.getLogger(__name__)


# ===========================================================================
# Guardrails
# ===========================================================================


class SafetyGuardrailHook:
    """Block the turn if the user input matches a blocklist.

    Fires at ``on_run_start`` so the LLM never sees disallowed input.
    Matching is case-insensitive substring by default; pass
    ``matcher=...`` to customize.

    Runs at priority 100 (the ``safety rails`` tier) so it fires before
    any detection or mutation hook has a chance to modify the payload.
    """

    points = frozenset({ON_RUN_START})
    priority = 100

    def __init__(
        self,
        *,
        blocklist: Iterable[str] = (),
        allowlist: Iterable[str] | None = None,
        reply: str = "I can't help with that.",
        matcher: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.blocklist = [b.lower() for b in blocklist]
        self.allowlist = [a.lower() for a in allowlist] if allowlist else None
        self.reply = reply
        self.matcher = matcher or (lambda needle, haystack: needle in haystack)

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        task = (payload or {}).get("task", "") if isinstance(payload, dict) else ""
        text = (task or "").lower()
        if self.allowlist is not None and not any(self.matcher(a, text) for a in self.allowlist):
            return HookResult.end(self.reply, reason="not in allowlist")
        for b in self.blocklist:
            if self.matcher(b, text):
                return HookResult.end(self.reply, reason=f"blocklist match: {b}")
        return None


# ===========================================================================
# Plan mode
# ===========================================================================


# A confirmer receives (tool_name, arguments, ctx) and returns True to allow.
Confirmer = Callable[[str, Any, "AgentContext"], bool]


class PlanModeHook:
    """Require explicit confirmation before dispatching sensitive tools.

    For a robot, wrap actions like ``grasp`` / ``place`` / ``navigate_to``.
    The ``approver`` callable is where the confirmation UI hooks in —
    CLI prompt, a Textual modal, a voice "yes/no" subagent, etc.

    If the approver returns False, the tool is skipped and a synthetic
    result ("user declined") is returned to the LLM so it can adapt.

    Priority 100 — plan-mode denial must beat any detection or
    mutation hook that might otherwise rewrite the call.
    """

    points = frozenset({BEFORE_TOOL})
    priority = 100

    def __init__(
        self,
        *,
        confirm: Iterable[str],
        approver: Confirmer,
        decline_result: str = "user declined",
    ) -> None:
        self.confirm = set(confirm)
        self.approver = approver
        self.decline_result = decline_result

    def __call__(self, point: str, ctx: AgentContext, payload: ToolCallRequest) -> HookResult | None:
        if payload.name not in self.confirm:
            return None
        # Decode JSON-string args to a dict for the approver — that's the
        # useful shape (UI will render keys/values, not raw JSON).
        args = payload.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"__raw__": payload.arguments}
        try:
            ok = self.approver(payload.name, args, ctx)
        except Exception:
            log.exception("PlanMode approver raised — treating as decline")
            ok = False
        if ok:
            return None
        payload.skip_dispatch = True
        payload.synthetic_result = {"ok": False, "error": self.decline_result}
        payload.skip_reason = "plan_mode_declined"
        return HookResult.replace(payload, reason="plan mode: declined")


def console_approver(tool: str, args: Any, ctx: AgentContext) -> bool:
    """Minimal terminal approver — prints a prompt and reads y/n.
    Intended for dev/demo use; production wires a real UI."""
    try:
        raw = input(f"[plan-mode] Approve {tool}({args})? [y/N] ").strip().lower()
    except EOFError:
        return False
    return raw in ("y", "yes")


# ===========================================================================
# Token budget / truncation
# ===========================================================================


class TokenBudgetHook:
    """Enforce a maximum context size on every LLM call.

    Fires at ``before_llm``. If estimated tokens exceed ``max_context_tokens``,
    drops oldest non-system messages until it fits (preserving system +
    last ``keep_last`` turns). Use :class:`ContextCompactionHook` instead
    if you want smart summarization; this hook is a hard safety net.
    """

    points = frozenset({BEFORE_LLM})

    def __init__(self, *, max_context_tokens: int = 4000, keep_last: int = 4) -> None:
        self.max_context_tokens = max_context_tokens
        self.keep_last = keep_last

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        messages: list[dict] = payload.get("messages", [])
        # Use the running LLM's tokenizer when present so "4000 tokens"
        # means actual tokens rather than the ``chars//4`` heuristic.
        llm = ctx.llm
        if estimate_tokens(messages, llm) <= self.max_context_tokens:
            return None
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        tail = messages[-self.keep_last :] if self.keep_last > 0 else []
        # Iteratively drop from the middle until we fit.
        trimmed = [*([system] if system else []), *tail]
        while estimate_tokens(trimmed, llm) > self.max_context_tokens and trimmed:
            # Drop the oldest non-system from tail.
            if len(trimmed) > (1 if system else 0):
                idx = 1 if system else 0
                trimmed.pop(idx)
        payload = dict(payload)
        payload["messages"] = trimmed
        return HookResult.replace(payload, reason=f"truncated to {self.max_context_tokens} tokens")


class ToolOutputTruncatorHook:
    """Truncate oversized tool results before they rejoin the context.

    Fires at ``after_tool``. Rough size-based truncation is enough for
    SLMs that suffer most from long JSON blobs (web scraper, filesystem
    walk, sensor dumps).
    """

    points = frozenset({AFTER_TOOL})
    priority = 40  # mutation tier

    def __init__(self, *, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def __call__(self, point: str, ctx: AgentContext, payload: ToolCallResult) -> HookResult | None:
        result = payload.result
        if isinstance(result, str) and len(result) > self.max_chars:
            payload.result = result[: self.max_chars] + f"\n… (truncated {len(result) - self.max_chars} chars)"
            return HookResult.replace(payload, reason="truncated tool output")
        if isinstance(result, (dict, list)):
            as_str = json.dumps(result, default=str)
            if len(as_str) > self.max_chars:
                payload.result = as_str[: self.max_chars] + f"\n… (truncated {len(as_str) - self.max_chars} chars)"
                return HookResult.replace(payload, reason="truncated tool output")
        return None


class TracingHook:
    """Emit OTel-compatible ``span_start`` / ``span_end`` events.

    On ``ON_RUN_START`` the hook:
      * generates a ``trace_id`` if the context doesn't already have
        one (a parent agent may have set it, see handoff plumbing);
      * generates a fresh ``span_id`` for this agent turn, stores it
        on the context so ``_drive``'s subsequent ``ctx.emit`` calls
        tag events with it, and records the previous
        ``parent_span_id`` so we can restore it at turn end.
      * emits a ``span_start`` event with ``name=agent_name``,
        ``trace_id``, ``span_id``, ``parent_span_id``, and a
        ``start_ns`` timestamp payload.

    On ``ON_RUN_END`` it closes the span with a ``span_end`` event
    carrying ``duration_ns`` + the run's final reply length as an
    attribute. Span ids use a 16-char hex prefix of a UUID, matching
    OTel's span-id width.

    Intended to be cheap: stdlib only, no IO, no threading. Wire an
    OTel exporter separately via :func:`install_otel_bridge` (in the
    ``[observability]`` extra) — that subscribes to ``span_start`` /
    ``span_end`` events and re-emits them as real OTel spans.
    """

    points = frozenset({ON_RUN_START, ON_RUN_END})
    priority = 10  # observability tier

    def __init__(self, *, service_name: str = "edgevox") -> None:
        self.service_name = service_name

    def _make_id(self, *, bits: int) -> str:
        import secrets

        return secrets.token_hex(bits // 8)

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> None:
        if point == ON_RUN_START:
            state = ctx.hook_state.setdefault(id(self), {})
            if not ctx.trace_id:
                ctx.trace_id = self._make_id(bits=128)
            span_id = self._make_id(bits=64)
            # Remember the incoming parent so we can restore it at
            # turn-end — nested sub-agents rely on the parent_span_id
            # pointing at the *caller's* active span while they're
            # running.
            state["prior_parent"] = ctx.parent_span_id
            state["span_id"] = span_id
            state["start_ns"] = time.monotonic_ns()
            ctx.parent_span_id = span_id
            ctx.emit(
                "span_start",
                ctx.agent_name or "anon",
                {
                    "service.name": self.service_name,
                    "name": ctx.agent_name or "run",
                    "trace_id": ctx.trace_id,
                    "span_id": span_id,
                    "parent_span_id": state["prior_parent"],
                    "start_ns": state["start_ns"],
                },
            )
            return

        # ON_RUN_END
        state = ctx.hook_state.get(id(self), {})
        start_ns = state.get("start_ns")
        span_id = state.get("span_id")
        duration_ns = time.monotonic_ns() - start_ns if start_ns else 0
        reply = ""
        if isinstance(payload, dict):
            reply = payload.get("reply") or ""
        ctx.emit(
            "span_end",
            ctx.agent_name or "anon",
            {
                "service.name": self.service_name,
                "trace_id": ctx.trace_id,
                "span_id": span_id,
                "parent_span_id": state.get("prior_parent"),
                "duration_ns": duration_ns,
                "reply_len": len(reply),
            },
        )
        # Restore the parent span for any code that continues to reuse
        # this context after the run (rare but possible; e.g. chaining
        # two runs on one context in a synchronous script).
        ctx.parent_span_id = state.get("prior_parent")


class ToolErrorRetryHook:
    """Track per-tool runtime errors and nudge the model after N failures.

    The base ``_drive`` loop already feeds a tool's error string back to
    the model via the ``tool``-role message on the next hop, so a small
    model already gets one chance to correct itself. What this hook
    adds is:

    * **A retry budget** per ``(tool_name, agent_name)`` — after
      ``max_retries`` failures on the same tool within a single turn,
      we replace the tool result with an actionable guidance payload
      (``{"ok": false, "error": ..., "suggest": "...", "retries_exhausted": true}``)
      so the model stops hammering a broken backend.
    * **Stable retry-vs-different-approach guidance** — the message
      is stable across runs, making a looping SLM easier to observe.

    Complementary to ``SchemaRetryHook`` (which handles ``bad
    arguments:`` / argument-shape errors): this one handles *runtime*
    errors (``ConnectionError`` / ``TimeoutError`` / stockfish crashes
    / permission denials). If the result passes ``SchemaRetryHook``'s
    ``is_argument_shape_error`` check, this hook treats it as
    already-handled and keeps its budget untouched.

    Priority 40 — mutation tier. Registered after ``SchemaRetryHook``
    so arg-shape errors consume the schema-retry path first.
    """

    points = frozenset({ON_RUN_START, AFTER_TOOL})
    priority = 40

    def __init__(
        self,
        *,
        max_retries: int = 2,
        exhausted_suggestion: str = "This tool is failing. Try a different tool or answer without it.",
    ) -> None:
        self.max_retries = int(max_retries)
        self.exhausted_suggestion = exhausted_suggestion

    def _budget(self, ctx: AgentContext) -> dict[str, int]:
        state = ctx.hook_state.setdefault(id(self), {})
        return state.setdefault("budget", {})

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if point == ON_RUN_START:
            ctx.hook_state[id(self)] = {"budget": {}}
            return None

        outcome: ToolCallResult = payload
        if outcome.ok:
            # A successful call resets the retry budget for that tool —
            # one transient failure followed by success shouldn't
            # permanently lower the ceiling inside this turn.
            self._budget(ctx).pop(outcome.name, None)
            return None

        # Let SchemaRetryHook own arg-shape errors (it runs at the same
        # priority, later in the priority-tie order by registration, so
        # if the result made it here without being replaced it's a
        # runtime error worth counting).
        from edgevox.llm._agent_harness import is_argument_shape_error

        if is_argument_shape_error(outcome.error):
            return None

        budget = self._budget(ctx)
        used = budget.get(outcome.name, 0) + 1
        budget[outcome.name] = used

        if used < self.max_retries:
            # Under budget — let the default path continue. The model
            # will see ``{"ok": false, "error": ...}`` on the next hop
            # and can self-correct. No mutation needed.
            return None

        # Budget exhausted — replace the outcome so the model stops
        # calling this tool in a tight loop. Keep the original error
        # string so the model knows what went wrong, and append a
        # structured suggestion that's stable across runs.
        replacement = {
            "ok": False,
            "error": outcome.error,
            "retries_exhausted": True,
            "suggest": self.exhausted_suggestion,
        }
        outcome.result = replacement
        outcome.error = None
        return HookResult.replace(outcome, reason="tool retry budget exhausted")


class OutputValidatorHook:
    """Composable post-tool output validator / redactor.

    Runs at ``AFTER_TOOL`` (priority 30 — mutation tier, after
    ``ToolOutputTruncatorHook`` at 40) and applies a pipeline of
    validators to ``ToolCallResult.result``. Each validator is a
    callable ``(value) -> value`` that can transform, redact, or
    raise ``ValidatorError`` to reject the result.

    Built-in validators ship as free functions in this module:

    * :func:`length_cap(max_chars)` — hard cap on str / json-dumped
      size, complementing ``ToolOutputTruncatorHook`` with a stricter
      budget for specific tools.
    * :func:`pii_redactor(patterns)` — regex-based replacement of
      common PII (email, phone, SSN, ...) with ``[REDACTED]``.
    * :func:`schema_check(json_schema)` — re-validate the output
      against a JSON schema fragment; rejects on mismatch so a
      hostile tool can't smuggle arbitrary data back into the LLM
      context.

    Usage::

        hook = OutputValidatorHook(validators=[
            length_cap(1500),
            pii_redactor(),
            schema_check({"type": "object", "required": ["status"]}),
        ])

    Validators are applied in list order. Any that raises
    ``ValidatorError`` replaces the outcome with ``{"ok": False,
    "error": "output validation: <reason>"}`` so the model can see
    why its tool result was rejected.
    """

    points = frozenset({AFTER_TOOL})
    priority = 30  # mutation tier, after truncation at 40

    def __init__(self, *, validators: list[Callable[[Any], Any]]) -> None:
        self.validators = list(validators)

    def __call__(self, point: str, ctx: AgentContext, payload: ToolCallResult) -> HookResult | None:
        if not payload.ok or not self.validators:
            return None
        current = payload.result
        for validator in self.validators:
            try:
                current = validator(current)
            except ValidatorError as e:
                payload.result = None
                payload.error = f"output validation: {e}"
                return HookResult.replace(payload, reason=f"output rejected: {e}")
        if current is payload.result:
            return None
        payload.result = current
        return HookResult.replace(payload, reason="output transformed by validators")


class ValidatorError(ValueError):
    """Raised by validators passed to :class:`OutputValidatorHook` to
    reject a tool's output wholesale. Distinguished from ``ValueError``
    so unrelated ``ValueError``-raising transforms aren't swallowed."""


def length_cap(max_chars: int) -> Callable[[Any], Any]:
    """Validator: hard cap total output size, dict/list via JSON dump.

    Different from :class:`ToolOutputTruncatorHook` in two ways:
    (a) you can set a *lower* cap for specific tools via a targeted
    validator pipeline, and (b) it raises ``ValidatorError`` when the
    capped output still exceeds a sensible bound — catching
    pathologically huge outputs that shouldn't just be silently
    truncated (streaming a 10 MB blob into the LLM context is a bug,
    not a truncation case).
    """

    def _apply(value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[:max_chars] + f"\n… (capped at {max_chars} chars)"
        if isinstance(value, (dict, list)):
            dumped = json.dumps(value, default=str)
            if len(dumped) <= max_chars:
                return value
            # Reject wholesale — a huge dict that we'd have to truncate
            # by re-parsing is almost always a sign of a misused tool
            # (scraper returned the whole page, filesystem walk returned
            # 100k paths, etc.). Surfacing the error lets the model
            # switch to a more scoped call.
            raise ValidatorError(f"output size {len(dumped)} > {max_chars} (too large; narrow the query)")
        return value

    return _apply


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("phone_us", re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)


def pii_redactor(
    patterns: tuple[tuple[str, re.Pattern[str]], ...] | None = None,
    *,
    marker: str = "[REDACTED]",
) -> Callable[[Any], Any]:
    """Validator: replace matches of common PII regexes with ``marker``.

    Default pattern set covers email, US phone, SSN, credit-card
    sequences. Pass your own tuple of ``(label, compiled_pattern)``
    pairs to cover domain-specific PII (employee id, patient mrn,
    dosage codes, ...). Matches inside nested dicts / lists are
    redacted recursively.
    """

    active = patterns or _PII_PATTERNS

    def _redact_str(s: str) -> str:
        for _label, pat in active:
            s = pat.sub(marker, s)
        return s

    def _apply(value: Any) -> Any:
        if isinstance(value, str):
            return _redact_str(value)
        if isinstance(value, dict):
            return {k: _apply(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_apply(v) for v in value]
        return value

    return _apply


def schema_check(json_schema: dict[str, Any]) -> Callable[[Any], Any]:
    """Validator: re-validate the output against a JSON schema fragment.

    Uses the same minimal JSON-schema subset as
    ``edgevox.llm.tools._validate_against_schema`` — ``type: object`` +
    flat properties + ``required`` + ``enum``. Raises
    ``ValidatorError`` on mismatch.
    """

    from edgevox.llm.tools import _validate_against_schema

    def _apply(value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValidatorError(f"expected object, got {type(value).__name__}")
        err = _validate_against_schema(value, json_schema)
        if err:
            raise ValidatorError(err)
        return value

    return _apply


class ContextWindowManager:
    """Unified context-window manager. Replaces three hand-composed
    hooks with one state-machine that owns the budget end-to-end.

    Strategy (Anthropic-style tiered compaction):

    1. ``after_tool`` — truncate oversized **tool result payloads**
       in-place. Cheap, safe mid-turn, biggest single byte saving on
       most SLM workloads (sensor dumps, web scrapes).
    2. ``before_llm`` — if the messages list is still over budget,
       drop the *bodies* of older tool-result messages while keeping
       the call history (the LLM still knows what it called and why).
       Also safe mid-turn: tool-call chains stay intact.
    3. ``on_run_start`` — only here, between turns, do we run the
       LLM-summary :class:`Compactor`. Mid-turn summarisation breaks
       tool-call chains and would invalidate the streamed `messages`.

    All three layers use the typed ``ctx.llm`` to count tokens
    exactly when available, falling back to the ``chars // 4``
    heuristic otherwise.

    Replaces the prior bundle of
    ``TokenBudgetHook + ToolOutputTruncatorHook + ContextCompactionHook``
    — keep using those individually if you only want one of the layers,
    or use this one for the recommended default.
    """

    points = frozenset({ON_RUN_START, BEFORE_LLM, AFTER_TOOL})
    priority = 40  # mutation tier (alongside other context-mutating hooks)

    def __init__(
        self,
        *,
        max_context_tokens: int = 4000,
        keep_last: int = 4,
        tool_output_max_chars: int = 2000,
        compactor: Compactor | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.keep_last = keep_last
        self.tool_output_max_chars = tool_output_max_chars
        # Compactor is optional — drop it for ``hard truncation only``
        # deployments that don't want to spend an extra LLM call between
        # turns.
        self.compactor = compactor

    # ----- ON_RUN_START: full LLM summarisation between turns -----

    def _on_run_start(self, ctx: AgentContext) -> None:
        if self.compactor is None:
            return
        session = ctx.session
        if not session.messages:
            return
        llm = ctx.llm or ctx.state.get("__llm__")
        stop_event = (
            ctx.interrupt.cancel_token if ctx.interrupt is not None and ctx.interrupt.policy.cancel_llm else None
        )
        new_messages = self.compactor.compact(session.messages, llm, stop_event=stop_event)
        if new_messages is not session.messages and new_messages != session.messages:
            session.messages[:] = new_messages
            log.info("ContextWindowManager: compacted to %d messages", len(new_messages))

    # ----- BEFORE_LLM: drop tool-result bodies, then hard truncate -----

    def _before_llm(self, ctx: AgentContext, payload: dict) -> HookResult | None:
        messages: list[dict] = list(payload.get("messages") or [])
        llm = ctx.llm
        if not messages or estimate_tokens(messages, llm) <= self.max_context_tokens:
            return None

        modified = False
        # Stage 1: blank out older tool-result bodies (keep envelope so
        # the call history stays parseable).
        for i, m in enumerate(messages[: -self.keep_last] if self.keep_last > 0 else messages):
            if m.get("role") == "tool" and m.get("content"):
                messages[i] = {**m, "content": "(truncated by context window manager)"}
                modified = True
                if estimate_tokens(messages, llm) <= self.max_context_tokens:
                    break

        # Stage 2: still over budget → preserve system + last keep_last,
        # drop the middle. This is the hard safety net.
        if estimate_tokens(messages, llm) > self.max_context_tokens:
            system = messages[0] if messages and messages[0].get("role") == "system" else None
            tail = messages[-self.keep_last :] if self.keep_last > 0 else []
            trimmed = [*([system] if system else []), *tail]
            while estimate_tokens(trimmed, llm) > self.max_context_tokens and len(trimmed) > (1 if system else 0):
                idx = 1 if system else 0
                trimmed.pop(idx)
            if trimmed != messages:
                messages = trimmed
                modified = True

        if not modified:
            return None
        out = dict(payload)
        out["messages"] = messages
        return HookResult.replace(out, reason="context window managed")

    # ----- AFTER_TOOL: truncate oversized payloads in place -----

    def _after_tool(self, payload: ToolCallResult) -> HookResult | None:
        result = payload.result
        if isinstance(result, str) and len(result) > self.tool_output_max_chars:
            payload.result = (
                result[: self.tool_output_max_chars]
                + f"\n… (truncated {len(result) - self.tool_output_max_chars} chars)"
            )
            return HookResult.replace(payload, reason="tool output truncated")
        if isinstance(result, (dict, list)):
            as_str = json.dumps(result, default=str)
            if len(as_str) > self.tool_output_max_chars:
                payload.result = (
                    as_str[: self.tool_output_max_chars]
                    + f"\n… (truncated {len(as_str) - self.tool_output_max_chars} chars)"
                )
                return HookResult.replace(payload, reason="tool output truncated")
        return None

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if point == ON_RUN_START:
            self._on_run_start(ctx)
            return None
        if point == BEFORE_LLM:
            return self._before_llm(ctx, payload)
        if point == AFTER_TOOL:
            return self._after_tool(payload)
        return None


# ===========================================================================
# Memory + compaction hooks
# ===========================================================================


class MemoryInjectionHook:
    """Inject :class:`MemoryStore`-rendered facts/episodes into the system prompt.

    Fires at ``before_llm``. Modifies the first system message of the
    payload's ``messages`` list by appending a ``## Memory`` block.
    Safe on empty memory (no-op) and idempotent across tool hops:
    the marker check runs **before** ``render_for_prompt`` so the
    hook's cost on subsequent hops within the same turn is a single
    substring check.
    """

    points = frozenset({BEFORE_LLM})

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        max_facts: int = 20,
        max_episodes: int = 5,
        header: str = "\n\n## Memory\n",
    ) -> None:
        self.memory_store = memory_store
        self.max_facts = max_facts
        self.max_episodes = max_episodes
        self.header = header

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        messages: list[dict] = list(payload.get("messages", []))
        if not messages or messages[0].get("role") != "system":
            return None
        base = messages[0].get("content") or ""
        marker = self.header.strip()
        # Idempotent: if we already injected this turn, short-circuit
        # before paying for memory rendering.
        if marker and marker in base:
            return None
        rendered = self.memory_store.render_for_prompt(
            max_facts=self.max_facts,
            max_episodes=self.max_episodes,
        )
        if not rendered:
            return None
        system = dict(messages[0])
        system["content"] = f"{base}{self.header}{rendered}"
        messages[0] = system
        payload = dict(payload)
        payload["messages"] = messages
        return HookResult.replace(payload, reason="memory injected")


class NotesInjectorHook:
    """Append the tail of a :class:`NotesFile` to the system prompt.

    Cheap long-term working memory: agents write notes via a tool,
    this hook re-feeds the most recent ones each turn.
    """

    points = frozenset({BEFORE_LLM})

    def __init__(self, notes: NotesFile, *, max_chars: int = 1500, header: str = "\n\n## Notes\n") -> None:
        self.notes = notes
        self.max_chars = max_chars
        self.header = header

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        messages: list[dict] = list(payload.get("messages", []))
        if not messages or messages[0].get("role") != "system":
            return None
        base = messages[0].get("content") or ""
        marker = self.header.strip()
        # Idempotent: short-circuit before re-reading notes on every hop.
        if marker and marker in base:
            return None
        tail = self.notes.tail(self.max_chars)
        if not tail.strip():
            return None
        system = dict(messages[0])
        system["content"] = f"{base}{self.header}{tail}"
        messages[0] = system
        payload = dict(payload)
        payload["messages"] = messages
        return HookResult.replace(payload, reason="notes injected")


class ContextCompactionHook:
    """Summarize old turns when the session crosses a token budget.

    Fires at ``on_run_start`` (between turns, never mid-turn — would
    break tool-call chains). Uses the same LLM bound to the agent via
    the context so no extra model needs to load.
    """

    points = frozenset({ON_RUN_START})

    def __init__(self, compactor: Compactor, *, llm_getter: Callable[[AgentContext], LLM | None] | None = None) -> None:
        self.compactor = compactor
        self.llm_getter = llm_getter

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        session = ctx.session
        if not session.messages:
            return None
        # Prefer the typed ``ctx.llm`` field; the legacy ``__llm__``
        # scratchpad key remains as a back-compat fallback for older
        # hook wiring.
        llm = self.llm_getter(ctx) if self.llm_getter else (ctx.llm or ctx.state.get("__llm__"))
        stop_event = (
            ctx.interrupt.cancel_token if ctx.interrupt is not None and ctx.interrupt.policy.cancel_llm else None
        )
        new_messages = self.compactor.compact(session.messages, llm, stop_event=stop_event)
        if new_messages is not session.messages and new_messages != session.messages:
            session.messages[:] = new_messages
            log.info("Context compacted: %d messages remain", len(new_messages))
        return None


# ===========================================================================
# Episode + audit + persistence
# ===========================================================================


class EpisodeLoggerHook:
    """Record every tool/skill outcome into a :class:`MemoryStore` as an episode."""

    points = frozenset({AFTER_TOOL})

    def __init__(self, memory_store: MemoryStore, *, agent_name: str = "") -> None:
        self.memory_store = memory_store
        self.agent_name = agent_name

    def __call__(self, point: str, ctx: AgentContext, payload: ToolCallResult) -> HookResult | None:
        outcome = "ok" if payload.ok else "failed"
        self.memory_store.add_episode(
            kind="tool_call",
            payload={"name": payload.name, "args": payload.arguments, "result": payload.result, "error": payload.error},
            outcome=outcome,
            agent=self.agent_name,
        )
        return None


@dataclass
class AuditLogHook:
    """Append-only JSONL log of fire-point events. Great for debugging
    SLM behavior offline — later you can replay and see exactly what the
    hook chain observed.
    """

    path: str | Path
    points: frozenset[str] = field(default_factory=lambda: frozenset({AFTER_LLM, AFTER_TOOL, ON_RUN_END}))

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        record = {
            "ts": time.time(),
            "point": point,
            "agent": getattr(ctx, "agent_name", ""),
            "payload": _safe_json(payload),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            log.exception("AuditLog write failed")
        return None


class PersistSessionHook:
    """Persist the session via a :class:`SessionStore` at end-of-run.

    Keyed by ``session_id``. A short-lived agent that should not persist
    (e.g. a router) can omit this hook.
    """

    points = frozenset({ON_RUN_END})

    def __init__(self, session_store: SessionStore, *, session_id: str) -> None:
        self.session_store = session_store
        self.session_id = session_id

    def __call__(self, point: str, ctx: AgentContext, payload: AgentResult) -> HookResult | None:
        try:
            self.session_store.save(self.session_id, ctx.session)
        except Exception:
            log.exception("PersistSessionHook save failed")
        return None


# ===========================================================================
# Misc useful hooks
# ===========================================================================


class TimingHook:
    """Record wall-clock time per fire point. Useful for profiling latency."""

    points = frozenset({BEFORE_LLM, AFTER_LLM, BEFORE_TOOL, AFTER_TOOL})

    def __init__(self) -> None:
        self.timings: list[tuple[str, float]] = []
        self._starts: dict[str, float] = {}

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        now = time.perf_counter()
        if point in (BEFORE_LLM, BEFORE_TOOL):
            self._starts[point] = now
        elif point == AFTER_LLM and BEFORE_LLM in self._starts:
            self.timings.append(("llm", now - self._starts.pop(BEFORE_LLM)))
        elif point == AFTER_TOOL and BEFORE_TOOL in self._starts:
            self.timings.append(("tool", now - self._starts.pop(BEFORE_TOOL)))
        return None


class EchoingHook:
    """Print every fire point — useful when debugging hook composition."""

    points = frozenset(
        {
            ON_RUN_START,
            BEFORE_LLM,
            AFTER_LLM,
            BEFORE_TOOL,
            AFTER_TOOL,
            ON_RUN_END,
        }
    )

    def __init__(self, logger: Callable[[str], None] = print) -> None:
        self.logger = logger

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        preview = _preview(payload)
        self.logger(f"[hook:{point}] {preview}")
        return None


class DebugTapHook:
    """Tap the agent loop and publish the payloads as structured events.

    Emits three payloads per turn:

    - ``{"kind": "messages", "messages": [...]}`` at ``before_llm`` — the
      full messages array the LLM is about to see, *after* every upstream
      hook has mutated it. Lets you verify briefings, memory blocks, and
      system prompts landed as intended.
    - ``{"kind": "raw_reply", "text": str}`` at ``after_llm`` — the
      model's unmodified output before downstream sanitisers
      (think-tag stripping, voice cleanup, sentence clipping) run.
    - ``{"kind": "final_reply", "text": str}`` at ``on_run_end`` — the
      sanitised text that the user actually sees / hears.

    Events are published via ``ctx.emit(event_kind, source, payload)`` so
    any UI — Qt chat bubble, TUI panel, WebSocket debug pane, plain log
    — can subscribe without knowing about the hook itself.

    The hook is intended to always be installed. Pass a ``enabled``
    predicate (callable) to gate emission live without rebuilding the
    agent; when ``enabled`` is off the hook does a single attribute read
    per fire point and returns.
    """

    points = frozenset({BEFORE_LLM, AFTER_LLM, ON_RUN_END})

    def __init__(
        self,
        *,
        enabled: bool | Callable[[], bool] = True,
        event_kind: str = "agent_debug",
        source: str = "debug",
    ) -> None:
        self._enabled = enabled
        self.event_kind = event_kind
        self.source = source

    def _is_on(self) -> bool:
        e = self._enabled
        return bool(e() if callable(e) else e)

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if not self._is_on():
            return None
        try:
            if point == BEFORE_LLM and isinstance(payload, dict):
                messages = payload.get("messages") or []
                ctx.emit(self.event_kind, self.source, {"kind": "messages", "messages": messages})
            elif point == AFTER_LLM and isinstance(payload, dict):
                raw = payload.get("content") or payload.get("reply") or ""
                if raw:
                    ctx.emit(self.event_kind, self.source, {"kind": "raw_reply", "text": str(raw)})
            elif point == ON_RUN_END and isinstance(payload, dict):
                reply = payload.get("reply")
                if reply:
                    ctx.emit(self.event_kind, self.source, {"kind": "final_reply", "text": str(reply)})
        except Exception:
            log.exception("DebugTapHook failed (non-fatal)")
        return None


# ===========================================================================
# Helpers
# ===========================================================================


def _safe_json(obj: Any) -> Any:
    """Best-effort JSON-serializable rendering for audit logs."""
    if hasattr(obj, "__dataclass_fields__"):
        try:
            return asdict(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _preview(obj: Any, max_chars: int = 120) -> str:
    try:
        s = json.dumps(_safe_json(obj), default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


__all__ = [
    "AuditLogHook",
    "ContextCompactionHook",
    "ContextWindowManager",
    "DebugTapHook",
    "EchoingHook",
    "EpisodeLoggerHook",
    "MemoryInjectionHook",
    "NotesInjectorHook",
    "PersistSessionHook",
    "PlanModeHook",
    "SafetyGuardrailHook",
    "TimingHook",
    "TokenBudgetHook",
    "ToolOutputTruncatorHook",
    "console_approver",
]
