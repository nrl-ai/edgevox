"""Tests for Tier-1 / Tier-2 agentic-practice upgrades:

1. ``ToolErrorRetryHook`` — runtime-error retry budget per tool.
2. ``OutputValidatorHook`` + ``length_cap`` / ``pii_redactor`` /
   ``schema_check``.
3. ``TracingHook`` — ``span_start`` / ``span_end`` events with
   ``trace_id`` / ``span_id`` / ``parent_span_id`` propagation.
4. ``Compactor(preserve_tool_calls=True)`` — selective compaction
   that keeps a tool-trace audit message.
5. Prompt-caching helpers: ``canonicalise_messages``,
   ``stable_tool_schemas``, ``tool_schema_fingerprint``, and
   ``ToolRegistry.fingerprint()``.
"""

from __future__ import annotations

from typing import Any

import pytest

from edgevox.agents import (
    AFTER_TOOL,
    ON_RUN_END,
    ON_RUN_START,
    AgentContext,
    Compactor,
    OutputValidatorHook,
    ToolErrorRetryHook,
    TracingHook,
    ValidatorError,
    length_cap,
    pii_redactor,
    schema_check,
)
from edgevox.agents.hooks import HookAction
from edgevox.llm.prompt_cache import (
    canonicalise_messages,
    stable_tool_schemas,
    tool_schema_fingerprint,
)
from edgevox.llm.tools import Tool, ToolCallResult, ToolRegistry

# ---------------------------------------------------------------------------
# 1. ToolErrorRetryHook
# ---------------------------------------------------------------------------


class TestToolErrorRetry:
    def _fire_failure(self, hook: ToolErrorRetryHook, ctx: AgentContext, name: str, err: str):
        outcome = ToolCallResult(name=name, arguments={}, error=err)
        return hook(AFTER_TOOL, ctx, outcome), outcome

    def test_on_run_start_resets_budget(self):
        hook = ToolErrorRetryHook()
        ctx = AgentContext()
        ctx.hook_state[id(hook)] = {"budget": {"get_temp": 99}}
        hook(ON_RUN_START, ctx, None)
        assert ctx.hook_state[id(hook)]["budget"] == {}

    def test_under_budget_is_passthrough(self):
        hook = ToolErrorRetryHook(max_retries=3)
        ctx = AgentContext()
        hook(ON_RUN_START, ctx, None)
        # First two runtime errors: no mutation, model sees the raw error.
        for _ in range(2):
            result, outcome = self._fire_failure(hook, ctx, "get_temp", "ConnectionError")
            assert result is None
            assert outcome.error == "ConnectionError"

    def test_exhausted_budget_replaces_outcome(self):
        hook = ToolErrorRetryHook(max_retries=2)
        ctx = AgentContext()
        hook(ON_RUN_START, ctx, None)
        self._fire_failure(hook, ctx, "get_temp", "ConnectionError")
        result, outcome = self._fire_failure(hook, ctx, "get_temp", "ConnectionError")
        assert result is not None
        assert result.action is HookAction.MODIFY
        assert outcome.result["retries_exhausted"] is True
        assert "different tool" in outcome.result["suggest"]
        assert outcome.error is None  # turned into a structured suggestion

    def test_successful_call_resets_counter(self):
        hook = ToolErrorRetryHook(max_retries=2)
        ctx = AgentContext()
        hook(ON_RUN_START, ctx, None)
        self._fire_failure(hook, ctx, "get_temp", "boom")
        hook(AFTER_TOOL, ctx, ToolCallResult(name="get_temp", arguments={}, result=22))
        # Fresh failure after success should be back to "first failure".
        result, _ = self._fire_failure(hook, ctx, "get_temp", "boom")
        assert result is None  # still under budget after reset

    def test_arg_shape_errors_pass_through(self):
        """Schema-shape errors are handled by ``SchemaRetryHook``; this
        hook must not consume their budget or interfere."""
        hook = ToolErrorRetryHook(max_retries=1)
        ctx = AgentContext()
        hook(ON_RUN_START, ctx, None)
        # ``bad arguments:`` prefix is the schema-shape signal.
        result, outcome = self._fire_failure(hook, ctx, "set_light", "bad arguments: missing required field 'name'")
        assert result is None
        assert outcome.error and outcome.error.startswith("bad arguments:")
        assert ctx.hook_state[id(hook)]["budget"] == {}  # no budget used

    def test_budgets_are_per_tool(self):
        hook = ToolErrorRetryHook(max_retries=2)
        ctx = AgentContext()
        hook(ON_RUN_START, ctx, None)
        self._fire_failure(hook, ctx, "tool_a", "err")
        self._fire_failure(hook, ctx, "tool_b", "err")
        # tool_a has 1 failure, tool_b has 1 failure — both under budget.
        assert ctx.hook_state[id(hook)]["budget"] == {"tool_a": 1, "tool_b": 1}


# ---------------------------------------------------------------------------
# 2. OutputValidatorHook
# ---------------------------------------------------------------------------


class TestOutputValidator:
    def test_length_cap_string_is_truncated(self):
        hook = OutputValidatorHook(validators=[length_cap(20)])
        ctx = AgentContext()
        outcome = ToolCallResult(name="t", arguments={}, result="a" * 50)
        result = hook(AFTER_TOOL, ctx, outcome)
        assert result is not None and result.action is HookAction.MODIFY
        assert len(outcome.result) <= 60  # capped + trailing marker

    def test_length_cap_dict_rejects_too_large(self):
        hook = OutputValidatorHook(validators=[length_cap(50)])
        ctx = AgentContext()
        huge = {"items": list(range(200))}
        outcome = ToolCallResult(name="t", arguments={}, result=huge)
        result = hook(AFTER_TOOL, ctx, outcome)
        assert result is not None
        assert "output validation" in outcome.error
        assert "too large" in outcome.error

    def test_pii_redactor_swaps_emails_and_phones(self):
        hook = OutputValidatorHook(validators=[pii_redactor()])
        ctx = AgentContext()
        outcome = ToolCallResult(
            name="t",
            arguments={},
            result="contact Ada at ada@example.com or 555-123-4567",
        )
        hook(AFTER_TOOL, ctx, outcome)
        assert "[REDACTED]" in outcome.result
        assert "@example.com" not in outcome.result
        assert "555" not in outcome.result

    def test_pii_redactor_walks_nested_structures(self):
        hook = OutputValidatorHook(validators=[pii_redactor()])
        ctx = AgentContext()
        outcome = ToolCallResult(
            name="t",
            arguments={},
            result={
                "contacts": [
                    {"email": "ada@example.com"},
                    {"note": "call 555-000-1111"},
                ]
            },
        )
        hook(AFTER_TOOL, ctx, outcome)
        assert outcome.result["contacts"][0]["email"] == "[REDACTED]"
        assert "[REDACTED]" in outcome.result["contacts"][1]["note"]

    def test_schema_check_passes_valid(self):
        hook = OutputValidatorHook(
            validators=[
                schema_check(
                    {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    }
                )
            ]
        )
        ctx = AgentContext()
        outcome = ToolCallResult(name="t", arguments={}, result={"status": "ok"})
        result = hook(AFTER_TOOL, ctx, outcome)
        assert result is None

    def test_schema_check_rejects_missing_required(self):
        hook = OutputValidatorHook(
            validators=[
                schema_check(
                    {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    }
                )
            ]
        )
        ctx = AgentContext()
        outcome = ToolCallResult(name="t", arguments={}, result={"unrelated": 1})
        hook(AFTER_TOOL, ctx, outcome)
        assert outcome.error is not None
        assert "output validation" in outcome.error

    def test_validators_compose_in_order(self):
        """PII redaction must run before length cap when both are
        present, so a redacted string doesn't accidentally become an
        over-cap one."""
        redactor_then_cap = OutputValidatorHook(validators=[pii_redactor(), length_cap(80)])
        ctx = AgentContext()
        outcome = ToolCallResult(
            name="t",
            arguments={},
            result="email ada@example.com and phone 555-123-4567 " + "x" * 50,
        )
        redactor_then_cap(AFTER_TOOL, ctx, outcome)
        assert "[REDACTED]" in outcome.result

    def test_failed_tool_is_not_validated(self):
        """If the tool already failed, validators shouldn't run — no
        point redacting / validating ``None``."""
        hook = OutputValidatorHook(validators=[schema_check({"type": "object"})])
        ctx = AgentContext()
        outcome = ToolCallResult(name="t", arguments={}, error="boom")
        result = hook(AFTER_TOOL, ctx, outcome)
        assert result is None

    def test_validator_error_is_raisable_by_custom_validators(self):
        def my_validator(value):
            raise ValidatorError("custom reason")

        hook = OutputValidatorHook(validators=[my_validator])
        ctx = AgentContext()
        outcome = ToolCallResult(name="t", arguments={}, result={"anything": 1})
        hook(AFTER_TOOL, ctx, outcome)
        assert outcome.error is not None
        assert "custom reason" in outcome.error


# ---------------------------------------------------------------------------
# 3. TracingHook
# ---------------------------------------------------------------------------


class TestTracingHook:
    def test_root_trace_id_is_generated(self):
        hook = TracingHook()
        ctx = AgentContext()
        seen: list = []
        ctx.bus.subscribe_all(seen.append)
        hook(ON_RUN_START, ctx, None)
        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 32  # 128-bit hex
        assert ctx.parent_span_id is not None
        assert len(ctx.parent_span_id) == 16  # 64-bit hex
        [evt] = [e for e in seen if e.kind == "span_start"]
        assert evt.payload["trace_id"] == ctx.trace_id

    def test_inherited_trace_id_is_preserved(self):
        """When a parent agent already set ``trace_id``, the sub-agent
        must reuse it (not generate a new one) so the whole turn is
        one connected trace."""
        hook = TracingHook()
        ctx = AgentContext()
        ctx.trace_id = "a" * 32
        ctx.parent_span_id = "b" * 16
        hook(ON_RUN_START, ctx, None)
        assert ctx.trace_id == "a" * 32
        # The hook pushes its own span_id as the new parent — the
        # saved "prior_parent" in state is the inherited b*16.
        state = ctx.hook_state[id(hook)]
        assert state["prior_parent"] == "b" * 16

    def test_span_end_records_duration_and_reply_len(self):
        hook = TracingHook()
        ctx = AgentContext()
        events: list = []
        ctx.bus.subscribe_all(events.append)
        hook(ON_RUN_START, ctx, None)
        hook(ON_RUN_END, ctx, {"reply": "hello world"})
        [end] = [e for e in events if e.kind == "span_end"]
        assert end.payload["duration_ns"] > 0
        assert end.payload["reply_len"] == len("hello world")

    def test_nested_spans_restore_parent(self):
        """Running two sequential turns on the same context should
        restore ``parent_span_id`` after each turn ends."""
        hook = TracingHook()
        ctx = AgentContext()
        # Prior state: no parent set.
        hook(ON_RUN_START, ctx, None)
        first_span = ctx.parent_span_id
        hook(ON_RUN_END, ctx, {"reply": "1"})
        assert ctx.parent_span_id is None
        # Second turn gets a fresh span.
        hook(ON_RUN_START, ctx, None)
        assert ctx.parent_span_id != first_span


# ---------------------------------------------------------------------------
# 4. Selective compaction
# ---------------------------------------------------------------------------


class TestSelectiveCompaction:
    def _msgs_with_tool_chain(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            # Long conversational middle
            *[{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 200} for i in range(12)],
            # Tool chain
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": '{"temp":22,"conditions":"sunny"}',
            },
            # Keep a few tail turns verbatim
            *[{"role": "assistant", "content": f"tail {i}"} for i in range(3)],
        ]

    def test_default_compactor_summarises_everything(self):
        c = Compactor(trigger_tokens=100, keep_last_turns=2)
        msgs = self._msgs_with_tool_chain()
        out = c.compact(msgs, llm=None)
        # One assistant summary + system + last 2 turns.
        assert out[0]["role"] == "system"
        assert "summary" in out[1]["content"].lower()
        # Tool-trace block is NOT separated out without the flag.
        assert not any("tool-call trace" in (m.get("content") or "") for m in out)

    def test_preserve_tool_calls_adds_audit_block(self):
        c = Compactor(trigger_tokens=100, keep_last_turns=2, preserve_tool_calls=True)
        msgs = self._msgs_with_tool_chain()
        out = c.compact(msgs, llm=None)
        # Expect a summary block AND an audit block.
        assert any("summary" in (m.get("content") or "").lower() for m in out)
        assert any("tool-call trace" in (m.get("content") or "") for m in out)
        # The audit block contains the call and its result.
        trace = next(m["content"] for m in out if "tool-call trace" in m["content"])
        assert "get_weather" in trace
        assert "sunny" in trace

    def test_below_trigger_returns_unchanged(self):
        c = Compactor(trigger_tokens=10_000, preserve_tool_calls=True)
        msgs = [{"role": "user", "content": "hi"}]
        assert c.compact(msgs, llm=None) == msgs


# ---------------------------------------------------------------------------
# 5. Prompt-caching helpers
# ---------------------------------------------------------------------------


class TestPromptCache:
    def test_canonicalise_sorts_dict_keys(self):
        a = canonicalise_messages([{"b": 2, "a": 1}])
        assert list(a[0].keys()) == ["a", "b"]

    def test_canonicalise_is_idempotent(self):
        once = canonicalise_messages([{"c": {"y": 2, "x": 1}}])
        twice = canonicalise_messages(once)
        assert once == twice

    def test_canonicalise_does_not_mutate_input(self):
        msgs = [{"z": 1, "a": 2}]
        original = {"z": 1, "a": 2}
        canonicalise_messages(msgs)
        assert msgs[0] == original  # untouched

    def test_stable_tool_schemas_reorders_keys(self):
        out = stable_tool_schemas([{"type": "object", "properties": {"x": {"type": "string"}}}])
        assert list(out[0].keys()) == ["properties", "type"]

    def test_fingerprint_is_order_independent(self):
        a = [{"type": "object", "name": "foo"}]
        b = [{"name": "foo", "type": "object"}]
        assert tool_schema_fingerprint(a) == tool_schema_fingerprint(b)

    def test_fingerprint_changes_when_schemas_differ(self):
        a = [{"type": "object", "properties": {"x": {"type": "string"}}}]
        b = [{"type": "object", "properties": {"x": {"type": "integer"}}}]
        assert tool_schema_fingerprint(a) != tool_schema_fingerprint(b)

    def test_tool_registry_fingerprint(self):
        reg_a = ToolRegistry()
        reg_b = ToolRegistry()
        tool = Tool(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )
        reg_a.register(tool)
        reg_b.register(tool)
        assert reg_a.fingerprint() == reg_b.fingerprint()

    def test_tool_registry_fingerprint_changes_after_register(self):
        reg = ToolRegistry()
        empty_fp = reg.fingerprint()
        reg.register(
            Tool(
                name="t",
                description="t",
                parameters={"type": "object", "properties": {}},
                func=lambda: None,
            )
        )
        assert reg.fingerprint() != empty_fp


# ---------------------------------------------------------------------------
# OTel bridge (optional dep)
# ---------------------------------------------------------------------------


class TestOTelBridge:
    def test_missing_otel_returns_silently(self, monkeypatch):
        """If ``opentelemetry`` isn't installed, ``install_otel_bridge``
        must log-and-return without raising so apps opting in by
        default degrade gracefully."""
        import sys

        # Simulate the missing module by removing it from sys.modules
        # if present, and blocking import.
        real_otel = sys.modules.pop("opentelemetry", None)

        class _Blocker:
            def find_module(self, name, path=None):
                if name.startswith("opentelemetry"):
                    return self

            def load_module(self, name):
                raise ModuleNotFoundError(name)

        try:
            sys.meta_path.insert(0, _Blocker())
            from edgevox.agents.bus import EventBus
            from edgevox.agents.tracing_otel import _reset_bridge_state, install_otel_bridge

            _reset_bridge_state()
            bus = EventBus()
            # Must not raise:
            install_otel_bridge(bus)
        finally:
            # Restore sys.meta_path
            sys.meta_path = [m for m in sys.meta_path if not isinstance(m, _Blocker)]
            if real_otel is not None:
                sys.modules["opentelemetry"] = real_otel

    def test_bridge_creates_spans_when_otel_available(self):
        pytest.importorskip("opentelemetry")
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from edgevox.agents.bus import EventBus
        from edgevox.agents.tracing_otel import _reset_bridge_state, install_otel_bridge

        _reset_bridge_state()
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        bus = EventBus()
        install_otel_bridge(bus, service_name="test")

        hook = TracingHook(service_name="test")
        ctx = AgentContext(bus=bus)
        hook(ON_RUN_START, ctx, None)
        hook(ON_RUN_END, ctx, {"reply": "done"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("edgevox.span_id") is not None
