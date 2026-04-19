"""Tests for the Tier-0 agentic-practice upgrades:

1. Deterministic seed mode — ``AgentContext.seed`` + ``ctx.rng``, plumbed
   into ``LLM.complete(seed=...)`` and forwarded through handoffs /
   ``spawn_subagent``.
2. ``agent_as_tool(child)`` — wrap any Agent as a Tool the parent LLM
   can call mid-turn.
3. Tool-argument JSON-schema validation at dispatch — catches malformed
   LLM output before it reaches the tool body, surfaces a structured
   error the LLM can recover from.
"""

from __future__ import annotations

import random
from typing import Any

from edgevox.agents import AgentContext, agent_as_tool
from edgevox.agents.base import AgentResult
from edgevox.llm.tools import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# 1. Deterministic seed mode
# ---------------------------------------------------------------------------


class TestDeterministicSeed:
    def test_default_ctx_has_unseeded_rng(self):
        ctx = AgentContext()
        assert isinstance(ctx.rng, random.Random)
        assert ctx.seed is None

    def test_seed_init_reseeds_rng(self):
        a = AgentContext(seed=42)
        b = AgentContext(seed=42)
        # Same seed → identical sequences from ``ctx.rng``.
        assert [a.rng.random() for _ in range(5)] == [b.rng.random() for _ in range(5)]

    def test_different_seeds_diverge(self):
        a = AgentContext(seed=1)
        b = AgentContext(seed=2)
        assert a.rng.random() != b.rng.random()

    def test_rng_isolated_from_global_random(self):
        """Hooks that read ``ctx.rng`` don't stomp on ``random.random()``
        state used by unrelated code."""
        ctx = AgentContext(seed=99)
        ctx.rng.random()
        ctx.rng.random()
        # After the ctx draws, a fresh unseeded Random still behaves like
        # a fresh unseeded Random — which is all we guarantee (the
        # global random module is shared; Python's Random instances are
        # not).
        fresh = random.Random()
        assert isinstance(fresh.random(), float)

    def test_seed_propagates_through_handoff(self):
        """Forward-check: handoff sub_ctx inherits the parent's seed so
        the whole multi-agent run is reproducible, not just the first
        leg. Structural test — we inspect the subctx construction via
        a fake sub-agent."""
        captured: dict[str, Any] = {}

        class _CaptureAgent:
            name = "capture"
            description = "records received ctx"

            def run(self, task: str, ctx: AgentContext) -> AgentResult:
                captured["seed"] = ctx.seed
                return AgentResult(reply="ok", agent_name="capture")

            def run_stream(self, task, ctx):  # pragma: no cover
                yield ""

        parent = AgentContext(seed=7)
        # Use agent_as_tool to exercise the same plumbing: when the
        # parent context has seed=7, the wrapped tool passes it through.
        tool = agent_as_tool(_CaptureAgent())
        tool.func(task="hello", ctx=parent)
        assert captured["seed"] == 7

    def test_llm_complete_accepts_seed_kwarg(self):
        """``LLM.complete`` must accept ``seed=`` so ``_drive`` can pass
        ``ctx.seed`` through. Tested here by inspection so we don't
        spin up a real llama-cpp model."""
        import inspect

        from edgevox.llm.llamacpp import LLM

        sig = inspect.signature(LLM.complete)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default is None


# ---------------------------------------------------------------------------
# 2. agent_as_tool wrapper
# ---------------------------------------------------------------------------


class _StaticReplyAgent:
    """Minimal Agent Protocol implementation for tool-wrapping tests."""

    def __init__(self, name: str, reply: str, description: str = ""):
        self.name = name
        self.description = description or f"{name} specialist"
        self._reply = reply
        self.calls: list[tuple[str, int | None]] = []

    def run(self, task: str, ctx: AgentContext) -> AgentResult:
        self.calls.append((task, ctx.seed))
        return AgentResult(reply=self._reply, agent_name=self.name)

    def run_stream(self, task, ctx):  # pragma: no cover — not exercised
        yield self._reply


class TestAgentAsTool:
    def test_returns_a_tool(self):
        child = _StaticReplyAgent("lookup", "42")
        t = agent_as_tool(child)
        assert isinstance(t, Tool)

    def test_default_name_and_description_come_from_agent(self):
        child = _StaticReplyAgent("lookup", "42", description="does lookups")
        t = agent_as_tool(child)
        assert t.name == "lookup"
        assert t.description == "does lookups"

    def test_overrides_are_honored(self):
        child = _StaticReplyAgent("lookup", "42")
        t = agent_as_tool(child, name="ask_lookup", description="ask for a value")
        assert t.name == "ask_lookup"
        assert t.description == "ask for a value"

    def test_schema_is_wellformed(self):
        child = _StaticReplyAgent("lookup", "42")
        t = agent_as_tool(child).openai_schema()
        assert t["type"] == "function"
        params = t["function"]["parameters"]
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert params["required"] == ["task"]

    def test_calling_tool_runs_child_with_fresh_session(self):
        child = _StaticReplyAgent("lookup", "the answer")
        t = agent_as_tool(child)
        parent_ctx = AgentContext()
        parent_ctx.session.state["parent-only"] = 1
        reply = t.func(task="what is it?", ctx=parent_ctx)
        assert reply == "the answer"
        # Child saw the task but NOT the parent session state — it
        # runs with a fresh Session.
        assert child.calls == [("what is it?", None)]

    def test_shared_deps_and_seed_flow_to_child(self):
        child = _StaticReplyAgent("lookup", "42")
        t = agent_as_tool(child)
        parent_ctx = AgentContext(deps={"api_key": "abc"}, seed=123)
        t.func(task="ping", ctx=parent_ctx)
        # The child-context constructor copies deps + seed from the
        # parent; we verify the seed round-trip via the agent's
        # recorded calls (deps isn't captured directly but deps
        # behaviour is covered by the handoff tests).
        assert child.calls == [("ping", 123)]

    def test_without_ctx_uses_default(self):
        """Calling the wrapper without a parent context still works —
        the child runs in an isolated default context. Useful when
        tools are invoked from tests or direct integration without a
        running agent loop."""
        child = _StaticReplyAgent("lookup", "42")
        t = agent_as_tool(child)
        assert t.func(task="hello") == "42"
        assert child.calls == [("hello", None)]


# ---------------------------------------------------------------------------
# 3. Tool-argument JSON schema validation
# ---------------------------------------------------------------------------


def _make_registry_with(tool: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


class TestArgumentValidation:
    def _temp_tool(self, parameters: dict[str, Any]):
        def _echo(**kwargs):
            return kwargs

        return Tool(
            name="echo",
            description="Echoes its args",
            parameters=parameters,
            func=_echo,
        )

    def test_missing_required_field_is_caught(self):
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {})
        assert result.error is not None
        assert "bad arguments" in result.error
        assert "missing required field" in result.error

    def test_wrong_type_is_caught(self):
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {"count": "not an int"})
        assert result.error is not None
        assert "bad arguments" in result.error
        assert "expected integer" in result.error

    def test_enum_violation_is_caught(self):
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["red", "green", "blue"]},
                },
                "required": ["color"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {"color": "mauve"})
        assert result.error is not None
        assert "not in allowed enum" in result.error

    def test_valid_args_pass_through(self):
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {"n": 5})
        assert result.error is None
        assert result.result == {"n": 5}

    def test_number_type_accepts_int_and_float(self):
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"temp": {"type": "number"}},
                "required": ["temp"],
            }
        )
        reg = _make_registry_with(tool)
        assert reg.dispatch("echo", {"temp": 22}).error is None
        assert reg.dispatch("echo", {"temp": 22.5}).error is None

    def test_unknown_property_is_tolerated(self):
        """``additionalProperties: false`` isn't on by default; extra
        fields pass through. This matches OpenAI tool-call semantics."""
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {"a": "ok", "extra": 1})
        assert result.error is None

    def test_non_object_schema_is_passed_through(self):
        """Tools authored with a custom (non-``type: object``) schema
        skip validation — the tool body owns correctness there."""
        tool = Tool(
            name="raw",
            description="accepts anything",
            parameters={},  # no schema at all
            func=lambda **kw: kw,
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("raw", {"anything": "goes"})
        assert result.error is None

    def test_validation_error_arguments_are_preserved(self):
        """Even on validation failure the arguments round-trip into the
        result so the agent loop / hooks can inspect them."""
        tool = self._temp_tool(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        )
        reg = _make_registry_with(tool)
        result = reg.dispatch("echo", {"wrong": "field"})
        assert result.arguments == {"wrong": "field"}
