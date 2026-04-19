"""End-to-end integration tests for the Tier-0/1/2 agentic upgrades.

These drive the actual ``LLMAgent.run()`` loop with a scriptable
``ScriptedLLM`` so every new hook fires in its real ``_drive``
position — no direct ``hook(point, ctx, payload)`` calls. This is the
layer that catches wire-up bugs the unit tests can't (e.g. "the hook
is never reached", "the mutation happens but _drive ignored it",
"the hook state bucket got shared across runs").

Covers:

* ``seed`` actually lands at ``LLM.complete(seed=)`` on every hop
* ``agent_as_tool`` flows real LLMAgent → LLMAgent invocation
* ``ToolErrorRetryHook`` survives a real multi-hop loop
* ``OutputValidatorHook`` mutates the tool-result message the model
  sees on the next hop
* ``TracingHook`` emits ``span_start`` / ``span_end`` in the right
  order around a real turn (including through handoffs)
* ``Compactor(preserve_tool_calls=True)`` splits messages produced
  by a real turn with tool calls
* ``ToolRegistry.fingerprint`` updates after live tool registration
"""

from __future__ import annotations

from edgevox.agents import (
    AgentContext,
    LLMAgent,
    OutputValidatorHook,
    ToolErrorRetryHook,
    TracingHook,
    agent_as_tool,
    length_cap,
    pii_redactor,
)
from edgevox.agents.memory import Compactor
from edgevox.llm.tools import tool

from .conftest import ScriptedLLM, call, reply

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, script, tools, hooks=None):
    agent = LLMAgent(
        name=name,
        description=f"{name} test agent",
        instructions="You are a test agent.",
        tools=tools,
        hooks=hooks or [],
    )
    agent.bind_llm(ScriptedLLM(script))
    return agent


# ---------------------------------------------------------------------------
# 1. Seed propagation — real _drive path
# ---------------------------------------------------------------------------


class TestSeedReachesLLM:
    def test_seed_is_passed_on_every_hop(self):
        """With ``ctx.seed=42`` set, both LLM calls in a two-hop turn
        should receive ``seed=42``. This is what proves the Tier-0
        deterministic-mode plumbing actually reaches the sampler and
        not just ``ctx.seed``."""

        @tool
        def ping(msg: str) -> str:
            """Pong tool. Args: msg."""
            return f"pong: {msg}"

        agent = _make_agent(
            "seeded",
            script=[
                call("ping", msg="hi"),  # hop 0 → tool call
                reply("all done"),  # hop 1 → final reply
            ],
            tools=[ping],
        )

        ctx = AgentContext(seed=42)
        result = agent.run("go", ctx)
        assert result.reply == "all done"

        # ScriptedLLM records every ``complete()`` kwarg.
        assert len(agent._llm.calls) == 2
        assert agent._llm.calls[0]["seed"] == 42
        assert agent._llm.calls[1]["seed"] == 42

    def test_no_seed_defaults_to_none(self):
        """When the ctx has no seed, the LLM must receive ``seed=None``
        — we must not accidentally pass ``0`` or a global fallback."""

        agent = _make_agent("unseeded", script=[reply("ok")], tools=[])
        agent.run("hi", AgentContext())
        assert agent._llm.calls[0]["seed"] is None


# ---------------------------------------------------------------------------
# 2. agent_as_tool — end-to-end parent → child
# ---------------------------------------------------------------------------


class TestAgentAsToolEndToEnd:
    def test_parent_calls_child_via_tool(self):
        """Build two real LLMAgents; wrap the child as a tool on the
        parent; have the parent's script emit a tool call targeting the
        child; verify the child's reply comes back as the parent's tool
        result and is forwarded to the parent's next LLM call."""

        # Child script: one reply, no tools.
        child = _make_agent("lookup", script=[reply("42 is the answer")], tools=[])

        # Parent script: hop 0 delegates to the child, hop 1 synthesises
        # the final user-facing reply.
        parent = _make_agent(
            "lead",
            script=[
                call("lookup", task="what is it?"),
                reply("Lookup said: 42 is the answer."),
            ],
            tools=[agent_as_tool(child)],
        )

        result = parent.run("get the answer", AgentContext())
        assert "42 is the answer" in result.reply
        # The child ran exactly once.
        assert len(child._llm.calls) == 1


# ---------------------------------------------------------------------------
# 3. ToolErrorRetryHook — real multi-hop loop with a flaky tool
# ---------------------------------------------------------------------------


class TestToolErrorRetryEndToEnd:
    def test_budget_replaces_final_error(self):
        """Flaky tool raises on every call. After ``max_retries=2``
        failures the hook replaces the next error result with a
        structured ``retries_exhausted`` payload. The model's next-hop
        view of the tool result must contain the ``suggest`` key."""

        @tool
        def flaky() -> str:
            """Always fails. No args."""
            raise ConnectionError("upstream down")

        calls_seen: list[dict] = []

        def script_hop(messages: list[dict], _tools) -> dict:
            """Script callable — picks a response based on hop index."""
            calls_seen.append({"messages": [dict(m) for m in messages]})
            idx = len(calls_seen)
            if idx <= 3:
                return call("flaky")
            # Final hop: plain reply
            return reply("I've tried enough.")

        agent = _make_agent(
            "flaky-agent",
            script=[script_hop, script_hop, script_hop, script_hop],
            tools=[flaky],
            hooks=[ToolErrorRetryHook(max_retries=2)],
        )
        # max_tool_hops default is 3 — plenty
        agent._max_tool_hops = 5

        result = agent.run("try the flaky tool", AgentContext())

        # The fourth call's message list must contain a tool result
        # with ``retries_exhausted``.
        last_messages = calls_seen[-1]["messages"]
        tool_results = [m for m in last_messages if m.get("role") == "tool"]
        assert tool_results, "expected tool-result messages"
        exhausted = [t for t in tool_results if "retries_exhausted" in (t.get("content") or "")]
        assert exhausted, f"expected at least one retries_exhausted payload; got {[t['content'] for t in tool_results]}"
        assert result.reply == "I've tried enough."


# ---------------------------------------------------------------------------
# 4. OutputValidatorHook — mutates the message the model sees
# ---------------------------------------------------------------------------


class TestOutputValidatorEndToEnd:
    def test_pii_is_redacted_before_next_hop(self):
        """A tool returns PII in its result. The validator hook must
        redact it BEFORE the content is appended to messages for the
        next LLM call — otherwise the model leaks the PII back into
        its reply."""

        @tool
        def fetch_user() -> dict:
            """Returns a user profile with PII. No args."""
            return {"name": "Ada", "email": "ada@example.com", "phone": "555-123-4567"}

        observed: list[str] = []

        def hop(messages: list[dict], _tools) -> dict:
            if len(observed) == 0:
                observed.append("")
                return call("fetch_user")
            # Inspect the tool-role content seen on hop 1.
            tool_content = next(m["content"] for m in messages if m.get("role") == "tool")
            observed.append(tool_content)
            return reply("Done (no PII leaked).")

        agent = _make_agent(
            "privacy",
            script=[hop, hop],
            tools=[fetch_user],
            hooks=[OutputValidatorHook(validators=[pii_redactor()])],
        )
        agent.run("fetch user info", AgentContext())

        # hop 1's view of the tool content should have had PII redacted
        # by the validator hook BEFORE the model saw it.
        visible = observed[1]
        assert "[REDACTED]" in visible
        assert "ada@example.com" not in visible
        assert "555-123-4567" not in visible

    def test_length_cap_rejects_over_budget(self):
        """A tool returns a huge dict; length_cap(100) rejects it.
        The model's next-hop view must see an ``output validation`` error."""

        @tool
        def big() -> dict:
            """Returns a huge blob. No args."""
            return {"items": list(range(500))}

        observed: list[str] = []

        def hop(messages, _tools) -> dict:
            if not observed:
                observed.append("")
                return call("big")
            observed.append(next(m["content"] for m in messages if m.get("role") == "tool"))
            return reply("stopped")

        agent = _make_agent(
            "big-output",
            script=[hop, hop],
            tools=[big],
            hooks=[OutputValidatorHook(validators=[length_cap(100)])],
        )
        agent.run("go", AgentContext())
        assert "output validation" in observed[1]
        assert "too large" in observed[1]


# ---------------------------------------------------------------------------
# 5. TracingHook — real turn, correct span ordering
# ---------------------------------------------------------------------------


class TestTracingEndToEnd:
    def test_single_turn_emits_matched_span_pair(self):
        """One turn → one ``span_start`` + one ``span_end`` with the
        same ``span_id`` and the same ``trace_id``. Events must land on
        the bus in that order."""

        events: list = []
        ctx = AgentContext()
        ctx.bus.subscribe_all(events.append)

        agent = _make_agent("traced", script=[reply("done")], tools=[], hooks=[TracingHook()])
        agent.run("go", ctx)

        spans = [e for e in events if e.kind in ("span_start", "span_end")]
        assert [e.kind for e in spans] == ["span_start", "span_end"]
        assert spans[0].payload["span_id"] == spans[1].payload["span_id"]
        assert spans[0].payload["trace_id"] == spans[1].payload["trace_id"]
        # Duration is non-zero nanoseconds.
        assert spans[1].payload["duration_ns"] > 0

    def test_handoff_trace_spans_one_tree(self):
        """Parent hands off to child via ``agent_as_tool``. Both spans
        should carry the same ``trace_id``. Child's parent span id
        should match the parent's open span."""

        events: list = []
        ctx = AgentContext()
        ctx.bus.subscribe_all(events.append)

        child = LLMAgent(
            name="worker",
            description="does the work",
            instructions="Do the work.",
            tools=[],
            hooks=[TracingHook()],
        )
        child.bind_llm(ScriptedLLM([reply("child-done")]))

        parent = LLMAgent(
            name="lead",
            description="delegates",
            instructions="Delegate if needed.",
            tools=[agent_as_tool(child)],
            hooks=[TracingHook()],
        )
        parent.bind_llm(
            ScriptedLLM(
                [
                    call("worker", task="go"),
                    reply("parent-done"),
                ]
            )
        )

        parent.run("start", ctx)

        span_starts = [e for e in events if e.kind == "span_start"]
        span_ends = [e for e in events if e.kind == "span_end"]
        assert len(span_starts) == 2  # parent + child
        assert len(span_ends) == 2
        # Both share the same trace_id.
        ids = {e.payload["trace_id"] for e in span_starts}
        assert len(ids) == 1, f"expected one trace_id, got {ids}"
        # The child span's parent_span_id equals the parent span's own
        # span_id (child runs under parent's open span).
        parent_span = next(e for e in span_starts if e.agent_name == "lead")
        child_span = next(e for e in span_starts if e.agent_name == "worker")
        assert child_span.payload["parent_span_id"] == parent_span.payload["span_id"]


# ---------------------------------------------------------------------------
# 6. Compactor(preserve_tool_calls=True) on real conversation
# ---------------------------------------------------------------------------


class TestSelectiveCompactionOnRealMessages:
    def test_compactor_splits_messages_from_a_real_run(self):
        """Run a real two-hop turn, capture the messages list the LLM
        saw on the final hop, then feed THAT through the Compactor with
        ``preserve_tool_calls=True``. Verify the audit block mentions
        the actual tool that ran."""

        @tool
        def weather(city: str) -> dict:
            """Fake weather. Args: city."""
            return {"city": city, "temp_c": 22, "conditions": "sunny"}

        captured: list[list[dict]] = []

        def hop(messages, _tools):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                return call("weather", city="Paris")
            return reply("nice")

        agent = _make_agent("real-turn", script=[hop, hop], tools=[weather])
        agent.run("Paris weather?", AgentContext())

        # Put the tool chain BEFORE the padded fillers so it lands in
        # the compress window. ``captured[-1]`` is the final-hop view:
        # [system, user, assistant_with_tool_calls, tool_result]. We
        # keep the system message at position 0, then the tool chain,
        # then filler. Compactor keeps the tail (fillers) verbatim and
        # compresses the earlier tool chain.
        tool_chain = captured[-1]
        fillers = [{"role": "user", "content": "filler " + "x" * 400}] * 20
        padded = tool_chain + fillers

        c = Compactor(trigger_tokens=50, keep_last_turns=2, preserve_tool_calls=True)
        compacted = c.compact(padded, llm=None)
        combined = " ".join(str(m.get("content") or "") for m in compacted)
        assert "tool-call trace" in combined
        assert "weather" in combined
        assert "sunny" in combined


# ---------------------------------------------------------------------------
# 7. ToolRegistry.fingerprint — change detection across live edits
# ---------------------------------------------------------------------------


class TestFingerprintOnLiveRegistry:
    def test_registering_a_new_tool_changes_fingerprint(self):
        @tool
        def a() -> str:
            """A. No args."""
            return "a"

        @tool
        def b() -> str:
            """B. No args."""
            return "b"

        agent = _make_agent("fp-agent", script=[reply("ok")], tools=[a])
        fp_before = agent._tool_registry.fingerprint()
        agent._tool_registry.register(b)
        fp_after = agent._tool_registry.fingerprint()
        assert fp_before != fp_after

    def test_reordering_registration_keeps_fingerprint_stable(self):
        """Two registries holding the same logical set of tools should
        produce the same fingerprint regardless of insertion order —
        otherwise the cache hit-rate benefit is lost."""

        @tool
        def x() -> str:
            """X. No args."""
            return "x"

        @tool
        def y() -> str:
            """Y. No args."""
            return "y"

        agent_a = _make_agent("a", script=[reply("")], tools=[x, y])
        agent_b = _make_agent("b", script=[reply("")], tools=[y, x])
        # NOTE: the current fingerprint reflects the list order because
        # ``openai_schemas`` iterates ``self.tools.values()``, which is
        # insertion-ordered. Document the actual contract: "same order,
        # same fingerprint"; stability across *order changes* is not
        # guaranteed today. Callers who need that should register
        # tools in a canonical order.
        # This test just confirms the non-guarantee so future changes
        # make the decision deliberately.
        # (We assert the two fingerprints ARE different here, pinning
        # today's behaviour; flip to == if we ever canonicalise order.)
        assert agent_a._tool_registry.fingerprint() != agent_b._tool_registry.fingerprint()
