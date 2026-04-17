"""Integration tests: hooks firing through real LLMAgent._drive with ScriptedLLM."""

from __future__ import annotations

from edgevox.agents.base import AgentContext, LLMAgent
from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_TOOL,
    ON_RUN_END,
    ON_RUN_START,
    HookRegistry,
    HookResult,
    ToolCallRequest,
    hook,
)
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, reply

# ---------------------------------------------------------------------------
# Minimum wiring helper
# ---------------------------------------------------------------------------


def make_agent(llm: ScriptedLLM, *, tools=None, hooks=None, skills=None) -> LLMAgent:
    return LLMAgent(
        name="tester",
        description="test",
        instructions="You are Tester.",
        tools=tools,
        hooks=hooks,
        skills=skills,
        llm=llm,
    )


@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@tool
def echo(s: str) -> str:
    """Echo s."""
    return s


# ---------------------------------------------------------------------------
# Fire-point coverage
# ---------------------------------------------------------------------------


class TestFirePointsAllFire:
    def test_every_point_fires_in_one_turn(self, scripted_llm_factory):
        seen = []

        @hook(ON_RUN_START, BEFORE_LLM, AFTER_LLM, BEFORE_TOOL, AFTER_TOOL, ON_RUN_END)
        def trace(point, ctx, payload):
            seen.append(point)
            return None

        llm = scripted_llm_factory([call("add", x=1, y=2), reply("three")])
        agent = make_agent(llm, tools=[add], hooks=[trace])
        result = agent.run("add 1 and 2")
        assert result.reply == "three"
        # Must include all 6 points at least once, in canonical order.
        assert seen[0] == ON_RUN_START
        assert seen[-1] == ON_RUN_END
        assert BEFORE_LLM in seen and AFTER_LLM in seen
        assert BEFORE_TOOL in seen and AFTER_TOOL in seen
        # Tool fires happened between the two LLM hops.
        before_llm_ix = [i for i, p in enumerate(seen) if p == BEFORE_LLM]
        before_tool_ix = [i for i, p in enumerate(seen) if p == BEFORE_TOOL]
        assert before_llm_ix[0] < before_tool_ix[0] < before_llm_ix[1]


# ---------------------------------------------------------------------------
# on_run_start — guardrail / end_turn
# ---------------------------------------------------------------------------


class TestOnRunStart:
    def test_end_turn_short_circuits_llm(self, scripted_llm_factory):
        @hook(ON_RUN_START)
        def block(ctx, payload):
            if "forbidden" in payload["task"].lower():
                return HookResult.end("I can't.", reason="guardrail")
            return None

        llm = scripted_llm_factory([])  # should never be called
        agent = make_agent(llm, hooks=[block])
        result = agent.run("do something FORBIDDEN")
        assert result.reply == "I can't."
        assert result.hook_ended == "guardrail"
        assert len(llm.calls) == 0

    def test_modify_rewrites_task(self, scripted_llm_factory):
        @hook(ON_RUN_START)
        def normalize(ctx, payload):
            return HookResult.replace({"task": payload["task"].strip().lower()})

        llm = scripted_llm_factory([reply("ok")])
        agent = make_agent(llm, hooks=[normalize])
        result = agent.run("  HELLO  ")
        assert result.reply == "ok"
        # The last user message inside the llm call should be the rewritten task.
        user_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "user"]
        assert user_msgs[-1]["content"] == "hello"


# ---------------------------------------------------------------------------
# before_llm — token truncation / message rewrite
# ---------------------------------------------------------------------------


class TestBeforeLLM:
    def test_modify_messages_is_observed_by_llm(self, scripted_llm_factory):
        @hook(BEFORE_LLM)
        def inject(ctx, payload):
            msgs = list(payload["messages"])
            msgs.insert(1, {"role": "user", "content": "(injected)"})
            new = dict(payload)
            new["messages"] = msgs
            return HookResult.replace(new)

        llm = scripted_llm_factory([reply("ok")])
        agent = make_agent(llm, hooks=[inject])
        agent.run("hi")
        # Injected message is visible to the LLM
        contents = [m.get("content") for m in llm.calls[0]["messages"]]
        assert "(injected)" in contents

    def test_end_turn_skips_llm(self, scripted_llm_factory):
        calls_seen: list[int] = []

        @hook(BEFORE_LLM)
        def stop(ctx, payload):
            calls_seen.append(payload["hop"])
            return HookResult.end("bounced", reason="test")

        llm = scripted_llm_factory([reply("ok")])
        agent = make_agent(llm, hooks=[stop])
        result = agent.run("hi")
        assert result.reply == "bounced"
        assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# after_llm — echo substitution
# ---------------------------------------------------------------------------


class TestAfterLLM:
    def test_modify_replaces_content(self, scripted_llm_factory):
        @hook(AFTER_LLM)
        def replace(ctx, payload):
            if payload["tool_calls"]:
                return None
            p = dict(payload)
            p["content"] = "filtered"
            return HookResult.replace(p)

        llm = scripted_llm_factory([reply("should be replaced")])
        agent = make_agent(llm, hooks=[replace])
        result = agent.run("hi")
        assert result.reply == "filtered"


# ---------------------------------------------------------------------------
# before_tool — skip_dispatch / argument rewrite
# ---------------------------------------------------------------------------


class TestBeforeTool:
    def test_skip_dispatch_uses_synthetic_result(self, scripted_llm_factory):
        actual_calls: list[int] = []

        @tool
        def counter(n: int) -> int:
            """Count."""
            actual_calls.append(n)
            return n

        @hook(BEFORE_TOOL)
        def fake_it(ctx, payload: ToolCallRequest):
            if payload.name == "counter":
                payload.skip_dispatch = True
                payload.synthetic_result = 999
                return HookResult.replace(payload, reason="faked")
            return None

        llm = scripted_llm_factory([call("counter", n=5), reply("done")])
        agent = make_agent(llm, tools=[counter], hooks=[fake_it])
        result = agent.run("count 5")
        assert result.reply == "done"
        # Tool body never ran.
        assert actual_calls == []
        # The message that went into the second LLM call contains 999.
        second_call_msgs = llm.calls[1]["messages"]
        tool_msg = [m for m in second_call_msgs if m.get("role") == "tool"]
        # fallback_mode may fold the tool result into a user message.
        tool_content = next(
            (m["content"] for m in second_call_msgs if "999" in str(m.get("content"))),
            "",
        )
        assert "999" in tool_content or any("999" in str(m.get("content", "")) for m in tool_msg)

    def test_rewrite_arguments(self, scripted_llm_factory):
        import json

        @hook(BEFORE_TOOL)
        def clamp(ctx, payload: ToolCallRequest):
            args = json.loads(payload.arguments) if isinstance(payload.arguments, str) else payload.arguments
            if "x" in args and args["x"] > 10:
                args["x"] = 10
                payload.arguments = args
                return HookResult.replace(payload, reason="clamped")
            return None

        llm = scripted_llm_factory([call("add", x=9999, y=0), reply("done")])
        agent = make_agent(llm, tools=[add], hooks=[clamp])
        agent.run("add big")
        # The tool should have been called with x=10 because of the clamp.
        # We can inspect via the final message that carries the result.
        # Just assert the run completed — if clamp wasn't applied, add would
        # still succeed but with a different value; assert the clamp fired
        # by observing the result contains '10' indirectly. Easier: recheck
        # via a second hook on after_tool.

        # Re-run with an after_tool inspector.
        seen: list = []

        @hook(AFTER_TOOL)
        def capture(ctx, payload):
            seen.append(payload.result)
            return None

        llm2 = ScriptedLLM([call("add", x=9999, y=0), reply("done")])
        agent2 = make_agent(llm2, tools=[add], hooks=[clamp, capture])
        agent2.run("add big")
        assert seen == [10]  # 10 + 0 after clamp


# ---------------------------------------------------------------------------
# after_tool — truncation, logging
# ---------------------------------------------------------------------------


class TestAfterTool:
    def test_modify_tool_result(self, scripted_llm_factory):
        @hook(AFTER_TOOL)
        def shorten(ctx, payload):
            if isinstance(payload.result, str):
                payload.result = payload.result[:3]
                return HookResult.replace(payload, reason="trimmed")
            return None

        llm = scripted_llm_factory([call("echo", s="hello world"), reply("done")])
        agent = make_agent(llm, tools=[echo], hooks=[shorten])
        agent.run("echo")
        # The truncated result must appear in the second LLM call's messages.
        found = False
        for m in llm.calls[1]["messages"]:
            content = str(m.get("content", ""))
            if "hel" in content and "hello world" not in content:
                found = True
        assert found


# ---------------------------------------------------------------------------
# on_run_end — persistence
# ---------------------------------------------------------------------------


class TestOnRunEnd:
    def test_modify_replaces_result(self, scripted_llm_factory):
        from edgevox.agents.base import AgentResult

        @hook(ON_RUN_END)
        def wrap(ctx, payload):
            new = AgentResult(reply="[wrapped] " + payload.reply, agent_name=payload.agent_name)
            return HookResult.replace(new)

        llm = scripted_llm_factory([reply("original")])
        agent = make_agent(llm, hooks=[wrap])
        result = agent.run("hi")
        assert result.reply == "[wrapped] original"

    def test_fires_even_on_early_end(self, scripted_llm_factory):
        saw_end = []

        @hook(ON_RUN_START)
        def block(ctx, payload):
            return HookResult.end("nope")

        @hook(ON_RUN_END)
        def observe(ctx, payload):
            saw_end.append(payload.reply)
            return None

        llm = scripted_llm_factory([])
        agent = make_agent(llm, hooks=[block, observe])
        agent.run("anything")
        assert saw_end == ["nope"]


# ---------------------------------------------------------------------------
# Ctx-level hooks compose with agent-level
# ---------------------------------------------------------------------------


class TestCtxAndAgentHooksCompose:
    def test_both_registries_fire(self, scripted_llm_factory):
        tags: list[str] = []

        @hook(ON_RUN_START)
        def agent_h(ctx, payload):
            tags.append("agent")
            return None

        @hook(ON_RUN_START)
        def ctx_h(ctx, payload):
            tags.append("ctx")
            return None

        llm = scripted_llm_factory([reply("ok")])
        agent = make_agent(llm, hooks=[agent_h])
        ctx = AgentContext(hooks=HookRegistry([ctx_h]))
        agent.run("hi", ctx)
        assert tags == ["agent", "ctx"]


# ---------------------------------------------------------------------------
# Context pop cleans up state
# ---------------------------------------------------------------------------


def test_ctx_state_tool_registry_pointer_cleared(scripted_llm_factory):
    """After run(), ctx.state must not leak private pointers back to the caller."""
    llm = scripted_llm_factory([reply("ok")])
    agent = make_agent(llm)
    ctx = AgentContext()
    agent.run("hi", ctx)
    assert "__tool_registry__" not in ctx.state
    assert "__llm__" not in ctx.state


def test_nested_run_restores_prior_pointers(scripted_llm_factory):
    """Nested run() (subagent pattern) must not clobber the outer run's pointers."""
    llm1 = scripted_llm_factory([reply("outer")])
    llm2 = scripted_llm_factory([reply("inner")])
    outer = make_agent(llm1)
    inner = make_agent(llm2)
    ctx = AgentContext()
    # Inject a sentinel before running; the agent's cleanup should restore it.
    ctx.state["__tool_registry__"] = "SENTINEL"
    outer.run("hi", ctx)
    # Restored to sentinel.
    assert ctx.state["__tool_registry__"] == "SENTINEL"
    # Inner works too.
    inner.run("hi", ctx)
    assert ctx.state["__tool_registry__"] == "SENTINEL"
