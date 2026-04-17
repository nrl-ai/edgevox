"""Edge-case stress tests covering things that should *not* happen but
occasionally do when a framework hits the real world.

- Hook registration races
- Deeply nested ``spawn_subagent`` calls
- Huge tool outputs + TokenBudgetHook interacting with NotesInjectorHook
- A hook that mutates the session mid-turn
- Blackboard with high fan-out watchers
- Entry-point discovery
- Malformed tool call arguments
- Skill cancellation from a hook
"""

from __future__ import annotations

import threading
import time

from edgevox.agents.base import AgentContext, AgentEvent, LLMAgent
from edgevox.agents.hooks import AFTER_TOOL, BEFORE_TOOL, HookResult, ToolCallRequest, hook
from edgevox.agents.hooks_builtin import (
    MemoryInjectionHook,
    NotesInjectorHook,
    TokenBudgetHook,
    ToolOutputTruncatorHook,
)
from edgevox.agents.memory import JSONMemoryStore
from edgevox.agents.multiagent import AgentPool, Blackboard
from edgevox.agents.skills import skill
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, reply

# ---------------------------------------------------------------------------
# Malformed arguments from LLM
# ---------------------------------------------------------------------------


class TestMalformedArgs:
    def test_bad_json_args_caught_by_dispatch(self):
        @tool
        def needs_num(n: int) -> int:
            """."""
            return n

        # Invalid JSON string in arguments.
        llm = ScriptedLLM(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "needs_num", "arguments": "{not valid json"},
                        }
                    ],
                },
                reply("recovered"),
            ]
        )
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            tools=[needs_num],
            llm=llm,
        )
        r = agent.run("x")
        assert r.reply == "recovered"

    def test_unknown_tool_returns_error_not_crash(self):
        llm = ScriptedLLM(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"id": "c", "function": {"name": "not_a_real_tool", "arguments": "{}"}},
                    ],
                },
                reply("recovered"),
            ]
        )
        agent = LLMAgent(name="t", description="", instructions="", llm=llm)
        r = agent.run("x")
        assert r.reply == "recovered"


# ---------------------------------------------------------------------------
# Nested subagents
# ---------------------------------------------------------------------------


class TestNestedSubagents:
    def test_three_level_spawn_all_complete(self):
        llm = ScriptedLLM([reply("root"), reply("L1"), reply("L2"), reply("L3")])
        root = LLMAgent(name="root", description="", instructions="You are root.", llm=llm)

        ctx = AgentContext()
        root.run("start", ctx)

        l1 = root.spawn_subagent("level 1", parent_ctx=ctx)
        assert l1.reply == "L1"

        # Spawn from within the subagent chain by calling again on root.
        l2 = root.spawn_subagent("level 2", parent_ctx=ctx)
        assert l2.reply == "L2"

        l3 = root.spawn_subagent("level 3", parent_ctx=ctx)
        assert l3.reply == "L3"


# ---------------------------------------------------------------------------
# Memory + token budget + notes + truncator: composed stack
# ---------------------------------------------------------------------------


class TestMemoryAndBudgetComposed:
    def test_big_tool_output_gets_truncated_before_budget(self, tmp_path, tmp_notes, tmp_memory_store):
        @tool
        def dump() -> str:
            """."""
            return "BIG" * 10_000

        tmp_memory_store.add_fact("user", "Anh")
        tmp_notes.append("user prefers tea")

        llm = ScriptedLLM([call("dump"), reply("ok")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="You are T.",
            tools=[dump],
            hooks=[
                MemoryInjectionHook(memory_store=tmp_memory_store),
                NotesInjectorHook(tmp_notes),
                ToolOutputTruncatorHook(max_chars=500),
                TokenBudgetHook(max_context_tokens=2_000),
            ],
            llm=llm,
        )
        r = agent.run("dump", AgentContext())
        assert r.reply == "ok"
        # Second-call messages must be under budget.
        total_chars = sum(len(str(m.get("content") or "")) for m in llm.calls[1]["messages"])
        assert total_chars < 10_000


# ---------------------------------------------------------------------------
# Hook races — hooks registered from multiple threads shouldn't race
# ---------------------------------------------------------------------------


class TestHookConcurrency:
    def test_registering_hooks_from_threads(self):
        from edgevox.agents.hooks import HookRegistry

        @hook("on_run_start")
        def h(ctx, payload):
            return None

        reg = HookRegistry()

        def worker():
            for _ in range(50):
                reg.register(h)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(reg) == 200


# ---------------------------------------------------------------------------
# Blackboard high fan-out
# ---------------------------------------------------------------------------


class TestBlackboardHighFanout:
    def test_100_watchers_all_fire(self):
        bb = Blackboard()
        counters = [0] * 100
        # Each watcher appends its index; use closure trick.
        for i in range(100):

            def make_w(ix):
                def w(k, old, new):
                    counters[ix] += 1

                return w

            bb.watch("k", make_w(i))
        bb.set("k", 1)
        assert all(c == 1 for c in counters)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class TestEntryPointDiscovery:
    def test_load_entry_point_hooks_tolerates_empty(self):
        from edgevox.agents.hooks import load_entry_point_hooks

        # No registered entry points in the test env — must return [].
        assert load_entry_point_hooks() == []


# ---------------------------------------------------------------------------
# Session persistence across ctx swap
# ---------------------------------------------------------------------------


class TestSessionPassThrough:
    def test_two_runs_same_ctx_share_session_messages(self):
        llm = ScriptedLLM([reply("r1"), reply("r2")])
        agent = LLMAgent(name="t", description="", instructions="", llm=llm)
        ctx = AgentContext()
        agent.run("first", ctx)
        n_after_first = len(ctx.session.messages)
        agent.run("second", ctx)
        n_after_second = len(ctx.session.messages)
        assert n_after_second > n_after_first

    def test_second_agent_swaps_system_prompt(self):
        llm = ScriptedLLM([reply("A-reply"), reply("B-reply")])
        a = LLMAgent(name="a", description="", instructions="You are A.", llm=llm)
        b = LLMAgent(name="b", description="", instructions="You are B.", llm=llm)
        ctx = AgentContext()
        a.run("hi", ctx)
        b.run("hi", ctx)
        # System prompt is B's now.
        system = ctx.session.messages[0]["content"]
        assert "B" in system


# ---------------------------------------------------------------------------
# Budget exhausted — hop limit
# ---------------------------------------------------------------------------


class TestHopBudget:
    def test_max_hops_reached_uses_fallback_reply(self):
        @tool
        def spin() -> str:
            """."""
            return "ok"

        # Always returns a tool call — never converges.
        llm = ScriptedLLM([call("spin", a=i) for i in range(10)])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            tools=[spin],
            max_tool_hops=2,
            llm=llm,
        )
        r = agent.run("x")
        assert "couldn't" in r.reply.lower() or "sorry" in r.reply.lower()


# ---------------------------------------------------------------------------
# Skill cancellation interaction with hooks
# ---------------------------------------------------------------------------


@skill(latency_class="fast", timeout_s=10)
def short_skill() -> str:
    """."""
    return "finished"


class TestSkillHookInteraction:
    def test_before_tool_skip_dispatch_for_skill(self):
        # A hook that swaps a skill's dispatch with a synthetic result.
        @hook(BEFORE_TOOL)
        def swap(ctx, payload: ToolCallRequest):
            if payload.is_skill and payload.name == "short_skill":
                payload.skip_dispatch = True
                payload.synthetic_result = "pretended"
                return HookResult.replace(payload, reason="swap")
            return None

        llm = ScriptedLLM([call("short_skill"), reply("saw pretended result")])
        agent = LLMAgent(
            name="bot",
            description="",
            instructions="",
            skills=[short_skill],
            hooks=[swap],
            llm=llm,
        )
        r = agent.run("x")
        assert r.reply == "saw pretended result"


# ---------------------------------------------------------------------------
# Multi-agent pool hammered with events
# ---------------------------------------------------------------------------


class TestPoolEventStorm:
    def test_background_agent_drains_burst(self):
        llm = ScriptedLLM([reply(f"r{i}") for i in range(10)])
        agent = LLMAgent(name="bg", description="", instructions="", llm=llm)

        pool = AgentPool()
        pool.register(agent)
        bg = pool.start_background("bg", trigger=lambda ev: "run" if ev.kind == "tick" else None)

        # Storm 200 events rapidly.
        for _ in range(200):
            pool.bus.publish(AgentEvent(kind="tick", agent_name="z"))
        time.sleep(0.2)
        pool.stop_all()
        # Processed some subset — bounded by max_queue (32 default).
        assert len(bg.results) > 0
        assert len(bg.results) <= 10  # script had 10 replies

    def test_background_agent_triggers_after_shutdown_are_dropped(self):
        llm = ScriptedLLM([reply("unused")])
        agent = LLMAgent(name="bg", description="", instructions="", llm=llm)
        pool = AgentPool()
        pool.register(agent)
        bg = pool.start_background("bg", trigger=lambda ev: "run")
        pool.stop_all()
        pool.bus.publish(AgentEvent(kind="any", agent_name="x"))
        assert len(bg.results) == 0


# ---------------------------------------------------------------------------
# Idempotency — MemoryInjectionHook shouldn't stack Memory sections
# ---------------------------------------------------------------------------


class TestHookIdempotency:
    def test_memory_injector_idempotent(self, tmp_memory_store):
        tmp_memory_store.add_fact("x", "1")
        llm = ScriptedLLM([reply("a"), reply("b")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="You are T.",
            hooks=[MemoryInjectionHook(memory_store=tmp_memory_store)],
            llm=llm,
        )
        ctx = AgentContext()
        agent.run("first", ctx)
        agent.run("second", ctx)
        # System prompt has exactly one "## Memory" section across turns.
        system2 = llm.calls[1]["messages"][0]["content"]
        assert system2.count("## Memory") == 1


# ---------------------------------------------------------------------------
# Hook that itself errors mid-loop shouldn't tank the run
# ---------------------------------------------------------------------------


class TestResilientLoop:
    def test_broken_before_tool_hook_still_allows_dispatch(self):
        @tool
        def ping() -> str:
            """."""
            return "pong"

        @hook(BEFORE_TOOL)
        def broken(ctx, payload):
            raise RuntimeError("hook crashed")

        llm = ScriptedLLM([call("ping"), reply("done")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            tools=[ping],
            hooks=[broken],
            llm=llm,
        )
        r = agent.run("x")
        assert r.reply == "done"

    def test_broken_after_tool_hook_still_finishes(self):
        @tool
        def ping() -> str:
            """."""
            return "pong"

        @hook(AFTER_TOOL)
        def broken(ctx, payload):
            raise RuntimeError("hook crashed")

        llm = ScriptedLLM([call("ping"), reply("done")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            tools=[ping],
            hooks=[broken],
            llm=llm,
        )
        r = agent.run("x")
        assert r.reply == "done"


# ---------------------------------------------------------------------------
# Memory reload survives crash
# ---------------------------------------------------------------------------


def test_memory_store_survives_flush_and_reload(tmp_path):
    mem1 = JSONMemoryStore(tmp_path / "m.json")
    for i in range(100):
        mem1.add_fact(f"k{i}", f"v{i}")
    mem1.flush()

    mem2 = JSONMemoryStore(tmp_path / "m.json")
    for i in range(100):
        assert mem2.get_fact(f"k{i}") == f"v{i}"
