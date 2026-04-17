"""Stress tests: long flows, diverse hook combos, multi-agent scenarios.

These are end-to-end tests that push the harness through realistic
sequences the way a voice-robot deployment would. Individual unit
tests live alongside each module; this file proves the pieces
compose under load and edge conditions.
"""

from __future__ import annotations

import threading
import time

from edgevox.agents.artifacts import InMemoryArtifactStore, text_artifact
from edgevox.agents.base import AgentContext, AgentEvent, LLMAgent, Session
from edgevox.agents.bus import EventBus
from edgevox.agents.hooks import BEFORE_TOOL, ON_RUN_START, HookResult, hook
from edgevox.agents.hooks_builtin import (
    AuditLogHook,
    ContextCompactionHook,
    EpisodeLoggerHook,
    MemoryInjectionHook,
    NotesInjectorHook,
    PersistSessionHook,
    PlanModeHook,
    SafetyGuardrailHook,
    TimingHook,
    TokenBudgetHook,
    ToolOutputTruncatorHook,
)
from edgevox.agents.interrupt import InterruptController
from edgevox.agents.memory import Compactor, JSONMemoryStore, JSONSessionStore
from edgevox.agents.multiagent import (
    AgentPool,
    send_message,
    subscribe_inbox,
)
from edgevox.agents.skills import GoalHandle, skill
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, calls, reply

# ---------------------------------------------------------------------------
# Kitchen-sink: every hook enabled at once
# ---------------------------------------------------------------------------


def _make_full_hook_stack(*, mem, sessions, notes, approver, audit_path):
    return [
        SafetyGuardrailHook(blocklist=["banned-word"]),
        MemoryInjectionHook(memory_store=mem),
        NotesInjectorHook(notes),
        TokenBudgetHook(max_context_tokens=5000),
        ContextCompactionHook(
            compactor=Compactor(trigger_tokens=3000, keep_last_turns=3),
            llm_getter=lambda _c: None,  # avoid LLM call in test
        ),
        ToolOutputTruncatorHook(max_chars=500),
        PlanModeHook(confirm=["grasp"], approver=approver),
        EpisodeLoggerHook(memory_store=mem, agent_name="kitchen"),
        AuditLogHook(path=audit_path),
        PersistSessionHook(session_store=sessions, session_id="kitchen-turn"),
        TimingHook(),
    ]


class TestKitchenSinkStack:
    def test_all_hooks_compose_and_run(self, tmp_path, tmp_memory_store, tmp_session_store, tmp_notes):
        @tool
        def weather(city: str) -> str:
            """Weather."""
            return f"{city}: sunny"

        @tool
        def grasp(item: str) -> str:
            """Grasp."""
            return f"grasped {item}"

        approver_calls = []

        def approve(tool, args, ctx):
            approver_calls.append((tool, args))
            return True

        llm = ScriptedLLM(
            [
                calls(("weather", {"city": "Paris"}), ("grasp", {"item": "cup"})),
                reply("ok, done"),
            ]
        )
        hooks = _make_full_hook_stack(
            mem=tmp_memory_store,
            sessions=tmp_session_store,
            notes=tmp_notes,
            approver=approve,
            audit_path=tmp_path / "audit.jsonl",
        )
        agent = LLMAgent(
            name="kitchen",
            description="k",
            instructions="You are a kitchen robot.",
            tools=[weather, grasp],
            hooks=hooks,
            llm=llm,
        )
        # Seed some memory so the injector has something to show.
        tmp_memory_store.add_fact("user_name", "Anh")
        tmp_notes.append("user prefers tea over coffee")

        result = agent.run("what's the weather and pick up the cup", AgentContext())
        assert result.reply == "ok, done"

        # All side effects happened.
        assert approver_calls == [("grasp", {"item": "cup"})]
        assert tmp_memory_store.recent_episodes(5)
        assert tmp_session_store.load("kitchen-turn") is not None
        assert (tmp_path / "audit.jsonl").exists()

        # Memory/notes were injected into system.
        system = llm.calls[0]["messages"][0]["content"]
        assert "Anh" in system
        assert "tea" in system


# ---------------------------------------------------------------------------
# Long-flow: 50 turns with compaction
# ---------------------------------------------------------------------------


class TestLongConversationCompaction:
    def test_50_turn_session_stays_bounded(self):
        @tool
        def noop() -> str:
            """."""
            return "x"

        # Alternate tool + reply to inflate the session 50 times.
        script = []
        for i in range(50):
            script.append(call("noop"))
            script.append(reply(f"turn-{i}"))
        llm = ScriptedLLM(script)

        compactor = Compactor(trigger_tokens=500, keep_last_turns=2)
        agent = LLMAgent(
            name="long",
            description="x",
            instructions="You are long.",
            tools=[noop],
            hooks=[ContextCompactionHook(compactor=compactor)],
            llm=llm,
        )
        ctx = AgentContext()

        for i in range(50):
            result = agent.run(f"q{i}", ctx)
            assert result.reply == f"turn-{i}"

        # After 50 turns, the session should have been aggressively compacted.
        assert len(ctx.session.messages) < 200  # naively would be ~200


# ---------------------------------------------------------------------------
# Parallel tool calls + hooks
# ---------------------------------------------------------------------------


class TestParallelToolsWithHooks:
    def test_parallel_tools_all_pass_through_before_after(self):
        seen_tools = []

        @hook(BEFORE_TOOL)
        def record(ctx, payload):
            seen_tools.append(payload.name)
            return None

        @tool
        def a() -> str:
            """."""
            return "A"

        @tool
        def b() -> str:
            """."""
            return "B"

        @tool
        def c() -> str:
            """."""
            return "C"

        llm = ScriptedLLM([calls(("a", {}), ("b", {}), ("c", {})), reply("done")])
        agent = LLMAgent(
            name="par",
            description="p",
            instructions="",
            tools=[a, b, c],
            hooks=[record],
            llm=llm,
        )
        r = agent.run("run all")
        assert r.reply == "done"
        assert set(seen_tools) == {"a", "b", "c"}

    def test_end_turn_during_parallel_tool_terminates(self):
        @tool
        def fine() -> str:
            """."""
            return "ok"

        @tool
        def bad() -> str:
            """."""
            return "no"

        @hook(BEFORE_TOOL)
        def abort(ctx, payload):
            if payload.name == "bad":
                return HookResult.end("stopped by hook", reason="bad tool")
            return None

        llm = ScriptedLLM([calls(("fine", {}), ("bad", {})), reply("should never reach")])
        agent = LLMAgent(
            name="p",
            description="p",
            instructions="",
            tools=[fine, bad],
            hooks=[abort],
            llm=llm,
        )
        r = agent.run("x")
        assert r.reply == "stopped by hook"


# ---------------------------------------------------------------------------
# Skill + hook preemption interaction
# ---------------------------------------------------------------------------


@skill(latency_class="fast")
def wait_a_bit(handle: GoalHandle) -> str:
    """Fast skill."""
    return "done"


class TestSkillPreemption:
    def test_ctx_stop_preempts_skill_dispatch(self):
        llm = ScriptedLLM([call("wait_a_bit"), reply("done")])
        agent = LLMAgent(
            name="robot",
            description="",
            instructions="",
            skills=[wait_a_bit],
            llm=llm,
        )
        ctx = AgentContext()
        ctx.stop.set()
        r = agent.run("do a thing", ctx)
        assert r.preempted is True
        assert r.reply == "Stopped."

    def test_interrupt_cleared_at_run_start(self):
        # An interrupt set *before* run() starts is deliberately reset by
        # the agent — otherwise a stale interrupt from the previous turn
        # would perma-block the agent. Verify that reset happens.
        ic = InterruptController()
        ic.trigger("user_barge_in")  # stale, from "last turn"
        assert ic.should_stop()
        llm = ScriptedLLM([reply("alive")])
        agent = LLMAgent(name="bot", description="", instructions="", llm=llm)
        result = agent.run("hi", AgentContext(interrupt=ic))
        # Reset occurred — the run proceeded normally.
        assert result.reply == "alive"
        assert not ic.should_stop()

    def test_interrupt_set_after_agent_start_preempts(self, scripted_llm_factory):
        # Trigger fires inside a hook — should be detected after that hop.
        from edgevox.agents.hooks import AFTER_LLM, hook

        ic = InterruptController()

        fired = []

        @hook(AFTER_LLM)
        def fire_interrupt(ctx, payload):
            # Only fire once.
            if not fired:
                fired.append(1)
                ctx.interrupt.trigger("user_barge_in")
            return None

        llm = ScriptedLLM(
            [
                call("wait_a_bit"),  # first hop: tool call
                reply("should not reach this"),  # blocked by interrupt at hop 2
            ]
        )
        agent = LLMAgent(
            name="bot",
            description="",
            instructions="",
            skills=[wait_a_bit],
            hooks=[fire_interrupt],
            llm=llm,
        )
        ctx = AgentContext(interrupt=ic)
        r = agent.run("do", ctx)
        # The interrupt was set after the first LLM hop.
        # After tool dispatch, the loop checks should_stop() and returns "Stopped."
        assert r.preempted is True
        assert r.reply == "Stopped."


# ---------------------------------------------------------------------------
# Multi-agent: supervisor + background worker + blackboard
# ---------------------------------------------------------------------------


class TestMultiAgentScenario:
    def test_supervisor_watches_blackboard_and_routes_to_worker(self):
        # Supervisor: fires when blackboard "task_queue" grows.
        # Worker: handles the task.
        # Blackboard: shared state.
        # Message bus: coordination.

        llm_sup = ScriptedLLM([reply("saw queue change")])
        llm_worker = ScriptedLLM([reply("did it")])

        sup = LLMAgent(name="supervisor", description="", instructions="", llm=llm_sup)
        worker = LLMAgent(name="worker", description="", instructions="", llm=llm_worker)

        pool = AgentPool()
        pool.register(sup)
        pool.register(worker)

        got_tasks = []

        def on_queue_change(key, old, new):
            # Supervisor reacts via a bus event, triggering its BG agent.
            pool.bus.publish(AgentEvent(kind="queue_changed", agent_name="blackboard"))

        pool.blackboard.watch("task_queue", on_queue_change)

        bg = pool.start_background(
            "supervisor",
            trigger=lambda ev: "watch" if ev.kind == "queue_changed" else None,
        )
        # Worker listens for direct messages.
        subscribe_inbox(
            pool.bus,
            agent_name="worker",
            handler=lambda m: got_tasks.append(m.content),
        )

        pool.blackboard.set("task_queue", ["grasp red block"])
        time.sleep(0.1)

        send_message(pool.bus, from_agent="supervisor", to="worker", content="grasp")
        time.sleep(0.05)
        pool.stop_all()

        assert len(bg.results) == 1
        assert bg.results[0].reply == "saw queue change"
        assert got_tasks == ["grasp"]


# ---------------------------------------------------------------------------
# Artifact handoff between agents
# ---------------------------------------------------------------------------


class TestArtifactHandoff:
    def test_parent_writes_subagent_reads(self):
        store = InMemoryArtifactStore()
        # Parent writes an artifact then spawns a subagent that reads it.
        store.write(text_artifact("plan.md", "step 1\nstep 2", tags=["plan"]))

        # The subagent simply asks the parent LLMAgent for a subagent run.
        # Here we simulate by running two agents sharing the artifact store.
        llm_parent = ScriptedLLM([reply("delegated")])
        llm_sub = ScriptedLLM([reply("read plan")])

        parent = LLMAgent(name="parent", description="", instructions="", llm=llm_parent)
        sub = LLMAgent(name="sub", description="", instructions="", llm=llm_sub)

        ctx = AgentContext(artifacts=store)

        # Parent runs in its ctx.
        parent.run("plan this", ctx)
        # Subagent inherits artifacts via parent ctx propagation.
        sub_ctx = AgentContext(artifacts=ctx.artifacts)
        sub_result = sub.run("do the plan", sub_ctx)
        assert sub_result.reply == "read plan"
        # Artifact still there.
        assert sub_ctx.artifacts.read("plan.md").content == "step 1\nstep 2"


# ---------------------------------------------------------------------------
# Memory persistence across sessions
# ---------------------------------------------------------------------------


class TestMemoryPersistence:
    def test_facts_survive_restart(self, tmp_path):
        mem1 = JSONMemoryStore(tmp_path / "mem.json")
        mem1.add_fact("user_name", "Anh")
        mem1.set_preference("voice", "concise")
        mem1.add_episode(kind="skill", payload={"a": "grasp"}, outcome="ok")
        mem1.flush()

        # Simulate restart.
        mem2 = JSONMemoryStore(tmp_path / "mem.json")
        assert mem2.get_fact("user_name") == "Anh"
        assert mem2.preferences()[0].value == "concise"
        assert len(mem2.recent_episodes(10)) == 1


# ---------------------------------------------------------------------------
# Session resume
# ---------------------------------------------------------------------------


class TestSessionResume:
    def test_load_continue(self, tmp_path):
        sessions = JSONSessionStore(tmp_path / "sess")
        s = Session(messages=[{"role": "user", "content": "earlier turn"}])
        sessions.save("sid", s)

        loaded = sessions.load("sid")
        assert loaded is not None

        llm = ScriptedLLM([reply("continued")])
        agent = LLMAgent(
            name="x",
            description="",
            instructions="",
            llm=llm,
        )
        ctx = AgentContext(session=loaded)
        result = agent.run("next turn", ctx)
        assert result.reply == "continued"
        # The session grew.
        assert any(m.get("content") == "next turn" for m in ctx.session.messages)


# ---------------------------------------------------------------------------
# Concurrency: multiple agents, one bus
# ---------------------------------------------------------------------------


class TestConcurrentAgents:
    def test_two_agents_on_same_bus_dont_corrupt_state(self):
        llm_a = ScriptedLLM([reply("A1"), reply("A2")])
        llm_b = ScriptedLLM([reply("B1"), reply("B2")])

        a = LLMAgent(name="a", description="", instructions="", llm=llm_a)
        b = LLMAgent(name="b", description="", instructions="", llm=llm_b)

        bus = EventBus()
        results_a, results_b = [], []

        def run_a():
            for _ in range(2):
                results_a.append(a.run("x", AgentContext(bus=bus)).reply)

        def run_b():
            for _ in range(2):
                results_b.append(b.run("x", AgentContext(bus=bus)).reply)

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results_a == ["A1", "A2"]
        assert results_b == ["B1", "B2"]


# ---------------------------------------------------------------------------
# Subagent spawn
# ---------------------------------------------------------------------------


class TestSubagentSpawn:
    def test_subagent_has_fresh_session(self):
        llm = ScriptedLLM([reply("parent"), reply("sub")])
        parent = LLMAgent(
            name="parent",
            description="",
            instructions="You are parent.",
            llm=llm,
        )
        ctx = AgentContext()
        parent.run("first", ctx)
        assert len(ctx.session.messages) > 0  # parent session has content
        sub_result = parent.spawn_subagent(
            "refine",
            parent_ctx=ctx,
            instructions="You are subagent.",
        )
        assert sub_result.reply == "sub"
        # Parent session unchanged by subagent.
        assert not any("sub" in str(m.get("content") or "") for m in ctx.session.messages)


# ---------------------------------------------------------------------------
# End-turn semantics propagate through hooks chain
# ---------------------------------------------------------------------------


def test_end_turn_from_ctx_hook_wins_over_agent_hook():
    # Agent hook modifies; ctx hook ends — ctx end wins.
    @hook(ON_RUN_START)
    def agent_h(ctx, payload):
        return HookResult.replace({"task": payload["task"] + "-a"})

    @hook(ON_RUN_START)
    def ctx_h(ctx, payload):
        return HookResult.end("ctx stopped")

    llm = ScriptedLLM([])
    agent = LLMAgent(
        name="x",
        description="",
        instructions="",
        hooks=[agent_h],
        llm=llm,
    )
    from edgevox.agents.hooks import HookRegistry

    r = agent.run("hi", AgentContext(hooks=HookRegistry([ctx_h])))
    assert r.reply == "ctx stopped"
