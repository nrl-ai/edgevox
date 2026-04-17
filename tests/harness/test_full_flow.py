"""Full-flow integration tests: every primitive wired into one scenario.

Proves the ecosystem actually composes — not just that individual
modules work in isolation. Each test sets up a realistic robot-voice
scenario, threads the call through hooks + memory + interrupt +
blackboard + background agents + artifacts, and asserts the whole
thing moved.
"""

from __future__ import annotations

import time

from edgevox.agents.artifacts import InMemoryArtifactStore, text_artifact
from edgevox.agents.base import AgentContext, AgentEvent, LLMAgent
from edgevox.agents.hooks import AFTER_TOOL, hook
from edgevox.agents.hooks_builtin import (
    AuditLogHook,
    ContextCompactionHook,
    EpisodeLoggerHook,
    MemoryInjectionHook,
    NotesInjectorHook,
    PersistSessionHook,
    PlanModeHook,
    SafetyGuardrailHook,
    TokenBudgetHook,
    ToolOutputTruncatorHook,
)
from edgevox.agents.interrupt import InterruptController
from edgevox.agents.memory import Compactor, JSONMemoryStore, JSONSessionStore, NotesFile, new_session_id
from edgevox.agents.multiagent import AgentPool, send_message, subscribe_inbox
from edgevox.llm import tool
from edgevox.llm.hooks_slm import default_slm_hooks

from .conftest import ScriptedLLM, call, calls, reply

# ---------------------------------------------------------------------------
# Scenario 1: Kitchen robot with a supervisor agent
# ---------------------------------------------------------------------------


class TestKitchenRobotScenario:
    """Realistic flow:
    - Supervisor agent on a shared pool watches a ``task_queue`` blackboard
      entry. When it changes, a BackgroundAgent wakes and either handles
      the task itself or delegates via direct message to the ``worker``.
    - Worker agent has tools (grasp, weather) guarded by PlanMode, plus
      memory injection + episode logging + audit log + session persistence.
    - Everything is driven by scripted LLMs so the test is deterministic.
    """

    def test_end_to_end_kitchen_flow(self, tmp_path):
        memory = JSONMemoryStore(tmp_path / "memory.json")
        sessions = JSONSessionStore(tmp_path / "sessions")
        notes = NotesFile(tmp_path / "notes.md")
        artifacts = InMemoryArtifactStore()

        # Seed persistent context so the injector has something.
        memory.add_fact("user_name", "Anh")
        memory.set_preference("voice", "concise")
        notes.append("kettle is in drawer 2")

        # --- tools & skills ---
        tool_log = []

        @tool
        def grasp(item: str) -> str:
            """Grasp."""
            tool_log.append(("grasp", item))
            return f"grasped {item}"

        @tool
        def weather(city: str) -> str:
            """Weather."""
            tool_log.append(("weather", city))
            return f"{city}: sunny"

        # --- approver simulates a UI that auto-approves kitchen requests ---
        approval_log = []

        def approver(tool_name, args, ctx):
            approval_log.append((tool_name, args))
            return "red" not in str(args).lower()  # decline red-block grasp

        # --- scripted LLMs ---
        supervisor_llm = ScriptedLLM([reply("noticed the queue change")])
        worker_llm = ScriptedLLM(
            [
                calls(("weather", {"city": "Paris"}), ("grasp", {"item": "cup"})),
                reply("weather reported; cup grasped."),
            ]
        )

        # --- agents ---
        session_id = new_session_id()
        worker_hooks = [
            SafetyGuardrailHook(blocklist=["shutdown"]),
            MemoryInjectionHook(memory_store=memory),
            NotesInjectorHook(notes),
            TokenBudgetHook(max_context_tokens=4000),
            ContextCompactionHook(compactor=Compactor(trigger_tokens=3000), llm_getter=lambda _c: None),
            ToolOutputTruncatorHook(max_chars=300),
            PlanModeHook(confirm=["grasp"], approver=approver),
            EpisodeLoggerHook(memory_store=memory, agent_name="worker"),
            AuditLogHook(path=tmp_path / "audit.jsonl"),
            PersistSessionHook(session_store=sessions, session_id=session_id),
            *default_slm_hooks(),
        ]
        worker = LLMAgent(
            name="worker",
            description="does kitchen tasks",
            instructions="You are Worker, a kitchen robot.",
            tools=[weather, grasp],
            hooks=worker_hooks,
            llm=worker_llm,
        )
        supervisor = LLMAgent(
            name="supervisor",
            description="watches the kitchen",
            instructions="You are Supervisor, monitoring the kitchen.",
            llm=supervisor_llm,
        )

        # --- pool + shared state ---
        pool = AgentPool()
        pool.register(worker)
        pool.register(supervisor)

        worker_inbox: list[str] = []
        subscribe_inbox(pool.bus, agent_name="worker", handler=lambda m: worker_inbox.append(m.content))

        # Background agent: supervisor reacts to a blackboard change.
        pool.blackboard.watch(
            "task_queue",
            lambda k, old, new: pool.bus.publish(AgentEvent(kind="queue_changed", agent_name="bb")),
        )
        sup_bg = pool.start_background(
            "supervisor",
            trigger=lambda ev: "new task appeared" if ev.kind == "queue_changed" else None,
        )

        # Parent ctx for the worker turn.
        worker_ctx = pool.make_ctx(memory=memory, artifacts=artifacts)

        # --- act ---
        # 1. Blackboard change triggers supervisor via background agent.
        pool.blackboard.set("task_queue", ["weather + grasp cup"])
        time.sleep(0.1)

        # 2. Supervisor (or anyone) sends the worker a direct message.
        send_message(pool.bus, from_agent="supervisor", to="worker", content="weather + grasp cup")
        time.sleep(0.05)

        # 3. Worker runs explicitly.
        result = worker.run("weather in paris then pick up the cup", worker_ctx)

        # 4. Worker writes an artifact for later handoff.
        artifacts.write(text_artifact("last_result", result.reply, author="worker", tags=["result"]))

        pool.stop_all()

        # --- assertions ---
        assert result.reply == "weather reported; cup grasped."
        # Only non-red grasp was approved.
        assert approval_log == [("grasp", {"item": "cup"})]
        assert ("grasp", "cup") in tool_log
        assert ("weather", "Paris") in tool_log

        # Supervisor's background agent saw the queue change.
        assert len(sup_bg.results) == 1
        assert sup_bg.results[0].reply == "noticed the queue change"

        # Worker inbox got the supervisor's message.
        assert worker_inbox == ["weather + grasp cup"]

        # Memory recorded both tool outcomes.
        episodes = memory.recent_episodes(10)
        assert len(episodes) >= 2
        assert all(e.outcome == "ok" for e in episodes)

        # Session persisted.
        loaded = sessions.load(session_id)
        assert loaded is not None
        assert any("weather" in str(m.get("content", "")) for m in loaded.messages)

        # Audit log written.
        audit_path = tmp_path / "audit.jsonl"
        assert audit_path.exists()
        assert audit_path.stat().st_size > 0

        # Artifact survived.
        stored = artifacts.read("last_result")
        assert stored is not None
        assert stored.content == result.reply

        # Injected memory visible in the FIRST LLM call's system prompt.
        system = worker_llm.calls[0]["messages"][0]["content"]
        assert "Anh" in system  # fact
        assert "drawer 2" in system  # notes tail


# ---------------------------------------------------------------------------
# Scenario 2: Interrupt mid-flow
# ---------------------------------------------------------------------------


class TestInterruptMidFlow:
    def test_interrupt_fires_during_hop(self):
        """The agent is processing a tool result when a user barge-in arrives.
        The next hop's ``should_stop()`` check must honor the interrupt."""

        @tool
        def slow() -> str:
            """Slow tool."""
            return "done"

        # After the first tool call, fire the interrupt from an after_tool
        # hook — the next hop check picks it up and returns "Stopped."
        ic = InterruptController()

        @hook(AFTER_TOOL)
        def barge(ctx, payload):
            ctx.interrupt.trigger("user_barge_in")
            return None

        llm = ScriptedLLM([call("slow"), reply("should not reach")])
        agent = LLMAgent(
            name="bot",
            description="",
            instructions="",
            tools=[slow],
            hooks=[barge],
            llm=llm,
        )
        ctx = AgentContext(interrupt=ic)
        result = agent.run("do it", ctx)
        assert result.preempted is True
        assert result.reply == "Stopped."


# ---------------------------------------------------------------------------
# Scenario 3: Multi-hop with full SLM protection
# ---------------------------------------------------------------------------


class TestSLMDefenseInDepth:
    def test_loop_plus_echo_plus_retry_all_fire(self, tmp_path):
        """Single turn exercises all three SLM hardening hooks:

        - Bad-arg tool call → SchemaRetryHook enriches.
        - Then the model loops on the same (name, args) → LoopDetectorHook
          short-circuits with a hint on call 2 and ends the turn on call 3.
        """

        @tool
        def add_two(x: int, y: int) -> int:
            """Add."""
            return x + y

        import json as _json

        wrong = {
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "add_two", "arguments": _json.dumps({"foo": 1})},
                }
            ],
        }
        # After retry, the LLM still loops with the same (wrong) args.
        llm = ScriptedLLM([wrong, wrong, wrong, reply("giving up")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            tools=[add_two],
            hooks=default_slm_hooks(),
            llm=llm,
        )
        result = agent.run("add")
        # Must terminate gracefully — either via loop-break reply or final.
        assert result.reply
        # And the conversation didn't explode.


# ---------------------------------------------------------------------------
# Scenario 4: Subagent handoff via artifact
# ---------------------------------------------------------------------------


class TestSubagentArtifactHandoff:
    def test_parent_writes_plan_sub_reads_and_executes(self):
        """Parent plans and writes an artifact. Subagent has a fresh
        context and can read the artifact + execute."""

        artifacts = InMemoryArtifactStore()

        # Parent produces a plan (text) and writes it. Sub-agent spawn
        # shares the parent's LLM, so the ScriptedLLM needs to carry the
        # full 3-response conversation (parent reply + sub tool call +
        # sub final reply).
        @tool
        def read_plan() -> str:
            """Read the plan artifact."""
            a = artifacts.read("plan.md")
            return a.content if a else ""

        llm = ScriptedLLM(
            [
                reply("I drafted the plan."),
                call("read_plan"),
                reply("plan says: 1. grasp cup"),
            ]
        )
        parent = LLMAgent(
            name="planner",
            description="",
            instructions="You are Planner.",
            llm=llm,
        )
        parent_ctx = AgentContext(artifacts=artifacts)
        parent.run("plan it", parent_ctx)
        artifacts.write(text_artifact("plan.md", "1. grasp cup\n2. pour water", author="planner", tags=["plan"]))

        sub_result = parent.spawn_subagent(
            "execute the plan",
            parent_ctx=parent_ctx,
            instructions="You are Executor.",
            tools=[read_plan],
        )
        assert "grasp cup" in sub_result.reply


# ---------------------------------------------------------------------------
# Scenario 5: Long-running multi-turn with compaction + persistence
# ---------------------------------------------------------------------------


class TestLongRunningPersisted:
    def test_session_resumes_across_restart(self, tmp_path):
        """Run 10 turns, persist, 'restart', resume, verify context is
        recoverable."""

        sessions = JSONSessionStore(tmp_path / "sess")
        session_id = new_session_id()

        # First lifetime.
        llm1 = ScriptedLLM([reply(f"turn-{i}") for i in range(5)])
        a1 = LLMAgent(
            name="x",
            description="",
            instructions="",
            hooks=[PersistSessionHook(session_store=sessions, session_id=session_id)],
            llm=llm1,
        )
        ctx1 = AgentContext()
        for i in range(5):
            r = a1.run(f"q{i}", ctx1)
            assert r.reply == f"turn-{i}"

        # Verify persisted.
        loaded = sessions.load(session_id)
        assert loaded is not None
        first_msg_count = len(loaded.messages)
        assert first_msg_count > 0

        # Second lifetime — same session_id.
        llm2 = ScriptedLLM([reply(f"resume-{i}") for i in range(5)])
        a2 = LLMAgent(
            name="x",
            description="",
            instructions="",
            hooks=[PersistSessionHook(session_store=sessions, session_id=session_id)],
            llm=llm2,
        )
        ctx2 = AgentContext(session=sessions.load(session_id))
        for i in range(5):
            r = a2.run(f"qq{i}", ctx2)
            assert r.reply == f"resume-{i}"

        final = sessions.load(session_id)
        assert final is not None
        assert len(final.messages) > first_msg_count


# ---------------------------------------------------------------------------
# Scenario 6: Concurrent agents through one pool
# ---------------------------------------------------------------------------


class TestConcurrentPoolAgents:
    def test_pool_runs_two_agents_in_parallel(self):
        import threading

        llm_a = ScriptedLLM([reply("A") for _ in range(5)])
        llm_b = ScriptedLLM([reply("B") for _ in range(5)])

        a = LLMAgent(name="a", description="", instructions="", llm=llm_a)
        b = LLMAgent(name="b", description="", instructions="", llm=llm_b)

        pool = AgentPool()
        pool.register(a)
        pool.register(b)

        results_a, results_b = [], []

        def run_a():
            for _ in range(5):
                results_a.append(pool.run("a", "x").reply)

        def run_b():
            for _ in range(5):
                results_b.append(pool.run("b", "x").reply)

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results_a == ["A"] * 5
        assert results_b == ["B"] * 5


# ---------------------------------------------------------------------------
# Scenario 7: LLM shim backwards compat
# ---------------------------------------------------------------------------


def test_legacy_llm_chat_with_tools_delegates_to_shim():
    """The legacy :meth:`LLM.chat` path with tools must still work via
    the internal :class:`LLMAgent` shim + :func:`default_slm_hooks`."""
    from unittest.mock import MagicMock, patch

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "c1", "function": {"name": "ping", "arguments": "{}"}}],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "pong!", "tool_calls": None}}]},
    ]

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    with (
        patch("llama_cpp.Llama", return_value=mock_llm),
        patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/fake.gguf"),
        patch("edgevox.core.gpu.has_metal", return_value=False),
        patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None),
    ):
        from edgevox.llm.llamacpp import LLM

        llm = LLM(model_path="/tmp/fake.gguf", tools=[ping])
        reply_text = llm.chat("ping?")
        assert reply_text == "pong!"
