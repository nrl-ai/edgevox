"""Tests for the built-in hook library (guardrails, plan mode, memory, ...)."""

from __future__ import annotations

import json

from edgevox.agents.base import AgentContext, LLMAgent
from edgevox.agents.hooks_builtin import (
    AuditLogHook,
    ContextCompactionHook,
    EchoingHook,
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
from edgevox.agents.memory import Compactor
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, reply


def _make_agent(llm, *, tools=None, hooks=None, skills=None, instructions="You are Testbot."):
    return LLMAgent(
        name="tester",
        description="test",
        instructions=instructions,
        tools=tools,
        skills=skills,
        hooks=hooks,
        llm=llm,
    )


# ---------------------------------------------------------------------------
# SafetyGuardrailHook
# ---------------------------------------------------------------------------


class TestSafetyGuardrail:
    def test_blocklist_blocks(self):
        llm = ScriptedLLM([])  # never called
        agent = _make_agent(
            llm,
            hooks=[SafetyGuardrailHook(blocklist=["forbidden"], reply="no way")],
        )
        r = agent.run("please do something FORBIDDEN")
        assert r.reply == "no way"
        assert llm.calls == []

    def test_allowlist_gates(self):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(
            llm,
            hooks=[SafetyGuardrailHook(allowlist=["weather"], reply="denied")],
        )
        r = agent.run("what's the weather")
        assert r.reply == "ok"
        r2 = agent.run("tell me a joke")
        assert r2.reply == "denied"

    def test_empty_input_passes(self):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[SafetyGuardrailHook(blocklist=["boom"])])
        r = agent.run("hello")
        assert r.reply == "ok"


# ---------------------------------------------------------------------------
# PlanModeHook
# ---------------------------------------------------------------------------


class TestPlanMode:
    def test_approver_accepts(self):
        calls_ok = []

        @tool
        def grasp(item: str) -> str:
            """Grasp."""
            calls_ok.append(item)
            return f"grasped {item}"

        def always_yes(tool, args, ctx):
            return True

        llm = ScriptedLLM([call("grasp", item="red_block"), reply("done")])
        agent = _make_agent(
            llm,
            tools=[grasp],
            hooks=[PlanModeHook(confirm=["grasp"], approver=always_yes)],
        )
        r = agent.run("grasp red block")
        assert r.reply == "done"
        assert calls_ok == ["red_block"]

    def test_approver_declines(self):
        calls_ok = []

        @tool
        def grasp(item: str) -> str:
            """Grasp."""
            calls_ok.append(item)
            return "grasped"

        def always_no(tool, args, ctx):
            return False

        llm = ScriptedLLM([call("grasp", item="hot"), reply("sorry, couldn't")])
        agent = _make_agent(
            llm,
            tools=[grasp],
            hooks=[PlanModeHook(confirm=["grasp"], approver=always_no)],
        )
        r = agent.run("grasp")
        assert r.reply == "sorry, couldn't"
        assert calls_ok == []  # tool body never ran

    def test_non_confirm_tools_unaffected(self):
        @tool
        def ping() -> str:
            """."""
            return "pong"

        def no(tool, args, ctx):
            return False

        llm = ScriptedLLM([call("ping"), reply("done")])
        agent = _make_agent(
            llm,
            tools=[ping],
            hooks=[PlanModeHook(confirm=["dangerous"], approver=no)],
        )
        assert agent.run("ping").reply == "done"

    def test_approver_exception_treated_as_decline(self):
        @tool
        def grasp() -> str:
            """."""
            return "grasped"

        def broken(tool, args, ctx):
            raise RuntimeError("UI crashed")

        llm = ScriptedLLM([call("grasp"), reply("declined")])
        agent = _make_agent(
            llm,
            tools=[grasp],
            hooks=[PlanModeHook(confirm=["grasp"], approver=broken)],
        )
        r = agent.run("grasp")
        assert r.reply == "declined"


# ---------------------------------------------------------------------------
# TokenBudgetHook
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_under_budget_no_change(self):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[TokenBudgetHook(max_context_tokens=10_000)])
        agent.run("hi")
        # Nothing trimmed.
        assert llm.calls[0]["messages"][-1]["content"] == "hi"

    def test_over_budget_truncates(self):
        # Seed session with many bloated turns.
        from edgevox.agents.base import Session

        ctx = AgentContext(session=Session(messages=[{"role": "system", "content": "sys"}]))
        for _ in range(30):
            ctx.session.messages.append({"role": "user", "content": "x" * 2000})
            ctx.session.messages.append({"role": "assistant", "content": "y"})

        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[TokenBudgetHook(max_context_tokens=500, keep_last=2)])
        agent.run("final", ctx)
        # The LLM saw a truncated list.
        msgs = llm.calls[0]["messages"]
        assert len(msgs) <= 8  # system + a few tail turns + the new user


# ---------------------------------------------------------------------------
# ToolOutputTruncatorHook
# ---------------------------------------------------------------------------


class TestToolOutputTruncator:
    def test_long_string_truncated(self):
        @tool
        def big() -> str:
            """."""
            return "a" * 5000

        llm = ScriptedLLM([call("big"), reply("done")])
        agent = _make_agent(llm, tools=[big], hooks=[ToolOutputTruncatorHook(max_chars=100)])
        agent.run("x")
        # Second call must carry truncated content.
        found = False
        for m in llm.calls[1]["messages"]:
            content = str(m.get("content", ""))
            if "truncated" in content:
                found = True
        assert found

    def test_short_result_unchanged(self):
        @tool
        def small() -> str:
            """."""
            return "short"

        llm = ScriptedLLM([call("small"), reply("ok")])
        agent = _make_agent(llm, tools=[small], hooks=[ToolOutputTruncatorHook(max_chars=1000)])
        agent.run("x")
        all_contents = " ".join(str(m.get("content", "")) for m in llm.calls[1]["messages"])
        assert "truncated" not in all_contents


# ---------------------------------------------------------------------------
# MemoryInjectionHook
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    def test_facts_injected_into_system(self, tmp_memory_store):
        tmp_memory_store.add_fact("user_name", "Anh")
        tmp_memory_store.add_fact("kettle_location", "drawer 2", scope="kitchen")

        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[MemoryInjectionHook(memory_store=tmp_memory_store)])
        agent.run("hi")

        system = llm.calls[0]["messages"][0]["content"]
        assert "Anh" in system
        assert "kettle_location" in system

    def test_empty_memory_no_change(self, tmp_memory_store):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[MemoryInjectionHook(memory_store=tmp_memory_store)])
        agent.run("hi")
        system = llm.calls[0]["messages"][0]["content"]
        assert "## Memory" not in system


# ---------------------------------------------------------------------------
# NotesInjectorHook
# ---------------------------------------------------------------------------


class TestNotesInjector:
    def test_tail_injected(self, tmp_notes):
        tmp_notes.append("The kettle is in drawer 2.")
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[NotesInjectorHook(tmp_notes)])
        agent.run("hi")
        system = llm.calls[0]["messages"][0]["content"]
        assert "drawer 2" in system


# ---------------------------------------------------------------------------
# ContextCompactionHook
# ---------------------------------------------------------------------------


class TestContextCompaction:
    def test_compacts_long_session(self):
        from edgevox.agents.base import Session

        ctx = AgentContext(
            session=Session(
                messages=[{"role": "system", "content": "sys"}]
                + [{"role": r, "content": "x" * 500} for r in ["user", "assistant"] * 20]
            )
        )
        initial_len = len(ctx.session.messages)

        compactor = Compactor(trigger_tokens=500, keep_last_turns=2)
        llm = ScriptedLLM([reply("ok")])
        # Skip LLM-based summarization in this unit test; the fallback
        # summarizer is exercised in test_memory.py::TestCompactor.
        agent = _make_agent(
            llm,
            hooks=[ContextCompactionHook(compactor=compactor, llm_getter=lambda _c: None)],
        )
        agent.run("new turn", ctx)

        # Session was compacted.
        assert len(ctx.session.messages) < initial_len


# ---------------------------------------------------------------------------
# EpisodeLoggerHook + PersistSessionHook
# ---------------------------------------------------------------------------


class TestEpisodeAndPersist:
    def test_episode_logger_records_tool_outcomes(self, tmp_memory_store):
        @tool
        def hello() -> str:
            """."""
            return "world"

        llm = ScriptedLLM([call("hello"), reply("done")])
        agent = _make_agent(
            llm,
            tools=[hello],
            hooks=[EpisodeLoggerHook(memory_store=tmp_memory_store, agent_name="alpha")],
        )
        agent.run("hi")
        episodes = tmp_memory_store.recent_episodes(5)
        assert len(episodes) == 1
        assert episodes[0].kind == "tool_call"
        assert episodes[0].outcome == "ok"
        assert episodes[0].agent == "alpha"

    def test_persist_session_saves(self, tmp_session_store):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(
            llm,
            hooks=[PersistSessionHook(session_store=tmp_session_store, session_id="abc")],
        )
        agent.run("hi")
        loaded = tmp_session_store.load("abc")
        assert loaded is not None
        assert any(m.get("content") == "hi" for m in loaded.messages)


# ---------------------------------------------------------------------------
# AuditLogHook
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_writes_jsonl(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[AuditLogHook(path=log)])
        agent.run("hi")
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        # At least one after_llm and one on_run_end record.
        points = {r["point"] for r in records}
        assert "after_llm" in points
        assert "on_run_end" in points


# ---------------------------------------------------------------------------
# Diagnostic hooks
# ---------------------------------------------------------------------------


class TestEchoingHook:
    def test_logs_every_point(self):
        events = []
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, hooks=[EchoingHook(logger=events.append)])
        agent.run("hi")
        # All 6 points observed.
        tagged = [line.split("]")[0] for line in events]
        assert any("on_run_start" in t for t in tagged)
        assert any("before_llm" in t for t in tagged)
        assert any("after_llm" in t for t in tagged)
        assert any("on_run_end" in t for t in tagged)


class TestTimingHook:
    def test_records_llm_and_tool_timings(self):
        @tool
        def ping() -> str:
            """."""
            return "pong"

        hook = TimingHook()
        llm = ScriptedLLM([call("ping"), reply("done")])
        agent = _make_agent(llm, tools=[ping], hooks=[hook])
        agent.run("hi")
        kinds = [k for k, _ in hook.timings]
        assert "llm" in kinds
        assert "tool" in kinds
