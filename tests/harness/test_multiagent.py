"""Tests for Blackboard, AgentMessage, BackgroundAgent, AgentPool."""

from __future__ import annotations

import threading
import time

import pytest

from edgevox.agents.base import AgentContext, AgentEvent, LLMAgent
from edgevox.agents.bus import EventBus
from edgevox.agents.multiagent import (
    AgentPool,
    BackgroundAgent,
    Blackboard,
    debounce_trigger,
    send_message,
    subscribe_inbox,
)

from .conftest import ScriptedLLM, reply

# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


class TestBlackboard:
    def test_get_set(self):
        bb = Blackboard()
        assert bb.get("k") is None
        bb.set("k", 42)
        assert bb.get("k") == 42

    def test_default(self):
        bb = Blackboard()
        assert bb.get("missing", default="fallback") == "fallback"

    def test_watch_fires(self):
        bb = Blackboard()
        seen = []
        bb.watch("x", lambda k, old, new: seen.append((k, old, new)))
        bb.set("x", 1)
        bb.set("x", 2)
        assert seen == [("x", None, 1), ("x", 1, 2)]

    def test_watch_wildcard(self):
        bb = Blackboard()
        seen = []
        bb.watch("*", lambda k, old, new: seen.append((k, new)))
        bb.set("a", 1)
        bb.set("b", 2)
        assert seen == [("a", 1), ("b", 2)]

    def test_unsubscribe(self):
        bb = Blackboard()
        seen = []
        unsub = bb.watch("k", lambda k, o, n: seen.append(n))
        bb.set("k", 1)
        unsub()
        bb.set("k", 2)
        assert seen == [1]

    def test_delete_fires_watcher(self):
        bb = Blackboard()
        seen = []
        bb.watch("k", lambda k, o, n: seen.append((o, n)))
        bb.set("k", 1)
        bb.delete("k")
        assert seen == [(None, 1), (1, None)]

    def test_snapshot_is_copy(self):
        bb = Blackboard()
        bb.set("a", 1)
        snap = bb.snapshot()
        bb.set("a", 2)
        assert snap == {"a": 1}

    def test_thread_safe_concurrent_writes(self):
        bb = Blackboard()

        def writer(prefix, n):
            for i in range(n):
                bb.set(f"{prefix}-{i}", i)

        threads = [threading.Thread(target=writer, args=(p, 100)) for p in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(bb.keys()) == 400


# ---------------------------------------------------------------------------
# Agent messaging over the bus
# ---------------------------------------------------------------------------


class TestMessaging:
    def test_direct_message_reaches_named_inbox(self):
        bus = EventBus()
        inbox = []
        subscribe_inbox(bus, agent_name="panda", handler=lambda m: inbox.append(m.content))
        send_message(bus, from_agent="home", to="panda", content="pick red block")
        assert inbox == ["pick red block"]

    def test_broadcast_reaches_all_inboxes(self):
        bus = EventBus()
        a_in, b_in = [], []
        subscribe_inbox(bus, agent_name="a", handler=lambda m: a_in.append(m.content))
        subscribe_inbox(bus, agent_name="b", handler=lambda m: b_in.append(m.content))
        send_message(bus, from_agent="x", to="*", content="hello")
        assert a_in == ["hello"] and b_in == ["hello"]

    def test_message_not_delivered_to_others(self):
        bus = EventBus()
        a_in = []
        b_in = []
        subscribe_inbox(bus, agent_name="a", handler=lambda m: a_in.append(m.content))
        subscribe_inbox(bus, agent_name="b", handler=lambda m: b_in.append(m.content))
        send_message(bus, from_agent="x", to="a", content="only-a")
        assert a_in == ["only-a"]
        assert b_in == []


# ---------------------------------------------------------------------------
# BackgroundAgent
# ---------------------------------------------------------------------------


def _make_agent(llm, name="bg", **kw):
    return LLMAgent(
        name=name,
        description="bg",
        instructions="You are BG.",
        llm=llm,
        **kw,
    )


class TestBackgroundAgent:
    def test_trigger_fires_agent(self):
        llm = ScriptedLLM([reply("saw it")])
        agent = _make_agent(llm)
        bus = EventBus()
        ctx = AgentContext(bus=bus)

        bg = BackgroundAgent(
            agent,
            trigger=lambda ev: "respond" if ev.kind == "sensor" else None,
        )
        bg.start(ctx, bus)
        bus.publish(AgentEvent(kind="sensor", agent_name="env"))
        # Give the worker thread time to pick it up.
        time.sleep(0.1)
        bg.stop()
        assert len(bg.results) == 1
        assert bg.results[0].reply == "saw it"

    def test_trigger_returning_none_skips(self):
        llm = ScriptedLLM([])  # must not be called
        agent = _make_agent(llm)
        bus = EventBus()
        ctx = AgentContext(bus=bus)

        bg = BackgroundAgent(agent, trigger=lambda ev: None)
        bg.start(ctx, bus)
        bus.publish(AgentEvent(kind="sensor", agent_name="env"))
        time.sleep(0.05)
        bg.stop()
        assert bg.results == []

    def test_stop_halts_cleanly(self):
        llm = ScriptedLLM([reply("x")] * 5)
        agent = _make_agent(llm)
        bus = EventBus()
        ctx = AgentContext(bus=bus)

        bg = BackgroundAgent(agent, trigger=lambda ev: "run" if ev.kind == "tick" else None)
        bg.start(ctx, bus)
        for _ in range(3):
            bus.publish(AgentEvent(kind="tick", agent_name="x"))
        time.sleep(0.1)
        bg.stop(timeout=1.0)
        # Thread must actually terminate.
        assert bg._thread is None

    def test_queue_backpressure_drops_oldest(self):
        from unittest.mock import MagicMock

        # Agent stub that blocks so the queue actually fills up.
        fake = MagicMock()
        fake.name = "slow"
        fake.run = MagicMock(side_effect=lambda t, c: (time.sleep(0.3), None)[1])

        bus = EventBus()
        ctx = AgentContext(bus=bus)
        bg = BackgroundAgent(fake, trigger=lambda ev: "go", max_queue=3)
        bg.start(ctx, bus)
        for _ in range(20):
            bus.publish(AgentEvent(kind="any", agent_name="z"))
        time.sleep(0.05)
        # Queue should never exceed max_queue.
        assert len(bg._queue) <= 3
        bg.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------


class TestAgentPool:
    def test_register_and_get(self):
        llm = ScriptedLLM([reply("a")])
        agent = _make_agent(llm, name="alpha")
        pool = AgentPool()
        pool.register(agent)
        assert pool.get("alpha") is agent
        assert "alpha" in pool.names()

    def test_run_through_pool(self):
        llm = ScriptedLLM([reply("ok")])
        agent = _make_agent(llm, name="alpha")
        pool = AgentPool()
        pool.register(agent)
        result = pool.run("alpha", "hi")
        assert result.reply == "ok"

    def test_run_unknown_raises(self):
        pool = AgentPool()
        with pytest.raises(KeyError):
            pool.run("nope", "x")

    def test_shared_bus_across_agents(self):
        llm_a = ScriptedLLM([reply("a")])
        llm_b = ScriptedLLM([reply("b")])
        a = _make_agent(llm_a, name="a")
        b = _make_agent(llm_b, name="b")
        pool = AgentPool()
        pool.register(a)
        pool.register(b)
        ctx = pool.make_ctx()
        assert ctx.bus is pool.bus
        assert ctx.blackboard is pool.blackboard

    def test_start_and_stop_background(self):
        llm = ScriptedLLM([reply("bg-saw")] * 2)
        agent = _make_agent(llm, name="watcher")
        pool = AgentPool()
        pool.register(agent)
        bg = pool.start_background("watcher", trigger=lambda ev: "go" if ev.kind == "tick" else None)
        pool.bus.publish(AgentEvent(kind="tick", agent_name="env"))
        time.sleep(0.1)
        pool.stop_all()
        assert bg._thread is None


# ---------------------------------------------------------------------------
# debounce_trigger
# ---------------------------------------------------------------------------


def test_debounce_trigger_limits_rate():
    # Trigger fires task for every event.
    base = lambda ev: "run"  # noqa: E731
    t = debounce_trigger(base, interval_s=0.2)

    class Dummy:
        kind = "k"
        agent_name = ""
        payload = None

    ev = Dummy()
    assert t(ev) == "run"
    assert t(ev) is None
    time.sleep(0.25)
    assert t(ev) == "run"
