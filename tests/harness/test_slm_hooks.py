"""Tests for LoopDetectorHook, EchoedPayloadHook, SchemaRetryHook."""

from __future__ import annotations

from edgevox.agents.base import LLMAgent
from edgevox.llm import tool
from edgevox.llm.hooks_slm import (
    EchoedPayloadHook,
    LoopDetectorHook,
    SchemaRetryHook,
    default_slm_hooks,
)

from .conftest import ScriptedLLM, call, reply


def _make_agent(llm, *, tools=None, hooks=None):
    return LLMAgent(
        name="slm",
        description="slm test",
        instructions="You are Testbot.",
        tools=tools,
        llm=llm,
        hooks=hooks,
    )


@tool
def ping() -> str:
    """Ping."""
    return "pong"


@tool
def add_two(x: int, y: int) -> int:
    """Add."""
    return x + y


# ---------------------------------------------------------------------------
# LoopDetectorHook
# ---------------------------------------------------------------------------


class TestLoopDetectorHook:
    def test_second_call_gets_hint(self):
        actual = []

        @tool
        def counted() -> int:
            """Counted."""
            actual.append(1)
            return len(actual)

        llm = ScriptedLLM(
            [
                call("counted"),
                call("counted"),
                reply("done"),
            ]
        )
        agent = _make_agent(llm, tools=[counted], hooks=[LoopDetectorHook()])
        agent.run("loop me")
        # Tool body runs only once; the hook substitutes on the 2nd call.
        assert len(actual) == 1

    def test_third_identical_call_ends_turn(self):
        actual = []

        @tool
        def counted() -> int:
            """Counted."""
            actual.append(1)
            return len(actual)

        llm = ScriptedLLM(
            [
                call("counted"),
                call("counted"),
                call("counted"),
                reply("never reached"),
            ],
        )
        agent = _make_agent(
            llm,
            tools=[counted],
            hooks=[LoopDetectorHook()],
        )
        result = agent.run("loop hard")
        # End-of-turn reply from loop detector.
        assert "couldn't" in result.reply.lower() or "loop" in (result.hook_ended or "").lower()
        # Tool ran once (2nd hint + 3rd break both short-circuit dispatch / end turn).
        assert len(actual) == 1

    def test_different_args_dont_trigger_loop(self):
        llm = ScriptedLLM(
            [
                call("add_two", x=1, y=2),
                call("add_two", x=3, y=4),  # different args — not a loop
                reply("done"),
            ]
        )
        agent = _make_agent(llm, tools=[add_two], hooks=[LoopDetectorHook()])
        result = agent.run("two adds")
        assert result.reply == "done"
        assert len(llm.calls) == 3

    def test_counts_reset_between_turns(self):
        # Use max_tool_hops=1 so each turn has at most 1 tool call.
        llm = ScriptedLLM(
            [
                call("ping"),
                reply("turn1"),
                call("ping"),  # same fingerprint across turns — should still go through
                reply("turn2"),
            ]
        )
        agent = _make_agent(llm, tools=[ping], hooks=[LoopDetectorHook()])
        r1 = agent.run("ping me")
        r2 = agent.run("ping again")
        assert r1.reply == "turn1"
        assert r2.reply == "turn2"


# ---------------------------------------------------------------------------
# EchoedPayloadHook
# ---------------------------------------------------------------------------


class TestEchoedPayloadHook:
    def test_echoed_json_replaced(self):
        llm = ScriptedLLM(
            [
                {
                    "content": '{"ok": true, "result": "42", "retry_hint": null}',
                    "tool_calls": None,
                },
            ]
        )
        agent = _make_agent(llm, hooks=[EchoedPayloadHook(fallback="sorry, no")])
        result = agent.run("hi")
        assert result.reply == "sorry, no"

    def test_normal_reply_passes_through(self):
        llm = ScriptedLLM([reply("Hello there!")])
        agent = _make_agent(llm, hooks=[EchoedPayloadHook()])
        result = agent.run("hi")
        assert result.reply == "Hello there!"

    def test_does_not_fire_when_tool_calls_present(self):
        # When tool_calls exist, the content isn't the final reply.
        llm = ScriptedLLM(
            [
                call("ping"),
                reply("done"),
            ]
        )
        agent = _make_agent(llm, tools=[ping], hooks=[EchoedPayloadHook()])
        result = agent.run("ping")
        assert result.reply == "done"


# ---------------------------------------------------------------------------
# SchemaRetryHook
# ---------------------------------------------------------------------------


class TestSchemaRetryHook:
    def test_bad_args_get_schema_hint(self):
        # The LLM calls `add_two` with a nonsense kwarg name, then correctly.
        import json

        # First call: wrong kwarg 'foo' → tool raises TypeError-style error.
        wrong = {
            "content": None,
            "tool_calls": [
                {
                    "id": "c_wrong",
                    "function": {"name": "add_two", "arguments": json.dumps({"foo": 1, "y": 2})},
                }
            ],
        }
        correct = call("add_two", x=1, y=2)
        llm = ScriptedLLM([wrong, correct, reply("three")])

        agent = _make_agent(llm, tools=[add_two], hooks=[SchemaRetryHook()])
        result = agent.run("add 1 and 2")
        assert result.reply == "three"

        # The 2nd LLM call's messages should carry the retry_hint.
        second_call_msgs = llm.calls[1]["messages"]
        all_content = " ".join(str(m.get("content") or "") for m in second_call_msgs)
        assert "retry_hint" in all_content or "ONLY" in all_content  # either format

    def test_budget_caps_retries(self):
        # Two consecutive bad calls — only the first gets a hint.
        import json

        bad1 = {
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "add_two", "arguments": json.dumps({"foo": 1})}},
            ],
        }
        bad2 = {
            "content": None,
            "tool_calls": [
                {"id": "c2", "function": {"name": "add_two", "arguments": json.dumps({"bar": 2})}},
            ],
        }
        llm = ScriptedLLM([bad1, bad2, reply("giving up")])
        agent = _make_agent(llm, tools=[add_two], hooks=[SchemaRetryHook(max_retries_per_tool=1)])
        result = agent.run("x")
        assert result.reply == "giving up"


# ---------------------------------------------------------------------------
# default_slm_hooks() bundle
# ---------------------------------------------------------------------------


def test_default_bundle_compose_without_error():
    llm = ScriptedLLM([reply("ok")])
    agent = _make_agent(llm, hooks=default_slm_hooks())
    result = agent.run("hi")
    assert result.reply == "ok"
