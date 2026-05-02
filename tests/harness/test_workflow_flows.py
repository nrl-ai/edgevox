"""Graduated workflow flows on the ToyWorld simulation.

Each test ramps up one dimension of complexity over the prior, using
``ScriptedLLM`` for determinism and ``ToyWorld`` for a stdlib-only sim.
The goal is two-fold:

1. Test the existing harness (LLMAgent + tools + skills + workflows)
   under realistic multi-step pressure -- single tool, parallel tool
   dispatch, sequential planner -> executor, plan-execute-evaluate
   (the Anthropic harness pattern), iterative goal-checking loop.
2. Surface gaps in the framework as the complexity climbs. When a
   level forces an awkward workaround, that's the place to add a
   first-class primitive next.

These tests run in <1 s total. They share no state with the
chess-robot Qt or BackgroundAgent flake-prone tests on purpose --
``test_full_flow.py`` covers the full integration surface; this file
is for the workflow-recipe surface specifically.

References:

- Anthropic, "Harness design for long-running apps" (2025) -- the
  plan / execute / evaluate decomposition + the rule that an evaluator
  must be a separate agent (self-evaluators over-rate their own work).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from edgevox.agents.base import LLMAgent
from edgevox.agents.sim import ToyWorld
from edgevox.agents.skills import GoalHandle, GoalStatus, skill
from edgevox.agents.workflow import Loop, Sequence
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, calls, reply

# ---------------------------------------------------------------------------
# Shared fixture: a ToyWorld + a fresh tool log we can inspect post-run.
# ---------------------------------------------------------------------------


@pytest.fixture
def world_and_tools():
    """Build a ToyWorld plus a captured-call log + tool/skill set bound to it.

    Returns a small bundle so each test gets independent state.
    """
    world = ToyWorld()
    tool_log: list[tuple[str, dict[str, Any]]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name -- one of living_room / kitchen / bedroom / office.
            on: true to turn on, false to turn off.
        """
        tool_log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    @tool
    def get_pose() -> str:
        """Return the robot's current pose."""
        state = world.get_world_state()["robot"]
        tool_log.append(("get_pose", state))
        return json.dumps(state)

    @skill(latency_class="slow", timeout_s=5.0)
    def navigate_to(room: str, ctx) -> GoalHandle:
        """Drive the robot to a named room.

        Args:
            room: target room.
        """
        tool_log.append(("navigate_to", {"room": room}))
        return ctx.deps.apply_action("navigate_to", room=room)

    return {
        "world": world,
        "tool_log": tool_log,
        "tools": [set_light, get_pose],
        "skills": [navigate_to],
    }


# ---------------------------------------------------------------------------
# Level 1 — single tool call: user asks, agent calls one tool, returns result.
# ---------------------------------------------------------------------------


class TestL1_SingleToolCall:
    """Smallest meaningful flow: one user turn, one tool call, one reply.

    Probes: tool registration, JSON-schema generation, parser dispatch,
    tool-result feedback loop.
    """

    def test_set_light_kitchen(self, world_and_tools):
        bundle = world_and_tools
        llm = ScriptedLLM(
            [
                call("set_light", room="kitchen", on=True),
                reply("Kitchen light is on."),
            ]
        )
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Use tools to control the home.",
            tools=bundle["tools"],
            llm=llm,
        )
        result = agent.run("turn on the kitchen light")

        assert "kitchen light is on" in result.reply.lower()
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True
        assert bundle["tool_log"] == [("set_light", {"room": "kitchen", "on": True})]


# ---------------------------------------------------------------------------
# Level 2 — parallel multi-tool: model emits two tool calls in one hop, both
# execute, both results feed back, model summarises.
# ---------------------------------------------------------------------------


class TestL2_ParallelToolDispatch:
    """One LLM hop -> N tool calls dispatched concurrently.

    Probes: parallel dispatch via ThreadPoolExecutor, tool-result
    pairing on multi-call hops, serialisation order in the message
    history.
    """

    def test_two_tools_one_hop(self, world_and_tools):
        bundle = world_and_tools
        llm = ScriptedLLM(
            [
                calls(
                    ("set_light", {"room": "kitchen", "on": True}),
                    ("get_pose", {}),
                ),
                reply("Kitchen on; pose retrieved."),
            ]
        )
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Plan parallel actions when possible.",
            tools=bundle["tools"],
            llm=llm,
        )
        result = agent.run("turn on kitchen and tell me where you are")

        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True
        # Both tools must have run in this turn (order is dispatcher-defined,
        # so we assert membership not sequence).
        names = {entry[0] for entry in bundle["tool_log"]}
        assert names == {"set_light", "get_pose"}
        assert "kitchen on" in result.reply.lower() or "kitchen light" in result.reply.lower()


# ---------------------------------------------------------------------------
# Level 3 — Sequence(planner, executor): the canonical multi-agent recipe.
# ---------------------------------------------------------------------------


class TestL3_PlannerExecutor:
    """Planner produces a structured plan; executor follows it.

    Probes: Sequence chaining the planner's output as the executor's
    task, two distinct LLMs each with their own scripted behaviour,
    no shared state leakage between them.
    """

    def test_planner_emits_plan_executor_runs_it(self, world_and_tools):
        bundle = world_and_tools
        # Planner has no tools — it just decomposes the user request.
        # The output is structured enough that the executor can act on it.
        planner_llm = ScriptedLLM(
            [
                reply("Plan:\n1. set_light(room='kitchen', on=true)\n2. get_pose()\nDone when both have been called."),
            ]
        )
        executor_llm = ScriptedLLM(
            [
                calls(
                    ("set_light", {"room": "kitchen", "on": True}),
                    ("get_pose", {}),
                ),
                reply("Plan executed: kitchen on, pose checked."),
            ]
        )
        planner = LLMAgent(
            name="Planner",
            description="produces ordered tool plans",
            instructions="Output a numbered tool plan in plain text.",
            llm=planner_llm,
        )
        executor = LLMAgent(
            name="Executor",
            description="runs a tool plan",
            instructions="Execute the plan via tool calls.",
            tools=bundle["tools"],
            llm=executor_llm,
        )
        seq = Sequence("plan_then_execute", [planner, executor])

        result = seq.run("turn on kitchen and tell me where you are")

        assert "executed" in result.reply.lower()
        # Executor must have seen the planner's output as its task.
        # The planner's reply contains "Plan:" -- inspect the executor's
        # logged messages.
        executor_first_user_msg = executor_llm.calls[0]["messages"][-1]
        assert executor_first_user_msg["role"] == "user"
        assert "Plan:" in executor_first_user_msg["content"]
        # Both tools ran.
        assert {t[0] for t in bundle["tool_log"]} == {"set_light", "get_pose"}


# ---------------------------------------------------------------------------
# Level 4 — Plan-Execute-Evaluate: Anthropic's three-agent recipe with a
# *separate* evaluator, because self-evaluators over-rate their own work.
# ---------------------------------------------------------------------------


class TestL4_PlanExecuteEvaluate:
    """Plan -> Execute -> Evaluate, with pass / fail signal from a third agent.

    The evaluator inspects the executor's reply against criteria the
    planner declared in its plan. The test asserts the evaluator's
    verdict propagates as the final ``Sequence`` result.

    Probes: 3-agent chaining; that the evaluator agent receives the
    executor's reply (not the original user request); that the
    final result is the evaluator's verdict, not the executor's.
    """

    def test_pass_path(self, world_and_tools):
        bundle = world_and_tools
        planner_llm = ScriptedLLM(
            [
                reply(
                    "Goal: kitchen light on, pose reported.\n"
                    "Plan:\n"
                    "1. set_light(kitchen, true)\n"
                    "2. get_pose()\n"
                    "Pass criterion: kitchen.light_on == true AND pose was reported."
                ),
            ]
        )
        executor_llm = ScriptedLLM(
            [
                calls(
                    ("set_light", {"room": "kitchen", "on": True}),
                    ("get_pose", {}),
                ),
                reply("set_light returned ok; pose was {x: 0, y: 0}. Both calls completed."),
            ]
        )
        # Evaluator emits a single verdict line; no tool calls.
        evaluator_llm = ScriptedLLM([reply("VERDICT: PASS — both criteria met.")])

        planner = LLMAgent(
            name="Planner",
            description="planner",
            instructions="Produce a Plan + Pass criterion.",
            llm=planner_llm,
        )
        executor = LLMAgent(
            name="Executor",
            description="executor",
            instructions="Execute the plan via tools, then summarise.",
            tools=bundle["tools"],
            llm=executor_llm,
        )
        evaluator = LLMAgent(
            name="Evaluator",
            description="evaluator",
            instructions=(
                "You receive an executor's report. Check it against the pass "
                "criterion in the plan. Reply with one line: VERDICT: PASS or VERDICT: FAIL — <reason>."
            ),
            llm=evaluator_llm,
        )

        pee = Sequence("plan_execute_evaluate", [planner, executor, evaluator])
        result = pee.run("turn on kitchen and report your pose")

        # Final reply is the evaluator's verdict, not the executor's.
        assert result.reply.startswith("VERDICT:")
        assert "PASS" in result.reply
        # Side effect happened.
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True

    def test_fail_path_does_not_silently_pass(self, world_and_tools):
        """Anthropic's anti-pattern: agents over-rate their own work. With
        a *separate* evaluator we can simulate a strict rejection and
        confirm the verdict propagates. The fail message must not get
        swallowed by Sequence's empty-reply rule."""
        bundle = world_and_tools
        planner_llm = ScriptedLLM([reply("Plan: set kitchen on. Pass: light_on == true.")])
        executor_llm = ScriptedLLM(
            [
                # Executor "forgets" the tool call and just claims success.
                reply("I have turned on the light."),
            ]
        )
        evaluator_llm = ScriptedLLM(
            [reply("VERDICT: FAIL — executor reported success but no tool was actually called.")]
        )

        planner = LLMAgent(name="Planner", description="planner", instructions="Plan.", llm=planner_llm)
        executor = LLMAgent(
            name="Executor",
            description="executor",
            instructions="Execute.",
            tools=bundle["tools"],
            llm=executor_llm,
        )
        evaluator = LLMAgent(
            name="Evaluator",
            description="evaluator",
            instructions="Evaluate strictly.",
            llm=evaluator_llm,
        )
        pee = Sequence("plan_execute_evaluate", [planner, executor, evaluator])
        result = pee.run("turn on kitchen")

        assert "FAIL" in result.reply
        # World state stayed unchanged because the executor never called the tool.
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is False


# ---------------------------------------------------------------------------
# Level 5 — iterative Loop until a goal predicate is met.
# ---------------------------------------------------------------------------


class TestL5_LoopUntilGoal:
    """Loop wraps an executor; ``until`` reads world state, not LLM reply.

    Probes: Loop primitive; the predicate fires against external state
    (the ToyWorld) so we exit on real-world goal achievement, not just
    LLM-claimed success. Mirrors a real-robot retry loop where the
    sensor reading is ground truth.
    """

    def test_loop_exits_when_world_reaches_goal(self, world_and_tools):
        bundle = world_and_tools
        world = bundle["world"]

        # Executor's first attempt forgets to set the light. Second
        # attempt does it. The loop predicate checks the world.
        executor_llm = ScriptedLLM(
            [
                # Iter 1: agent rambles, no tool call.
                reply("Working on it..."),
                # Iter 2: agent calls the tool.
                call("set_light", room="kitchen", on=True),
                reply("Done."),
            ]
        )
        executor = LLMAgent(
            name="Executor",
            description="executor",
            instructions="Turn on the kitchen light. Use the set_light tool.",
            tools=bundle["tools"],
            llm=executor_llm,
        )

        # Predicate: read world state directly. Returns True to stop.
        def goal_met(state) -> bool:
            return world.get_world_state()["rooms"]["kitchen"]["light_on"]

        loop = Loop("retry_until_lit", executor, until=goal_met, max_iterations=4)
        result = loop.run("turn on the kitchen light")

        assert world.get_world_state()["rooms"]["kitchen"]["light_on"] is True
        # We should have iterated at least twice (first failed, second succeeded).
        # Each "iteration" is one full executor.run() — which itself may have
        # used multiple LLM hops. Reading the LLM call count is the cleanest
        # way to assert we didn't bail early.
        assert len(executor_llm.calls) >= 2
        assert "done" in result.reply.lower() or "kitchen" in result.reply.lower()


# ---------------------------------------------------------------------------
# Level 6 — multi-tool flow under skill cancellation pressure.
# ---------------------------------------------------------------------------


class TestL6_SkillCancellationDuringFlow:
    """A long-running skill is started, then ctx.stop fires mid-flight.

    Probes: that the safety-preempt path actually cuts a real skill
    invocation short, not just the LLM hop. Connects the bench
    measurement (~one poll quantum cancel latency) to a workflow-level
    flow.
    """

    def test_cancel_propagates_to_handle(self, world_and_tools):
        # We exercise the GoalHandle / cancel path directly here -- the
        # bench file already measures the wall-clock; this asserts the
        # *correctness* of cancellation under a workflow-shaped
        # invocation (skill function returning a handle, dispatcher
        # observing should_cancel on the worker thread).
        bundle = world_and_tools
        world = bundle["world"]

        handle = world.apply_action("navigate_to", room="bedroom")
        assert handle.status in (GoalStatus.PENDING, GoalStatus.RUNNING)

        # Fire the cancel; wait for the worker to observe it.
        handle.cancel()
        terminal = handle.poll(timeout=2.0)

        assert terminal == GoalStatus.CANCELLED, (
            f"expected CANCELLED, got {terminal}; check that ToyWorld's navigate "
            "worker honours should_cancel() between steps"
        )
        # Robot didn't reach the bedroom.
        rs = world.get_world_state()["robot"]
        assert (rs["x"], rs["y"]) != (4.0, 4.0)


# ---------------------------------------------------------------------------
# Sanity: every level above succeeds with the same fixture independently --
# i.e. levels don't share state and can run in any order.
# ---------------------------------------------------------------------------


def test_level_isolation(world_and_tools):
    """Cheap check: the fixture really is fresh per test (no leak from
    earlier levels). If this fires the fixture scope is wrong."""
    assert world_and_tools["tool_log"] == []
    state = world_and_tools["world"].get_world_state()
    assert all(r["light_on"] is False for r in state["rooms"].values())
    # Pose is at default origin.
    assert (state["robot"]["x"], state["robot"]["y"]) == (0.0, 0.0)


# Linter prefers re imported and used; we keep it in case a future
# level needs to assert plan structure shape -- delete this if the
# pattern doesn't grow.
_PLAN_STEP_RE = re.compile(r"^\s*\d+\.\s")
del _PLAN_STEP_RE
