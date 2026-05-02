"""Tests for PlanThenLoop -- iterative plan-execute-evaluate."""

from __future__ import annotations

import json

import pytest

from edgevox.agents.sim import ToyWorld
from edgevox.agents.workflow_recipes import PlanThenLoop
from edgevox.llm import tool

from .conftest import ScriptedLLM, calls, reply


@pytest.fixture
def world_and_tools():
    world = ToyWorld()
    log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
            on: true to turn on, false to turn off.
        """
        log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    @tool
    def get_pose() -> str:
        """Return the robot's current pose."""
        return json.dumps(world.get_world_state()["robot"])

    return {"world": world, "log": log, "tools": [set_light, get_pose]}


class TestPlanThenLoop:
    def test_pass_on_first_iteration_exits_immediately(self, world_and_tools):
        bundle = world_and_tools
        recipe = PlanThenLoop.build(
            planner_llm=ScriptedLLM([reply("Plan: set_light(kitchen, true). Pass: light on.")]),
            executor_llm=ScriptedLLM(
                [
                    calls(("set_light", {"room": "kitchen", "on": True})),
                    reply("Set kitchen light on."),
                ]
            ),
            evaluator_llm=ScriptedLLM([reply("VERDICT: PASS")]),
            tools=bundle["tools"],
            max_iterations=3,
        )
        result = recipe.run("turn on kitchen")

        assert result.reply.startswith("VERDICT: PASS")
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True

    def test_fail_then_pass_loops_correctly(self, world_and_tools):
        """First iteration fails (executor doesn't call tool), second succeeds.

        The loop must feed iteration 1's verdict forward into iteration 2's
        task, then exit on the second PASS.
        """
        bundle = world_and_tools
        # Three full iteration scripts on each LLM. The recipe runs the
        # inner Sequence each iteration so each LLM gets one hop per
        # iteration (planner + evaluator) or two (executor with a tool
        # call). Iteration 1 fails: planner plans, executor only replies
        # (no tool), evaluator emits FAIL. Iteration 2 succeeds: planner
        # re-plans with feedback, executor calls the tool, evaluator PASSes.
        recipe = PlanThenLoop.build(
            planner_llm=ScriptedLLM(
                [
                    reply("Plan v1: set_light(kitchen, true). Pass: light on."),
                    reply("Plan v2: set_light(kitchen, true). Pass: light on. (re-plan)"),
                ]
            ),
            executor_llm=ScriptedLLM(
                [
                    # Iteration 1: no tool call, just claims success.
                    reply("I have turned on the light."),
                    # Iteration 2: actually calls the tool.
                    calls(("set_light", {"room": "kitchen", "on": True})),
                    reply("kitchen light is now on (confirmed)."),
                ]
            ),
            evaluator_llm=ScriptedLLM(
                [
                    reply("VERDICT: FAIL -- no tool was called; light state is still off."),
                    reply("VERDICT: PASS -- light is on."),
                ]
            ),
            tools=bundle["tools"],
            max_iterations=3,
        )
        result = recipe.run("turn on kitchen")

        assert result.reply.startswith("VERDICT: PASS")
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True

    def test_max_iterations_cap_returns_last_verdict(self, world_and_tools):
        """When the executor never succeeds, the loop must give up at
        max_iterations and return the last (FAIL) verdict cleanly."""
        bundle = world_and_tools
        n_iters = 2
        recipe = PlanThenLoop.build(
            planner_llm=ScriptedLLM([reply(f"Plan v{i}.") for i in range(n_iters)]),
            executor_llm=ScriptedLLM([reply("could not do it") for _ in range(n_iters)]),
            evaluator_llm=ScriptedLLM(
                [reply(f"VERDICT: FAIL -- attempt {i + 1} did not call any tool.") for i in range(n_iters)]
            ),
            tools=bundle["tools"],
            max_iterations=n_iters,
        )
        result = recipe.run("turn on kitchen")

        assert "FAIL" in result.reply
        # Light never turned on.
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is False

    def test_world_predicate_overrides_lying_llm(self, world_and_tools):
        """The strongest hardening: even if the LLM keeps emitting FAIL,
        if the world predicate sees success, the loop exits with PASS.

        Mirrors a real robot scenario where the sensor reading is ground
        truth and the LLM just hasn't noticed yet.
        """
        bundle = world_and_tools
        # Executor *does* call set_light on iteration 1. World will be
        # green after the first executor turn. But the evaluator is
        # broken / paranoid and emits FAIL anyway. World predicate
        # overrides.
        recipe = PlanThenLoop.build(
            planner_llm=ScriptedLLM([reply("Plan: set_light(kitchen, true). Pass: light on."), reply("Plan v2.")]),
            executor_llm=ScriptedLLM(
                [
                    calls(("set_light", {"room": "kitchen", "on": True})),
                    reply("done"),
                    calls(("set_light", {"room": "kitchen", "on": True})),
                    reply("done"),
                ]
            ),
            evaluator_llm=ScriptedLLM([reply("VERDICT: FAIL -- bogus reason"), reply("VERDICT: FAIL -- still bogus")]),
            tools=bundle["tools"],
            world_predicate=lambda: bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"],
            max_iterations=3,
        )
        result = recipe.run("turn on kitchen")

        # World predicate fired on iteration 1 -- recipe exits with
        # the wrapper PASS message even though the LLM said FAIL.
        assert result.reply.startswith("VERDICT: PASS")
        assert "world predicate satisfied" in result.reply
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True
