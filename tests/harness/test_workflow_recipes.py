"""Tests for higher-level workflow recipes.

The recipe must produce the same observable behaviour as a manually-wired
``Sequence([planner, executor, evaluator])`` -- L4 in
``test_workflow_flows.py`` is the manual-wiring reference test, and the
tests here re-run that same scenario through ``PlanExecuteEvaluate.build``
to confirm the recipe is a drop-in replacement.
"""

from __future__ import annotations

import json

import pytest

from edgevox.agents.sim import ToyWorld
from edgevox.agents.workflow_recipes import PlanExecuteEvaluate
from edgevox.llm import tool

from .conftest import ScriptedLLM, calls, reply


@pytest.fixture
def world_and_tools():
    world = ToyWorld()
    tool_log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
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

    return {"world": world, "tool_log": tool_log, "tools": [set_light, get_pose]}


class TestPlanExecuteEvaluateRecipe:
    def test_pass_path_via_recipe(self, world_and_tools):
        bundle = world_and_tools
        planner_llm = ScriptedLLM(
            [
                reply(
                    "Goal: kitchen light on, pose reported.\n"
                    "Plan:\n1. set_light(kitchen, true)\n2. get_pose()\n"
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
                reply("set_light returned ok; pose reported {x:0,y:0}. Both calls completed."),
            ]
        )
        evaluator_llm = ScriptedLLM([reply("VERDICT: PASS -- both criteria met.")])

        recipe = PlanExecuteEvaluate.build(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            evaluator_llm=evaluator_llm,
            tools=bundle["tools"],
        )

        result = recipe.run("turn on kitchen and report your pose")

        assert result.reply.startswith("VERDICT:")
        assert "PASS" in result.reply
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True
        # Two scripted hops on the executor (one tool-call hop, one summary).
        assert len(executor_llm.calls) == 2
        # Single hop on planner + evaluator (each is a one-shot reply).
        assert len(planner_llm.calls) == 1
        assert len(evaluator_llm.calls) == 1

    def test_fail_path_via_recipe(self, world_and_tools):
        """Strict evaluator catches an executor that claims success without
        actually calling tools -- the harness's anti-self-rating contract."""
        bundle = world_and_tools
        planner_llm = ScriptedLLM([reply("Plan: set kitchen on. Pass: kitchen.light_on == true.")])
        executor_llm = ScriptedLLM([reply("I have turned on the light.")])
        evaluator_llm = ScriptedLLM([reply("VERDICT: FAIL -- no tool was actually called; light state unchanged.")])

        recipe = PlanExecuteEvaluate.build(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            evaluator_llm=evaluator_llm,
            tools=bundle["tools"],
        )
        result = recipe.run("turn on kitchen")

        assert "FAIL" in result.reply
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is False

    def test_recipe_returns_sequence_so_it_composes(self, world_and_tools):
        """Recipe output is a Sequence -- composes with Loop / Retry / Timeout
        without rewrapping. Quick assertion to lock in that contract."""
        from edgevox.agents.workflow import Sequence

        bundle = world_and_tools
        recipe = PlanExecuteEvaluate.build(
            planner_llm=ScriptedLLM([reply("plan")]),
            executor_llm=ScriptedLLM([reply("did it")]),
            evaluator_llm=ScriptedLLM([reply("VERDICT: PASS")]),
            tools=bundle["tools"],
        )
        assert isinstance(recipe, Sequence)
        # The 3 children carry the expected names.
        assert [a.name for a in recipe._children] == ["Planner", "Executor", "Evaluator"]

    def test_recipe_accepts_per_role_instruction_overrides(self, world_and_tools):
        """When the canonical instructions don't fit, callers can replace
        them. Verify the override actually lands on the agent."""
        bundle = world_and_tools
        recipe = PlanExecuteEvaluate.build(
            planner_llm=ScriptedLLM([reply("plan")]),
            executor_llm=ScriptedLLM([reply("did it")]),
            evaluator_llm=ScriptedLLM([reply("VERDICT: PASS")]),
            tools=bundle["tools"],
            planner_instructions="CUSTOM PLANNER PROMPT",
            executor_instructions="CUSTOM EXECUTOR PROMPT",
            evaluator_instructions="CUSTOM EVALUATOR PROMPT",
        )

        # Run once so the LLMs see the system prompt that includes
        # the custom instructions.
        recipe.run("anything")

        assert "CUSTOM PLANNER PROMPT" in recipe._children[0].instructions
        assert "CUSTOM EXECUTOR PROMPT" in recipe._children[1].instructions
        assert "CUSTOM EVALUATOR PROMPT" in recipe._children[2].instructions
