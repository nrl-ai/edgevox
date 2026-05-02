"""Tests for ``PlannedToolDispatcher`` recipe.

Pattern: planner emits JSON plan -> Python loop direct-dispatches each
step -> synthesiser writes one-sentence reply. The planner and synth
are LLM-driven; the dispatch in the middle is deterministic Python.

These tests use ToyWorld (stdlib, no extra deps) and ScriptedLLM to
keep them fast (<1 s) and deterministic.
"""

from __future__ import annotations

import json

import pytest

from edgevox.agents.sim import ToyWorld
from edgevox.agents.skills import GoalHandle, skill
from edgevox.agents.workflow_recipes import PlannedToolDispatcher
from edgevox.llm import tool

from .conftest import ScriptedLLM, reply


@pytest.fixture
def world_and_actions():
    world = ToyWorld()
    log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
            on: true for on.
        """
        log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    @tool
    def get_pose() -> str:
        """Return robot pose."""
        log.append(("get_pose", {}))
        return json.dumps(world.get_world_state()["robot"])

    @skill(latency_class="slow", timeout_s=5.0)
    def navigate_to(room: str, ctx) -> GoalHandle:
        """Drive to a room.

        Args:
            room: target room.
        """
        log.append(("navigate_to", {"room": room}))
        return ctx.deps.apply_action("navigate_to", room=room)

    return {
        "world": world,
        "log": log,
        "tools": [set_light, get_pose],
        "skills": [navigate_to],
    }


class TestPlannedToolDispatcher:
    def test_happy_path_two_step_plan(self, world_and_actions):
        bundle = world_and_actions
        # Planner emits valid JSON plan.
        planner_llm = ScriptedLLM(
            [
                reply(
                    '[{"tool": "set_light", "args": {"room": "kitchen", "on": true}}, {"tool": "get_pose", "args": {}}]'
                )
            ]
        )
        # Synthesiser writes a friendly reply over the executed plan.
        synth_llm = ScriptedLLM([reply("Kitchen is on; pose reported.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
        )
        result = agent.run("turn on kitchen and tell me where you are")

        assert "kitchen" in result.reply.lower()
        # Both tools fired against the ToyWorld.
        names = [n for n, _ in bundle["log"]]
        assert "set_light" in names
        assert "get_pose" in names
        # World state mutated.
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True

    def test_empty_plan_no_dispatches(self, world_and_actions):
        """Planner outputs ``[]`` for chit-chat / impossible asks."""
        bundle = world_and_actions
        planner_llm = ScriptedLLM([reply("[]")])
        synth_llm = ScriptedLLM([reply("Nothing to do for that request.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
        )
        result = agent.run("how's the weather on Mars?")

        assert "nothing" in result.reply.lower() or "request" in result.reply.lower()
        assert bundle["log"] == []

    def test_invalid_json_plan_recovers_to_empty(self, world_and_actions):
        """Planner emits prose instead of JSON. Parser falls back to []."""
        bundle = world_and_actions
        planner_llm = ScriptedLLM([reply("I'd like to set the light, then check pose.")])
        synth_llm = ScriptedLLM([reply("Sorry, I couldn't produce a valid plan.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
        )
        result = agent.run("turn on kitchen")

        # No tools fired because the plan didn't parse.
        assert bundle["log"] == []
        assert "couldn" in result.reply.lower() or "valid" in result.reply.lower()

    def test_unknown_tool_in_plan_stops_execution(self, world_and_actions):
        """Plan references a tool that isn't in the catalog -> stop, surface error."""
        bundle = world_and_actions
        planner_llm = ScriptedLLM(
            [
                reply(
                    '[{"tool": "fabricated_tool", "args": {}}, {"tool": "set_light", "args": {"room": "kitchen", "on": true}}]'
                )
            ]
        )
        synth_llm = ScriptedLLM([reply("First step asked for an unknown action; nothing was done.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
        )
        result = agent.run("anything")

        # set_light should NOT have run -- execution stops at first failure.
        assert ("set_light", {"room": "kitchen", "on": True}) not in bundle["log"]
        assert "unknown" in result.reply.lower() or "nothing" in result.reply.lower()

    def test_skill_in_plan_dispatched_via_handle(self, world_and_actions):
        """``@skill`` actions go through ``Skill.start`` -> handle.poll path,
        not through the tool registry."""
        bundle = world_and_actions
        planner_llm = ScriptedLLM([reply('[{"tool": "navigate_to", "args": {"room": "kitchen"}}]')])
        synth_llm = ScriptedLLM([reply("Heading to the kitchen.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
            skills=bundle["skills"],
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=bundle["world"])
        result = agent.run("drive to the kitchen", ctx)

        assert "kitchen" in result.reply.lower()
        # navigate_to skill body fired.
        assert any(name == "navigate_to" for name, _ in bundle["log"])

    def test_step_error_stops_chain_and_surfaces_to_synth(self, world_and_actions):
        """When a tool returns an error mid-plan, the dispatcher stops
        and the synthesiser sees the failure in its results summary."""
        bundle = world_and_actions
        # Bad args (room='nowhere' isn't a valid ToyWorld room) -> set_light
        # raises KeyError on world._rooms[room]. The dispatcher catches
        # that as a tool error.
        planner_llm = ScriptedLLM(
            [
                reply(
                    '[{"tool": "set_light", "args": {"room": "nowhere", "on": true}}, {"tool": "get_pose", "args": {}}]'
                )
            ]
        )
        synth_llm = ScriptedLLM([reply("First step failed (unknown room); stopped early.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
        )
        result = agent.run("anything")

        # get_pose must not have run -- chain stopped after set_light failure.
        names = [n for n, _ in bundle["log"]]
        assert "get_pose" not in names
        assert "fail" in result.reply.lower() or "stopped" in result.reply.lower()

    def test_max_steps_truncates_long_plan(self, world_and_actions):
        """A planner that emits 20 steps gets truncated at max_steps."""
        bundle = world_and_actions
        many = [
            {"tool": "set_light", "args": {"room": "kitchen", "on": True}},
        ] * 20
        planner_llm = ScriptedLLM([reply(json.dumps(many))])
        synth_llm = ScriptedLLM([reply("Did some lights.")])

        agent = PlannedToolDispatcher.build(
            planner_llm=planner_llm,
            synthesiser_llm=synth_llm,
            tools=bundle["tools"],
            max_steps=3,
        )
        agent.run("light spam")

        # Only 3 set_light calls fired despite 20 in the plan.
        n_set = sum(1 for n, _ in bundle["log"] if n == "set_light")
        assert n_set == 3
