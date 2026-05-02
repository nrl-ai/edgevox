"""Tests for ``ReActAgent`` recipe.

Pattern: a single LLMAgent loop with ReAct-tuned prompt + high hop budget.
Each test scripts the LLM through several think-act-observe cycles so
the agent fires multiple tools across a single ``run()``.
"""

from __future__ import annotations

import pytest

from edgevox.agents.sim import ToyWorld
from edgevox.agents.skills import GoalHandle, skill
from edgevox.agents.workflow_recipes import ReActAgent
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, calls, reply


@pytest.fixture
def world_and_actions():
    world = ToyWorld()
    log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
            on: true to turn on.
        """
        log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    @tool
    def list_rooms() -> str:
        """List rooms the robot knows about."""
        log.append(("list_rooms", {}))
        return ", ".join(world.room_names())

    @skill(latency_class="slow", timeout_s=5.0)
    def navigate_to(room: str, ctx) -> GoalHandle:
        """Drive to a named room.

        Args:
            room: target.
        """
        log.append(("navigate_to", {"room": room}))
        return ctx.deps.apply_action("navigate_to", room=room)

    return {
        "world": world,
        "log": log,
        "tools": [set_light, list_rooms],
        "skills": [navigate_to],
    }


class TestReActAgent:
    def test_multi_step_loop_fires_each_tool_separately(self, world_and_actions):
        """Three discovery + action cycles in one run.

        Hop 1: think + list_rooms (discover what's available)
        Hop 2: think + set_light(kitchen, on)
        Hop 3: think + set_light(living_room, on)
        Hop 4: stop -- emit final summary, no tool calls
        """
        bundle = world_and_actions
        scripted = ScriptedLLM(
            [
                # Hop 1: discover via list_rooms.
                call("list_rooms"),
                # Hop 2: turn on kitchen.
                call("set_light", room="kitchen", on=True),
                # Hop 3: turn on living_room.
                call("set_light", room="living_room", on=True),
                # Hop 4: stop with summary including the termination marker.
                reply("TASK COMPLETE: listed rooms and turned on the kitchen + living_room lights."),
            ]
        )
        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            max_iterations=10,
        )
        result = agent.run("turn on kitchen and living_room lights")

        # All three tool calls fired.
        names = [n for n, _ in bundle["log"]]
        assert "list_rooms" in names
        # set_light fired twice with the right rooms.
        rooms_lit = [a["room"] for n, a in bundle["log"] if n == "set_light"]
        assert "kitchen" in rooms_lit
        assert "living_room" in rooms_lit
        # World state matches.
        rooms = bundle["world"].get_world_state()["rooms"]
        assert rooms["kitchen"]["light_on"] is True
        assert rooms["living_room"]["light_on"] is True
        # Final reply mentions completion.
        assert "kitchen" in result.reply.lower() and "living_room" in result.reply.lower()

    def test_completion_check_vetos_early_termination(self, world_and_actions):
        """Model claims done after one tool, but completion_check sees
        the world isn't matching the goal yet, so the agent gets a
        re-prompt and continues."""
        bundle = world_and_actions
        world = bundle["world"]

        scripted = ScriptedLLM(
            [
                # First inner.run: model fires one tool, claims done WITH marker
                # (so the marker check passes -- but the predicate sees only
                # one light is on and overrides).
                call("set_light", room="kitchen", on=True),
                reply("TASK COMPLETE: kitchen light is on."),
                # Re-prompt forces another round; model fires the
                # second tool then emits a real completion.
                call("set_light", room="living_room", on=True),
                reply("TASK COMPLETE: now both lights are on."),
            ]
        )

        # Predicate: BOTH lights must be on.
        def both_on(ctx) -> bool:
            rs = world.get_world_state()["rooms"]
            return rs["kitchen"]["light_on"] and rs["living_room"]["light_on"]

        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            max_iterations=5,
            completion_check=both_on,
        )
        result = agent.run("turn on kitchen and living_room")

        # Both fired.
        rooms_lit = {a["room"] for n, a in bundle["log"] if n == "set_light"}
        assert rooms_lit == {"kitchen", "living_room"}
        # Both lights on now -- check fired and unblocked the second pass.
        rs = world.get_world_state()["rooms"]
        assert rs["kitchen"]["light_on"] and rs["living_room"]["light_on"]
        assert "both" in result.reply.lower() or "now" in result.reply.lower()

    def test_completion_check_passes_first_time_no_extra_hop(self, world_and_actions):
        """When the predicate says done after the model's natural
        termination, the agent doesn't fire a re-prompt."""
        bundle = world_and_actions
        world = bundle["world"]
        scripted = ScriptedLLM(
            [
                call("set_light", room="kitchen", on=True),
                reply("TASK COMPLETE: kitchen light is on."),
            ]
        )

        def kitchen_on(ctx) -> bool:
            return world.get_world_state()["rooms"]["kitchen"]["light_on"]

        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            max_iterations=5,
            completion_check=kitchen_on,
        )
        agent.run("turn on the kitchen light")

        # Only one set_light fired -- no re-prompt round.
        n_set = sum(1 for n, _ in bundle["log"] if n == "set_light")
        assert n_set == 1

    def test_skill_dispatch_in_loop(self, world_and_actions):
        """``@skill`` (cancellable, returns GoalHandle) dispatches the
        same way ``@tool`` does inside the loop."""
        bundle = world_and_actions
        scripted = ScriptedLLM(
            [
                call("navigate_to", room="kitchen"),
                reply("TASK COMPLETE: heading to the kitchen."),
            ]
        )
        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            skills=bundle["skills"],
            max_iterations=5,
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=bundle["world"])
        agent.run("drive to the kitchen", ctx)

        assert any(n == "navigate_to" for n, _ in bundle["log"])

    def test_parallel_calls_in_one_hop_supported(self, world_and_actions):
        """The ReAct prompt asks for one tool per response, but the
        loop must still cope with a hop that emits multiple calls
        (older models, structured-output forcing). All calls in that
        hop dispatch in parallel; the next hop sees all results."""
        bundle = world_and_actions
        scripted = ScriptedLLM(
            [
                # One hop emits two tool calls in parallel -- the
                # framework dispatches both before the next hop.
                calls(
                    ("set_light", {"room": "kitchen", "on": True}),
                    ("set_light", {"room": "living_room", "on": True}),
                ),
                reply("TASK COMPLETE: both lights are on."),
            ]
        )
        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            max_iterations=5,
        )
        agent.run("kitchen + living room please")

        rooms_lit = {a["room"] for n, a in bundle["log"] if n == "set_light"}
        assert rooms_lit == {"kitchen", "living_room"}

    def test_max_iterations_bounds_runaway_loop(self, world_and_actions):
        """A model that never emits the termination marker is bounded
        by max_iterations * (max_completion_recheck_attempts + 1).
        With max_iterations=3 and max_recheck=0, only 3 tool calls fire."""
        bundle = world_and_actions
        # Script enough calls that the budget would be exceeded if
        # unbounded.
        many = []
        for _ in range(30):
            many.append(call("list_rooms"))
        scripted = ScriptedLLM(many)
        agent = ReActAgent.build(
            llm=scripted,
            tools=bundle["tools"],
            max_iterations=3,
            max_completion_recheck_attempts=0,  # disable re-prompt for clean bound
        )
        agent.run("loop forever")

        # list_rooms called at most max_iterations times.
        n_list = sum(1 for n, _ in bundle["log"] if n == "list_rooms")
        assert n_list <= 3
