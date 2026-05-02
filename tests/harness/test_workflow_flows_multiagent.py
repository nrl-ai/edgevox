"""Multi-agent workflow flows on ToyWorld -- L7 handoff, L8 router, L9 supervisor.

Continuation of ``test_workflow_flows.py`` for patterns where the LLM
delegates to another agent rather than (or in addition to) calling
tools. Same fixtures, same determinism via ``ScriptedLLM``.
"""

from __future__ import annotations

import json

import pytest

from edgevox.agents.base import LLMAgent
from edgevox.agents.sim import ToyWorld
from edgevox.agents.workflow import Router, Supervisor
from edgevox.llm import tool

from .conftest import ScriptedLLM, call, calls, reply


@pytest.fixture
def world_and_specialists():
    """Build ToyWorld + a small kitchen specialist + a small mobility specialist.

    Each specialist is its own LLMAgent with its own scripted LLM and a
    narrow tool list. The router or supervisor in each test then
    dispatches the user request to the right one.
    """
    world = ToyWorld()
    log: list[tuple[str, str, dict]] = []  # (agent_name, tool, args)

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
            on: true to turn on.
        """
        log.append(("kitchen_specialist", "set_light", {"room": room, "on": on}))
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} light -> {'on' if on else 'off'}"

    @tool
    def navigate_to(room: str) -> str:
        """Drive the robot to a named room."""
        log.append(("mobility_specialist", "navigate_to", {"room": room}))
        handle = world.apply_action("navigate_to", room=room)
        handle.poll(timeout=5.0)
        rs = world.get_world_state()["robot"]
        return f"robot now at ({rs['x']:.1f},{rs['y']:.1f})"

    @tool
    def get_pose() -> str:
        """Return the robot's current pose."""
        return json.dumps(world.get_world_state()["robot"])

    return {
        "world": world,
        "log": log,
        "kitchen_tools": [set_light],
        "mobility_tools": [navigate_to, get_pose],
    }


# ---------------------------------------------------------------------------
# L7 — Handoff: parent LLMAgent dispatches to a child via synthetic
# handoff_to_<name> tool emitted in its tool_calls.
# ---------------------------------------------------------------------------


class TestL7_Handoff:
    """Parent agent receives a request, decides it's not its job, hands off.

    Probes: the synthetic ``handoff_to_<name>`` tool that LLMAgent
    auto-registers when ``handoffs=[...]``; that the target agent's
    ``run()`` is invoked with the original task as input; that the
    final reply is the target's reply, not the parent's empty hop.
    """

    def test_parent_hands_off_to_specialist(self, world_and_specialists):
        bundle = world_and_specialists

        # Mobility specialist owns navigate_to.
        mobility_llm = ScriptedLLM(
            [
                calls(("navigate_to", {"room": "bedroom"})),
                reply("Moved to bedroom."),
            ]
        )
        mobility = LLMAgent(
            name="mobility",
            description="navigation specialist",
            instructions="Drive the robot. Use navigate_to.",
            tools=bundle["mobility_tools"],
            llm=mobility_llm,
        )

        # Parent looks at the request, decides it needs the mobility
        # specialist, emits handoff_to_mobility.
        parent_llm = ScriptedLLM(
            [
                call("handoff_to_mobility"),
            ]
        )
        parent = LLMAgent(
            name="dispatcher",
            description="dispatcher",
            instructions="If the request is about driving, hand off to mobility.",
            handoffs=[mobility],
            llm=parent_llm,
        )

        result = parent.run("drive the robot to the bedroom")

        # Final reply came from mobility, not the parent's empty hop.
        assert "bedroom" in result.reply.lower()
        # Tool dispatched on the specialist's side, with the correct args.
        assert any(name == "navigate_to" and args["room"] == "bedroom" for _, name, args in bundle["log"])
        # Robot actually moved toward the bedroom. Don't assert exact
        # coords -- ToyWorld navigation is timeout-bounded and may stop
        # short on a heavily loaded test runner. Direction + nontrivial
        # displacement is enough.
        rs = bundle["world"].get_world_state()["robot"]
        assert rs["x"] > 1.0 and rs["y"] > 1.0, f"robot didn't move toward bedroom: {rs}"


# ---------------------------------------------------------------------------
# L8 — Router: one LLM call picks the specialist; specialist runs.
# ---------------------------------------------------------------------------


class TestL8_Router:
    """``Router.build(routes={...})`` -- canonical voice-agent multi-agent
    pattern. Probes: that a single router LLM hop dispatches to the
    right leaf, that the leaf's tool actually executes, and that the
    final reply is the leaf's reply (not the router's hand-off hop).
    """

    def test_router_dispatches_to_kitchen_specialist(self, world_and_specialists):
        bundle = world_and_specialists

        kitchen_llm = ScriptedLLM(
            [
                calls(("set_light", {"room": "kitchen", "on": True})),
                reply("Kitchen light on."),
            ]
        )
        mobility_llm = ScriptedLLM([reply("(unused)")])

        kitchen = LLMAgent(
            name="kitchen",
            description="lights specialist",
            instructions="Toggle lights.",
            tools=bundle["kitchen_tools"],
            llm=kitchen_llm,
        )
        mobility = LLMAgent(
            name="mobility",
            description="navigation specialist",
            instructions="Drive the robot.",
            tools=bundle["mobility_tools"],
            llm=mobility_llm,
        )

        router_llm = ScriptedLLM([call("handoff_to_kitchen")])
        router = Router.build(
            "router",
            "Pick the right specialist based on the request.",
            routes={"kitchen": kitchen, "mobility": mobility},
        )
        # Router.build returns an LLMAgent; bind the scripted LLM.
        router._llm = router_llm

        result = router.run("turn on the kitchen light")

        assert "kitchen light on" in result.reply.lower() or "kitchen" in result.reply.lower()
        assert bundle["world"].get_world_state()["rooms"]["kitchen"]["light_on"] is True
        # Mobility specialist must NOT have been engaged.
        assert mobility_llm.calls == [], "mobility specialist should not have been called"

    def test_router_dispatches_to_mobility_specialist(self, world_and_specialists):
        """Same router topology, different request -> different leaf."""
        bundle = world_and_specialists

        kitchen_llm = ScriptedLLM([reply("(unused)")])
        mobility_llm = ScriptedLLM(
            [
                calls(("navigate_to", {"room": "office"})),
                reply("Moved to office."),
            ]
        )
        kitchen = LLMAgent(
            name="kitchen",
            description="lights",
            instructions="Lights.",
            tools=bundle["kitchen_tools"],
            llm=kitchen_llm,
        )
        mobility = LLMAgent(
            name="mobility",
            description="navigation",
            instructions="Drive.",
            tools=bundle["mobility_tools"],
            llm=mobility_llm,
        )
        router_llm = ScriptedLLM([call("handoff_to_mobility")])
        router = Router.build(
            "router",
            "Dispatch.",
            routes={"kitchen": kitchen, "mobility": mobility},
        )
        router._llm = router_llm

        result = router.run("drive to the office")

        assert "office" in result.reply.lower()
        rs = bundle["world"].get_world_state()["robot"]
        # Office is at (0, 4) so y should grow while x stays small.
        assert rs["y"] > 1.0 and rs["x"] < 1.0, f"didn't move toward office: {rs}"
        assert kitchen_llm.calls == []


# ---------------------------------------------------------------------------
# L9 — Supervisor: same wire shape as Router but forces dispatch.
# ---------------------------------------------------------------------------


class TestL9_Supervisor:
    """``Supervisor.build`` requires the model to dispatch to a worker on
    the first hop -- canonical SLM loop-break for multi-agent.

    Probes: same end-to-end behaviour as Router but with the
    required_first_hop tool-choice policy enforced.
    """

    def test_supervisor_forces_first_hop_dispatch(self, world_and_specialists):
        bundle = world_and_specialists

        worker_llm = ScriptedLLM(
            [
                calls(("set_light", {"room": "bedroom", "on": True})),
                reply("Bedroom light on."),
            ]
        )
        worker = LLMAgent(
            name="worker",
            description="generic worker",
            instructions="Do the thing.",
            tools=bundle["kitchen_tools"],
            llm=worker_llm,
        )
        supervisor_llm = ScriptedLLM([call("handoff_to_worker")])
        supervisor = Supervisor.build(
            "boss",
            "Dispatch every request to the worker.",
            workers={"worker": worker},
        )
        supervisor._llm = supervisor_llm

        result = supervisor.run("turn on the bedroom light")

        assert "bedroom" in result.reply.lower()
        assert bundle["world"].get_world_state()["rooms"]["bedroom"]["light_on"] is True
