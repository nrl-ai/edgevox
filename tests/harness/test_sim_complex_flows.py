"""Comprehensive multi-step flows that exercise the FULL action surface
of each sim. Companion to ``test_sim_scenarios.py`` -- those tests prove
the harness works on each sim with one or two actions; these tests
prove the agent can plan and execute over the entire surface.

Action surface per sim:

  IR-SIM (Tier 1)
    get_pose, battery_level, list_rooms, navigate_to(room|x,y)

  MuJoCo arm (Tier 2a)
    move_to(x,y,z), grasp(object), release, get_ee_pose, list_objects,
    goto_home

The flows below pick scenarios that *force* the harness to chain multiple
actions, observe state between steps, and react to results -- not just
fire one tool and report.

All deterministic via ScriptedLLM. Sims run real but headless.
"""

from __future__ import annotations

import json
import time

import pytest

from edgevox.agents.base import LLMAgent
from edgevox.agents.skills import GoalStatus, skill
from edgevox.llm import tool

from .conftest import ScriptedLLM, calls, reply

# ===========================================================================
# IR-SIM -- whole action surface
# ===========================================================================


@pytest.fixture(scope="module")
def irsim_world():
    pytest.importorskip("irsim")
    from edgevox.integrations.sim.irsim import IrSimEnvironment

    env = IrSimEnvironment(render=False, tick_interval=0.02)
    yield env
    env._phys_stop.set()
    env._phys_thread.join(timeout=1.0)


def _make_irsim_tools(world):
    """Wrap the FULL IR-SIM action surface as @tools / @skills."""
    events: list[tuple[str, dict]] = []

    @tool
    def get_pose() -> str:
        """Return the robot's current pose (x, y, heading_deg, battery, moving)."""
        h = world.apply_action("get_pose")
        h.poll(timeout=1.0)
        events.append(("get_pose", h.result))
        return json.dumps(h.result)

    @tool
    def battery_level() -> str:
        """Return the robot's battery level percentage."""
        h = world.apply_action("battery_level")
        h.poll(timeout=1.0)
        events.append(("battery_level", h.result))
        return json.dumps(h.result)

    @tool
    def list_rooms() -> str:
        """List the named rooms the robot knows about."""
        h = world.apply_action("list_rooms")
        h.poll(timeout=1.0)
        events.append(("list_rooms", {"result": h.result}))
        return ", ".join(h.result)

    @skill(latency_class="slow", timeout_s=10.0)
    def navigate_to(room: str, ctx):
        """Drive the robot to a named room.

        Args:
            room: target room name.
        """
        events.append(("navigate_to", {"room": room}))
        return ctx.deps.apply_action("navigate_to", room=room)

    return events, [get_pose, battery_level, list_rooms], [navigate_to]


class TestIrSimComprehensiveFlows:
    def test_full_action_surface_in_one_flow(self, irsim_world):
        """Single agent uses every action in the IR-SIM surface across
        multiple LLM hops.

        Hop 1 -- list_rooms + battery_level (parallel observation)
        Hop 2 -- navigate_to(kitchen) (skill)
        Hop 3 -- get_pose (final state read)
        Hop 4 -- summarising reply
        """
        events, tools, skills_list = _make_irsim_tools(irsim_world)

        agent = LLMAgent(
            name="ApartmentExplorer",
            description="comprehensive nav agent",
            instructions=(
                "Use tools in this order: list_rooms + battery_level (parallel), "
                "then navigate_to the requested room, then get_pose. "
                "Reply with one sentence summarising battery + arrival."
            ),
            tools=tools,
            skills=skills_list,
            llm=ScriptedLLM(
                [
                    calls(("list_rooms", {}), ("battery_level", {})),
                    calls(("navigate_to", {"room": "kitchen"})),
                    calls(("get_pose", {})),
                    reply("Battery 95%; navigation kicked off; final pose reported."),
                ]
            ),
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=irsim_world)
        agent.run("explore: where can you go, what's the battery, then go to the kitchen", ctx)

        # All four action types fired through.
        names = {n for n, _ in events}
        assert names == {"list_rooms", "battery_level", "navigate_to", "get_pose"}

    def test_battery_aware_decision(self, irsim_world):
        """Realistic robotics pattern: read battery first, decide based
        on state. The agent only commits to a long navigation if
        battery > threshold. Demonstrates state-driven branching from
        a tool result."""
        events, tools, skills_list = _make_irsim_tools(irsim_world)

        agent = LLMAgent(
            name="CautiousNav",
            description="battery-aware nav agent",
            instructions=(
                "Read battery_level first. If battery >= 50%, navigate to "
                "the requested room. Otherwise reply that you'll wait."
            ),
            tools=tools,
            skills=skills_list,
            llm=ScriptedLLM(
                [
                    # Hop 1: check battery.
                    calls(("battery_level", {})),
                    # Hop 2: branch on the result. IR-SIM starts at 95%, so we proceed.
                    calls(("navigate_to", {"room": "office"})),
                    # Hop 3: report.
                    reply("Battery healthy at 95%; navigating to office."),
                ]
            ),
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=irsim_world)
        agent.run("drive to the office if you can spare the energy", ctx)

        # Ensure battery_level was consulted BEFORE navigate_to fired.
        battery_idx = next(i for i, (n, _) in enumerate(events) if n == "battery_level")
        nav_idx = next(i for i, (n, _) in enumerate(events) if n == "navigate_to")
        assert battery_idx < nav_idx, "agent navigated before checking battery"


# ===========================================================================
# MuJoCo arm -- whole action surface
# ===========================================================================


@pytest.fixture(scope="module")
def mujoco_world():
    pytest.importorskip("mujoco")
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    env = MujocoArmEnvironment(
        model_source="gantry",
        render=False,
        allow_hf_download=False,
    )
    yield env


def _make_mujoco_tools(world):
    """Wrap the FULL MuJoCo arm action surface as @tools / @skills."""
    events: list[tuple[str, dict]] = []

    @tool
    def list_objects() -> str:
        """List all freejoint objects on the table with their positions."""
        h = world.apply_action("list_objects")
        h.poll(timeout=2.0)
        events.append(("list_objects", {"result": h.result}))
        return json.dumps(h.result)

    @tool
    def get_ee_pose() -> str:
        """Return the end-effector pose (x, y, z, possibly orientation)."""
        h = world.apply_action("get_ee_pose")
        h.poll(timeout=2.0)
        events.append(("get_ee_pose", h.result))
        return json.dumps(h.result)

    @skill(latency_class="slow", timeout_s=15.0)
    def move_to(x: float, y: float, z: float, ctx):
        """Move the arm's end effector to a Cartesian target.

        Args:
            x: target x coordinate (m).
            y: target y coordinate (m).
            z: target z coordinate (m).
        """
        events.append(("move_to", {"x": x, "y": y, "z": z}))
        return ctx.deps.apply_action("move_to", x=x, y=y, z=z)

    @skill(latency_class="slow", timeout_s=15.0)
    def grasp(object: str, ctx):
        """Grasp a named freejoint object body.

        Args:
            object: body name (red_cube / green_cube / blue_cube).
        """
        events.append(("grasp", {"object": object}))
        return ctx.deps.apply_action("grasp", object=object)

    @skill(latency_class="slow", timeout_s=15.0)
    def release(ctx):
        """Release whatever the gripper is currently holding."""
        events.append(("release", {}))
        return ctx.deps.apply_action("release")

    @skill(latency_class="slow", timeout_s=15.0)
    def goto_home(ctx):
        """Return the arm to its home pose."""
        events.append(("goto_home", {}))
        return ctx.deps.apply_action("goto_home")

    return events, [list_objects, get_ee_pose], [move_to, grasp, release, goto_home]


class TestMujocoComprehensiveFlows:
    def test_pick_and_place_red_to_drop_zone(self, mujoco_world):
        """Cool flow: list -> grasp red -> move_to drop zone -> release -> goto_home.

        Five distinct skill / tool dispatches across three LLM hops.
        Exercises every action type the arm sim exposes.
        """
        events, tools, skills_list = _make_mujoco_tools(mujoco_world)

        agent = LLMAgent(
            name="PickerBot",
            description="pick-and-place arm",
            instructions=(
                "Plan: list_objects, grasp the red cube, move_to(0.0, 0.3, 0.5) "
                "(drop zone), release, goto_home. Reply with one sentence."
            ),
            tools=tools,
            skills=skills_list,
            max_tool_hops=8,
            llm=ScriptedLLM(
                [
                    calls(("list_objects", {})),
                    calls(("grasp", {"object": "red_cube"})),
                    calls(("move_to", {"x": 0.0, "y": 0.3, "z": 0.5})),
                    calls(("release", {})),
                    calls(("goto_home", {})),
                    reply("Red cube relocated to drop zone; arm at home."),
                ]
            ),
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=mujoco_world)
        agent.run("pick the red cube and put it at (0.0, 0.3, 0.5)", ctx)

        # All five action types fired in the right order.
        names = [n for n, _ in events]
        assert names == ["list_objects", "grasp", "move_to", "release", "goto_home"]

    def test_observe_then_pick_specific_color(self, mujoco_world):
        """Observation-driven branching: list_objects returns positions
        for all cubes, agent picks the one whose name matches the user
        request. Tests that the agent's plan can use earlier-step
        results (list_objects) to inform later-step args (grasp's
        object name)."""
        events, tools, skills_list = _make_mujoco_tools(mujoco_world)

        agent = LLMAgent(
            name="ColorPicker",
            description="picks the requested color",
            instructions=(
                "First call list_objects to see what's on the table. "
                "Then grasp the cube whose name matches the user's color. "
                "Then goto_home. Keep the reply to one sentence."
            ),
            tools=tools,
            skills=skills_list,
            llm=ScriptedLLM(
                [
                    calls(("list_objects", {})),
                    calls(("grasp", {"object": "blue_cube"})),
                    calls(("goto_home", {})),
                    reply("Blue cube picked up; arm returned home."),
                ]
            ),
        )
        from edgevox.agents.base import AgentContext

        ctx = AgentContext(deps=mujoco_world)
        agent.run("please pick the blue one", ctx)

        # list_objects fired BEFORE grasp.
        list_idx = next(i for i, (n, _) in enumerate(events) if n == "list_objects")
        grasp_idx = next(i for i, (n, _) in enumerate(events) if n == "grasp")
        assert list_idx < grasp_idx
        # The grasped object matches the user's color request.
        grasp_args = next(a for n, a in events if n == "grasp")
        assert grasp_args["object"] == "blue_cube"

    def test_long_running_skill_cancellation_on_arm(self, mujoco_world):
        """The arm's grasp / move_to skills run on a worker thread. Cancel
        propagates the same way it does on ToyWorld and IR-SIM. This is
        the safety-monitor companion test on the manipulation sim."""
        handle = mujoco_world.apply_action("move_to", x=0.4, y=0.0, z=0.6)
        # Let the skill actually start.
        time.sleep(0.05)
        handle.cancel()
        terminal = handle.poll(timeout=3.0)
        assert terminal in (GoalStatus.CANCELLED, GoalStatus.SUCCEEDED), (
            f"expected CANCELLED or SUCCEEDED (race-finish before cancel), got {terminal}"
        )
