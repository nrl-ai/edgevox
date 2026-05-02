"""Panda in MuJoCo — voice-controlled tabletop pick-and-place.

This is the Tier-2a simulation demo: a Franka Emika Panda (or the
bundled XYZ gantry fallback) drives a 3D arm to pick, move, and place
coloured cubes on a table in a MuJoCo viewer window.

The agent uses :class:`~edgevox.agents.workflow_recipes.ReActAgent`
by default (think -> act -> observe loop, adapts to tool results,
naturally multi-step). Pass ``--planned`` for the upfront-plan
:class:`PlannedToolDispatcher` recipe (faster for known-shape tasks
but doesn't adapt). ``--legacy`` falls back to the single-LLMAgent
dispatch -- useful for benchmarking the chain-handling difference
but expect sycophancy on weak local models.

Launch:

    edgevox-agent robot-panda                 # full TUI voice (default ReAct)
    edgevox-agent robot-panda --simple-ui     # lightweight CLI voice
    edgevox-agent robot-panda --text-mode     # keyboard chat + MuJoCo viewer
    edgevox-agent robot-panda --no-render     # headless, tests only
    edgevox-agent robot-panda --gantry        # force the bundled gantry fallback
    edgevox-agent robot-panda --planned       # PlannedToolDispatcher (upfront plan)
    edgevox-agent robot-panda --legacy        # single-LLMAgent (no planner/synth)

Requires ``pip install 'edgevox[sim-mujoco]'`` (or ``pip install 'mujoco>=3.2'``).
The Franka scene is auto-fetched from HuggingFace Hub on first run (~33 MB).
"""

from __future__ import annotations

import argparse
from typing import Any

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.examples.agents.framework import dispatch_mode_extra_args as _dispatch_extras
from edgevox.llm import tool

PANDA_PERSONA = (
    "You are Panda, a tabletop pick-and-place robot arm. The table has "
    "three coloured cubes: red_cube, green_cube, blue_cube.\n\n"
    "Available actions:\n"
    "- move_to_point(x, y, z) — move the tool to a raw position\n"
    "- move_above_object(object) — hover 5 cm above a cube\n"
    "- grasp(object) — close the gripper on a named cube\n"
    "- release() — open the gripper\n"
    "- goto_home() — retract the arm to the ready pose\n"
    "- list_objects() / locate_object(name) — read cube positions\n"
    "- get_gripper_state() / get_ee_pose() — read your own state\n\n"
    "VOCABULARY MAPPING -- map user intents to ACTUAL physical actions:\n"
    "- 'sort A, B, C' / 'arrange A, B, C' / 'reorder' / 'rearrange' / "
    "'put in order' / 'organise' = physically MOVE each cube to its new "
    "spot via grasp -> move_to_point -> release. Reading positions is "
    "NOT sorting; you must physically pick each cube up and move it.\n"
    "- 'pick up X' = grasp(X) followed by move_to_point above the table.\n"
    "- 'put X at (x,y,z)' / 'place X at...' = (assuming holding X) "
    "move_to_point + release.\n"
    "- 'swap X and Y' = move X to a temp spot, move Y to X's spot, "
    "move X to Y's spot.\n\n"
    "RULES:\n"
    "1. CALL ONE TOOL PER RESPONSE. After your tool returns, I will send "
    "you the result and you can call the next one. Do NOT batch tool "
    "calls; do NOT claim a multi-step plan is done before each step has "
    "actually returned.\n"
    "2. To MOVE an object: grasp(object) -> move_to_point(x, y, z) -> "
    "release() -> goto_home(). Each step is a separate response.\n"
    "3. list_objects / locate_object / get_gripper_state / get_ee_pose "
    "are DIAGNOSTIC only -- they don't satisfy a 'sort' / 'arrange' / "
    "'move' request. You MUST follow with actual action calls.\n"
    "4. If a tool returns an error, report it and stop. Do not retry "
    "blindly.\n\n"
    "Never read raw JSON aloud; keep replies one short sentence."
)

_HOVER_CLEARANCE_M = 0.05


@skill(latency_class="slow", timeout_s=30.0)
def move_to_point(x: float, y: float, z: float, ctx: AgentContext) -> GoalHandle:
    """Move the end-effector to an explicit x/y/z position in metres.

    Args:
        x: target x position.
        y: target y position.
        z: target z position.
    """
    return ctx.deps.apply_action("move_to", x=x, y=y, z=z)


@skill(latency_class="slow", timeout_s=30.0)
def move_above_object(object: str, ctx: AgentContext) -> GoalHandle:
    """Hover the end-effector directly above a named cube (5 cm clearance).

    Args:
        object: name of the cube to hover above (red_cube, green_cube, or blue_cube).
    """
    world = ctx.deps.get_world_state()
    objs = world.get("objects", {})
    if object not in objs:
        h = GoalHandle()
        h.fail(f"unknown object {object!r}; known: {sorted(objs)}")
        return h
    pos = objs[object]
    return ctx.deps.apply_action(
        "move_to",
        x=float(pos["x"]),
        y=float(pos["y"]),
        z=float(pos["z"]) + _HOVER_CLEARANCE_M,
    )


@skill(latency_class="slow", timeout_s=30.0)
def grasp(object: str, ctx: AgentContext) -> GoalHandle:
    """Approach and grasp a named cube on the table.

    Args:
        object: name of the cube to grasp (red_cube, green_cube, or blue_cube).
    """
    return ctx.deps.apply_action("grasp", object=object)


@skill(latency_class="slow", timeout_s=15.0)
def release(ctx: AgentContext) -> GoalHandle:
    """Open the gripper and release the currently held object."""
    return ctx.deps.apply_action("release")


@skill(latency_class="slow", timeout_s=30.0)
def goto_home(ctx: AgentContext) -> GoalHandle:
    """Return the arm to its home (ready) position."""
    return ctx.deps.apply_action("goto_home")


@skill(latency_class="fast")
def get_ee_pose(ctx: AgentContext) -> dict:
    """Report the end-effector's current x/y/z position."""
    h = ctx.deps.apply_action("get_ee_pose")
    return h.result


@tool
def list_objects(ctx: AgentContext) -> list[dict[str, Any]]:
    """List every object on the table with its current x/y/z position."""
    h = ctx.deps.apply_action("list_objects")
    return h.result


@tool
def locate_object(name: str, ctx: AgentContext) -> dict[str, Any]:
    """Return the x/y/z position of a single cube by name.

    Args:
        name: cube name (red_cube, green_cube, or blue_cube).
    """
    world = ctx.deps.get_world_state()
    objs = world.get("objects", {})
    if name not in objs:
        return {"error": f"unknown object: {name}", "known": sorted(objs)}
    pos = objs[name]
    return {
        "object": name,
        "x": round(float(pos["x"]), 3),
        "y": round(float(pos["y"]), 3),
        "z": round(float(pos["z"]), 3),
    }


@tool
def get_gripper_state(ctx: AgentContext) -> dict[str, Any]:
    """Return whether the gripper is holding an object and if so, which one."""
    world = ctx.deps.get_world_state()
    held = world.get("grasped")
    return {"holding": held, "open": held is None}


def _pre_run(args: argparse.Namespace) -> None:
    from edgevox.examples.agents.framework import apply_dispatch_mode
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    source = "gantry" if getattr(args, "gantry", False) else "franka"
    APP.deps = MujocoArmEnvironment(
        model_source=source,
        render=not getattr(args, "no_render", False),
    )
    # Default: ReAct loop. Override with --legacy or --planned.
    apply_dispatch_mode(APP, args)


APP = AgentApp(
    name="Panda",
    description="Voice-controlled tabletop arm running in MuJoCo.",
    instructions=PANDA_PERSONA,
    tools=[list_objects, locate_object, get_gripper_state],
    skills=[move_to_point, move_above_object, grasp, release, goto_home, get_ee_pose],
    deps=None,
    stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    greeting=(
        "Panda online. I see three cubes on the table — red, green, and blue. What would you like me to pick up?"
    ),
    # 10 hops gives the model room for the canonical 4-step pick+place
    # chain (grasp -> move -> release -> goto_home) plus a few
    # observation calls (list_objects, locate_object, get_ee_pose). The
    # framework caps anyway, so a higher number costs nothing on
    # short tasks and unblocks long ones.
    max_tool_hops=10,
    extra_args=[
        (("--no-render",), {"action": "store_true", "help": "Run headless (no MuJoCo viewer)."}),
        (("--gantry",), {"action": "store_true", "help": "Use the bundled gantry arm instead of Franka."}),
        *_dispatch_extras(),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
