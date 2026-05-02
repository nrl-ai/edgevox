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
    "PHYSICAL ACTION ORDER (critical -- do not break this order):\n"
    "    grasp(cube)  ->  move_to_point(x, y, z)  ->  release()\n"
    "Calling move_to_point WITHOUT a prior grasp moves only the empty\n"
    "gripper. The cube STAYS where it was. To move a cube you MUST\n"
    "grasp it first. After release(), call goto_home() to retract.\n\n"
    "VOCABULARY MAPPING -- map user intents to physical action chains:\n"
    "- 'sort A, B, C' / 'arrange in order' / 'reorder' / 'rearrange' /\n"
    "  'put in order' = for EACH cube in the requested order:\n"
    "      grasp(cube) -> move_to_point(slot_x, slot_y, slot_z) ->\n"
    "      release(). After all cubes: goto_home().\n"
    "- 'pick up X' = grasp(X). That's one tool call.\n"
    "- 'put X at (x,y,z)' (assuming you already grasped X) =\n"
    "      move_to_point(x, y, z) -> release().\n"
    "- 'swap X and Y' = move X to a temp spot, then Y to X's spot,\n"
    "  then X to Y's spot.\n\n"
    "WORKED EXAMPLE -- 'sort cubes blue, red, green' (left-to-right at y=0):\n"
    "  grasp(blue_cube)\n"
    "  move_to_point(0.5, 0.30, 0.50)\n"
    "  release()\n"
    "  grasp(red_cube)\n"
    "  move_to_point(0.5, 0.00, 0.50)\n"
    "  release()\n"
    "  grasp(green_cube)\n"
    "  move_to_point(0.5, -0.30, 0.50)\n"
    "  release()\n"
    "  goto_home()\n"
    "  TASK COMPLETE: sorted blue, red, green left-to-right.\n\n"
    "RULES:\n"
    "1. CALL ONE TOOL PER RESPONSE. After your tool returns, I will send\n"
    "you the result and you can call the next one.\n"
    "2. list_objects / locate_object / get_gripper_state / get_ee_pose\n"
    "are DIAGNOSTIC only. They don't satisfy a 'sort'/'arrange'/'move'\n"
    "request. You MUST follow them with grasp/move/release.\n"
    "3. If a tool returns an error, report it and stop -- don't retry.\n"
    "4. Never read raw JSON aloud; keep replies short.\n"
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


_PHYSICAL_TOOLS = frozenset({"grasp", "move_to_point", "move_above_object", "release", "goto_home"})


def _pre_run(args: argparse.Namespace) -> None:
    from edgevox.examples.agents.framework import (
        apply_dispatch_mode,
        physical_action_check,
    )
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    source = "gantry" if getattr(args, "gantry", False) else "franka"
    APP.deps = MujocoArmEnvironment(
        model_source=source,
        render=not getattr(args, "no_render", False),
    )

    # Track every tool/skill call so the recheck loop can refuse to
    # accept TASK COMPLETE on a physical-action request when the
    # model has only fired diagnostics. The hook updates this list;
    # the predicate reads it.
    physical_log: list[tuple[str, dict]] = []
    APP._physical_log = physical_log  # type: ignore[attr-defined]

    completion_check = physical_action_check(physical_log, _PHYSICAL_TOOLS)

    # Default: ReAct loop with physical-action backstop. Override with
    # --legacy or --planned.
    apply_dispatch_mode(APP, args, completion_check=completion_check)

    # Wire an after_tool hook to the resulting agent so we record
    # every tool/skill that fires. ReAct mode wired the trace hooks
    # already; this APPENDS our recorder.
    inner = getattr(APP.agent, "_inner", None)
    if inner is not None:
        import contextlib as _contextlib

        from edgevox.agents.hooks import AFTER_TOOL, hook

        @hook(AFTER_TOOL)
        def _record(point, ctx, payload):
            if hasattr(payload, "name"):
                physical_log.append((payload.name, getattr(payload, "arguments", {})))
                return None
            with _contextlib.suppress(AttributeError):
                physical_log.append((payload.get("name", "?"), payload.get("arguments", {})))
            return None

        inner._hooks.register(_record)


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
