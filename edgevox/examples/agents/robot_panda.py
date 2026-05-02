"""Panda in MuJoCo — voice-controlled tabletop pick-and-place.

This is the Tier-2a simulation demo: a Franka Emika Panda (or the
bundled XYZ gantry fallback) drives a 3D arm to pick, move, and place
coloured cubes on a table in a MuJoCo viewer window.

The agent uses :class:`~edgevox.agents.workflow_recipes.PlannedToolDispatcher`
by default (planner emits a JSON plan, Python loop direct-dispatches
each step, synthesiser writes the user-facing reply). Pass ``--legacy``
to fall back to the single-LLMAgent dispatch -- useful for benchmarking
the chain-handling difference but expect sycophancy on weak local models.

Launch:

    edgevox-agent robot-panda                 # full TUI voice (default)
    edgevox-agent robot-panda --simple-ui     # lightweight CLI voice
    edgevox-agent robot-panda --text-mode     # keyboard chat + MuJoCo viewer
    edgevox-agent robot-panda --no-render     # headless, tests only
    edgevox-agent robot-panda --gantry        # force the bundled gantry fallback
    edgevox-agent robot-panda --legacy        # single-LLMAgent (no planner/synth)

Requires ``pip install 'edgevox[sim-mujoco]'`` (or ``pip install 'mujoco>=3.2'``).
The Franka scene is auto-fetched from HuggingFace Hub on first run (~33 MB).
"""

from __future__ import annotations

import argparse
from typing import Any

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.agents.trace_hooks import terminal_trace_hooks
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

PANDA_PERSONA = (
    "You are Panda, a terse pick-and-place robot arm on a tabletop. "
    "The table has three coloured cubes: red_cube, green_cube, blue_cube.\n\n"
    "Available actions:\n"
    "- move_to_point(x, y, z) — move the tool to a raw position\n"
    "- move_above_object(object) — hover 5 cm above a cube\n"
    "- grasp(object) — close the gripper on a named cube\n"
    "- release() — open the gripper\n"
    "- goto_home() — retract the arm to the ready pose\n"
    "- list_objects() / locate_object(name) — read cube positions\n"
    "- get_gripper_state() / get_ee_pose() — read your own state\n\n"
    "RULES:\n"
    "1. CALL ONE TOOL PER RESPONSE. After your tool returns, I will send "
    "you the result and you can call the next one. Do NOT batch tool "
    "calls; do NOT claim a multi-step plan is done before each step has "
    "actually returned.\n"
    "2. To MOVE an object: grasp(object) -> move_to_point(x, y, z) -> "
    "release() -> goto_home(). Each step is a separate response.\n"
    "3. When the task is fully complete (every step has returned), reply "
    "with one short plain-text sentence summarising what happened. Do "
    "NOT call any further tool on that final reply.\n"
    "4. If a tool returns an error, report it and stop. Do not retry "
    "blindly.\n\n"
    "Never read raw JSON aloud; keep replies one sentence."
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
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    source = "gantry" if getattr(args, "gantry", False) else "franka"
    APP.deps = MujocoArmEnvironment(
        model_source=source,
        render=not getattr(args, "no_render", False),
    )

    # Pick the dispatch architecture. Three options:
    #
    #   --legacy  : single LLMAgent with all tools attached. Sycophants
    #               on chains for weak local models (Gemma 4 E2B/E4B):
    #               model claims release / goto_home without actually
    #               calling them. Kept for back-compat / benchmarking.
    #
    #   --react   : ReActAgent loop -- think -> act -> observe -> repeat.
    #               LLM iterates until it stops emitting tool calls or
    #               hits max_iterations. Adapts to tool results. Slower
    #               (one LLM hop per step), more flexible. Good for
    #               search / exploration / branching tasks.
    #
    #   default   : PlannedToolDispatcher -- planner emits JSON plan,
    #               Python loop direct-dispatches each step, synthesiser
    #               writes user-facing reply. Fastest, most reliable for
    #               tasks where the plan can be determined upfront.
    common_skills = [move_to_point, move_above_object, grasp, release, goto_home, get_ee_pose]
    common_tools = [list_objects, locate_object, get_gripper_state]

    if getattr(args, "legacy", False):
        # Keep the auto-built single-LLMAgent that AgentApp.__post_init__
        # already wired -- nothing to do here.
        pass
    elif getattr(args, "react", False):
        from edgevox.agents.workflow_recipes import ReActAgent

        APP.agent = ReActAgent.build(
            llm=None,
            tools=common_tools,
            skills=common_skills,
            max_iterations=20,
        )
    else:
        from edgevox.agents.workflow_recipes import PlannedToolDispatcher

        APP.agent = PlannedToolDispatcher.build(
            planner_llm=None,
            synthesiser_llm=None,
            tools=common_tools,
            skills=common_skills,
            max_steps=10,
        )


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
    # Print live trace (LLM reasoning + tool calls + tool returns) to
    # stderr so the operator can see exactly what the arm is doing in
    # the terminal alongside the MuJoCo viewer animation.
    hooks=terminal_trace_hooks(),
    extra_args=[
        (("--no-render",), {"action": "store_true", "help": "Run headless (no MuJoCo viewer)."}),
        (("--gantry",), {"action": "store_true", "help": "Use the bundled gantry arm instead of Franka."}),
        (
            ("--legacy",),
            {
                "action": "store_true",
                "help": (
                    "Use the legacy single-LLMAgent dispatch (sycophants "
                    "on multi-step chains with weak models -- kept for "
                    "back-compat and benchmarking)."
                ),
            },
        ),
        (
            ("--react",),
            {
                "action": "store_true",
                "help": (
                    "Use the ReActAgent loop (think -> act -> observe -> "
                    "repeat) instead of the default PlannedToolDispatcher. "
                    "Right when next action depends on previous tool "
                    "result -- search, exploration, recovery."
                ),
            },
        ),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
