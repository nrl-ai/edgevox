"""Plan-Execute-Evaluate demo on the ToyWorld simulation.

Wires :class:`~edgevox.agents.workflow_recipes.PlanExecuteEvaluate`
to a real LLM and a stdlib-only sim so the recipe can be exercised
end-to-end with a single ``python -m`` invocation. The same task
runs three times with role-specialised system prompts:

  Planner   -> reads the user request, emits Goal + Plan + Pass criterion
  Executor  -> executes the plan via real tools on ToyWorld
  Evaluator -> reads the executor's report, emits VERDICT: PASS / FAIL

Run with a real model::

    python -m edgevox.examples.agents.plan_execute_evaluate_demo \\
        --model hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf

Run with a local GGUF::

    python -m edgevox.examples.agents.plan_execute_evaluate_demo \\
        --model /path/to/your/model.gguf

Run offline (deterministic ScriptedLLM, no model required, useful for
CI smoke tests of this script itself)::

    python -m edgevox.examples.agents.plan_execute_evaluate_demo --scripted

What you should see (real LLM, single run):

    [stage 1/3] Planner -> Goal + Plan + Pass criterion (~5-15 s on CPU)
    [stage 2/3] Executor -> tool dispatch on ToyWorld (~5-30 s)
    [stage 3/3] Evaluator -> VERDICT: PASS or VERDICT: FAIL -- ...

The world state is printed after each stage so you can see the sim
actually mutate. The verdict is the recipe's headline output.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading

from edgevox.agents.sim import ToyWorld
from edgevox.agents.workflow_recipes import PlanExecuteEvaluate
from edgevox.llm import tool

DEFAULT_TASK = "Turn on the kitchen light, then drive to the bedroom."


def _build_world_and_tools() -> tuple[ToyWorld, list, list[tuple[str, dict]]]:
    """Build a fresh ToyWorld + a tool set bound to it.

    Returns ``(world, tools_list, call_log)`` so the caller can inspect
    state changes after running the recipe. The demo uses tools only
    (no @skill cancellable actions) to keep the wire format simple and
    the recipe pattern in focus -- ``test_workflow_flows.py::TestL6``
    covers the cancellable-skill correctness path.
    """
    world = ToyWorld()
    call_log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name -- one of living_room / kitchen / bedroom / office.
            on: true to turn on, false to turn off.
        """
        call_log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            if room not in world._rooms:
                return f"unknown room {room!r}"
            world._rooms[room].light_on = on
        return f"{room} light is now {'on' if on else 'off'}"

    @tool
    def get_pose() -> str:
        """Return the robot's current pose (x, y, heading) and battery level."""
        state = world.get_world_state()["robot"]
        call_log.append(("get_pose", state))
        return json.dumps(state)

    @tool
    def list_rooms() -> str:
        """List the rooms the robot knows about."""
        rooms = world.room_names()
        call_log.append(("list_rooms", {"rooms": rooms}))
        return ", ".join(rooms)

    @tool
    def navigate_to(room: str) -> str:
        """Drive the robot to a named room (synchronous in this demo).

        Args:
            room: target room.
        """
        call_log.append(("navigate_to", {"room": room}))
        # ToyWorld's apply_action runs on a worker thread + returns a
        # GoalHandle. For the demo we wait for the goal to finish so
        # the tool is synchronous from the LLM's perspective.
        handle = world.apply_action("navigate_to", room=room)
        handle.poll(timeout=5.0)
        rs = world.get_world_state()["robot"]
        return f"robot now at ({rs['x']:.1f},{rs['y']:.1f})"

    return world, [set_light, get_pose, list_rooms, navigate_to], call_log


def _print_world_state(world: ToyWorld, label: str) -> None:
    state = world.get_world_state()
    rooms = state["rooms"]
    on_rooms = [r for r, info in rooms.items() if info["light_on"]]
    print(
        f"  [world @ {label}] robot=({state['robot']['x']:.1f},{state['robot']['y']:.1f}) "
        f"battery={state['robot']['battery_pct']:.0f}%  "
        f"lights_on={on_rooms or '(none)'}"
    )


# ----- LLM acquisition -----


def _build_real_llm(model_path: str):
    """Load a real llama-cpp-backed LLM. One instance, shared across the
    three roles -- the agents serialise on the LLM's inference lock so
    role separation is by *prompt*, not by *process*."""
    from edgevox.llm import LLM

    print(f"  Loading model: {model_path}", file=sys.stderr)
    return LLM(model_path=model_path)


def _build_scripted_llm():
    """Deterministic stand-in for offline / CI. Returns an LLM-shaped
    object whose responses are pre-canned. Useful for verifying the
    recipe wiring without a model download."""
    # Imported lazily so the file works in environments where pytest
    # isn't installed. The harness module is a regular .py file so a
    # plain import is fine.
    import threading as _threading

    class _ScriptedDemoLLM:
        """Tiny ScriptedLLM equivalent with the right shape for LLMAgent."""

        def __init__(self, script: list[dict]) -> None:
            self._script = list(script)
            self._language = "en"
            self.calls: list[dict] = []
            self._inference_lock = _threading.RLock()

        def complete(
            self,
            messages,
            *,
            tools=None,
            tool_choice=None,
            max_tokens=256,
            temperature=0.7,
            stream=False,
            stop_event=None,
            grammar=None,
            seed=None,
        ):
            self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
            if not self._script:
                raise RuntimeError(f"_ScriptedDemoLLM exhausted after {len(self.calls)} calls")
            return {"choices": [{"message": self._script.pop(0)}]}

    planner_script = [
        {
            "content": (
                "Goal: kitchen light on, robot in bedroom.\n"
                "Plan:\n"
                "1. set_light(room='kitchen', on=true)\n"
                "2. navigate_to(room='bedroom')\n"
                "Pass criterion: kitchen.light_on == true AND robot at bedroom."
            ),
            "tool_calls": None,
        },
    ]
    executor_script = [
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "c0",
                    "function": {"name": "set_light", "arguments": json.dumps({"room": "kitchen", "on": True})},
                },
                {
                    "id": "c1",
                    "function": {"name": "navigate_to", "arguments": json.dumps({"room": "bedroom"})},
                },
            ],
        },
        {"content": "Set kitchen light on; navigate_to(bedroom) launched.", "tool_calls": None},
    ]
    evaluator_script = [
        {
            "content": "VERDICT: PASS -- kitchen light on; navigation kicked off.",
            "tool_calls": None,
        },
    ]
    return (
        _ScriptedDemoLLM(planner_script),
        _ScriptedDemoLLM(executor_script),
        _ScriptedDemoLLM(evaluator_script),
    )


# ----- Main -----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="Model path or hf: shorthand. Required unless --scripted.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use a deterministic ScriptedLLM (no model required).",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"User request to run through the recipe. Default: {DEFAULT_TASK!r}",
    )
    args = parser.parse_args(argv)

    if not args.scripted and not args.model:
        parser.error("--model is required (or pass --scripted for offline mode)")

    print("=" * 72)
    print("Plan-Execute-Evaluate demo on ToyWorld")
    print("=" * 72)
    print(f"  Task: {args.task!r}")
    print(f"  Mode: {'scripted (offline)' if args.scripted else 'real LLM'}")
    print()

    world, tools, call_log = _build_world_and_tools()
    _print_world_state(world, "before")
    print()

    if args.scripted:
        planner_llm, executor_llm, evaluator_llm = _build_scripted_llm()
    else:
        shared = _build_real_llm(args.model)
        planner_llm = executor_llm = evaluator_llm = shared

    recipe = PlanExecuteEvaluate.build(
        planner_llm=planner_llm,
        executor_llm=executor_llm,
        evaluator_llm=evaluator_llm,
        tools=tools,
    )

    # AgentContext.deps lets skills reach the world via ctx.deps.apply_action(...).
    from edgevox.agents.base import AgentContext

    ctx = AgentContext(deps=world, stop=threading.Event())

    print("[stage 1/3] Planner ...", flush=True)
    print("[stage 2/3] Executor ... (tools dispatch here)")
    print("[stage 3/3] Evaluator ...")
    result = recipe.run(args.task, ctx)
    print()

    print("--- Recipe output ---")
    print(result.reply)
    print()

    _print_world_state(world, "after")
    print()
    print("Tool / skill calls observed during the run:")
    for name, args_ in call_log:
        print(f"  - {name}({args_})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
