"""Multi-step agentic demos across the three simulation tiers.

One ``python -m`` invocation per scenario. Each demo wires a real LLM
to the simulation it actually exercises, then prints the world state
before, during, and after so the run is observable.

Subcommands:

  toyworld_morning_routine   Tier 0 -- 3-step ToyWorld home flow with a
                             single-agent multi-tool dispatch.
  toyworld_approval_then_run Tier 0 -- ApprovalGate around a privileged
                             navigation action.
  irsim_patrol               Tier 1 -- IR-SIM list_rooms -> navigate
                             pipeline driven by ScriptedLLM (real-LLM
                             with --model is opt-in; needs ~30s of
                             physics).
  mujoco_pick_red            Tier 2a -- MuJoCo arm gantry: list_objects
                             -> grasp red_cube -> goto_home.

Run:

    python -m edgevox.examples.agents.multistep_demos toyworld_morning_routine \\
        --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf

Each scenario also accepts ``--scripted`` to run with a deterministic
``ScriptedLLM`` that mirrors the matching scenario test in
``tests/harness/test_sim_scenarios.py`` -- useful as a smoke check
when no model is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading

from edgevox.agents.base import AgentContext, LLMAgent
from edgevox.agents.workflow_recipes import ApprovalGate
from edgevox.llm import tool

# ---- Shared LLM acquisition ----


def _build_real_llm(model_path: str):
    from edgevox.llm import LLM

    print(f"[demo] loading model: {model_path}", file=sys.stderr)
    return LLM(model_path=model_path)


def _scripted(messages: list[dict]):
    """Tiny ScriptedLLM equivalent so demo files don't depend on test
    fixtures."""
    import threading as _t

    class _S:
        def __init__(self):
            self._script = list(messages)
            self._language = "en"
            self.calls: list[dict] = []
            self._inference_lock = _t.RLock()

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
            self.calls.append({"messages": [dict(m) for m in messages]})
            if not self._script:
                raise RuntimeError(f"scripted LLM exhausted after {len(self.calls)} calls")
            return {"choices": [{"message": self._script.pop(0)}]}

    return _S()


def _reply(content: str) -> dict:
    return {"content": content, "tool_calls": None}


def _calls(*specs: tuple[str, dict]) -> dict:
    return {
        "content": None,
        "tool_calls": [
            {"id": f"c{i}_{n}", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(specs)
        ],
    }


# ---- Scenario A: ToyWorld morning routine ----


def _toyworld_morning_routine(args: argparse.Namespace) -> int:
    from edgevox.agents.sim import ToyWorld

    world = ToyWorld()
    log: list[tuple[str, dict]] = []

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name -- one of living_room / kitchen / bedroom / office.
            on: true to turn on.
        """
        log.append(("set_light", {"room": room, "on": on}))
        with world._lock:
            if room not in world._rooms:
                return f"unknown room {room!r}"
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    @tool
    def get_pose() -> str:
        """Return the robot's current pose."""
        log.append(("get_pose", {}))
        return json.dumps(world.get_world_state()["robot"])

    if args.scripted:
        llm = _scripted(
            [
                _calls(
                    ("set_light", {"room": "living_room", "on": True}),
                    ("set_light", {"room": "kitchen", "on": True}),
                    ("get_pose", {}),
                ),
                _reply("Living room and kitchen lights are on; pose reported."),
            ]
        )
    else:
        llm = _build_real_llm(args.model)

    agent = LLMAgent(
        name="HomeBot",
        description="home assistant",
        instructions=(
            "You are a household assistant. The user issues a multi-step "
            "request. Plan everything, then issue all tool calls in one "
            "hop -- the framework dispatches them in parallel. Reply with "
            "a one-sentence summary of what you did."
        ),
        tools=[set_light, get_pose],
        llm=llm,
    )

    print(f"[task] {args.task!r}")
    print(
        f"[before] lights_on={[r for r, info in world.get_world_state()['rooms'].items() if info['light_on']] or '(none)'}"
    )
    result = agent.run(args.task)
    print(
        f"[after]  lights_on={[r for r, info in world.get_world_state()['rooms'].items() if info['light_on']] or '(none)'}"
    )
    print()
    print("--- Reply ---")
    print(result.reply)
    print()
    print("Tool calls:")
    for n, a in log:
        print(f"  - {n}({a})")
    return 0


# ---- Scenario B: ToyWorld approval gate ----


def _toyworld_approval_gate(args: argparse.Namespace) -> int:
    from edgevox.agents.sim import ToyWorld

    world = ToyWorld()
    log: list[tuple[str, dict]] = []

    @tool
    def navigate_to(room: str) -> str:
        """Drive the robot to a named room.

        Args:
            room: target room.
        """
        log.append(("navigate_to", {"room": room}))
        handle = world.apply_action("navigate_to", room=room)
        handle.poll(timeout=5.0)
        rs = world.get_world_state()["robot"]
        return f"robot moved toward {room} -- now at ({rs['x']:.1f},{rs['y']:.1f})"

    if args.scripted:
        # Deterministic scripted path: proposer drafts, approver denies,
        # executor never runs.
        proposer_llm = _scripted([_reply("Plan: navigate_to(bedroom). Privileged.")])
        approver_llm = _scripted([_reply("DENIED -- bedroom is privacy-restricted; require explicit user opt-in.")])
        executor_llm = _scripted([_calls(("navigate_to", {"room": "bedroom"})), _reply("done")])
    else:
        # Real LLM mode: share one model across all three roles. Each
        # role's system prompt is different so the model plays each part.
        shared = _build_real_llm(args.model)
        proposer_llm = approver_llm = executor_llm = shared

    executor = LLMAgent(
        name="Driver",
        description="executes navigation",
        instructions="You are a driver. Use navigate_to to drive the robot.",
        tools=[navigate_to],
        llm=executor_llm,
    )
    gate = ApprovalGate.build(
        proposer_llm=proposer_llm,
        approver_llm=approver_llm,
        executor_agent=executor,
    )

    print(f"[task] {args.task!r}")
    print(f"[before] robot at {world.get_world_state()['robot']}")
    result = gate.run(args.task)
    print(f"[after]  robot at {world.get_world_state()['robot']}")
    print()
    print("--- Verdict / Reply ---")
    print(result.reply)
    print()
    print("Tool calls observed:", log)
    return 0


# ---- Scenario C: MuJoCo pick-red-cube ----


def _mujoco_pick_red(args: argparse.Namespace) -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[error] mujoco is not installed -- run `pip install mujoco`", file=sys.stderr)
        return 2

    from edgevox.agents.skills import skill
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    print("[demo] loading MuJoCo gantry scene (headless)...", file=sys.stderr)
    world = MujocoArmEnvironment(
        model_source="gantry",
        render=False,
        allow_hf_download=False,
    )

    events: list[tuple[str, dict]] = []

    @tool
    def list_objects() -> str:
        """List the freejoint objects the arm can grasp."""
        h = world.apply_action("list_objects")
        h.poll(timeout=2.0)
        events.append(("list_objects", {"result": h.result}))
        return json.dumps(h.result)

    @skill(latency_class="slow", timeout_s=15.0)
    def grasp(object: str, ctx):
        """Grasp a named object.

        Args:
            object: body name (red_cube / green_cube / blue_cube).
        """
        events.append(("grasp", {"object": object}))
        return ctx.deps.apply_action("grasp", object=object)

    @skill(latency_class="slow", timeout_s=15.0)
    def goto_home(ctx):
        """Return the arm to its home pose."""
        events.append(("goto_home", {}))
        return ctx.deps.apply_action("goto_home")

    if args.scripted:
        llm = _scripted(
            [
                _calls(("list_objects", {})),
                _calls(("grasp", {"object": "red_cube"})),
                _calls(("goto_home", {})),
                _reply("Picked the red cube and returned home."),
            ]
        )
    else:
        llm = _build_real_llm(args.model)

    agent = LLMAgent(
        name="PickerBot",
        description="tabletop manipulator",
        instructions=(
            "You drive a robot arm. Use list_objects to see what's on the "
            "table, then grasp the requested object, then goto_home. Reply "
            "with one sentence summarising what happened."
        ),
        tools=[list_objects],
        skills=[grasp, goto_home],
        llm=llm,
    )

    ctx = AgentContext(deps=world, stop=threading.Event())
    print(f"[task] {args.task!r}")
    result = agent.run(args.task, ctx)
    print()
    print("--- Reply ---")
    print(result.reply)
    print()
    print("Events:")
    for n, a in events:
        print(f"  - {n}({a})")
    return 0


# ---- Scenario D: IR-SIM patrol ----


def _irsim_patrol(args: argparse.Namespace) -> int:
    try:
        import irsim  # noqa: F401
    except ImportError:
        print("[error] irsim is not installed -- run `pip install ir-sim`", file=sys.stderr)
        return 2

    from edgevox.agents.skills import skill
    from edgevox.integrations.sim.irsim import IrSimEnvironment

    print("[demo] loading IR-SIM apartment (headless)...", file=sys.stderr)
    world = IrSimEnvironment(render=False, tick_interval=0.02)
    try:
        events: list[tuple[str, dict]] = []

        @tool
        def list_rooms() -> str:
            """List the rooms the robot knows about."""
            h = world.apply_action("list_rooms")
            h.poll(timeout=1.0)
            events.append(("list_rooms", {"result": h.result}))
            return ", ".join(h.result)

        @skill(latency_class="slow", timeout_s=10.0)
        def navigate_to(room: str, ctx):
            """Drive the robot to a named room.

            Args:
                room: target room.
            """
            events.append(("navigate_to", {"room": room}))
            return ctx.deps.apply_action("navigate_to", room=room)

        if args.scripted:
            llm = _scripted(
                [
                    _calls(("list_rooms", {})),
                    _calls(("navigate_to", {"room": "kitchen"})),
                    _reply("Listed rooms; navigating to kitchen."),
                ]
            )
        else:
            llm = _build_real_llm(args.model)

        agent = LLMAgent(
            name="ApartmentBot",
            description="2D nav agent",
            instructions="Use list_rooms to see options, then navigate_to the requested target. Reply briefly.",
            tools=[list_rooms],
            skills=[navigate_to],
            llm=llm,
        )

        ctx = AgentContext(deps=world, stop=threading.Event())
        print(f"[task] {args.task!r}")
        result = agent.run(args.task, ctx)
        print()
        print("--- Reply ---")
        print(result.reply)
        print()
        print("Events:")
        for n, a in events:
            print(f"  - {n}({a})")
        return 0
    finally:
        world._phys_stop.set()
        world._phys_thread.join(timeout=1.0)


# ---- Scenario E: MuJoCo pick-and-place (full action surface) ----


def _mujoco_pick_and_place(args: argparse.Namespace) -> int:
    """Cool flow: list -> grasp red -> move_to drop zone -> release -> goto_home.

    Exercises every action the arm sim exposes in a single agent run.
    """
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[error] mujoco is not installed", file=sys.stderr)
        return 2

    from edgevox.agents.skills import skill
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    print("[demo] loading MuJoCo gantry scene (headless)...", file=sys.stderr)
    world = MujocoArmEnvironment(model_source="gantry", render=False, allow_hf_download=False)
    events: list[tuple[str, dict]] = []

    @tool
    def list_objects() -> str:
        """List freejoint objects on the table with their positions."""
        h = world.apply_action("list_objects")
        h.poll(timeout=2.0)
        events.append(("list_objects", {"result": h.result}))
        return json.dumps(h.result)

    @skill(latency_class="slow", timeout_s=15.0)
    def grasp(object: str, ctx):
        """Grasp a named object body.

        Args:
            object: body name (red_cube / green_cube / blue_cube).
        """
        events.append(("grasp", {"object": object}))
        return ctx.deps.apply_action("grasp", object=object)

    @skill(latency_class="slow", timeout_s=15.0)
    def move_to(x: float, y: float, z: float, ctx):
        """Move arm end effector to (x, y, z).

        Args:
            x: target x (m).
            y: target y (m).
            z: target z (m).
        """
        events.append(("move_to", {"x": x, "y": y, "z": z}))
        return ctx.deps.apply_action("move_to", x=x, y=y, z=z)

    @skill(latency_class="slow", timeout_s=15.0)
    def release(ctx):
        """Release the gripper."""
        events.append(("release", {}))
        return ctx.deps.apply_action("release")

    @skill(latency_class="slow", timeout_s=15.0)
    def goto_home(ctx):
        """Return to home pose."""
        events.append(("goto_home", {}))
        return ctx.deps.apply_action("goto_home")

    if args.scripted:
        llm = _scripted(
            [
                _calls(("list_objects", {})),
                _calls(("grasp", {"object": "red_cube"})),
                _calls(("move_to", {"x": 0.0, "y": 0.3, "z": 0.5})),
                _calls(("release", {})),
                _calls(("goto_home", {})),
                _reply("Red cube relocated to (0.0, 0.3, 0.5); arm returned home."),
            ]
        )
    else:
        llm = _build_real_llm(args.model)

    agent = LLMAgent(
        name="PickerBot",
        description="full-surface arm",
        instructions=(
            "You drive a robot arm with these primitives: list_objects, grasp, "
            "move_to(x,y,z), release, goto_home. To MOVE an object, you must: "
            "1) grasp it, 2) move_to the destination, 3) release, 4) goto_home. "
            "Plan the full sequence; reply with one sentence."
        ),
        tools=[list_objects],
        skills=[grasp, move_to, release, goto_home],
        max_tool_hops=8,
        llm=llm,
    )

    ctx = AgentContext(deps=world, stop=threading.Event())
    print(f"[task] {args.task!r}")
    result = agent.run(args.task, ctx)
    print()
    print("--- Reply ---")
    print(result.reply)
    print()
    print("Events:")
    for n, a in events:
        print(f"  - {n}({a})")
    return 0


# ---- Scenario F: IR-SIM battery-aware ----


def _irsim_battery_aware(args: argparse.Namespace) -> int:
    """Read battery first; navigate only if healthy. State-driven branching."""
    try:
        import irsim  # noqa: F401
    except ImportError:
        print("[error] irsim not installed", file=sys.stderr)
        return 2

    from edgevox.agents.skills import skill
    from edgevox.integrations.sim.irsim import IrSimEnvironment

    print("[demo] loading IR-SIM apartment...", file=sys.stderr)
    world = IrSimEnvironment(render=False, tick_interval=0.02)
    try:
        events: list[tuple[str, dict]] = []

        @tool
        def battery_level() -> str:
            """Return battery percentage."""
            h = world.apply_action("battery_level")
            h.poll(timeout=1.0)
            events.append(("battery_level", h.result))
            return json.dumps(h.result)

        @tool
        def list_rooms() -> str:
            """List rooms."""
            h = world.apply_action("list_rooms")
            h.poll(timeout=1.0)
            events.append(("list_rooms", {"result": h.result}))
            return ", ".join(h.result)

        @skill(latency_class="slow", timeout_s=10.0)
        def navigate_to(room: str, ctx):
            """Drive to a named room.

            Args:
                room: target room.
            """
            events.append(("navigate_to", {"room": room}))
            return ctx.deps.apply_action("navigate_to", room=room)

        if args.scripted:
            llm = _scripted(
                [
                    _calls(("battery_level", {})),
                    _calls(("navigate_to", {"room": "office"})),
                    _reply("Battery healthy at 95%; navigating to office."),
                ]
            )
        else:
            llm = _build_real_llm(args.model)

        agent = LLMAgent(
            name="CautiousNav",
            description="battery-aware nav",
            instructions=(
                "Read battery_level first. If battery >= 50%, navigate to the "
                "requested room. Else reply you'll wait. Keep replies brief."
            ),
            tools=[battery_level, list_rooms],
            skills=[navigate_to],
            llm=llm,
        )

        ctx = AgentContext(deps=world, stop=threading.Event())
        print(f"[task] {args.task!r}")
        result = agent.run(args.task, ctx)
        print()
        print("--- Reply ---")
        print(result.reply)
        print()
        print("Events:")
        for n, a in events:
            print(f"  - {n}({a})")
        return 0
    finally:
        world._phys_stop.set()
        world._phys_thread.join(timeout=1.0)


# ---- Main ----


_DISPATCH = {
    "toyworld_morning_routine": (
        _toyworld_morning_routine,
        "Turn on the living room and kitchen lights, then report your pose.",
    ),
    "toyworld_approval_then_run": (_toyworld_approval_gate, "drive to the bedroom (privileged)"),
    "irsim_patrol": (_irsim_patrol, "list known rooms then drive to the kitchen"),
    "irsim_battery_aware": (_irsim_battery_aware, "drive to the office if you have enough battery"),
    "mujoco_pick_red": (_mujoco_pick_red, "pick up the red cube and return home"),
    "mujoco_pick_and_place": (_mujoco_pick_and_place, "move the red cube to (0.0, 0.3, 0.5)"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=sorted(_DISPATCH), help="Scenario subcommand")
    parser.add_argument("--model", help="Model path or hf: shorthand. Required unless --scripted.")
    parser.add_argument("--scripted", action="store_true", help="Use deterministic ScriptedLLM (no model needed).")
    parser.add_argument("--task", help="Override the default task string for the chosen scenario.")
    args = parser.parse_args(argv)

    handler, default_task = _DISPATCH[args.scenario]
    if args.task is None:
        args.task = default_task
    if not args.scripted and not args.model:
        parser.error("--model is required (or pass --scripted)")

    print("=" * 78)
    print(f"Multi-step demo: {args.scenario}")
    print("=" * 78)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
