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

    if args.render:
        print("[demo] loading MuJoCo Franka scene (with viewer)...", file=sys.stderr)
    else:
        print("[demo] loading MuJoCo Franka scene (headless)...", file=sys.stderr)
    # Use Franka (the real model) instead of gantry: gantry has a physics
    # bug that rockets cubes on arm contact (verified May 2026).
    world = MujocoArmEnvironment(model_source="franka", render=args.render)

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
    if args.render:
        # Pump the viewer for a few seconds so the user can see the
        # final scene state before the daemon thread tears down.
        import time as _t

        print("\n[viewer] holding open for 8 s -- close the window or wait...")
        for _ in range(80):
            world.pump_render()
            _t.sleep(0.1)
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

    if args.render:
        print("[demo] loading MuJoCo Franka scene (with viewer)...", file=sys.stderr)
    else:
        print("[demo] loading MuJoCo Franka scene (headless)...", file=sys.stderr)
    # Franka grasps cleanly. The gantry model has a physics bug where the
    # actuators rocket the cube on contact (verified May 2026). Until the
    # gantry is fixed, real demos run on Franka.
    world = MujocoArmEnvironment(model_source="franka", render=args.render)
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

    # Choreography: pick three cubes one at a time, drop each in a
    # different zone, then return home. Drives the sim end-to-end so
    # the user can watch real motion in the viewer. Each segment fires
    # the @skill the same way an LLM would; the only difference is
    # determinism -- no model in the loop.
    drop_zones = {
        "red_cube": (0.5, 0.30, 0.50),
        "green_cube": (0.5, 0.00, 0.50),
        "blue_cube": (0.5, -0.30, 0.50),
    }

    if args.scripted or args.render:
        # In scripted/render mode, drive the sim directly so we don't
        # depend on the LLM cooperating. The agent's @skill wrappers
        # are still in place; we just drive them via ctx.deps directly.
        # (This is also what an LLM would compile to internally.)
        from edgevox.agents.skills import GoalStatus as _GS

        def _wait(handle, label, timeout_s=12.0):
            print(f"  -> {label}", flush=True)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < timeout_s:
                if handle.status in (_GS.SUCCEEDED, _GS.FAILED, _GS.CANCELLED):
                    break
                time.sleep(0.1)
            return handle.status

        import time

        print(f"[task] {args.task!r}")
        print(f"[choreography] pick + place {len(drop_zones)} cubes, then home")

        for cube, (x, y, z) in drop_zones.items():
            events.append(("grasp", {"object": cube}))
            h = world.apply_action("grasp", object=cube)
            _wait(h, f"grasp {cube}")

            events.append(("move_to", {"x": x, "y": y, "z": z}))
            h = world.apply_action("move_to", x=x, y=y, z=z)
            _wait(h, f"move_to ({x}, {y}, {z})")

            events.append(("release", {}))
            h = world.apply_action("release")
            _wait(h, "release")

        events.append(("goto_home", {}))
        h = world.apply_action("goto_home")
        _wait(h, "goto_home")

        print()
        print("--- Final cube positions ---")
        for obj in world.apply_action("list_objects").result or []:
            print(f"  {obj['name']}: ({obj['x']:.2f}, {obj['y']:.2f}, {obj['z']:.2f})")

        if args.render:
            print("\n[viewer] holding open for 8 s -- close the window or wait...")
            for _ in range(80):
                world.pump_render()
                time.sleep(0.1)
        return 0

    # Real-LLM path (model-driven).
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


# ---- Scenario G: MuJoCo LLM-controlled choreography ----


def _mujoco_llm_choreography(args: argparse.Namespace) -> int:
    """LLM drives the arm through pick + place + release for each cube,
    then returns home. Uses an outer Python plan + per-step mini-agents
    so a small model only has to call ONE skill per LLM call -- which
    is a load Gemma 4 E4B handles reliably, even though it sycophants
    on long single-agent skill chains.

    The whole sequence is observable in the MuJoCo viewer when --render
    is set. ~12 mini-agent calls, one LLM hop each, ~60-180 s end to end.
    """
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[error] mujoco not installed", file=sys.stderr)
        return 2

    from edgevox.agents.skills import skill
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    if args.render:
        print("[demo] loading MuJoCo Franka scene (with viewer)...", file=sys.stderr)
    else:
        print("[demo] loading MuJoCo Franka scene (headless)...", file=sys.stderr)
    world = MujocoArmEnvironment(model_source="franka", render=args.render)
    events: list[tuple[str, dict]] = []

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
        """Move the arm end effector to (x, y, z).

        Args:
            x: target x in metres.
            y: target y in metres.
            z: target z in metres.
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
        # 12 turns: 3 cubes x (grasp, move, release) + final home.
        scripted_msgs: list[dict] = []
        plan = [
            ("red_cube", 0.5, 0.30, 0.50),
            ("green_cube", 0.5, 0.00, 0.50),
            ("blue_cube", 0.5, -0.30, 0.50),
        ]
        for cube, x, y, z in plan:
            scripted_msgs.append(_calls(("grasp", {"object": cube})))
            scripted_msgs.append(_calls(("move_to", {"x": x, "y": y, "z": z})))
            scripted_msgs.append(_calls(("release", {})))
        scripted_msgs.append(_calls(("goto_home", {})))
        llm = _scripted(scripted_msgs)
    else:
        llm = _build_real_llm(args.model)

    ctx = AgentContext(deps=world, stop=threading.Event())

    # Live-trace hooks: prints LLM thinking + tool calls + results as
    # they fire, so the operator can read the agent's reasoning in the
    # terminal alongside the MuJoCo viewer animation. Shared module so
    # other demos / agents reuse the same prefix style.
    from edgevox.agents.trace_hooks import terminal_trace_hooks

    trace_hooks = terminal_trace_hooks()

    def _step(skill_obj, instructions: str, task: str, label: str) -> None:
        """One mini-agent run -- the agent has exactly one skill so the
        LLM has no other option but to call it."""
        import time as _t

        t0 = _t.perf_counter()
        print(f"  [step] {label}", flush=True)
        sub = LLMAgent(
            name="Step",
            description="single-skill executor",
            instructions=instructions,
            skills=[skill_obj],
            llm=llm,
            hooks=trace_hooks,
            max_tool_hops=2,
        )
        sub.run(task, ctx)
        print(f"        took {_t.perf_counter() - t0:.1f}s", flush=True)

    # Default: all 3 cubes. Pass --single-cube to run only red cube (fast).
    plan = [
        ("red_cube", 0.5, 0.30, 0.50),
        ("green_cube", 0.5, 0.00, 0.50),
        ("blue_cube", 0.5, -0.30, 0.50),
    ]
    if getattr(args, "single_cube", False):
        plan = plan[:1]

    print(f"[task] {args.task!r}", flush=True)
    print(f"[plan] {len(plan)} cube(s); per-cube: grasp -> move_to -> release; then goto_home", flush=True)

    # Run the choreography on a worker thread so the main thread can
    # pump the MuJoCo viewer continuously. Without this, pump_render
    # only fires after all LLM hops complete -- the arm physically
    # moves in the background but the viewer shows a frozen frame
    # the whole time.
    import time as _t

    choreography_done = threading.Event()
    choreography_error: list[Exception] = []

    def _choreography_worker() -> None:
        try:
            for cube, x, y, z in plan:
                _step(grasp, "Call the grasp skill with the named object.", f"Grasp the {cube}.", f"grasp {cube}")
                _step(
                    move_to,
                    "Call the move_to skill with the given x, y, z coordinates.",
                    f"Move arm to ({x}, {y}, {z}).",
                    f"move_to ({x},{y},{z})",
                )
                _step(release, "Call the release skill.", "Release the gripper.", "release")
            _step(goto_home, "Call the goto_home skill.", "Return to home pose.", "goto_home")
        except Exception as e:
            choreography_error.append(e)
        finally:
            choreography_done.set()

    worker = threading.Thread(target=_choreography_worker, name="choreography", daemon=True)
    worker.start()

    if args.render:
        # Pump the viewer at ~30 Hz while the worker runs. This is the
        # critical change -- without main-thread pumping the viewer
        # never refreshes during long LLM hops.
        while not choreography_done.is_set():
            world.pump_render()
            _t.sleep(0.033)
    else:
        worker.join()

    if choreography_error:
        raise choreography_error[0]

    print()
    print("--- Final cube positions ---")
    for obj in world.apply_action("list_objects").result or []:
        print(f"  {obj['name']}: ({obj['x']:.2f}, {obj['y']:.2f}, {obj['z']:.2f})")
    print()
    print("Skill calls observed:")
    for n, a in events:
        print(f"  - {n}({a})")

    if args.render:
        print("\n[viewer] holding open for 8 s -- close the window or wait...")
        for _ in range(240):
            world.pump_render()
            _t.sleep(0.033)
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
    "mujoco_llm_choreography": (
        _mujoco_llm_choreography,
        "sort all 3 cubes into separate zones, then go home",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=sorted(_DISPATCH), help="Scenario subcommand")
    parser.add_argument("--model", help="Model path or hf: shorthand. Required unless --scripted.")
    parser.add_argument("--scripted", action="store_true", help="Use deterministic ScriptedLLM (no model needed).")
    parser.add_argument("--task", help="Override the default task string for the chosen scenario.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a viewer window for sims that support it (MuJoCo, IR-SIM matplotlib).",
    )
    parser.add_argument(
        "--single-cube",
        action="store_true",
        help="LLM choreography only: run one cube instead of three (faster).",
    )
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
