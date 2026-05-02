"""Framework for building voice-agent example apps.

The goal: example authors declare a list of ``@tool``-decorated
functions plus a name/greeting, then call ``AgentApp(...).run()``. The
framework handles everything else — argv parsing, model resolution with
a first-run download banner, mode selection (TUI / simple CLI voice /
text chat), and tool-call tracing.

Three launch modes:

- **Default (TUI)** — full voice pipeline backed by the Textual UI
  shipped with EdgeVox. Users press-to-talk (or use wakeword) and the
  agent can call tools in response.
- ``--simple-ui`` — the plain rich-based CLI voice loop
  (``edgevox.cli.main.VoiceBot``). Lighter than the TUI, still voice.
- ``--text-mode`` — a keyboard REPL that runs the same ``LLM.chat``
  path, useful for debugging tools without a microphone.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from edgevox.agents import (
    Agent,
    AgentContext,
    AgentEvent,
    Hook,
    LLMAgent,
    Session,
    Skill,
)
from edgevox.llm import Tool, ToolCallResult, ToolRegistry
from edgevox.llm.llamacpp import DEFAULT_HF_FILE, DEFAULT_HF_REPO

ToolCallback = Callable[[ToolCallResult], None]
ToolsArg = Iterable[Callable[..., object] | Tool] | ToolRegistry
SkillsArg = Iterable[Skill]
HooksArg = Iterable[Hook]

DEFAULT_PERSONA = "You are an EdgeVox voice assistant. Keep responses concise and conversational — 1-3 sentences."


def _parse_hf_shorthand(model: str | None) -> tuple[str, str] | None:
    """Return ``(repo, filename)`` for an ``hf:repo:file`` shorthand or
    the default Gemma pair when ``model`` is ``None``. Returns ``None``
    if ``model`` looks like a local file path (no download step needed).
    """
    if model is None:
        return DEFAULT_HF_REPO, DEFAULT_HF_FILE
    if model.startswith("hf:"):
        rest = model[3:]
        if ":" not in rest:
            raise ValueError(f"Expected 'hf:repo/name:filename.gguf', got {model!r}")
        repo, _, filename = rest.partition(":")
        return repo, filename
    return None


def _is_cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=repo, filename=filename)
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:
        return False


def _announce_model(console: Console, model: str | None) -> None:
    """Print a friendly notice about model resolution so the first-run
    download doesn't look like the app has hung."""
    hf_pair = _parse_hf_shorthand(model)
    if hf_pair is None:
        console.print(f"[dim]model: {model}[/]")
        return
    repo, filename = hf_pair
    target = f"{repo}/{filename}"
    if _is_cached(repo, filename):
        console.print(f"[dim]model: {target} (cached)[/]")
    else:
        console.print(
            Panel(
                f"[bold]First-run model download[/bold]\n"
                f"Fetching [cyan]{target}[/] into your Hugging Face cache.\n"
                f"This happens once per model — later runs start in a few seconds.\n"
                f"[dim]You'll see a progress bar from huggingface_hub below.[/]",
                title="[yellow]downloading model[/]",
                border_style="yellow",
            )
        )


# ---------------------------------------------------------------------------
# Dispatch-mode helpers (shared across robot examples)
# ---------------------------------------------------------------------------


def dispatch_mode_extra_args() -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Standard argparse extra_args for robot agents that want all
    three dispatch modes (default ReAct + ``--legacy`` single agent +
    ``--planned`` upfront-plan dispatcher).

    Use in an :class:`AgentApp` like::

        APP = AgentApp(
            ...,
            extra_args=[
                *dispatch_mode_extra_args(),
                # robot-specific args
            ],
            pre_run=_my_pre_run,
        )

    And in ``_my_pre_run`` call :func:`apply_dispatch_mode` to swap the
    agent based on the parsed flags.
    """
    return [
        (
            ("--legacy",),
            {
                "action": "store_true",
                "help": (
                    "Use the legacy single-LLMAgent dispatch (sycophants on "
                    "multi-step chains with weak local models -- kept for "
                    "back-compat / benchmarking)."
                ),
            },
        ),
        (
            ("--planned",),
            {
                "action": "store_true",
                "help": (
                    "Use PlannedToolDispatcher (planner emits JSON plan up-"
                    "front, Python loop direct-dispatches each step) instead "
                    "of the default ReAct loop. Faster for tasks where the "
                    "plan is determinable from the user request."
                ),
            },
        ),
    ]


def apply_dispatch_mode(
    app: AgentApp,
    args: argparse.Namespace,
    *,
    tools: Any = None,
    skills: Any = None,
    default_mode: str = "react",
    completion_check: Any = None,
) -> None:
    """Swap ``app.agent`` based on argparse flags.

    Call this from a robot example's ``pre_run`` callback after building
    the sim deps. The resulting agent is one of:

    - ``--legacy``: keep the auto-built single :class:`LLMAgent` (does
      nothing -- the framework's ``__post_init__`` already wired it).
    - ``--planned``: replace with :class:`PlannedToolDispatcher`
      (planner -> Python loop -> synthesiser).
    - default (ReAct): replace with :class:`ReActAgent` (think -> act ->
      observe loop with TASK COMPLETE termination marker + recheck).

    The ReAct path automatically attaches :func:`terminal_trace_hooks`
    so the operator sees the agent's reasoning and tool calls in
    ``--text-mode``.

    Args:
        app: the :class:`AgentApp` instance (typically the module-level
            ``APP`` constant).
        args: parsed argparse namespace from :meth:`AgentApp.run`.
        tools: tool list to attach to the recipe agent. If None,
            ``app.tools`` is used.
        skills: skill list. If None, ``app.skills`` is used.
        default_mode: which mode wins when no flag is passed. ``"react"``
            (the canonical agentic-loop default), ``"planned"`` or
            ``"legacy"``.
    """
    tools = tools if tools is not None else (app.tools or [])
    skills = skills if skills is not None else (app.skills or [])

    if getattr(args, "legacy", False):
        return  # auto-built LLMAgent stays
    if getattr(args, "planned", False):
        from edgevox.agents.workflow_recipes import PlannedToolDispatcher

        app.agent = PlannedToolDispatcher.build(
            planner_llm=None,
            synthesiser_llm=None,
            tools=tools,
            skills=skills,
            max_steps=10,
        )
        return

    # Default: ReAct loop with terminal trace hooks.
    if default_mode == "legacy":
        return
    if default_mode == "planned":
        from edgevox.agents.workflow_recipes import PlannedToolDispatcher

        app.agent = PlannedToolDispatcher.build(
            planner_llm=None,
            synthesiser_llm=None,
            tools=tools,
            skills=skills,
            max_steps=10,
        )
        return

    from edgevox.agents.trace_hooks import terminal_trace_hooks
    from edgevox.agents.workflow_recipes import ReActAgent

    app.agent = ReActAgent.build(
        llm=None,
        tools=tools,
        skills=skills,
        hooks=terminal_trace_hooks(),
        max_iterations=20,
        completion_check=completion_check,
    )


@dataclass
class AgentApp:
    """Declarative description of a voice-agent example.

    You must provide one of:

    - ``agent=`` — a pre-constructed :class:`~edgevox.agents.Agent`
      (workflow or ``LLMAgent``). Use this when you need multi-agent
      handoffs, sequences, loops, or a router.
    - ``tools=`` / ``skills=`` — a bag of ``@tool`` functions and/or
      ``@skill`` Skill instances. The framework wraps them in a default
      single-agent ``LLMAgent`` using ``instructions`` as the persona.

    Attributes:
        name: human-readable display name shown in banners.
        agent: pre-built Agent or workflow. Mutually exclusive with tools/skills.
        tools: list of ``@tool``-decorated functions.
        skills: list of ``@skill``-decorated Skills (cancellable actions).
        instructions: persona / system prompt for the default LLMAgent
            when ``agent`` is not provided.
        deps: user-supplied dependency object passed through
            ``AgentContext.deps``. Typically a ``ToyWorld`` or
            ``SimEnvironment``.
        stop_words: override the default SafetyMonitor stop-word list.
        greeting: optional opening line shown in text mode.
        language: default language code.
        default_model: LLM model path or ``hf:repo:file`` shorthand.
        description: shown in ``--help``.
        extra_args: optional argparse additions.
        pre_run: optional callback invoked before the agent launches.
    """

    name: str
    agent: Agent | None = None
    tools: ToolsArg | None = None
    skills: SkillsArg | None = None
    hooks: HooksArg | None = None
    instructions: str | None = None
    deps: Any = None
    stop_words: tuple[str, ...] | None = None
    greeting: str | None = None
    language: str = "en"
    default_model: str | None = None
    description: str = ""
    extra_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    pre_run: Callable[[argparse.Namespace], None] | None = None
    # Hop budget for the auto-built LLMAgent. Default mirrors LLMAgent's
    # default (3) so existing apps don't change behaviour. Robot demos
    # with chains like grasp -> move -> release -> goto_home need 4+;
    # crank to 8-10 for those.
    max_tool_hops: int = 3

    def __post_init__(self) -> None:
        if self.agent is None and self.tools is None and self.skills is None:
            raise ValueError("AgentApp requires one of: agent=..., tools=..., or skills=...")
        if self.agent is None:
            self.agent = LLMAgent(
                name=self.name,
                description=self.description or f"{self.name} voice agent",
                instructions=self.instructions or DEFAULT_PERSONA,
                tools=self.tools,
                skills=self.skills,
                hooks=self.hooks,
                max_tool_hops=self.max_tool_hops,
            )

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=self.name,
            description=self.description or f"{self.name} — EdgeVox agent example",
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--simple-ui",
            action="store_true",
            help="Use the simple rich-based CLI voice loop instead of the Textual TUI.",
        )
        mode.add_argument(
            "--text-mode",
            action="store_true",
            help="Text chat REPL only — no microphone, no TTS playback.",
        )
        parser.add_argument(
            "--model",
            default=self.default_model,
            help="LLM model path or hf:repo/name:file.gguf shorthand (default: Gemma 4 E2B).",
        )
        parser.add_argument("--language", default=self.language, help="Language code (default: %(default)s).")
        parser.add_argument("--voice", default=None, help="TTS voice name override (voice modes only).")
        parser.add_argument("--tts", default=None, help="TTS backend override, e.g. kokoro, piper.")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
        parser.add_argument(
            "--ros2",
            action="store_true",
            help=(
                "Attach a ROS2 bridge alongside the selected UI mode. Publishes "
                "robot_state/agent_event/state/response and accepts text_input "
                "commands over /edgevox/*. Requires a sourced ROS2 workspace."
            ),
        )
        parser.add_argument(
            "--ros2-namespace",
            default="/edgevox",
            help="ROS2 namespace for the agent's bridge node (default: %(default)s).",
        )
        parser.add_argument(
            "--ros2-state-hz",
            type=float,
            default=10.0,
            help="Rate at which the sim/robot world state is republished (default: %(default)s Hz).",
        )
        for args, kwargs in self.extra_args:
            parser.add_argument(*args, **kwargs)
        return parser

    # ----- ROS2 bridge helpers --------------------------------------------

    def _start_ros2_bridge(
        self,
        args: argparse.Namespace,
        console: Console,
    ):
        """Bring up a :class:`ROS2Bridge` wired into the agent's event
        stream and the ``SimEnvironment`` world state.

        Returns ``(bridge, stop_event, world_thread, text_input_queue,
        on_event_wrap)`` — ``stop_event.set()`` + ``world_thread.join()``
        cleanly shut everything down. ``on_event_wrap`` should be used in
        place of the caller's own ``on_event`` so agent events reach
        ROS2 too. ``text_input_queue`` receives strings that arrived on
        ``/<ns>/text_input`` — the REPL drains it alongside stdin.
        """
        import queue
        import threading as _threading

        from edgevox.integrations.ros2_bridge import NullBridge, create_bridge

        bridge = create_bridge(enabled=True, namespace=args.ros2_namespace)
        if isinstance(bridge, NullBridge):
            console.print("[yellow]⚠ --ros2 requested but rclpy was not importable; running without ROS2.[/]")
            return None, None, None, None, None

        console.print(f"[dim]ROS2 bridge up at [cyan]{args.ros2_namespace}[/] (state @ {args.ros2_state_hz:.1f} Hz)[/]")

        # --- Robot adapter (TF2 / cmd_vel / PoseStamped / LaserScan / Image) ---
        try:
            from edgevox.integrations.ros2_robot import create_robot_adapter

            adapter = create_robot_adapter(bridge._node, self.deps)
            if adapter is not None:
                bridge.attach_robot_adapter(adapter)
                caps = []
                if adapter._has_pose2d:
                    caps.append("pose2d+TF")
                if adapter._has_ee_pose:
                    caps.append("ee_pose+TF")
                if adapter._has_velocity:
                    caps.append("cmd_vel")
                if adapter._has_lidar:
                    caps.append("scan")
                if adapter._has_camera:
                    caps.append("image_raw")
                console.print(f"[dim]  robot adapter: {', '.join(caps) or 'nothing'}[/]")
        except Exception:
            logging.getLogger(__name__).debug("robot adapter init failed", exc_info=True)

        # --- Skill ActionServer (requires edgevox_msgs package) ---------------
        try:
            from edgevox.integrations.ros2_actions import (
                create_skill_action_server,
                is_available,
            )

            if is_available() and self.deps is not None:

                def _dispatch(skill_name: str, kwargs: dict[str, Any]) -> Any:
                    apply = getattr(self.deps, "apply_action", None)
                    if apply is None:
                        raise RuntimeError("deps has no apply_action; can't dispatch skill")
                    return apply(skill_name, **kwargs)

                action_server = create_skill_action_server(bridge._node, _dispatch)
                if action_server is not None:
                    bridge.attach_skill_action_server(action_server)
                    console.print("[dim]  execute_skill action server: up[/]")
            elif self.deps is not None:
                console.print(
                    "[dim]  execute_skill action server: "
                    "skipped (edgevox_msgs not built — "
                    "`colcon build --packages-select edgevox_msgs`)[/]"
                )
        except Exception:
            logging.getLogger(__name__).debug("skill action server init failed", exc_info=True)

        stop_event = _threading.Event()
        text_input_queue: queue.Queue[str] = queue.Queue()

        bridge.set_text_input_callback(lambda t: text_input_queue.put(t))

        def _state_loop() -> None:
            period = 1.0 / max(0.1, args.ros2_state_hz)
            while not stop_event.is_set():
                try:
                    get_ws = getattr(self.deps, "get_world_state", None)
                    if callable(get_ws):
                        bridge.publish_robot_state(get_ws())
                except Exception:
                    logging.getLogger(__name__).debug("robot_state publish failed", exc_info=True)
                stop_event.wait(period)

        world_thread: _threading.Thread | None = None
        if self.deps is not None:
            world_thread = _threading.Thread(target=_state_loop, name="ros2-world-state", daemon=True)
            world_thread.start()

        bridge.publish_state("idle")

        def on_event_wrap(inner: Callable[[AgentEvent], None]):
            def _wrapped(e: AgentEvent) -> None:
                try:
                    payload = getattr(e, "payload", None)
                    if e.kind == "tool_call" and hasattr(payload, "name"):
                        bridge.publish_agent_event(
                            {
                                "kind": e.kind,
                                "agent": getattr(e, "agent_name", None),
                                "tool": payload.name,
                                "ok": getattr(payload, "ok", None),
                                "arguments": getattr(payload, "arguments", None),
                                "result": getattr(payload, "result", None),
                                "error": getattr(payload, "error", None),
                            }
                        )
                    else:
                        bridge.publish_agent_event(
                            {
                                "kind": e.kind,
                                "agent": getattr(e, "agent_name", None),
                                "payload": payload,
                            }
                        )
                except Exception:
                    logging.getLogger(__name__).debug("agent_event publish failed", exc_info=True)
                inner(e)

            return _wrapped

        return bridge, stop_event, world_thread, text_input_queue, on_event_wrap

    def _stop_ros2_bridge(self, bridge, stop_event, world_thread) -> None:
        if stop_event is not None:
            stop_event.set()
        if world_thread is not None:
            world_thread.join(timeout=2.0)
        if bridge is not None:
            with contextlib.suppress(Exception):
                bridge.publish_state("shutting_down")
            bridge.shutdown()

    # ----- mode launchers -----

    def _on_tool(self, console: Console) -> ToolCallback:
        def callback(result: ToolCallResult) -> None:
            if result.ok:
                console.print(f"[dim cyan]↳ {result.name}({result.arguments}) → {result.result}[/]")
            else:
                console.print(f"[red]↳ {result.name} failed: {result.error}[/]")

        return callback

    def _input_with_pump(self, prompt: str, pump: Callable[[], None] | None) -> str:
        """Read a line from stdin while periodically pumping a GUI event
        loop. Lets matplotlib (Tk/Qt) stay responsive between REPL turns.

        Falls back to blocking ``input()`` on platforms where
        ``select.select()`` on stdin isn't supported (e.g. Windows).
        """
        import select
        import sys

        sys.stdout.write(prompt)
        sys.stdout.flush()

        if pump is None or not sys.stdin.isatty():
            try:
                return input()
            except EOFError:
                raise

        try:
            while True:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                except (OSError, ValueError):
                    return input()
                if rlist:
                    line = sys.stdin.readline()
                    if not line:
                        raise EOFError
                    return line.rstrip("\n")
                with contextlib.suppress(Exception):
                    pump()
        except KeyboardInterrupt:
            raise

    def _run_text(self, args: argparse.Namespace, console: Console) -> None:
        # Agent-driven text REPL. Runs LLMAgent.run() directly so the
        # same code path is exercised whether the app is a single
        # LLMAgent, a Router, a Sequence, or any nested workflow.
        import queue as _queue
        import threading

        from edgevox.examples.agents._repl import _load_llm_with_progress

        console.print(Panel.fit(self.name, border_style="green"))
        llm = _load_llm_with_progress(
            console,
            model=args.model,
            language=args.language,
            tools=None,  # we attach tools via the agent, not via LLM
        )

        # Bind the shared LLM into every LLMAgent leaf of the composite.
        from edgevox.agents.workflow import _bind_llm_recursive

        _bind_llm_recursive(self.agent, llm)

        stop_event = threading.Event()

        def on_event(e: AgentEvent) -> None:
            if e.kind == "tool_call":
                r = e.payload
                if getattr(r, "ok", False):
                    console.print(f"[dim cyan]↳ {r.name}({r.arguments}) → {r.result}[/]")
                else:
                    console.print(f"[red]↳ {r.name} failed: {getattr(r, 'error', '?')}[/]")
            elif e.kind == "skill_goal":
                console.print(f"[dim yellow]⇒ skill {e.payload['skill']} started[/]")
            elif e.kind == "skill_cancelled":
                console.print(f"[bold red]⇒ skill {e.payload.get('skill', '?')} cancelled[/]")
            elif e.kind == "handoff":
                console.print(f"[dim magenta]→ handoff {e.agent_name} → {e.payload['target']}[/]")
            elif e.kind == "safety_preempt":
                console.print(f"[bold red]⚠ safety preempt: {e.payload}[/]")

        # Wire up ROS2 if requested. The bridge runs alongside stdin:
        # every AgentEvent is republished on /agent_event, the sim
        # world state is refreshed on /robot_state at a fixed Hz, and
        # inbound /text_input messages are processed as if typed.
        bridge = None
        ros2_stop = None
        ros2_thread = None
        ros2_queue: _queue.Queue[str] | None = None
        if getattr(args, "ros2", False):
            bridge, ros2_stop, ros2_thread, ros2_queue, on_event_wrap = self._start_ros2_bridge(args, console)
            if bridge is not None and on_event_wrap is not None:
                on_event = on_event_wrap(on_event)

        def _publish_state(s: str) -> None:
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.publish_state(s)

        def _publish_transcription(t: str) -> None:
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.publish_transcription(t)

        def _publish_response(t: str) -> None:
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.publish_response(t)

        registered_tools: list[str] = []
        if isinstance(self.agent, LLMAgent):
            registered_tools = [t.name for t in self.agent.tools]
        console.print(f"[dim]agent: {self.agent.name} · tools/skills: {', '.join(registered_tools) or '(none)'}[/]")

        if self.greeting:
            console.print(f"[bold green]{self.agent.name}:[/] {self.greeting}")
            _publish_response(self.greeting)

        stop_words = {w.lower() for w in (self.stop_words or ("stop", "halt", "freeze", "abort", "emergency"))}

        pump = getattr(self.deps, "pump_events", None)
        session = Session()

        # Agent dispatch is single-threaded: stdin turns and ROS2 turns
        # serialise on this lock so we don't run two ``agent.run`` calls
        # against the same LLM at the same time.
        dispatch_lock = threading.Lock()

        def _dispatch(user: str, source: str) -> None:
            nonlocal stop_event
            tokens = {t.strip(".,!?").lower() for t in user.split()}
            if tokens & stop_words:
                stop_event.set()
                console.print(f"[bold red]⚠ safety preempt: stop-word in {user!r}[/]")
                console.print(f"[bold green]{self.agent.name}:[/] Stopped.")
                _publish_state("interrupted")
                _publish_response("Stopped.")
                stop_event = threading.Event()
                return

            _publish_transcription(user)
            _publish_state("thinking")
            ctx = AgentContext(
                session=session,
                deps=self.deps,
                on_event=on_event,
                stop=stop_event,
            )
            result = self.agent.run(user, ctx)
            console.print(f"[bold green]{self.agent.name} ({source}):[/] {result.reply}")
            _publish_response(result.reply)
            _publish_state("listening")
            if result.preempted:
                stop_event = threading.Event()

        ros2_worker: threading.Thread | None = None
        if ros2_queue is not None and ros2_stop is not None:

            def _ros2_worker() -> None:
                while not ros2_stop.is_set():
                    try:
                        user = ros2_queue.get(timeout=0.25)
                    except _queue.Empty:
                        continue
                    if not user:
                        continue
                    with dispatch_lock:
                        _dispatch(user, "ros2")

            ros2_worker = threading.Thread(target=_ros2_worker, name="ros2-text-input-worker", daemon=True)
            ros2_worker.start()

        try:
            _publish_state("listening")
            while True:
                try:
                    console.print("[bold]you:[/] ", end="")
                    user = self._input_with_pump("", pump).strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    return
                if not user:
                    continue
                if user.lower() in {"quit", "exit", "bye"}:
                    return
                with dispatch_lock:
                    _dispatch(user, "stdin")
        finally:
            self._stop_ros2_bridge(bridge, ros2_stop, ros2_thread)
            if ros2_worker is not None:
                ros2_worker.join(timeout=1.0)

    def _needs_agent_path(self) -> bool:
        """Always route through :class:`LLMAgent`.

        Previous revisions let tools-only examples fall through to the
        legacy ``LLM.chat_stream`` shim. That bypassed hooks, the event
        bus, ``ctx`` injection, and parallel tool dispatch — one code
        path for skills / deps / workflows, another for plain tools.
        One path is simpler, more testable, and lets downstream
        features (audit logging, memory injection, interrupt
        plumbing) work identically for every example.
        """
        return True

    def _build_event_printer(self, console: Console):
        """Build an on_event handler that colour-codes AgentEvents in
        the rich console (used by voice modes for live feedback)."""

        def on_event(e) -> None:
            if e.kind == "tool_call":
                r = e.payload
                if getattr(r, "ok", False):
                    console.print(f"[dim cyan]↳ {r.name}({r.arguments}) → {r.result}[/]")
                else:
                    console.print(f"[red]↳ {r.name} failed: {getattr(r, 'error', '?')}[/]")
            elif e.kind == "skill_goal":
                console.print(f"[dim yellow]⇒ skill {e.payload['skill']} started[/]")
            elif e.kind == "skill_cancelled":
                console.print(f"[bold red]⇒ skill {e.payload.get('skill', '?')} cancelled[/]")
            elif e.kind == "handoff":
                console.print(f"[dim magenta]→ handoff {e.agent_name} → {e.payload['target']}[/]")
            elif e.kind == "safety_preempt":
                console.print(f"[bold red]⚠ safety preempt: {e.payload}[/]")

        return on_event

    def _run_simple_voice(self, args: argparse.Namespace, console: Console) -> None:
        from edgevox.cli.main import VoiceBot

        console.print(Panel.fit(f"{self.name} — voice mode", border_style="green"))
        _announce_model(console, args.model)
        use_agent_path = self._needs_agent_path() or self.agent is not None
        bot = VoiceBot(
            llm_model=args.model,
            tts_backend=args.tts,
            voice=args.voice,
            language=args.language,
            tools=None if use_agent_path else self.tools,
            on_tool_call=None if use_agent_path else self._on_tool(console),
            agent=self.agent if use_agent_path else None,
            deps=self.deps if use_agent_path else None,
            on_event=self._build_event_printer(console) if use_agent_path else None,
        )
        bot.run()

    def _run_tui(self, args: argparse.Namespace, console: Console) -> None:
        from edgevox.tui import EdgeVoxApp

        console.print(Panel.fit(f"{self.name} — launching full TUI", border_style="green"))
        _announce_model(console, args.model)
        # Agent-aware path: when skills, deps, or a custom agent are
        # wired we route the TUI through ``AgentProcessor`` so the full
        # LLMAgent loop (hooks, ctx injection, handoffs, cancellable
        # skills) runs identically to ``--simple-ui``. The legacy
        # tools-only / no-tools path keeps using ``LLMProcessor`` so
        # the streaming-token chat stays cheap when no agent surface
        # is needed.
        use_agent_path = self._needs_agent_path() or self.agent is not None
        app = EdgeVoxApp(
            llm_model=args.model,
            tts_backend=args.tts,
            voice=args.voice,
            language=args.language,
            tools=None if use_agent_path else self.tools,
            on_tool_call=None if use_agent_path else self._on_tool(console),
            agent=self.agent if use_agent_path else None,
            skills=self.skills if use_agent_path else None,
            deps=self.deps if use_agent_path else None,
            on_event=None,  # the TUI installs its own chat-panel sink
            banner_title=self.name,
        )
        app.run()

    # ----- public entry point -----

    def run(self, argv: list[str] | None = None) -> None:
        """Parse CLI args and dispatch to the selected mode."""
        parser = self._build_parser()
        args = parser.parse_args(argv)

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.WARNING,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        console = Console()

        # Centralised Ctrl+C handler: one SIGINT cleans up sim deps
        # (closes viewer, stops physics thread, releases GL), ROS2
        # bridge, and any policy thread; a second SIGINT hard-exits.
        # This matters most for the MuJoCo viewer — if the process
        # dies without closing the viewer the GLFW window lingers.
        # ``signal.signal`` only works from the main thread; tests and
        # other callers that invoke ``AgentApp.run`` from a worker
        # thread just skip the handler silently.
        _sigint_count = {"n": 0}

        def _hard_shutdown(_sig: int, _frame: object) -> None:
            _sigint_count["n"] += 1
            if _sigint_count["n"] >= 2:
                console.print("[red]force exit.[/]")
                os._exit(130)
            console.print("\n[yellow]stopping... (press Ctrl+C again to force-exit)[/]")
            try:
                close = getattr(self.deps, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            sys.exit(130)

        import threading as _threading_mod

        if _threading_mod.current_thread() is _threading_mod.main_thread():
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGINT, _hard_shutdown)
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTERM, _hard_shutdown)

        if self.pre_run is not None:
            self.pre_run(args)

        try:
            if args.text_mode:
                self._run_text(args, console)
            elif args.simple_ui:
                self._run_simple_voice(args, console)
            else:
                self._run_tui(args, console)
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted.[/]")
            sys.exit(130)
        finally:
            with contextlib.suppress(Exception):
                close = getattr(self.deps, "close", None)
                if callable(close):
                    close()
