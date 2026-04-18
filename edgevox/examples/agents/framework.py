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
        for args, kwargs in self.extra_args:
            parser.add_argument(*args, **kwargs)
        return parser

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

        registered_tools: list[str] = []
        if isinstance(self.agent, LLMAgent):
            registered_tools = [t.name for t in self.agent.tools]
        console.print(f"[dim]agent: {self.agent.name} · tools/skills: {', '.join(registered_tools) or '(none)'}[/]")

        if self.greeting:
            console.print(f"[bold green]{self.agent.name}:[/] {self.greeting}")

        stop_words = {w.lower() for w in (self.stop_words or ("stop", "halt", "freeze", "abort", "emergency"))}

        pump = getattr(self.deps, "pump_events", None)
        session = Session()
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

            # Hard-coded stop-word preempt BEFORE the LLM sees the text.
            tokens = {t.strip(".,!?").lower() for t in user.split()}
            if tokens & stop_words:
                stop_event.set()
                console.print(f"[bold red]⚠ safety preempt: stop-word in {user!r}[/]")
                console.print(f"[bold green]{self.agent.name}:[/] Stopped.")
                # Reset for next turn
                stop_event = threading.Event()
                continue

            ctx = AgentContext(
                session=session,
                deps=self.deps,
                on_event=on_event,
                stop=stop_event,
            )
            result = self.agent.run(user, ctx)
            console.print(f"[bold green]{self.agent.name}:[/] {result.reply}")
            if result.preempted:
                stop_event = threading.Event()

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
