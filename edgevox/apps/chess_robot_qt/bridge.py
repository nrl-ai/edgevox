"""Agent <-> Qt bridge.

The EdgeVox ``LLMAgent`` runs blocking LLM + engine calls on a worker
thread; Qt's event loop runs on the main thread. This module owns the
hand-off: a ``RookBridge`` exposes Qt signals that the UI binds to, and
under the hood schedules ``agent.run`` on a :class:`QThreadPool` worker
while translating agent events back into ``Signal.emit`` calls.

All chess-specific hooks live exactly where they did before
(:mod:`edgevox.examples.agents.chess_robot`). We reuse:

- :class:`MoveInterceptHook` — deterministic move application so the
  LLM can't freeze the board with a missed tool call.
- :class:`RichChessAnalyticsHook` — hidden system-role briefing with
  perspective, eval, opening, threats.
- :class:`RobotFaceHook` — emits robot_face events → translated to
  ``face_changed`` Qt signal here.
- :class:`MoveCommentaryHook` — captures last-move outcome for
  downstream reading.
- :class:`ThinkTagStripHook` / :class:`VoiceCleanupHook` /
  :class:`SentenceClipHook` — TTS sanitation before reply reaches the
  chat bubble.

The bridge deliberately does NOT launch an HTTP server. Everything runs
in one Python process: Qt UI ↔ bridge ↔ agent ↔ stockfish subprocess.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from edgevox.agents import AgentContext, LLMAgent
from edgevox.examples.agents.chess_partner import CHESS_TOOLS
from edgevox.examples.agents.chess_robot.face_hook import RobotFaceHook
from edgevox.examples.agents.chess_robot.move_intercept import MoveInterceptHook
from edgevox.examples.agents.chess_robot.rich_board import RichChessAnalyticsHook
from edgevox.examples.agents.chess_robot.sanitize import (
    SentenceClipHook,
    ThinkTagStripHook,
    VoiceCleanupHook,
)
from edgevox.integrations.chess import (
    ChessEnvironment,
    EngineUnavailable,
    MoveCommentaryHook,
    build_engine,
    resolve_persona,
)
from edgevox.llm import LLM
from edgevox.llm.hooks_slm import default_slm_hooks

log = logging.getLogger(__name__)


@dataclass
class RookConfig:
    """User-configurable knobs. Mirrors the old CLI flags."""

    persona: str = "casual"
    user_plays: str = "white"
    engine_kind: str | None = None  # None → persona default (with stockfish fallback)
    stockfish_skill: int | None = None
    maia_weights: str | None = None
    # LLM path/spec. ``None`` → framework default (Gemma-4-E2B). Pass a
    # local ``.gguf`` path or the ``hf:repo:file`` shorthand
    # ``_resolve_model_path`` understands (see
    # :func:`edgevox.llm.llamacpp._resolve_model_path`). We default to a
    # tool-call-capable small model for Rook — Llama-3.2-1B — because
    # MoveInterceptHook handles the chess tools and the LLM only needs
    # natural conversation.
    llm_path: str | None = "hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    @classmethod
    def from_env(cls) -> RookConfig:
        """Pull overrides from the same ``EDGEVOX_CHESS_*`` env vars the
        chess_robot server factory honours, so migrating from the old
        flow to the Qt app doesn't change the config surface."""
        cfg = cls()
        if v := os.environ.get("EDGEVOX_CHESS_PERSONA"):
            cfg.persona = v
        if v := os.environ.get("EDGEVOX_CHESS_ENGINE"):
            cfg.engine_kind = v
        if v := os.environ.get("EDGEVOX_CHESS_USER_PLAYS"):
            cfg.user_plays = v
        if v := os.environ.get("EDGEVOX_CHESS_STOCKFISH_SKILL"):
            cfg.stockfish_skill = int(v)
        if v := os.environ.get("EDGEVOX_CHESS_MAIA_WEIGHTS"):
            cfg.maia_weights = v
        return cfg


class _Signals(QObject):
    """The signal bus surfaced to the UI.

    Qt's signal-slot system requires signals to live on a QObject; the
    bridge itself is a plain class so the worker thread can hold
    references without Qt parenting complications.
    """

    state_changed = Signal(str)  # "thinking" / "speaking" / "listening"
    chess_state_changed = Signal(object)  # ChessState dict
    face_changed = Signal(dict)  # robot_face payload
    reply_finalised = Signal(str)  # rook's final spoken reply
    user_echo = Signal(str)  # what the user just said (text mirror)
    error = Signal(str)  # user-facing error string
    ready = Signal()  # emitted once the LLM has finished loading
    load_progress = Signal(str)  # human-readable step during boot ("downloading …" etc.)


class _TurnJob(QRunnable):
    """Runs one ``agent.run`` call in the QThreadPool."""

    def __init__(self, bridge: RookBridge, text: str) -> None:
        super().__init__()
        self._bridge = bridge
        self._text = text

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._bridge._run_turn_blocking(self._text)


class RookBridge:
    """Orchestrates agent + env + LLM + Qt signals for the app window."""

    def __init__(self, config: RookConfig | None = None) -> None:
        self.config = config or RookConfig.from_env()
        self.signals = _Signals()
        self._env: ChessEnvironment | None = None
        self._agent: LLMAgent | None = None
        self._llm: LLM | None = None
        self._ctx_session = None  # created after agent built
        self._pool = QThreadPool.globalInstance()
        self._busy = threading.Event()
        self._loaded = False

    # ----- lifecycle -----

    def start(self) -> None:
        """Launch a background job to load LLM + engine + build agent.

        We don't do this synchronously because the user's window needs
        to paint before model load (2-10 s) completes.
        """
        job = _LoadJob(self)
        self._pool.start(job)

    def close(self) -> None:
        """Release the chess engine subprocess and LLM handle cleanly.

        Waits up to 3 seconds for any in-flight agent turn to finish so
        we don't leak a llama-cpp thread across process exit. The
        QThreadPool itself is global — we only clear our own pending
        turn job; other users of the pool are untouched.
        """
        # Signal any running turn to stop via the interrupt mechanism —
        # the agent loop polls ``ctx.interrupt`` / ``ctx.stop`` between
        # hops. Then wait briefly.
        self._busy.wait(timeout=3.0)
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                log.exception("env.close() failed")
        # Drop the LLM ref so llama-cpp can release its VRAM.
        self._llm = None
        self._agent = None

    # ----- public ops (called from UI thread) -----

    def submit_text(self, text: str) -> None:
        """Send a user message through the agent. Non-blocking; reply
        arrives via ``reply_finalised`` / ``face_changed``."""
        if not self._loaded:
            self.signals.error.emit("Still loading — give me a moment.")
            return
        if self._busy.is_set():
            self.signals.error.emit("Still thinking — one sec.")
            return
        self._busy.set()
        self.signals.state_changed.emit("thinking")
        self._pool.start(_TurnJob(self, text))

    def reset_game(self) -> None:
        """Start a fresh board. Fires chess_state_changed."""
        if self._env is None:
            return
        self._env.new_game()

    def snapshot(self):
        """Latest ChessState for the UI to render initially."""
        return None if self._env is None else self._env.snapshot()

    @property
    def env(self) -> ChessEnvironment | None:
        return self._env

    # ----- worker-thread bodies -----

    def _run_turn_blocking(self, text: str) -> None:
        assert self._agent is not None
        ctx = AgentContext(
            session=self._ctx_session,
            deps=self._env,
            on_event=self._on_agent_event,
        )
        try:
            self.signals.user_echo.emit(text)
            result = self._agent.run(text, ctx)
            self.signals.reply_finalised.emit((result.reply or "").strip())
        except Exception as e:
            log.exception("agent run failed")
            self.signals.error.emit(f"Agent error: {e}")
        finally:
            self._busy.clear()
            self.signals.state_changed.emit("listening")

    def _on_agent_event(self, event) -> None:
        """Translate AgentEvent → Qt signal. Runs on the worker thread;
        Qt's Direct/Queued connection mechanics deliver to the UI thread."""
        kind = getattr(event, "kind", None)
        payload = getattr(event, "payload", None)
        if kind == "robot_face":
            self.signals.face_changed.emit(dict(payload or {}))

    def _build(self) -> None:
        """Worker-thread: load LLM, build engine + env + agent.

        Emits :attr:`ready` when done so the UI can transition from
        "loading…" to fully interactive.
        """
        try:
            self.signals.load_progress.emit("checking chess engine…")
            # Pre-flight: build the engine first — if stockfish is
            # missing the user gets an immediate, specific error
            # instead of a cryptic LLM-load timeout.
            persona = resolve_persona(self.config.persona)
            engine_kind_preview = self.config.engine_kind or persona.engine_kind
            if engine_kind_preview == "stockfish":
                # build_engine raises EngineUnavailable with a useful
                # message when stockfish isn't on PATH.
                import shutil as _shutil

                if not _shutil.which("stockfish"):
                    raise EngineUnavailable(
                        "stockfish not found on PATH — install with `apt install stockfish` / `brew install stockfish`."
                    )

            self.signals.load_progress.emit("loading language model…")
            log.info("Loading LLM (%s)...", self.config.llm_path or "<default>")
            self._llm = LLM(model_path=self.config.llm_path, language="en")
            self.signals.load_progress.emit("starting chess engine…")
            persona = resolve_persona(self.config.persona)
            engine_kind = self.config.engine_kind or persona.engine_kind

            engine_kwargs: dict[str, Any] = dict(persona.engine_options) if engine_kind == persona.engine_kind else {}
            if self.config.stockfish_skill is not None:
                engine_kwargs["skill"] = self.config.stockfish_skill
            if engine_kind == "maia":
                if not self.config.maia_weights:
                    # Desktop-friendly fallback: if no maia weights,
                    # drop to stockfish at a similar strength.
                    log.warning(
                        "persona %r defaults to maia, no weights — falling back to stockfish",
                        persona.slug,
                    )
                    engine_kind = "stockfish"
                    elo = int(persona.engine_options.get("elo", 1500))
                    engine_kwargs = {"skill": max(0, min(20, round((elo - 800) / 100)))}
                else:
                    engine_kwargs["weights"] = self.config.maia_weights

            engine = build_engine(engine_kind, **engine_kwargs)
            self._env = ChessEnvironment(engine, user_plays=self.config.user_plays)
            self.signals.load_progress.emit("wiring the agent…")

            agent = LLMAgent(
                name=f"Rook — {persona.display_name}",
                description="Voice-controlled chess robot.",
                instructions=_compose_instructions(persona.system_prompt),
                tools=CHESS_TOOLS,
                hooks=[
                    MoveInterceptHook(),
                    MoveCommentaryHook(),
                    RobotFaceHook(persona=persona.slug),
                    RichChessAnalyticsHook(),
                    ThinkTagStripHook(),
                    VoiceCleanupHook(),
                    SentenceClipHook(max_sentences=2),
                    *default_slm_hooks(),
                ],
            )
            agent.bind_llm(self._llm)
            self._agent = agent

            # One Session shared across all turns so the chat history
            # persists (the agent prepends ``ctx.session.messages`` each
            # run). Create it lazily once the agent exists.
            from edgevox.agents import Session

            self._ctx_session = Session()

            # Prime the UI with the starting chess state.
            self.signals.chess_state_changed.emit(self._env.snapshot())
            # Subscribe the env so every move pushes a state update.
            self._env.subscribe(lambda s: self.signals.chess_state_changed.emit(s))

            self._loaded = True
            self.signals.ready.emit()
            log.info("Rook ready.")
        except EngineUnavailable as e:
            self.signals.error.emit(f"Chess engine unavailable: {e}. Install stockfish and restart.")
        except Exception as e:
            log.exception("Rook bridge load failed")
            self.signals.error.emit(f"Startup failed: {e}")


class _LoadJob(QRunnable):
    def __init__(self, bridge: RookBridge) -> None:
        super().__init__()
        self._bridge = bridge

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._bridge._build()


_ROOK_TOOL_GUIDANCE = """\
/no_think
You are Rook, a chess robot playing against a human. You speak with your persona's voice — see the persona block below. Think of yourself as sitting across a table from a friend, not writing a chess report.

Before every turn the system shows you a [CHESS BRIEFING ...] block with the current position, your side, the user's side, the evaluation, and the top engine lines. Read it quietly. Do NOT recite it — use it as background knowledge while you speak naturally.

Speaking rules:
- Talk like a person, not a search engine. Short, natural, one or two sentences. Contractions are good.
- Your replies are spoken by a TTS engine later: no markdown, no asterisks, no bullets, no emoji, no <think> tags, no lists.
- Only reference moves that actually happened (the last move by the user and your reply). Never invent or speculate.
- Vary your phrasing. If you catch yourself starting several turns the same way ("X is a bold move…"), change it up.
- Don't explain chess theory unless the user explicitly asks for analysis."""


def _compose_instructions(persona_prompt: str) -> str:
    return f"{_ROOK_TOOL_GUIDANCE}\n\n---\n\n{persona_prompt}"


__all__ = ["RookBridge", "RookConfig"]
