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

import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QStandardPaths, QThreadPool, Signal, Slot

from edgevox.agents import AgentContext, LLMAgent
from edgevox.agents.hooks_builtin import (
    ContextCompactionHook,
    DebugTapHook,
    MemoryInjectionHook,
    NotesInjectorHook,
    PersistSessionHook,
    TokenBudgetHook,
)
from edgevox.agents.interrupt import InterruptController
from edgevox.agents.memory import Compactor, NotesFile, SQLiteMemoryStore
from edgevox.examples.agents.chess_robot.commentary_gate import CommentaryGateHook
from edgevox.examples.agents.chess_robot.face_hook import RobotFaceHook
from edgevox.examples.agents.chess_robot.move_intercept import MoveInterceptHook
from edgevox.examples.agents.chess_robot.rich_board import RichChessAnalyticsHook
from edgevox.examples.agents.chess_robot.sanitize import (
    BriefingLeakGuard,
    SentenceClipHook,
    SilenceSentinelHook,
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

_SESSION_ID = "rook-default"


def _data_dir() -> Path:
    """Platform-appropriate per-user data dir for RookApp.

    Uses Qt's ``QStandardPaths`` so we land in the right spot on every
    platform. Falls back to ``~/.rookapp`` if QStandardPaths returns an
    empty string (headless CI builds sometimes do).
    """
    # AppDataLocation already appends the org + app names we set in main().
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    path = Path(base) if base else Path.home() / ".rookapp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _try_session_store(path: Path):
    """Best-effort JSON session store. Older EdgeVox builds may not
    expose :class:`JSONSessionStore`; in that case we skip persistence
    rather than crash startup."""
    try:
        from edgevox.agents.memory import JSONSessionStore

        return JSONSessionStore(path)
    except Exception:
        log.debug("JSONSessionStore unavailable; sessions won't persist", exc_info=True)
        return None


def _migrate_legacy_memory(data_dir: Path, store: SQLiteMemoryStore) -> None:
    """Move a pre-SQLite ``memory.json`` into the new store.

    Older RookApp installs wrote memory to ``memory.json``; the SQLite
    migration happens transparently on first open of the new store:
    facts / preferences / episodes are replayed through the Protocol,
    then the legacy file is renamed to ``memory.json.migrated`` so a
    second launch doesn't re-import duplicates. Idempotent and safe to
    call on fresh installs (no-op when there's no legacy file).
    """
    legacy = data_dir / "memory.json"
    if not legacy.exists():
        return
    try:
        from edgevox.agents.memory import JSONMemoryStore as _Legacy

        old = _Legacy(legacy)
        for f in old.facts():
            # Skip facts the SQLite store already knows about; the
            # legacy file may have been migrated partially on a prior
            # crashed run.
            if store.get_fact(f.key, scope=f.scope) is not None:
                continue
            store.add_fact(f.key, f.value, scope=f.scope, source=f.source)
        for p in old.preferences():
            store.set_preference(p.key, p.value)
        for e in old.recent_episodes(n=500):
            store.add_episode(e.kind, e.payload, e.outcome, agent=e.agent)
        legacy.rename(legacy.with_suffix(".json.migrated"))
        log.info("migrated legacy memory.json to SQLite")
    except Exception:
        log.exception("legacy memory migration failed (non-fatal)")


@dataclass
class RookConfig:
    """User-configurable knobs. Mirrors the old CLI flags."""

    persona: str = "casual"
    user_plays: str = "white"
    engine_kind: str | None = None  # None → persona default (with stockfish fallback)
    stockfish_skill: int | None = None
    maia_weights: str | None = None
    # LLM path/spec. ``None`` → framework default (Gemma-4-E2B). Pass a
    # local ``.gguf`` path, the ``hf:repo:file`` shorthand, or a preset
    # slug that ``_resolve_model_path`` understands (see
    # :func:`edgevox.llm.llamacpp._resolve_model_path` /
    # :mod:`edgevox.llm.models`). Rook's LLM only has to *talk*
    # naturally (MoveInterceptHook handles the chess tools).
    #
    # Default ``gemma-4-e2b`` — picked by the LLM eval harness
    # (``scripts/eval_llm_commentary.py``). At the three sizes the
    # Settings dialog exposes, Gemma 4 E2B gave the cleanest, shortest,
    # most in-persona replies with correct pronouns. Llama 3.2 1B
    # fabricated piece identities; Qwen 3 1.7B leaked ``<think>`` tags
    # and had tone mismatches. Gemma's extra ~1 GB vs Llama 1B is
    # worth the quality jump. Window overrides this at launch from the
    # persisted ``Settings.llm_model``, so users can swap models
    # without editing this file.
    llm_path: str | None = "gemma-4-e2b"

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
    # Debug stream — only emits when ``debug_mode`` is on.
    # ``title`` is a short tag ("SYSTEM PROMPT", "USER", "RAW REPLY", …);
    # ``body`` is the full text to dump into a monospace chat bubble.
    debug_event = Signal(str, str)
    # Analytics event — a structured per-turn summary (piece / square /
    # capture / eval). Rendered as a system-info bubble in the chat
    # regardless of whether Rook spoke this turn. ``headline`` is a
    # one-line summary; ``body`` is a multi-line detail block.
    analytics_event = Signal(str, str)


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
        # One controller per bridge instance. Reused across turns —
        # ``.reset()`` at the top of each turn so a stale barge-in flag
        # can't poison the next one. ``cancel_llm=True`` is the default,
        # which plumbs the cancel-token into ``LLM.complete(stop_event=…)``
        # so llama-cpp actually halts mid-decode on barge-in.
        self._interrupt = InterruptController()
        self._active_ctx: AgentContext | None = None
        # Paths retained so ``clear_memory`` can unlink backing files
        # for stores with no explicit ``.clear()`` method.
        self._data_dir: Path | None = None
        self._memory: SQLiteMemoryStore | None = None
        self._notes: NotesFile | None = None
        self._sessions = None
        self._game_path: Path | None = None
        # Debug-mode flag. Read per fire point by the installed
        # :class:`DebugTapHook` via a predicate closure, flipped live
        # via :meth:`set_debug_mode`. The hook is always installed; the
        # flag controls whether it emits anything. Keeps the cost off
        # the hot path when debugging is off without requiring an agent
        # rebuild.
        self._debug_mode = False

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
        # Reset the controller per turn so a stale barge-in from a
        # previous reply doesn't cancel the new one before it starts.
        self._interrupt.reset()
        ctx = AgentContext(
            session=self._ctx_session,
            deps=self._env,
            on_event=self._on_agent_event,
            interrupt=self._interrupt,
        )
        self._active_ctx = ctx
        try:
            self.signals.user_echo.emit(text)
            result = self._agent.run(text, ctx)
            reply = (result.reply or "").strip()
            # Silently drop the reply when the turn was cancelled by a
            # barge-in — the user isn't waiting for it.
            if reply and not self._interrupt.should_stop():
                self.signals.reply_finalised.emit(reply)
        except Exception as e:
            log.exception("agent run failed")
            self.signals.error.emit(f"Agent error: {e}")
        finally:
            # Re-broadcast the final env state so the UI has a
            # guaranteed opportunity to converge on the authoritative
            # board even if any in-turn ``chess_state_changed``
            # emissions were dropped (cross-thread signal queuing can
            # drop an event if the sender is deleted mid-queue, and
            # we've seen user reports of "my move didn't appear on the
            # board"). Cheap — the UI's ``set_state`` is a no-op when
            # the FEN hasn't changed since last apply.
            if self._env is not None:
                try:
                    self.signals.chess_state_changed.emit(self._env.snapshot())
                except Exception:
                    log.exception("post-turn chess_state_changed emit failed (non-fatal)")
            self._active_ctx = None
            self._busy.clear()
            self.signals.state_changed.emit("listening")

    # ----- live persona swap -----

    def set_persona(self, slug: str) -> None:
        """Apply a persona swap live so the next reply uses the new
        voice and the face animation switches to the matching persona.

        What this does *not* swap: the chess engine kind / strength.
        That requires a fresh ``ChessEnvironment`` (and on Maia, model
        weights), which would clobber the current game. Engine strength
        therefore picks up the new persona on the next ``new game`` /
        app restart — a documented limitation.
        """
        if not slug or slug == self.config.persona:
            return
        try:
            persona = resolve_persona(slug)
        except ValueError:
            return
        self.config.persona = slug
        if self._agent is not None:
            self._agent.instructions = _compose_instructions(persona.system_prompt)
            for h in self._agent.hooks:
                if isinstance(h, RobotFaceHook):
                    h.persona = slug
                    break
        # Push a face_changed event so the UI refreshes the face widget
        # and persona label even before the next agent reply fires.
        self.signals.face_changed.emit({"mood": "calm", "tempo": "idle", "persona": slug})

    # ----- session replay -----

    def session_messages(self) -> list[tuple[str, str]]:
        """Return the restored chat transcript as ``(role, text)`` pairs
        for the UI to replay on launch.

        Filters to the two roles the chat widget renders — ``"user"`` and
        ``"assistant"``. System messages (the persona prompt + the
        runtime-injected chess briefing) are context the agent needs but
        not speech the user ever said or heard, so they're skipped.
        Empty-content messages are dropped too; those are silent-reply
        turns, and replaying them as blank bubbles would be misleading.
        """
        if self._ctx_session is None:
            return []
        out: list[tuple[str, str]] = []
        for msg in self._ctx_session.messages:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            out.append((role, content))
        return out

    # ----- debug -----

    def set_debug_mode(self, on: bool) -> None:
        """Toggle inline-in-chat logging of LLM input + raw reply.

        Flips a flag read by the installed :class:`DebugTapHook` on
        every fire point — no agent rebuild, no restart. Safe to call
        before the bridge has finished loading (the hook reads this
        attribute at call time, not init).
        """
        self._debug_mode = bool(on)

    # ----- barge-in / memory management -----

    def cancel_turn(self, *, reason: str = "user_barge_in") -> None:
        """Barge-in entry point.

        Triggers the interrupt controller (sets ``cancel_token`` → llama-cpp
        halts mid-decode) and flips ``ctx.stop`` so the agent loop exits
        between hops. Safe to call when idle — no-ops.
        """
        self._interrupt.trigger(reason)
        ctx = self._active_ctx
        if ctx is not None:
            ctx.stop.set()

    def clear_memory(self) -> None:
        """Wipe per-session memory, notes, chat history, and persisted
        session on disk. Called on ``new game`` so stale commentary
        ("nice Nf3!") can't leak into a fresh board.

        Idempotent: safe to call before the bridge finishes loading.
        """
        # Reset in-memory chat history first — agent sees an empty
        # session on the next turn even if the persistence path fails.
        if self._ctx_session is not None:
            try:
                self._ctx_session.reset()
            except Exception:
                log.exception("session.reset failed")
        # Wipe the notes file (unlinks the backing markdown).
        if self._notes is not None:
            try:
                self._notes.clear()
            except Exception:
                log.exception("notes.clear failed")
        # SQLiteMemoryStore has no public clear() — close the handle,
        # unlink the DB file (and its WAL / SHM sidecars), rebuild an
        # empty instance in place. WAL artefacts are safe to delete
        # after close().
        if self._memory is not None and self._data_dir is not None:
            try:
                self._memory.close()
                for suffix in ("", "-wal", "-shm"):
                    p = self._data_dir / f"memory.db{suffix}"
                    if p.exists():
                        p.unlink()
                self._memory = SQLiteMemoryStore(self._data_dir / "memory.db")
            except Exception:
                log.exception("memory wipe failed")
        # Drop the persisted session so the next launch starts clean
        # instead of restoring the previous game's chat.
        if self._sessions is not None:
            try:
                self._sessions.delete(_SESSION_ID)
            except Exception:
                log.exception("sessions.delete failed")
        # Board + chat are persisted as one unit — wipe the game file
        # too so next launch doesn't resurrect the match we just
        # abandoned.
        if self._game_path is not None and self._game_path.exists():
            try:
                self._game_path.unlink()
            except OSError:
                log.exception("game.json unlink failed")
        # Rewire hooks that hold their own store references so they
        # observe the newly-empty stores on the next turn.
        self._rebind_memory_hooks()

    def _try_restore_game(self) -> bool:
        """Read ``game.json`` and rehydrate the board. Returns True if a
        live (non-finished) game was restored, False otherwise. Falls
        back silently on any parse/FEN error — a corrupt save file
        shouldn't keep the app from starting."""
        if self._env is None or self._game_path is None or not self._game_path.exists():
            return False
        try:
            data = json.loads(self._game_path.read_text())
        except Exception:
            log.exception("game.json unreadable; ignoring")
            return False
        if data.get("is_game_over"):
            # Saved game is finished — drop the file so we don't re-check
            # it on every launch.
            with contextlib.suppress(OSError):
                self._game_path.unlink()
            return False
        fen = data.get("fen")
        if not fen:
            return False
        try:
            self._env.restore(
                fen=fen,
                san_history=data.get("san_history") or [],
                last_move_uci=data.get("last_move_uci"),
                user_plays=data.get("user_plays"),
            )
        except ValueError:
            log.exception("saved FEN rejected; starting fresh")
            return False
        # Keep config in sync so the UI orientation matches the restored
        # side-to-move preference.
        if saved_side := data.get("user_plays"):
            self.config.user_plays = saved_side
        log.info("restored saved game at ply %s", data.get("ply"))
        return True

    def _persist_game_state(self, state) -> None:
        """Auto-save listener — writes ``game.json`` on every state
        change, or deletes it when the game ends so next launch starts
        fresh."""
        if self._game_path is None:
            return
        try:
            if state.is_game_over:
                if self._game_path.exists():
                    self._game_path.unlink()
                return
            payload = {
                "fen": state.fen,
                "ply": state.ply,
                "turn": state.turn,
                "san_history": list(state.san_history),
                "last_move_uci": state.last_move_uci,
                "user_plays": self.config.user_plays,
                "is_game_over": False,
            }
            self._game_path.write_text(json.dumps(payload))
        except Exception:
            log.exception("game.json write failed (non-fatal)")

    def _purge_coupled_persistence(self) -> None:
        """No live game to restore → also drop any stale chat session +
        memory on disk so rook doesn't reference a match that isn't
        there. Board state is kept in lockstep with the chat context."""
        if self._data_dir is None:
            return
        try:
            if self._memory is not None:
                self._memory.close()
            for suffix in ("", "-wal", "-shm"):
                p = self._data_dir / f"memory.db{suffix}"
                if p.exists():
                    p.unlink()
            self._memory = SQLiteMemoryStore(self._data_dir / "memory.db")
        except Exception:
            log.exception("stale memory wipe failed")
        if self._sessions is not None:
            try:
                self._sessions.delete(_SESSION_ID)
            except Exception:
                log.exception("stale session wipe failed")

    def _rebind_memory_hooks(self) -> None:
        """After ``clear_memory`` swaps the SQLiteMemoryStore, walk the
        agent's hook list and point MemoryInjectionHook at the new
        store so the next turn's context renders the empty memory.
        Hooks that take a store by reference otherwise keep the old,
        stale handle."""
        if self._agent is None:
            return
        for hook in getattr(self._agent, "hooks", []) or []:
            if isinstance(hook, MemoryInjectionHook):
                hook.memory_store = self._memory

    def _on_agent_event(self, event) -> None:
        """Translate AgentEvent → Qt signal. Runs on the worker thread;
        Qt's Direct/Queued connection mechanics deliver to the UI thread."""
        kind = getattr(event, "kind", None)
        payload = getattr(event, "payload", None)
        if kind == "robot_face":
            self.signals.face_changed.emit(dict(payload or {}))
        elif kind == "rook_debug":
            self._emit_debug(payload or {})
        elif kind == "move_analytics":
            self._emit_analytics(payload or {})

    def _emit_analytics(self, payload: dict) -> None:
        """Render a ``move_analytics`` payload as a (headline, body)
        pair for the chat's system-info bubble.

        Gated behind ``debug_mode`` — the structured YOU / ROOK /
        engine-eval breakdown is useful for developers debugging the
        commentary pipeline but clutters the regular chat, so it only
        renders when the user has Debug mode switched on in Settings.
        """
        if not self._debug_mode:
            return
        user_desc = payload.get("user_desc")
        engine_desc = payload.get("engine_desc")
        # User-facing variant — "you" means the user reading the chat,
        # not the LLM. Falls back to the Rook-POV line if the user-
        # facing one is somehow missing (back-compat with older payload
        # shapes in persisted sessions).
        score_line = payload.get("score_line_user") or payload.get("score_line")
        classification = payload.get("classification")

        headline_bits: list[str] = []
        if user_desc:
            headline_bits.append(f"you: {user_desc}")
        if engine_desc:
            headline_bits.append(f"rook: {engine_desc}")
        headline = " · ".join(headline_bits) if headline_bits else "analytics"

        body_lines: list[str] = []
        if user_desc:
            body_lines.append(f"YOU — {user_desc}")
        if engine_desc:
            body_lines.append(f"ROOK — {engine_desc}")
        if classification and classification not in ("best", "good"):
            body_lines.append(f"classification: {classification}")
        if score_line:
            body_lines.append(score_line)
        body = "\n".join(body_lines)
        self.signals.analytics_event.emit(headline, body)

    def _emit_debug(self, payload: dict) -> None:
        """Render a ``DebugTapHook`` payload as a (title, body) pair
        for the chat. Gated by ``debug_mode`` so stale events from a
        lingering worker don't spam the UI after the user toggles it
        off."""
        if not self._debug_mode:
            return
        dkind = payload.get("kind")
        if dkind == "messages":
            messages = payload.get("messages") or []
            body = _format_messages(messages)
            title = f"LLM INPUT · {len(messages)} msg"
        elif dkind == "raw_reply":
            body = str(payload.get("text", ""))
            title = "RAW REPLY (pre-sanitize)"
        else:
            return
        self.signals.debug_event.emit(title, body)

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

            # Persistent memory + notes directory — one per OS user.
            # QStandardPaths gives us the right spot on every platform
            # (~/.local/share/EdgeVox/RookApp on Linux,
            # ~/Library/Application Support/EdgeVox/RookApp on macOS,
            # %APPDATA%/EdgeVox/RookApp on Windows).
            data_dir = _data_dir()
            self._data_dir = data_dir
            # SQLiteMemoryStore is crash-safe (WAL-mode atomic writes)
            # and multi-process-safe, so a kill-9 mid-turn can't lose
            # the fact rook just learned. A legacy ``memory.json`` from
            # older installs is migrated lazily on first open.
            self._memory = SQLiteMemoryStore(data_dir / "memory.db")
            _migrate_legacy_memory(data_dir, self._memory)
            self._notes = NotesFile(data_dir / "notes.md")
            self._sessions = _try_session_store(data_dir / "sessions.json")
            self._game_path = data_dir / "game.json"
            log.info("Rook memory dir: %s", data_dir)

            # Rehydrate the board before the agent is built so the first
            # briefing sees the restored position. Board and chat session
            # are treated as one unit: if the saved game is gone or
            # finished, we also wipe the stale session + memory so rook
            # never chats about a game that isn't on the board.
            game_restored = self._try_restore_game()
            if not game_restored:
                self._purge_coupled_persistence()

            # No ``tools=`` here on purpose. ``MoveInterceptHook``
            # applies moves deterministically before the LLM runs, and
            # ``RichChessAnalyticsHook`` hands the position over as a
            # system-role briefing. The LLM's job is to *talk*, not to
            # call tools — exposing tool schemas to a 1B model just
            # tempts it to regurgitate the JSON schema as its reply
            # (seen in the Llama-3.2-1B output). If a stronger model is
            # bound later we can reintroduce CHESS_TOOLS behind a flag.
            hooks: list[Any] = [
                MoveInterceptHook(),
                # Must run after MoveInterceptHook (priority 90) so the
                # gate sees the post-move snapshot. Ends the turn
                # silently for quiet moves — the LLM never runs, no
                # reply bubble, zero chatter. For notable moves it
                # stashes a verified-facts directive that
                # RichChessAnalyticsHook turns into a GROUND TRUTH line.
                CommentaryGateHook(),
                MoveCommentaryHook(),
                RobotFaceHook(persona=persona.slug),
                RichChessAnalyticsHook(),
                MemoryInjectionHook(memory_store=self._memory),
                NotesInjectorHook(notes=self._notes),
                ContextCompactionHook(compactor=Compactor()),
                # Silence sentinel runs first among the AFTER_LLM
                # sanitizers (priority 80) so empty-reply fallbacks in
                # the other hooks don't resurrect a reply Rook chose to
                # skip.
                SilenceSentinelHook(),
                ThinkTagStripHook(),
                BriefingLeakGuard(),
                VoiceCleanupHook(),
                # Clip runaway template loops but leave room for a
                # natural, complete reply. Earlier `max_sentences=2`
                # was chopping the second half off normal answers.
                SentenceClipHook(max_sentences=6),
                TokenBudgetHook(max_context_tokens=3500, keep_last=6),
                *default_slm_hooks(),
                # Always installed, no-ops unless ``self._debug_mode`` is
                # on. Kept last so it observes the fully-composed payload
                # after every other hook has mutated it. Uses
                # ``event_kind="rook_debug"`` so the existing chat-bubble
                # renderer picks the events up unchanged.
                DebugTapHook(
                    enabled=lambda: self._debug_mode,
                    event_kind="rook_debug",
                    source="rook",
                ),
            ]
            if self._sessions is not None:
                hooks.append(PersistSessionHook(session_store=self._sessions, session_id=_SESSION_ID))
            agent = LLMAgent(
                name=f"Rook — {persona.display_name}",
                description="Voice-controlled chess robot.",
                instructions=_compose_instructions(persona.system_prompt),
                hooks=hooks,
            )
            agent.bind_llm(self._llm)
            self._agent = agent

            # Resume last session only if the board was restored — keeps
            # board + chat in lockstep.
            from edgevox.agents import Session

            restored = None
            if game_restored and self._sessions is not None:
                try:
                    restored = self._sessions.load(_SESSION_ID)
                except Exception:
                    log.exception("failed to restore session")
            self._ctx_session = restored or Session()

            # Prime the UI with the starting (possibly restored) state.
            self.signals.chess_state_changed.emit(self._env.snapshot())
            # Subscribe two listeners: the UI signal forwarder, and an
            # auto-save so every applied move persists the board. Saves
            # are best-effort — an I/O hiccup shouldn't block gameplay.
            self._env.subscribe(lambda s: self.signals.chess_state_changed.emit(s))
            self._env.subscribe(self._persist_game_state)

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
I am Rook, a chess robot playing against a human. My persona — see the block below — is the whole point: the user is here for MY voice, not a chess report. I tease, gloat, sigh, joke, trash-talk, sound impressed — whatever my persona does, I do. A flat factual summary is worse than silence.

PRONOUN DISCIPLINE — the single hardest rule to follow. I always refer to myself in the first person: "I played", "my knight", "I'm winning", "I missed it". I use "you" and "your" EXCLUSIVELY when speaking TO the user about THEIR moves or THEIR pieces. I never paste my own move onto the user ("you captured with the pawn" when I did). If I catch myself starting a sentence with "You're" while describing something I just did, I am wrong and must rewrite.

CRITICAL — the user's message will often say "I just played X. You reply with Y." In THAT message, "I" is the user talking to me about THEIR move X, and "you" is me. When I write my reply I switch to MY perspective: my move Y becomes "I played Y" / "my Y"; the user's X becomes "your X" / "you played X". I never restate the user's "I played X" as if I had played X.

When the briefing has a GROUND TRUTH section, its bullets are the event I'm reacting to. Everything else in the briefing is background context.

Speaking rules:
- Lead with personality. React emotionally, not clinically. One short sentence is usually plenty.
- Spoken-English only: no markdown, no asterisks, no bullets, no emoji, no <think> tags, no lists. Contractions welcome.
- Stay grounded. I don't invent tactics the briefing didn't declare — no made-up pins, forks, or specific attacks on pieces. Vague reactions ("hmm", "tough one", "well played") are fine; hallucinated specifics are not.
- I don't recite the briefing or quote SAN notation. The user already sees the moves.
- Vary my phrasing between turns.

If the moment really doesn't call for a reaction — or I genuinely have nothing in character to add — I reply with exactly `<silent>` and nothing else."""


def _compose_instructions(persona_prompt: str) -> str:
    return f"{_ROOK_TOOL_GUIDANCE}\n\n---\n\n{persona_prompt}"


def _format_messages(messages: list[dict]) -> str:
    """Flatten the LLM messages array into a single readable block for
    the debug bubble. Preserves role order and full content — the chat
    widget itself truncates if the body gets absurdly long."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?")).upper()
        content = msg.get("content")
        if content is None:
            # Tool / function messages sometimes store the body under
            # ``tool_calls`` or similar. Show the raw msg dict so we
            # can see everything without blessing one schema.
            extras = {k: v for k, v in msg.items() if k != "role"}
            content = repr(extras) if extras else ""
        lines.append(f"── {role} ──\n{content}")
    return "\n\n".join(lines) if lines else "(empty)"


__all__ = ["RookBridge", "RookConfig"]
