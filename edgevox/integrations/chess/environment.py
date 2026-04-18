"""ChessEnvironment — owns the board + engine, sits under ``ctx.deps``.

Implements the :class:`~edgevox.agents.sim.SimEnvironment` protocol so
the harness treats it identically to a robot sim. Agent tools call
methods on this object to read or mutate board state; the environment
emits a typed :class:`ChessState` snapshot after every mutation so UI
sinks (Textual widget, WebSocket forwarder) can redraw without
peeking into private fields.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import chess

from edgevox.agents.skills import GoalHandle
from edgevox.integrations.chess.analytics import (
    MoveClassification,
    classify_move,
    opening_name,
    win_probability,
)
from edgevox.integrations.chess.engine import ChessEngine, EngineMove

log = logging.getLogger(__name__)

StateListener = Callable[["ChessState"], None]


@dataclass
class ChessState:
    """Serialisable snapshot — what the UI and event bus consume.

    The fields are exactly what the React ``ChessBoard`` / ``EvalBar`` /
    ``MoveList`` components render from, plus enough structured state
    that the LLM prompt injection can reconstitute a compact summary.
    """

    fen: str
    ply: int
    turn: str  # "white" | "black"
    last_move_uci: str | None = None
    last_move_san: str | None = None
    last_move_classification: MoveClassification | None = None
    san_history: list[str] = field(default_factory=list)
    eval_cp: int | None = None
    mate_in: int | None = None
    win_prob_white: float = 0.5
    opening: str | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        if self.last_move_classification is not None:
            out["last_move_classification"] = self.last_move_classification.value
        return out


class ChessEnvironment:
    """Chess-game deps object plugged into ``ctx.deps``.

    Responsibilities:

    - own the canonical :class:`chess.Board` across turns,
    - apply user moves (with UCI or SAN strings) and engine moves,
    - wrap the engine call so agents never see python-chess types
      directly,
    - compute an :class:`EngineMove` for the current position on
      request (for analytics / eval bar),
    - publish a :class:`ChessState` snapshot to any registered
      listeners after every mutation.

    Thread safety: every mutation takes ``self._lock`` so parallel
    tool dispatch (LLM emitting multiple calls in one batch) can't
    interleave half-applied moves.
    """

    # SimEnvironment protocol attributes are implemented below by
    # reset / step / get_world_state / apply_action / render.

    def __init__(
        self,
        engine: ChessEngine,
        *,
        user_plays: str = "white",
        engine_skill: int | None = None,
        analyse_depth: int = 12,
        analyse_time: float = 0.5,
    ) -> None:
        self._engine = engine
        self._user_plays = chess.WHITE if user_plays.lower().startswith("w") else chess.BLACK
        self._engine_skill = engine_skill
        self._analyse_depth = analyse_depth
        self._analyse_time = analyse_time
        self._board = chess.Board()
        self._san_history: list[str] = []
        self._last_move: chess.Move | None = None
        self._last_classification: MoveClassification | None = None
        self._last_eval: EngineMove | None = None
        self._lock = threading.RLock()
        self._listeners: list[StateListener] = []

    # ----- public plumbing -----

    @property
    def engine(self) -> ChessEngine:
        return self._engine

    @property
    def user_plays(self) -> str:
        return "white" if self._user_plays == chess.WHITE else "black"

    @property
    def engine_plays(self) -> str:
        return "black" if self._user_plays == chess.WHITE else "white"

    def subscribe(self, listener: StateListener) -> Callable[[], None]:
        """Register a listener that receives every :class:`ChessState` update.

        Used by the TUI widget and the WebSocket forwarder. Listeners
        fire under ``self._lock`` in registration order; raise at your
        own risk — exceptions are logged and do not break the mutation.

        Returns an ``unsubscribe()`` function. Calling it removes the
        listener; calling it a second time is a no-op. Existing callers
        that don't capture the return value are unaffected — the
        returned callable is simply discarded. The server's per-session
        forwarder uses this to avoid stacking subscribers across turns.
        """
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            # ``remove`` raises ``ValueError`` if the listener was
            # already removed; be idempotent.
            with self._lock, contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return _unsubscribe

    def snapshot(self) -> ChessState:
        """Return the current :class:`ChessState` without mutating anything."""
        with self._lock:
            return self._snapshot_unlocked()

    # ----- SimEnvironment protocol -----

    def reset(self) -> None:
        """Start a fresh game with the same engine and side-to-move."""
        with self._lock:
            self._board = chess.Board()
            self._san_history.clear()
            self._last_move = None
            self._last_classification = None
            self._last_eval = None
        self._publish()

    def step(self, dt: float) -> None:  # pragma: no cover — event-driven
        del dt

    def get_world_state(self) -> dict[str, Any]:
        """Return the snapshot as a plain dict — used by the sim protocol."""
        return self.snapshot().to_json()

    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle:
        """Dispatch an action to the matching ``_action_*`` handler.

        Mirrors :class:`ToyWorld` / :class:`MujocoArmEnvironment` so any
        existing sim tooling that routes through ``apply_action`` keeps
        working. Agents in this example call the typed methods
        (:meth:`play_user_move`, :meth:`engine_move`, …) directly
        because those are clearer in the tool bodies.
        """
        handle = GoalHandle()
        dispatcher = getattr(self, f"_action_{action}", None)
        if dispatcher is None:
            handle.fail(f"unknown chess action {action!r}")
            return handle
        try:
            result = dispatcher(**kwargs)
            handle.succeed(result)
        except Exception as e:
            handle.fail(f"{type(e).__name__}: {e}")
        return handle

    def render(self) -> None:  # pragma: no cover — TUI owns rendering
        pass

    # ----- high-level API (what the agent tools use) -----

    def new_game(self, *, user_plays: str | None = None, engine_skill: int | None = None) -> ChessState:
        """Reset and optionally reconfigure the engine for a new game."""
        with self._lock:
            if user_plays is not None:
                self._user_plays = chess.WHITE if user_plays.lower().startswith("w") else chess.BLACK
            if engine_skill is not None:
                self._engine_skill = int(engine_skill)
                # Re-configure Stockfish skill on the fly if supported.
                configure = getattr(self._engine, "configure", None)
                if callable(configure):
                    try:
                        configure({"Skill Level": max(0, min(20, int(engine_skill)))})
                    except Exception:
                        log.debug("Engine ignored Skill Level reconfig", exc_info=True)
        self.reset()
        return self.snapshot()

    def list_legal_moves(self) -> list[str]:
        """Return the side-to-move's legal moves as UCI strings."""
        with self._lock:
            return [m.uci() for m in self._board.legal_moves]

    def play_user_move(self, move: str) -> ChessState:
        """Apply a user move given in UCI (``e2e4``) or SAN (``e4``).

        Raises :class:`ValueError` on illegal or unparseable moves so
        the tool wrapper can surface a clean error to the LLM. Applies
        a move-classification based on how much it loses relative to
        the engine's top choice before the move.
        """
        with self._lock:
            if self._board.is_game_over():
                raise ValueError("game is already over — call new_game first")
            if self._board.turn != self._user_plays:
                raise ValueError("it's the engine's turn — call engine_move instead")
            baseline = self._safe_analyse_unlocked()
            parsed = self._parse_move_unlocked(move)
            self._apply_move_unlocked(parsed, baseline)
        self._publish()
        return self.snapshot()

    def engine_move(self) -> tuple[ChessState, EngineMove]:
        """Ask the engine for its move, apply it, return (state, move).

        The engine call happens with the environment lock *released* —
        engine subprocesses can take hundreds of milliseconds and we
        don't want to block a concurrent ``snapshot()`` call from a UI
        thread that long.
        """
        with self._lock:
            if self._board.is_game_over():
                raise ValueError("game is over")
            if self._board.turn == self._user_plays:
                raise ValueError("it's the user's turn — call play_user_move first")
            baseline = self._safe_analyse_unlocked()
            board_snapshot = self._board.copy(stack=False)
        engine_move = self._engine.bestmove(board_snapshot, time_limit=self._analyse_time)
        with self._lock:
            parsed = chess.Move.from_uci(engine_move.uci)
            self._apply_move_unlocked(parsed, baseline, engine_move=engine_move)
        self._publish()
        return self.snapshot(), engine_move

    def analyse(self, depth: int | None = None) -> EngineMove:
        """Return the engine's static eval + best continuation for the
        current position. Does not mutate state."""
        effective = depth if depth is not None else self._analyse_depth
        with self._lock:
            board_copy = self._board.copy(stack=False)
        return self._engine.analyse(board_copy, depth=effective)

    def restore(
        self,
        *,
        fen: str,
        san_history: list[str] | None = None,
        last_move_uci: str | None = None,
        user_plays: str | None = None,
    ) -> ChessState:
        """Rehydrate a saved position. No engine analysis is run — eval
        fields stay ``None`` until the next real move refreshes them.
        Raises :class:`ValueError` if ``fen`` is malformed; caller should
        fall back to a fresh board.
        """
        with self._lock:
            if user_plays is not None:
                self._user_plays = chess.WHITE if user_plays.lower().startswith("w") else chess.BLACK
            self._board = chess.Board(fen)
            self._san_history = list(san_history or [])
            self._last_move = chess.Move.from_uci(last_move_uci) if last_move_uci else None
            self._last_classification = None
            self._last_eval = None
        self._publish()
        return self.snapshot()

    def undo_last_move(self) -> ChessState:
        """Roll back the most recent half-move. Used for "wait, I meant…"
        voice corrections."""
        with self._lock:
            if not self._board.move_stack:
                raise ValueError("no moves to undo")
            self._board.pop()
            if self._san_history:
                self._san_history.pop()
            self._last_move = self._board.peek() if self._board.move_stack else None
            self._last_classification = None
            self._last_eval = None
        self._publish()
        return self.snapshot()

    def close(self) -> None:
        """Shut the engine subprocess down."""
        self._engine.close()

    # ----- apply_action handlers (SimEnvironment glue) -----

    def _action_new_game(self, user_plays: str | None = None, engine_skill: int | None = None) -> dict[str, Any]:
        return self.new_game(user_plays=user_plays, engine_skill=engine_skill).to_json()

    def _action_list_legal_moves(self) -> list[str]:
        return self.list_legal_moves()

    def _action_get_state(self) -> dict[str, Any]:
        return self.snapshot().to_json()

    # ----- internals -----

    def _parse_move_unlocked(self, move: str) -> chess.Move:
        """Accept UCI (``e2e4``, ``e7e8q``) or SAN (``e4``, ``Nxd5``)."""
        text = move.strip()
        if not text:
            raise ValueError("empty move string")
        try:
            parsed = chess.Move.from_uci(text.lower())
            if parsed in self._board.legal_moves:
                return parsed
        except ValueError:
            pass
        try:
            return self._board.parse_san(text)
        except ValueError as e:
            raise ValueError(f"could not parse move {move!r}: {e}") from e

    def _apply_move_unlocked(
        self,
        move: chess.Move,
        baseline: EngineMove | None,
        *,
        engine_move: EngineMove | None = None,
    ) -> None:
        """Push a move, update history, recompute eval + classification."""
        san = self._board.san(move)
        self._board.push(move)
        self._san_history.append(san)
        self._last_move = move
        self._last_eval = engine_move or self._safe_analyse_unlocked()
        self._last_classification = self._classify_unlocked(baseline)

    def _safe_analyse_unlocked(self) -> EngineMove | None:
        """Best-effort analysis for classification / eval display.

        Wrapped in a try/except because we don't want a flaky engine
        subprocess to block a legal move from being applied. Callers
        see ``None`` and treat it as "no baseline, can't classify".
        """
        try:
            board_copy = self._board.copy(stack=False)
            return self._engine.analyse(board_copy, depth=self._analyse_depth)
        except Exception:
            log.debug("Analysis failed; skipping classification", exc_info=True)
            return None

    def _classify_unlocked(self, baseline: EngineMove | None) -> MoveClassification | None:
        """Compare the played move's resulting eval to the pre-move best."""
        if baseline is None or self._last_eval is None:
            return None
        pre = baseline.score_from_white
        post = self._last_eval.score_from_white
        if pre is None or post is None:
            return None
        # Swing from the *mover's* perspective: a white move dropping
        # 200 cp means the swing is -200 for white, but the mover
        # (white) lost 200. We normalise to "how many cp did the
        # mover lose" using the pre-move turn. When it's black's turn
        # after the move, white just moved; otherwise black moved.
        swing = pre - post if self._board.turn == chess.BLACK else -(pre - post)
        return classify_move(swing)

    def _snapshot_unlocked(self) -> ChessState:
        eval_cp = self._last_eval.score_from_white if self._last_eval else None
        mate_in = self._last_eval.mate_in if self._last_eval else None
        winner: str | None = None
        reason: str | None = None
        if self._board.is_game_over():
            result = self._board.result()
            reason = (
                "checkmate"
                if self._board.is_checkmate()
                else "stalemate"
                if self._board.is_stalemate()
                else "insufficient material"
                if self._board.is_insufficient_material()
                else "75-move / fivefold"
                if self._board.is_seventyfive_moves() or self._board.is_fivefold_repetition()
                else "rules"
            )
            winner = "white" if result == "1-0" else "black" if result == "0-1" else None
        return ChessState(
            fen=self._board.fen(),
            ply=self._board.ply(),
            turn="white" if self._board.turn == chess.WHITE else "black",
            last_move_uci=self._last_move.uci() if self._last_move else None,
            last_move_san=self._san_history[-1] if self._san_history else None,
            last_move_classification=self._last_classification,
            san_history=list(self._san_history),
            eval_cp=eval_cp,
            mate_in=mate_in,
            win_prob_white=win_probability(eval_cp, mate_in=mate_in),
            opening=opening_name(self._board),
            is_game_over=self._board.is_game_over(),
            game_over_reason=reason,
            winner=winner,
        )

    def _publish(self) -> None:
        state = self.snapshot()
        for listener in list(self._listeners):
            try:
                listener(state)
            except Exception:
                log.exception("ChessState listener failed")


__all__ = ["ChessEnvironment", "ChessState"]
