"""Rich chess analytics as a hidden system message.

:class:`RichChessAnalyticsHook` fires at :data:`BEFORE_LLM`, builds a
compact but comprehensive "briefing card" from the current
:class:`ChessEnvironment` + engine analysis, and injects it as a
``role=system`` message into the messages list just before the last
user message.

Why a system message (vs. prepending to the user task):

1. **Clean user chat** — the React chat log shows exactly what the
   user typed; rich analytics stay out of the transcript.
2. **Fresh at every hop** — the hook re-runs each time the agent
   loops, so the briefing reflects the latest position (after any
   tool-induced mutations).
3. **Semantically correct** — context that is "about the game" isn't
   user speech; system role is the appropriate channel.

The briefing is written in PLAIN ENGLISH, not JSON. Small LMs (1-2 B)
handle natural prose markedly better than structured data, and our
chess content is short enough that parsing cost is zero.

The briefing makes PERSPECTIVE explicit — which colour is Rook, which
is the user — because small models routinely confuse whose move was
just played.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chess

from edgevox.agents.hooks import BEFORE_LLM, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment, ChessState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Natural-language helpers — small, side-effect-free
# ---------------------------------------------------------------------------


def _piece_names(symbol: str) -> str:
    """Map a piece letter to a spoken name."""
    return {
        "k": "king",
        "q": "queen",
        "r": "rook",
        "b": "bishop",
        "n": "knight",
        "p": "pawn",
    }[symbol.lower()]


def _material_count(board: chess.Board) -> tuple[dict[str, int], dict[str, int]]:
    """Return (white_counts, black_counts) keyed by piece letter."""
    white = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
    black = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece or piece.piece_type == chess.KING:
            continue
        sym = piece.symbol()
        target = white if sym.isupper() else black
        target[sym.upper()] = target.get(sym.upper(), 0) + 1
    return white, black


_PIECE_VALUE = {"Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}


def _material_line(board: chess.Board) -> str:
    """Compact one-line material description + imbalance."""
    white, black = _material_count(board)
    white_pts = sum(_PIECE_VALUE[p] * n for p, n in white.items())
    black_pts = sum(_PIECE_VALUE[p] * n for p, n in black.items())
    diff = white_pts - black_pts
    if diff == 0:
        imbalance = "material is even"
    elif diff > 0:
        imbalance = f"white is up {diff} point{'s' if diff != 1 else ''}"
    else:
        imbalance = f"black is up {-diff} point{'s' if diff != -1 else ''}"
    return imbalance


def _phase(board: chess.Board) -> str:
    """Classify game phase from total non-king material (Reinfeld values).

    Endgame threshold is traditional: ≤ ~13 total points (e.g. rook +
    bishop + a few pawns vs. rook + bishop). Middlegame kicks in once
    both sides have completed their openings (either queen traded or
    >10 plies played)."""
    white, black = _material_count(board)
    total = sum(_PIECE_VALUE[p] * (n_w + n_b) for p, n_w in white.items() for n_b in [black[p]])
    if total <= 26:
        return "endgame"
    if board.fullmove_number <= 10 and total >= 70:
        return "opening"
    return "middlegame"


def _eval_description(eval_cp: int | None, mate_in: int | None) -> str:
    """Translate an eval into a natural-language sentence.

    Rough bands mirror what strong humans say over the board: ±50 cp =
    equal, ±150 = slight edge, ±300 = clear edge, ±500 = winning."""
    if mate_in is not None:
        who = "white" if mate_in > 0 else "black"
        return f"mate in {abs(mate_in)} for {who}"
    if eval_cp is None:
        return "evaluation unavailable"
    pawns = eval_cp / 100
    mag = abs(pawns)
    if mag < 0.3:
        return f"{pawns:+.2f} pawns (equal)"
    if mag < 1.0:
        side = "white" if pawns > 0 else "black"
        return f"{pawns:+.2f} pawns (slight edge to {side})"
    if mag < 3.0:
        side = "white" if pawns > 0 else "black"
        return f"{pawns:+.2f} pawns (clear advantage to {side})"
    side = "white" if pawns > 0 else "black"
    return f"{pawns:+.2f} pawns ({side} is winning)"


def _king_safety(board: chess.Board) -> str:
    """Terse description of both kings' safety state."""
    parts = []
    for colour, name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        king_sq = board.king(colour)
        if king_sq is None:
            parts.append(f"{name} king missing")
            continue
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        castled = False
        # Simple heuristic: if king is on g-file (file 6) or c-file (file 2)
        # and back rank, treat as castled.
        back_rank = 0 if colour == chess.WHITE else 7
        if rank == back_rank and file in (2, 6):
            castled = True
        start_square = 4
        if rank == back_rank and file == start_square:
            parts.append(f"{name} uncastled (king still on e{back_rank + 1})")
        elif castled:
            side = "kingside" if file == 6 else "queenside"
            parts.append(f"{name} castled {side}")
        else:
            parts.append(f"{name} king on {chess.square_name(king_sq)} — exposed")
    return "; ".join(parts)


def _attacked_pieces(board: chess.Board, side: chess.Color) -> list[str]:
    """Piece names on ``side`` currently attacked by the opponent.

    Pawns are cheap noise; we only flag minor+ pieces so the briefing
    stays short. Returns human-readable entries like
    ``"black queen on d8 under attack"``."""
    opp = not side
    threats: list[str] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece or piece.color != side:
            continue
        if piece.piece_type == chess.PAWN:
            continue
        if board.is_attacked_by(opp, square):
            # Defended? Don't flag obvious chumps.
            if board.is_attacked_by(side, square):
                continue
            colour_name = "white" if side == chess.WHITE else "black"
            threats.append(f"{colour_name} {_piece_names(piece.symbol())} on {chess.square_name(square)} under attack")
    return threats


def _last_move_context(
    san: str | None,
    classification: str | None,
) -> str:
    """Human-readable last-move line, with quality tag when present."""
    if not san:
        return "no last move (starting position)"
    if classification is None or classification == "unclassified":
        return f"{san}"
    return f"{san} ({classification})"


def _engine_side_explanation(engine_plays: str) -> str:
    """Explicit perspective sentence — top of the briefing.

    Small LMs routinely confuse whose-move-it-was if told only "side to
    move: black". Be unmistakable about which colour Rook is and which
    the user is."""
    engine = engine_plays.lower()
    user = "black" if engine == "white" else "white"
    return (
        f"You (Rook) are playing the {engine.upper()} pieces. "
        f"The user is playing {user.upper()}. "
        f"Any sentence starting with 'You' / 'your' refers to Rook ({engine}); "
        f"any sentence about 'the user' refers to {user}."
    )


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------


class RichChessAnalyticsHook:
    """Inject a hidden system-role briefing with rich chess context.

    Fires at :data:`BEFORE_LLM`. On the first hop of a run, inserts
    the briefing; on subsequent hops (tool-loop continuation) the
    position may have changed after a move tool, so we re-inject with
    fresh numbers.

    The briefing lives under an ``__analytics_idx__`` key in
    ``ctx.hook_state`` so we can replace the previous briefing in
    place — prevents the messages list from growing unboundedly as
    the agent takes multiple hops.
    """

    points = frozenset({BEFORE_LLM})
    priority = 75

    def __init__(self, *, analyse_depth: int = 8, max_pv: int = 3) -> None:
        self.analyse_depth = analyse_depth
        self.max_pv = max_pv

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        if not isinstance(payload, dict):
            return None
        env = _chess_env(ctx)
        if env is None:
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None

        briefing = self._build_briefing(env, ctx)
        if not briefing:
            return None

        # Place the briefing just before the LAST user message so the
        # LLM sees it as the freshest context. If no user message
        # exists yet, prepend it right after the initial system
        # prompt.
        bucket = ctx.hook_state.setdefault(id(self), {})
        prev_idx = bucket.get("idx")
        new_msg = {"role": "system", "content": briefing}

        new_messages = list(messages)
        # Drop the previous briefing if it's still in place so we
        # don't accumulate one per hop.
        if isinstance(prev_idx, int) and 0 <= prev_idx < len(new_messages):
            prev = new_messages[prev_idx]
            if isinstance(prev, dict) and prev.get("role") == "system" and _is_analytics(prev.get("content", "")):
                new_messages.pop(prev_idx)

        insert_at = _index_of_last_user_message(new_messages)
        if insert_at is None:
            # No user messages yet — put right after the initial system
            # prompt (index 0), if any.
            insert_at = 1 if new_messages and new_messages[0].get("role") == "system" else 0

        new_messages.insert(insert_at, new_msg)
        bucket["idx"] = insert_at

        return HookResult.replace(
            {**payload, "messages": new_messages},
            reason="rich chess analytics injected",
        )

    # ----- briefing builder -----

    def _build_briefing(self, env: ChessEnvironment, ctx: AgentContext | None = None) -> str:
        """Assemble the briefing card.

        When :class:`CommentaryGateHook` has stashed a
        ``commentary_directive`` on ``ctx.session.state``, the briefing
        switches to a tight GROUND TRUTH-only shape: perspective +
        recent moves + the directive. The rich FEN / PV / king-safety
        lines are omitted because 1-2B models can't read them
        accurately and their presence invites fabrication (claiming
        pins / forks / attacks that don't exist).

        Without a directive, the full analytics card is rendered as
        before — useful for stronger models or for diagnostic runs
        where commentary quality isn't gated on small-model limits.
        """
        state = env.snapshot()
        try:
            board = chess.Board(state.fen)
        except ValueError:
            return ""

        directive = None
        if ctx is not None:
            directive = ctx.session.state.get("commentary_directive")

        if directive:
            return self._build_focused_briefing(env, state, directive)
        return self._build_full_briefing(env, state, board)

    def _build_focused_briefing(self, env: ChessEnvironment, state: ChessState, directive: str) -> str:
        """Tight, fabrication-resistant briefing for small models.

        Only contains the perspective note, the last few moves in SAN,
        and the verified-facts directive from
        :class:`CommentaryGateHook`. No FEN, no eval number, no PV — all
        of which a 1-2B model will otherwise paraphrase into hallucinated
        tactical claims.
        """
        lines: list[str] = ["[CHESS BRIEFING — internal context, do not read aloud verbatim]"]
        lines.append(_engine_side_explanation(env.engine_plays))
        if state.san_history:
            tail = state.san_history[-6:]
            lines.append(f"Recent moves: {' '.join(tail)}")
        lines.append(directive)
        lines.append("[END BRIEFING]")
        return "\n".join(lines)

    def _build_full_briefing(self, env: ChessEnvironment, state: ChessState, board: chess.Board) -> str:
        """Full analytics card — legacy shape, still used when no
        commentary gate is wired in (or for debugging)."""
        lines: list[str] = ["[CHESS BRIEFING — internal context, do not read aloud verbatim]"]
        lines.append(_engine_side_explanation(env.engine_plays))
        lines.append(f"Position (FEN): {state.fen}")

        to_move = "white" if board.turn == chess.WHITE else "black"
        whose_turn = "yours (Rook to move)" if to_move == env.engine_plays else "the user's"
        lines.append(f"To move: {to_move} — {whose_turn}")

        lines.append(f"Phase: {_phase(board)}")
        if state.opening:
            lines.append(f"Opening: {state.opening}")

        lines.append(f"Material: {_material_line(board)}")
        lines.append(f"Evaluation: {_eval_description(state.eval_cp, state.mate_in)}")
        lines.append(f"King safety: {_king_safety(board)}")

        threats = _attacked_pieces(board, chess.WHITE) + _attacked_pieces(board, chess.BLACK)
        if threats:
            lines.append(f"Under attack: {', '.join(threats)}")

        if board.is_check():
            side_in_check = "you" if to_move == env.engine_plays else "the user"
            lines.append(f"{side_in_check} is in CHECK")

        lines.append(f"Last move: {_last_move_context(state.last_move_san, _class_value(state))}")

        # Engine analysis — top line (who's to move + their best reply).
        try:
            analysis = env.analyse(depth=self.analyse_depth)
        except Exception:
            log.debug("RichChessAnalyticsHook: analyse failed", exc_info=True)
            analysis = None
        if analysis and analysis.pv:
            pv_moves = analysis.pv[: self.max_pv]
            pv_str = " ".join(pv_moves)
            leader_label = "Your" if to_move == env.engine_plays else "Engine-top"
            lines.append(f"{leader_label} top line: {pv_str}")

        if state.san_history:
            tail = state.san_history[-6:]
            lines.append(f"Recent moves: {' '.join(tail)}")

        if state.is_game_over:
            lines.append(f"GAME OVER: {state.game_over_reason}, winner: {state.winner or 'draw'}")

        lines.append("[END BRIEFING]")
        return "\n".join(lines)


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "snapshot") and hasattr(deps, "engine_plays") and hasattr(deps, "analyse"):
        return deps  # type: ignore[return-value]
    return None


def _index_of_last_user_message(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return None


def _is_analytics(content: str) -> bool:
    return isinstance(content, str) and content.startswith("[CHESS BRIEFING")


def _class_value(state) -> str | None:
    """Pull the classification enum value off a state, surviving both
    dataclass and dict forms."""
    cls = getattr(state, "last_move_classification", None)
    if cls is None and isinstance(state, dict):
        cls = state.get("last_move_classification")
    if cls is None:
        return None
    return getattr(cls, "value", cls if isinstance(cls, str) else None)


__all__ = ["RichChessAnalyticsHook"]
