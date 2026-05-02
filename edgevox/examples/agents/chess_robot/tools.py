"""Chess tool surface for the conversational Rook agent. (PARKED)

Replaces the regex-based ``MoveInterceptHook`` path with ``@tool``
functions the LLM dispatches. Each tool closes over a ``RookBridge``
so it can read the live :class:`ChessEnvironment` and call back into
bridge-level actions (persona swap, etc.) — keeping the tools
per-bridge instead of module-global.

Status: **NOT wired into the default RookApp bridge.** A live test
on 2026-05-02 with the default Gemma-4-E2B model + 9 tools +
``tool_choice_policy="required_first_hop"`` produced two failure
modes the project's own docs already flagged:

1. **Semantic loop / freeze** — the agent kept dispatching tools
   across hops instead of producing a final reply (``tool-calling.md``
   §"hooks remain registered as a safety net for … semantic looping").
   Process pinned at 99% CPU, no chat updates, board frozen.
2. **Misrouted dispatch** — the SLM picked board-mutation tools
   (``new_game(side='black')``) on conversational input where
   ``chitchat()`` was correct (``move_intercept.py``:6-13: *"Small
   models are unreliable at emitting tool calls on every turn"*).

Re-enable when the project ships with a stronger default LLM (e.g.
Gemma-4-E4B / Llama-3-8B) that handles tool dispatch reliably. To
wire it in:

1. ``from .tools import build_chess_tools`` in
   ``edgevox/apps/chess_robot_qt/bridge.py``.
2. Add ``tools=build_chess_tools(self),
   tool_choice_policy="required_first_hop"`` to the ``LLMAgent(...)``
   call.
3. Drop ``MoveInterceptHook()`` from the hook chain (or keep it as
   a regex fallback before the LLM runs).
4. Swap ``ROOK_TOOL_GUIDANCE`` for ``ROOK_TOOL_AGENT_GUIDANCE`` in
   ``_compose_instructions``.
5. Re-run ``edgevox-chess-robot`` and watch for the loop / misroute
   failure modes above.

Tools intentionally return short, factual strings (not user-facing
sentences). The LLM composes the spoken reply *after* seeing the tool
result, so its persona shapes the wording. Pre-formatting here would
flatten the persona's voice.

The ``chitchat`` tool is the always-valid escape hatch for inputs that
don't need any board action (greetings, thanks, opinions, questions
about Rook itself). With ``tool_choice_policy="required_first_hop"``
the grammar forces the LLM to call *some* tool on hop 0; ``chitchat``
gives it a no-op when conversation is all that's needed.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from edgevox.llm import tool

if TYPE_CHECKING:
    from edgevox.apps.chess_robot_qt.bridge import RookBridge


def build_chess_tools(bridge: RookBridge) -> list[Any]:
    """Construct the per-bridge chess tool list.

    Closure factory: each tool captures ``bridge`` so it can reach the
    live :class:`ChessEnvironment` via ``bridge.env`` (which is ``None``
    until the load job finishes — every tool guards against that).
    """

    @tool
    def play_move(move: str) -> str:
        """Apply the user's chess move, then make the engine reply.

        Use when the user specifies a move — explicit notation like
        ``e4``, ``Nxd5``, ``O-O``, ``e2e4``, or natural phrasing like
        ``knight to c3``, ``castle kingside``, ``pawn takes d5``.

        Args:
            move: Chess move in UCI (e2e4, e7e8q) or SAN (e4, Nxd5, O-O).
        """
        env = bridge.env
        if env is None:
            return "board not loaded yet"
        try:
            env.play_user_move(move)
        except ValueError as e:
            return f"illegal: {e}"
        state = env.snapshot()
        if state.is_game_over:
            winner = state.winner or "draw"
            return f"applied {state.last_move_san}; GAME OVER — {state.game_over_reason}, winner {winner}"
        try:
            _, engine_move = env.engine_move()
        except Exception as e:
            return f"applied {state.last_move_san}; engine reply failed: {e}"
        post = env.snapshot()
        eval_str = _eval_str(post.eval_cp, post.mate_in)
        cls = post.last_move_classification.value if post.last_move_classification else "unknown"
        return (
            f"user played {state.last_move_san}; engine replied {engine_move.san}; "
            f"eval {eval_str} (white pov); engine move classification {cls}"
        )

    @tool
    def get_position() -> dict:
        """Return the current board state — eval, whose turn, last move, opening.

        Use when the user asks about the position generally: "am I winning?",
        "what's the eval?", "how am I doing?", "what's happening on the board?".
        """
        env = bridge.env
        if env is None:
            return {"error": "board not loaded yet"}
        s = env.snapshot()
        return {
            "fen": s.fen,
            "turn": s.turn,
            "user_plays": env.user_plays,
            "engine_plays": env.engine_plays,
            "last_move_san": s.last_move_san,
            "last_move_classification": (s.last_move_classification.value if s.last_move_classification else None),
            "eval_cp_white_pov": s.eval_cp,
            "mate_in": s.mate_in,
            "win_prob_white": round(s.win_prob_white, 3),
            "opening": s.opening,
            "ply": s.ply,
            "is_game_over": s.is_game_over,
        }

    @tool
    def get_top_moves(n: int = 3) -> list:
        """Return the engine's top candidate moves for the side to move.

        Use when the user asks for hints, suggestions, or analysis:
        "what should I play?", "any hints?", "show me a good move",
        "what's the best continuation?".

        Args:
            n: Number of top moves to return, between 1 and 5.
        """
        env = bridge.env
        if env is None:
            return [{"error": "board not loaded yet"}]
        n = max(1, min(5, int(n)))
        try:
            best = env.analyse(depth=14)
        except Exception as e:
            return [{"error": f"analysis failed: {e}"}]
        out: list[dict] = [
            {
                "move_uci": best.uci,
                "san": _safe_san(env, best.uci),
                "eval_cp_white_pov": best.score_from_white,
                "mate_in": best.mate_in,
            }
        ]
        # The engine wrapper returns the principal variation but only the
        # top move's ``score_from_white`` is reliable — secondary lines
        # need a separate analyse() pass. For SLM cost reasons we just
        # surface the top move with high confidence and the next few PV
        # plies as candidate alternatives without per-move evals. The
        # LLM phrases this back to the user as "best is X; Y and Z are
        # also reasonable" without misleading them with fake numbers.
        for ply_uci in (getattr(best, "pv", []) or [])[1:n]:
            out.append(
                {"move_uci": ply_uci, "san": _safe_san(env, ply_uci), "eval_cp_white_pov": None, "mate_in": None}
            )
        return out

    @tool
    def get_legal_moves() -> list:
        """Return every legal move from the current position as UCI strings.

        Use when the user explicitly asks for the move list: "what are my
        options?", "list legal moves", "can I check the king?".
        """
        env = bridge.env
        if env is None:
            return ["board not loaded yet"]
        return env.list_legal_moves()

    @tool
    def undo_move() -> str:
        """Take back the most recent move pair so the user can retry.

        Use for "undo", "take back", "wait I meant", "let me redo that".
        Pops both the engine's reply and the user's move when both exist;
        otherwise pops whatever's on top.
        """
        env = bridge.env
        if env is None:
            return "board not loaded yet"
        # Pop engine + user. If only one exists (mid-engine-move corner
        # case) the second pop will raise; swallow it so the partial
        # undo still succeeds.
        try:
            env.undo_last_move()
        except ValueError as e:
            return f"nothing to undo: {e}"
        with contextlib.suppress(ValueError):
            env.undo_last_move()
        return f"undone; now ply {env.snapshot().ply}, {env.snapshot().turn} to move"

    @tool
    def new_game(side: str = "white") -> str:
        """Start a fresh game with the user playing the given side.

        Use for "new game", "reset", "let's start over", "play me as black".

        Args:
            side: "white" or "black" — which side the user plays.
        """
        env = bridge.env
        if env is None:
            return "board not loaded yet"
        side_norm = "white" if side.lower().startswith("w") else "black"
        bridge.config.user_plays = side_norm
        env.new_game(user_plays=side_norm)
        # When the user takes black, the engine opens immediately so the
        # turn that follows has a move to talk about.
        if side_norm == "black":
            try:
                _, engine_move = env.engine_move()
                return f"new game; user plays black; engine opened with {engine_move.san}"
            except Exception as e:
                return f"new game; user plays black; engine opening failed: {e}"
        return "new game; user plays white; user to move"

    @tool
    def resign() -> str:
        """The user resigns the current game.

        Use for "I resign", "I give up", "you win this one".
        Marks the game as ended; a new game requires ``new_game``.
        """
        env = bridge.env
        if env is None:
            return "board not loaded yet"
        # python-chess has no built-in resign mutator; the cleanest signal
        # to the rest of the stack is to start a new game on the same
        # side after recording the resignation in the tool result. The
        # LLM's spoken reply carries the "you got me" in persona.
        side = env.user_plays
        env.new_game(user_plays=side)
        return f"user resigned; engine wins; new game ready, user still plays {side}"

    @tool
    def set_persona(name: str) -> str:
        """Switch Rook's persona — voice, tone, and engine strength.

        Use when the user names a persona or asks for a different style:
        "switch to grandmaster", "be more casual", "trash talk me".

        Args:
            name: One of "casual", "grandmaster", "trash_talker".
        """
        try:
            bridge.set_persona(name)
        except Exception as e:
            return f"persona swap failed: {e}"
        return f"persona switched to {name}; engine strength change applies on next new game"

    @tool
    def chitchat() -> str:
        """Conversational input that needs no board action.

        Use for greetings ("hi", "good morning"), thanks, opinions
        ("how do you like this opening?"), questions about Rook itself
        ("what's your name?", "are you AI?"), or any small talk where
        the user doesn't want a move, hint, or eval.

        Always-valid escape hatch under ``required_first_hop`` — the
        grammar forces a tool call on hop 0, this one is the no-op.
        """
        return "ok"

    return [
        play_move,
        get_position,
        get_top_moves,
        get_legal_moves,
        undo_move,
        new_game,
        resign,
        set_persona,
        chitchat,
    ]


def _eval_str(cp: int | None, mate_in: int | None) -> str:
    if mate_in is not None:
        return f"M{mate_in:+d}"
    if cp is None:
        return "?"
    return f"{cp:+d}cp"


def _safe_san(env: Any, uci: str) -> str:
    """Best-effort UCI→SAN. Returns the UCI on parse failure."""
    try:
        import chess

        board = chess.Board(env.snapshot().fen)
        return board.san(chess.Move.from_uci(uci))
    except Exception:
        return uci


__all__ = ["build_chess_tools"]
