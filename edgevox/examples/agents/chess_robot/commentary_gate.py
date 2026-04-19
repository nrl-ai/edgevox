"""CommentaryGateHook — decide when Rook should speak, based on real signals.

Pacing (current calibration):

* Opening (first 2 plies of the game, flagged by ``ctx.session.state
  ["greeted"]``): always speak.
* Game over / check / checkmate / any capture: speak.
* Inaccuracy, mistake, blunder classifications: speak.
* Otherwise: increment a ``quiet_streak`` counter; after three quiet
  turns in a row, speak a casual remark so Rook doesn't feel dead.

Tune ``_QUIET_STREAK_LIMIT`` if the chatter/silence ratio feels off —
lowering makes Rook more talkative, raising makes him quieter.

History:
* First pass gated too hard (mistakes / blunders / rook-or-queen
  captures only) and Rook went silent for entire games. Now biased
  toward speaking on any move that has *any* verifiable hook, with the
  anti-fabrication guard kept in the briefing itself.


Small instruction-tuned LLMs (1-2B) are unreliable at two things the
original prompt asked of them:

1. Staying silent on routine moves. They will emit filler every turn no
   matter how many "you don't have to speak" directives the system
   prompt contains.
2. Not fabricating tactical claims. Given a FEN + eval + top-line
   briefing, they'll cheerfully invent pins, forks, and attacks that
   don't exist on the board.

This hook addresses both deterministically — it inspects the
:class:`ChessEnvironment` snapshot after :class:`MoveInterceptHook` has
applied the user's move and the engine's reply, decides whether this
moment is noteworthy, and either:

* suppresses the turn entirely with ``HookResult.end("")`` — the LLM
  never runs, the chat stays quiet, the face still updates via
  :class:`RobotFaceHook`;
* or stashes a ``commentary_directive`` on ``ctx.session.state`` that
  :class:`RichChessAnalyticsHook` surfaces to the model as a
  ``GROUND TRUTH`` block. The model can only narrate what's in that
  block, which keeps commentary grounded in verified facts instead of
  hallucinated tactics.

Runs at priority 85 — after :class:`MoveInterceptHook` (90) applies the
moves, before the analytics briefing (75) injects into messages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import chess

from edgevox.agents.hooks import ON_RUN_START, HookResult
from edgevox.integrations.chess.analytics import MoveClassification

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment, ChessState

log = logging.getLogger(__name__)


# After this many consecutive quiet turns the gate will force a casual
# remark, so Rook doesn't go dead during long strategic phases.
_QUIET_STREAK_LIMIT = 3


# Templated game-end lines per (persona, outcome). Game-over results are
# fully determined facts (winner, reason) — letting an LLM phrase them
# was where attribution failures clustered: small models said "I'll
# keep playing" after being mated, claimed "you missed it" to the
# winner, or echoed the SAN. Replacing that turn with a hand-written
# persona-appropriate line eliminates the failure mode at zero LLM
# cost. The gate fires ``HookResult.end(line)`` so the agent loop
# short-circuits — no model call.
#
# ``canned_game_end_index`` on ``ctx.session.state`` rotates picks so
# back-to-back games don't repeat the same closer.
_GAME_END_LINES: dict[str, dict[str, tuple[str, ...]]] = {
    "casual": {
        "won": (
            "GG! That was a fun one.",
            "Got you. Nice game.",
            "Mate. Tough luck.",
            "And that's the game — well played.",
        ),
        "lost": (
            "You got me. Well played!",
            "Nice finish — GG.",
            "Clean win, congrats.",
            "That was sharp. Good game.",
        ),
        "draw": (
            "Draw it is. Even match.",
            "Stalemate — fair enough.",
            "Half a point each, then.",
        ),
    },
    "grandmaster": {
        "won": (
            "Mate. Well played.",
            "Game.",
            "An instructive finish.",
            "And that concludes the game.",
        ),
        "lost": (
            "A fine game. Congratulations.",
            "Well played. Resigning.",
            "Excellent finish.",
        ),
        "draw": (
            "A draw. Reasonable conclusion.",
            "Half a point each.",
        ),
    },
    "trash_talker": {
        "won": (
            "Easy. Don't sweat it.",
            "And that's the game! Better luck next time.",
            "Mate. Try harder.",
            "Tag, you're it. GG.",
        ),
        "lost": (
            "You got me. This time.",
            "Ugh, fine. Well played.",
            "Lucky shot. I'll be back.",
        ),
        "draw": (
            "A draw. Boring.",
            "Stalemate, huh? Anticlimactic.",
        ),
    },
}


class CommentaryGateHook:
    """Gate Rook's speech on real board signals.

    Reads the post-move snapshot (``MoveInterceptHook`` ran at
    priority 90) and decides whether Rook has something grounded to
    say. Silent turns terminate the run without ever invoking the LLM.

    ``persona`` selects the canned game-end line set. The bridge
    passes the same persona it uses to compose system instructions
    so closers feel consistent with the rest of Rook's voice.
    """

    points = frozenset({ON_RUN_START})
    priority = 85

    def __init__(self, *, persona: str = "casual") -> None:
        self.persona = persona

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        env = _chess_env(ctx)
        if env is None:
            return None  # Fail open: no env, let the LLM run normally.
        state = env.snapshot()
        # Game over is 100% deterministic — winner / reason are facts,
        # not interpretations. Skip the LLM entirely and emit a canned
        # persona line. Eliminates the attribution failures small
        # models hit on game-over (claiming "I'll keep playing" after
        # being mated, etc.) at zero inference cost.
        if state.is_game_over:
            line = _canned_game_end(state, env, ctx.session.state, self.persona)
            ctx.emit(
                "commentary_gate",
                getattr(ctx, "agent_name", None) or "rook",
                {"decision": "canned-game-end", "line": line},
            )
            return HookResult.end(line, reason="canned game-end reply")
        current_ply = len(state.san_history)
        last_gate_ply = ctx.session.state.get("last_gate_ply")
        if last_gate_ply is not None and current_ply == last_gate_ply:
            # The board didn't advance since we last fired — the user
            # sent a non-move input (e.g. a question) that
            # MoveInterceptHook passed through. Don't re-emit analytics,
            # don't re-inject a directive, don't re-greet. Clear any
            # stale directive so RichChessAnalyticsHook falls back to
            # its full briefing and the LLM answers the question
            # naturally.
            ctx.session.state.pop("commentary_directive", None)
            return None
        ctx.session.state["last_gate_ply"] = current_ply
        # Record per-turn eval + classification FIRST so the directive
        # can reference the running history, and so silent turns still
        # contribute context to the next turn Rook actually speaks on.
        _record_turn_history(state, env, ctx.session.state)
        agent_name = getattr(ctx, "agent_name", None) or "rook"
        # Emit a UI-facing analytics event regardless of speak/silent —
        # the user sees the structured description ("user bishop f1→f7
        # capturing pawn · eval +0.40") as a system info bubble in the
        # chat even when Rook stays quiet.
        analytics_payload = _build_analytics_payload(state, env, ctx.session.state)
        if analytics_payload is not None:
            ctx.emit("move_analytics", agent_name, analytics_payload)
        facts = _build_ground_truth(state, env, ctx.session.state)
        if facts is None:
            # Quiet turn — bump the streak counter and decide whether
            # we've been quiet for too long.
            streak = int(ctx.session.state.get("quiet_streak", 0)) + 1
            if streak >= _QUIET_STREAK_LIMIT:
                ctx.session.state["quiet_streak"] = 0
                keepalive = _keepalive_directive(state, env, ctx.session.state)
                ctx.session.state["commentary_directive"] = keepalive
                ctx.emit(
                    "commentary_gate",
                    agent_name,
                    {"decision": "speak", "reason": "quiet streak fallback"},
                )
                return None
            ctx.session.state["quiet_streak"] = streak
            ctx.emit(
                "commentary_gate",
                agent_name,
                {"decision": "silent", "streak": streak},
            )
            return HookResult.end("", reason="commentary gate: quiet turn")
        # Speak — reset the streak, hand the facts to the briefing.
        ctx.session.state["quiet_streak"] = 0
        ctx.session.state["commentary_directive"] = facts
        ctx.emit(
            "commentary_gate",
            agent_name,
            {"decision": "speak", "facts": facts},
        )
        return None


def _build_analytics_payload(
    state: ChessState,
    env: ChessEnvironment,
    session_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Assemble the structured payload emitted as a ``move_analytics``
    :class:`AgentEvent`. Consumed by the Qt bridge to render a system
    info bubble in the chat.

    Returns ``None`` on the very first call before any move exists, so
    the UI doesn't render an empty analytics bubble at game start.
    """
    if not state.san_history:
        return None
    user_san, engine_san = _split_last_pair(state.san_history, env.engine_plays)
    user_desc, engine_desc = _describe_turn_pair(state.san_history, user_san, engine_san)
    history = session_state.get("turn_history", [])
    last_entry = history[-1] if history else {}
    return {
        "user_san": user_san,
        "engine_san": engine_san,
        "user_desc": user_desc,
        "engine_desc": engine_desc,
        "eval_cp": state.eval_cp,
        "prev_eval_cp": last_entry.get("prev_eval_cp"),
        # Two variants: ``score_line`` (Rook POV) is what the LLM
        # directive uses; ``score_line_user`` (user POV) is what the
        # chat analytics bubble renders. Having both avoids the
        # "you are winning" footgun where the bubble addressed the
        # user with Rook's pronouns.
        "score_line": _score_line(state, env),
        "score_line_user": _score_line_user_facing(state, env),
        "classification": state.last_move_classification.value if state.last_move_classification else None,
        "is_game_over": state.is_game_over,
    }


def _build_ground_truth(
    state: ChessState,
    env: ChessEnvironment,
    session_state: dict[str, Any],
    *,
    ablate: frozenset[str] | set[str] | None = None,
) -> str | None:
    """Return a short GROUND TRUTH block, or ``None`` to stay silent.

    Each returned string is designed to be dropped into the briefing
    verbatim. Every claim is derived from verified env state; nothing
    mentions piece positions or tactics the engine didn't report.

    Takes ``session_state`` so it can record and read the
    ``greeted`` flag — whether Rook has already greeted the user for
    the current game. ``clear_memory`` wipes the session, so a new
    game starts ungreeted even within the same app process.

    ``ablate`` is a harness-only knob for the prompt-ablation bench
    (``scripts/bench_prompt_ablation.py``). Pass a set of component
    IDs — ``{"move_desc", "classification", "material", "score",
    "situation"}`` — to suppress those sections from the returned
    directive. ``role_header`` and ``footer`` were dropped
    unconditionally after the ablation sweep found them redundant
    with ``ROOK_TOOL_GUIDANCE``; those keys are accepted for
    backward compat but have no effect.
    """
    ablate = frozenset(ablate or ())
    # Game-over is handled in the gate's ``__call__`` via canned
    # persona lines (``HookResult.end``) — by the time we reach this
    # builder the run is alive and the LLM is going to be asked to
    # speak, so we only handle the not-game-over path here.

    # Opening greeting. By the time this hook runs, MoveInterceptHook
    # has already applied the user's move AND the engine's reply, so
    # ``san_history`` is never empty on a normal turn. We gate on a
    # persistent ``greeted`` flag in session state instead — cleared on
    # new-game via ``clear_memory`` — so the first turn of each game
    # always triggers a greeting regardless of how quickly the user
    # pushed their first move.
    if not session_state.get("greeted", False):
        session_state["greeted"] = True
        user_first = state.san_history[0] if state.san_history else None
        engine_first = state.san_history[1] if len(state.san_history) >= 2 else None
        opener = _opener_line(user_first, engine_first, env.engine_plays)
        return (
            "OPENING MOVE — this is my first line of the game. I greet "
            "the user in my persona's voice: cocky, warm, wry, whatever "
            "fits. One short sentence with character — not a chess-report "
            "recap. I might comment on the opening's vibe, tease them, "
            "or set the tone for the match.\n"
            f"- {opener}"
        )

    engine_plays = env.engine_plays
    user_san, engine_san = _split_last_pair(state.san_history, engine_plays)

    # Build rich move descriptions: piece names, from/to squares, what
    # got captured, check/mate flags, promotion, castling. The 1-2B LLM
    # can't read the board, so we hand it plain English for every fact
    # that matters. This is the single biggest lever against hallucinated
    # tactics — if the prompt spells out "the user moved their knight from
    # f6 to e4, capturing your bishop, giving check", the model no longer
    # has to guess what piece went where.
    user_desc, engine_desc = _describe_turn_pair(state.san_history, user_san, engine_san)

    # Is this turn noteworthy enough to speak on? We still gate speech
    # (a strict match of "user pushed a pawn one square" doesn't merit a
    # remark), but now the check reads concrete data: was there a
    # capture, a check / mate, a promotion, a classification worse than
    # GOOD? The rich descriptions above drive both the decision and the
    # directive body.
    cls = state.last_move_classification
    has_capture = bool((user_san and "x" in user_san) or (engine_san and "x" in engine_san))
    has_check = bool(
        (user_san and (user_san.endswith("+") or user_san.endswith("#")))
        or (engine_san and (engine_san.endswith("+") or engine_san.endswith("#")))
    )
    has_promotion = bool((user_san and "=" in user_san) or (engine_san and "=" in engine_san))
    has_castle = bool(
        (user_san and user_san.rstrip("+#") in ("O-O", "O-O-O"))
        or (engine_san and engine_san.rstrip("+#") in ("O-O", "O-O-O"))
    )
    has_notable_classification = cls in (
        MoveClassification.BLUNDER,
        MoveClassification.MISTAKE,
        MoveClassification.INACCURACY,
    )
    if not (has_capture or has_check or has_promotion or has_castle or has_notable_classification):
        return None

    facts: list[str] = []
    if user_desc and "move_desc" not in ablate:
        facts.append(f"The user's move: {user_desc}")
    if engine_desc and "move_desc" not in ablate:
        facts.append(f"My move (Rook): {engine_desc}")
    if has_notable_classification and "classification" not in ablate:
        mover = "my" if (engine_san and engine_san == state.last_move_san) else "the user's"
        facts.append(
            f"Classification: that {mover} last move was a {cls.value} — "
            f"the engine thinks there was a clearly better choice."
        )
    if "material" not in ablate:
        material_line = _material_change_line(state.san_history, env)
        if material_line:
            facts.append(material_line)
    if "score" not in ablate:
        score_line = _score_line(state, env)
        if score_line:
            facts.append(score_line)

    # Minimalist directive shape. Prior versions prepended a role
    # header (pronoun discipline) and appended an anti-fabrication
    # footer, but both duplicated rules already spelled out in
    # ``ROOK_TOOL_GUIDANCE`` — the prompt-ablation sweep
    # (``scripts/bench_prompt_ablation.py``) measured no quality
    # change on Gemma 4 E2B or Llama 3.2 1B when dropped, and saved
    # ~130 tokens per turn. Section header is ``SITUATION`` (third-
    # person, declarative) rather than ``MY REACTION TONE`` (first-
    # person, narrative) — 1B models will paste first-person
    # instructions verbatim into their reply if we give them that
    # shape.
    facts_line = ". ".join(facts) if facts else "(no extra facts)"
    sections: list[str] = [f"FACTS — just happened: {facts_line}."]
    if "situation" not in ablate:
        situation = _situation_summary(state, env, user_san, engine_san)
        if situation:
            sections.append(f"SITUATION: {situation}")
    return "\n".join(sections)


def _situation_summary(
    state: ChessState,
    env: ChessEnvironment,
    user_san: str | None,
    engine_san: str | None,
) -> str | None:
    """One-line summary of what just happened + required reaction tone.

    The LLM sees this as the last line of the briefing, which is the
    slot small models actually follow. Keep it short and prescriptive —
    every failure case we've hit in eval corresponds to the model
    missing a cue buried deeper in the directive.
    """
    # Compute material delta from Rook's perspective for the punch
    # line. Duplicates what ``_material_change_line`` does, but we want
    # the number here for a tight summary.
    delta_rook = 0
    if len(state.san_history) >= 2:
        if len(state.san_history) % 2 == 0:
            pre = _replay_up_to(state.san_history, count=len(state.san_history) - 2)
        else:
            pre = _replay_up_to(state.san_history, count=len(state.san_history) - 1)
        post = _replay_up_to(state.san_history, count=len(state.san_history))
        raw = _material_balance(post) - _material_balance(pre)
        delta_rook = raw if env.engine_plays == "white" else -raw

    # Phrasing rule: describe the situation as neutral state, NOT as
    # first-person narrative instructions. 1B models will paste phrases
    # like "I concede in persona" or "I react with confidence" verbatim
    # into the reply if the briefing narrates in first person. Keep
    # this section declarative — "situation is X, tone should be Y" —
    # so there's nothing quotable to leak.
    check_flag = ""
    if user_san and user_san.endswith("#"):
        return "User just delivered checkmate. Tone: resigned, gracious, or defiant in character. Do not suggest the game continues."
    if engine_san and engine_san.endswith("#"):
        return (
            "Rook just delivered checkmate. Tone: triumphant or sharply satisfied in character. Do not suggest danger."
        )
    if engine_san and engine_san.endswith("+"):
        check_flag = " Rook gave check."
    elif user_san and user_san.endswith("+"):
        check_flag = " User gave check."

    if delta_rook >= 3:
        return f"Rook gained material this turn (+{delta_rook} points).{check_flag} Tone: confident; do not praise the user's move."
    if delta_rook <= -3:
        return f"Rook lost material this turn ({delta_rook} points).{check_flag} Tone: rattled, rueful, or wry; do not claim an advantage."
    if delta_rook > 0:
        return f"Rook came out slightly ahead in the exchange.{check_flag} Tone: brief and in character."
    if delta_rook < 0:
        return f"Rook came out slightly behind in the exchange.{check_flag} Tone: brief and in character; no congratulating the user."
    if check_flag:
        return f"Even exchange this turn.{check_flag}"
    # No material change, no check — still called here because something
    # else (classification, opening turn) triggered speech. Keep it open.
    return None


def _material_change_line(san_history: list[str], env: ChessEnvironment) -> str | None:
    """Compute the material delta over this turn (user move + engine
    reply) and render it as a plain-English line.

    Without this, the 1-2B model sees "capture" as uniformly positive —
    it doesn't notice that a capture can follow a blunder where the
    user just handed the piece over. The material-change line tells
    the model explicitly whether this turn was a gain or a loss *for
    Rook*.
    """
    if len(san_history) < 2:
        return None
    # Re-use the same pairing logic as ``_describe_turn_pair`` to
    # figure out which board states bracket this turn.
    if len(san_history) % 2 == 0:
        pre_board = _replay_up_to(san_history, count=len(san_history) - 2)
    else:
        # Terminal user move; only one new ply this turn.
        pre_board = _replay_up_to(san_history, count=len(san_history) - 1)
    post_board = _replay_up_to(san_history, count=len(san_history))
    pre_white_adv = _material_balance(pre_board)
    post_white_adv = _material_balance(post_board)
    delta_white = post_white_adv - pre_white_adv
    if delta_white == 0:
        return "Material change this turn: roughly even exchange."
    delta_for_rook = delta_white if env.engine_plays == "white" else -delta_white
    pts = abs(delta_for_rook)
    if delta_for_rook > 0:
        return (
            f"Material change this turn: YOU gained {pts} point"
            f"{'s' if pts != 1 else ''} of material (the user came out "
            "worse in the exchange). React accordingly — this is a "
            "good turn for you."
        )
    return (
        f"Material change this turn: YOU lost {pts} point"
        f"{'s' if pts != 1 else ''} of material (the user came out "
        "better in the exchange). React accordingly — this hurt."
    )


def _material_balance(board: chess.Board) -> int:
    """Return (white_points - black_points) in standard piece values."""
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sign = 1 if piece.color == chess.WHITE else -1
        total += sign * values[piece.piece_type]
    return total


def _mood_cue(state: ChessState, env: ChessEnvironment) -> str | None:
    """Give the model an explicit tone signal based on the current
    eval. Small models don't reliably derive mood from a cp number —
    spelling it out ("sound confident" / "sound rattled") gates the
    model into the right persona register for this turn.
    """
    if state.mate_in is not None:
        rook_wins = (state.mate_in > 0) == (env.engine_plays == "white")
        return (
            "MOOD CUE: I have a forced mate — I sound sharp, focused, final."
            if rook_wins
            else "MOOD CUE: I am being mated — I sound resigned or defiant, whatever my persona does under pressure."
        )
    if state.eval_cp is None:
        return None
    pawns_for_rook = state.eval_cp / 100.0
    if env.engine_plays == "black":
        pawns_for_rook = -pawns_for_rook
    if pawns_for_rook >= 3.0:
        return "MOOD CUE: I am winning decisively — I sound confident, or playfully smug in persona. I do NOT praise the user's move."
    if pawns_for_rook >= 1.0:
        return "MOOD CUE: I have a clear edge — I sound pleased with my position."
    if pawns_for_rook <= -3.0:
        return "MOOD CUE: the user is winning decisively — I sound rattled, wry, or defiant. I do NOT congratulate the user generically."
    if pawns_for_rook <= -1.0:
        return "MOOD CUE: the user has a clear edge — I sound mildly concerned or focused."
    return "MOOD CUE: the position is close — neutral, curious, engaged."


# --- move / score description helpers ------------------------------------


_PIECE_NAME = {
    "K": "king",
    "Q": "queen",
    "R": "rook",
    "B": "bishop",
    "N": "knight",
    "P": "pawn",
}


def _describe_turn_pair(
    san_history: list[str],
    user_san: str | None,
    engine_san: str | None,
) -> tuple[str | None, str | None]:
    """Produce rich-English descriptions of the last user and engine moves.

    Replays ``san_history`` minus the last move(s) to reconstruct the
    pre-move board — that's what lets us name the captured piece and
    the moving piece, which SAN alone doesn't expose for pawns.

    Returns ``(user_description, engine_description)``; either can be
    ``None`` if the move can't be parsed (shouldn't happen in practice,
    but we fail soft rather than crash the turn).
    """
    user_desc: str | None = None
    engine_desc: str | None = None

    if engine_san and user_san:
        # Normal turn: user then engine.
        pre_user_board = _replay_up_to(san_history, count=len(san_history) - 2)
        user_desc = _describe_move(user_san, pre_user_board)
        pre_engine_board = _replay_up_to(san_history, count=len(san_history) - 1)
        engine_desc = _describe_move(engine_san, pre_engine_board)
    elif user_san:
        # Terminal user move — engine never replied.
        pre_user_board = _replay_up_to(san_history, count=len(san_history) - 1)
        user_desc = _describe_move(user_san, pre_user_board)
    elif engine_san:
        pre_engine_board = _replay_up_to(san_history, count=len(san_history) - 1)
        engine_desc = _describe_move(engine_san, pre_engine_board)

    return user_desc, engine_desc


def _replay_up_to(san_history: list[str], *, count: int) -> chess.Board:
    """Return a ``chess.Board`` after replaying the first ``count`` moves.

    Used to get the pre-move position so we can resolve which piece
    moved and what piece (if any) got captured. We clamp ``count`` so
    callers don't need to worry about off-by-ones at game start.
    """
    board = chess.Board()
    for san in san_history[: max(0, count)]:
        try:
            board.push_san(san)
        except ValueError:
            break
    return board


def _describe_move(san: str, pre_board: chess.Board) -> str | None:
    """Return a one-sentence English description of ``san`` played from
    ``pre_board``. Handles pieces, pawns, captures (with captured-piece
    name), checks / mates, promotions, castling.
    """
    if not san:
        return None
    clean = san.rstrip("+#")
    suffix = ""
    if san.endswith("#"):
        suffix = ", delivering checkmate"
    elif san.endswith("+"):
        suffix = ", giving check"

    # Castling is a special SAN shape python-chess parses but we render
    # it in plain English for the LLM rather than leaving the O-O in.
    if clean in ("O-O", "O-O-O"):
        side = "kingside" if clean == "O-O" else "queenside"
        return f"castled {side}{suffix} ({san})"

    try:
        move = pre_board.parse_san(san)
    except ValueError:
        log.debug("describe_move: SAN %r unparseable on pre-move board", san, exc_info=True)
        return None

    piece = pre_board.piece_at(move.from_square)
    if piece is None:
        return None
    piece_name = _PIECE_NAME.get(piece.symbol().upper(), "piece")
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    # Promotion — describe the resulting piece, not the pawn.
    promo_text = ""
    if move.promotion:
        promo_symbol = chess.piece_symbol(move.promotion).upper()
        promo_text = f", promoting to a {_PIECE_NAME.get(promo_symbol, 'piece')}"

    if pre_board.is_capture(move):
        if pre_board.is_en_passant(move):
            captured_name = "pawn (en passant)"
        else:
            captured = pre_board.piece_at(move.to_square)
            captured_name = _PIECE_NAME.get(captured.symbol().upper(), "piece") if captured else "piece"
        return f"{piece_name} from {from_sq} to {to_sq}, capturing a {captured_name}{promo_text}{suffix} ({san})"

    return f"{piece_name} from {from_sq} to {to_sq}{promo_text}{suffix} ({san})"


def _score_line(state: ChessState, env: ChessEnvironment) -> str | None:
    """Rook-facing score line — for the LLM directive. ``you`` = Rook."""
    return _score_line_from_perspective(state, env, rook_pov=True)


def _score_line_user_facing(state: ChessState, env: ChessEnvironment) -> str | None:
    """User-facing score line — for the chat analytics bubble.

    The bubble is displayed to the user reading the chat, so ``you``
    must mean *the user*, not Rook. Same math, flipped pronouns.
    Fixes the bug where the bubble previously said "you are winning"
    to the user when Rook was winning.
    """
    return _score_line_from_perspective(state, env, rook_pov=False)


def _score_line_from_perspective(
    state: ChessState,
    env: ChessEnvironment,
    *,
    rook_pov: bool,
) -> str | None:
    """Render the engine eval as a plain-English line.

    ``rook_pov=True`` → addressed to Rook (LLM directive); "you" is
    Rook. ``rook_pov=False`` → addressed to the user (chat bubble);
    "you" is the user. Sign of the centipawn eval and the "who is
    winning" text are inverted accordingly.
    """
    if state.mate_in is not None:
        # ``mate_in`` > 0 means white is delivering mate. Translate to
        # "you" / "the user" for whichever perspective we're rendering.
        white_mating = state.mate_in > 0
        if rook_pov:
            subject = "you" if white_mating == (env.engine_plays == "white") else "the user"
        else:
            subject = "you" if white_mating == (env.user_plays == "white") else "the opponent"
        return f"Engine evaluation: {subject} have mate in {abs(state.mate_in)}."
    if state.eval_cp is None:
        return None
    pawns = state.eval_cp / 100.0
    if rook_pov:
        pawns_for_me = pawns if env.engine_plays == "white" else -pawns
        them = "the user"
    else:
        pawns_for_me = pawns if env.user_plays == "white" else -pawns
        them = "Rook"
    mag = abs(pawns_for_me)
    if mag < 0.3:
        mood = "the position is roughly level"
    elif mag < 1.0:
        leader = "you have" if pawns_for_me > 0 else f"{them} has"
        mood = f"{leader} a slight edge"
    elif mag < 3.0:
        leader = "you have" if pawns_for_me > 0 else f"{them} has"
        mood = f"{leader} a clear advantage"
    else:
        leader = "you are" if pawns_for_me > 0 else f"{them} is"
        mood = f"{leader} winning decisively"
    label = "from your side" if rook_pov else "from your side (white/black as you play)"
    return f"Engine evaluation ({label}): {pawns_for_me:+.2f} pawns — {mood}."


def _keepalive_directive(state: ChessState, env: ChessEnvironment, session_state: dict[str, Any]) -> str:
    """Low-intensity remark after too many quiet moves in a row.

    Deliberately doesn't hand the model any tactical signal — just the
    recent move list and running score — because quiet moves by
    definition don't have a hook. Invites personality-driven small
    talk so Rook feels alive, while still forbidding made-up tactical
    claims.
    """
    history_block = _render_move_history(session_state.get("turn_history", []), env, limit=5)
    sections = [
        "QUIET PHASE — nothing dramatic just happened, but it's been a "
        "few turns since I said anything. I drop a short in-character "
        "remark — banter, a sigh, a tease, an observation about the pace "
        "or mood. I don't claim tactics. I speak about myself in first "
        "person; I speak to the user as 'you'."
    ]
    if history_block:
        sections.append(history_block)
    sections.append("If nothing truly comes to mind in persona, I reply exactly `<silent>`.")
    return "\n".join(sections)


def _opener_line(user_first: str | None, engine_first: str | None, engine_plays: str) -> str:
    """A hint about the opening that's safe for the model to quote.

    We include both the user's first move and the engine's reply when
    present — that's enough for a greeting that feels game-specific
    without inviting tactical invention.
    """
    if engine_plays == "white" and engine_first and user_first:
        return f"You opened with {engine_first}; the user replied {user_first}."
    if user_first and engine_first:
        return f"The user opened with {user_first}; you replied {engine_first}."
    if user_first:
        return f"The user opened with {user_first}."
    return ""


def _record_turn_history(state: ChessState, env: ChessEnvironment, session_state: dict[str, Any]) -> None:
    """Append this turn's outcome to a rolling per-game history.

    Stored under ``session_state["turn_history"]``. Each entry captures
    the move pair just played, the resulting eval, the eval delta from
    the prior turn (useful for "score swung by N pawns" messaging),
    and the last classification. We dedupe on ply so repeat calls
    within the same turn don't bloat the list — chess.Board's
    ``fullmove_number`` isn't convenient here because MoveInterceptHook
    applies two half-moves in one gate call.
    """
    history = session_state.setdefault("turn_history", [])
    user_san, engine_san = _split_last_pair(state.san_history, env.engine_plays)
    if not user_san and not engine_san:
        return
    ply = len(state.san_history)
    if history and history[-1].get("ply") == ply:
        return
    prev_eval_cp = history[-1].get("eval_cp") if history else None
    entry: dict[str, Any] = {
        "ply": ply,
        "user_san": user_san,
        "engine_san": engine_san,
        "eval_cp": state.eval_cp,
        "classification": state.last_move_classification.value if state.last_move_classification else None,
        "prev_eval_cp": prev_eval_cp,
    }
    history.append(entry)
    # Cap at 16 turns so the session file doesn't grow unboundedly.
    if len(history) > 16:
        del history[:-16]


def _render_move_history(history: list[dict[str, Any]], env: ChessEnvironment, *, limit: int) -> str | None:
    """Render the last ``limit`` entries as a plain-English block.

    Shape:
        MOVE HISTORY (last N turns, eval from your side in pawns):
        - 5: user knight g1→f3, you knight b8→c6 · eval +0.10 (level)
        - 6: user bishop f1→c4, you knight g8→f6 · eval +0.05
        - 7: user bishop c4→f7 capturing pawn · eval -0.40 (slight edge to them, swung -0.45)

    Meant to give the 1-2B LLM enough narrative to reference ("you've
    been trading pieces", "the eval just swung in their favour") without
    having to read the FEN.
    """
    if not history:
        return None
    tail = history[-limit:]
    lines = ["MOVE HISTORY (last turns, eval from your side in pawns — negative means the user is ahead):"]
    for entry in tail:
        turn_no = (entry["ply"] + 1) // 2
        pieces: list[str] = []
        if entry["user_san"]:
            pieces.append(
                f"user {_short_move_desc(entry['user_san'], pre_moves=_pre_moves(history, entry, side='user'))}"
            )
        if entry["engine_san"]:
            pieces.append(
                f"you {_short_move_desc(entry['engine_san'], pre_moves=_pre_moves(history, entry, side='engine'))}"
            )
        score_part = _short_score_annotation(entry, env)
        joined = ", ".join(pieces) if pieces else "(no moves)"
        lines.append(f"- turn {turn_no}: {joined} · {score_part}")
    return "\n".join(lines)


def _pre_moves(history: list[dict[str, Any]], current: dict[str, Any], *, side: str) -> list[str]:
    """Build the SAN sequence leading up to the move on ``side`` of the
    ``current`` history entry, so ``_short_move_desc`` can resolve the
    piece name. ``side`` is ``"user"`` or ``"engine"``."""
    moves: list[str] = []
    for entry in history:
        if entry["ply"] < current["ply"] - 1:
            if entry["user_san"]:
                moves.append(entry["user_san"])
            if entry["engine_san"]:
                moves.append(entry["engine_san"])
        elif entry["ply"] == current["ply"] - 1:
            # Same-turn interleaving.
            if entry["user_san"]:
                moves.append(entry["user_san"])
            if entry["engine_san"]:
                moves.append(entry["engine_san"])
    if side == "engine" and current["user_san"]:
        moves.append(current["user_san"])
    return moves


def _short_move_desc(san: str, *, pre_moves: list[str]) -> str:
    """Compact description used inside the history block — shorter than
    ``_describe_move`` because history is a summary list, not the focal
    GROUND TRUTH fact.
    """
    if not san:
        return "(none)"
    clean = san.rstrip("+#")
    if clean in ("O-O", "O-O-O"):
        side = "kingside" if clean == "O-O" else "queenside"
        return f"castled {side}"
    try:
        board = chess.Board()
        for m in pre_moves:
            board.push_san(m)
        move = board.parse_san(san)
    except ValueError:
        return san
    piece = board.piece_at(move.from_square)
    if piece is None:
        return san
    piece_name = _PIECE_NAME.get(piece.symbol().upper(), "piece")
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    if board.is_capture(move):
        captured_name = "pawn"
        if not board.is_en_passant(move):
            captured = board.piece_at(move.to_square)
            if captured:
                captured_name = _PIECE_NAME.get(captured.symbol().upper(), "piece")
        desc = f"{piece_name} {from_sq}→{to_sq} capturing {captured_name}"
    else:
        desc = f"{piece_name} {from_sq}→{to_sq}"
    if san.endswith("#"):
        desc += " ##"
    elif san.endswith("+"):
        desc += " +"
    return desc


def _short_score_annotation(entry: dict[str, Any], env: ChessEnvironment) -> str:
    """One-line score annotation for a history entry."""
    eval_cp = entry.get("eval_cp")
    prev_cp = entry.get("prev_eval_cp")
    if eval_cp is None:
        return "eval n/a"
    pawns = eval_cp / 100.0
    # Flip for Rook's side so the sign reads naturally for persona.
    pawns_for_rook = pawns if env.engine_plays == "white" else -pawns
    parts = [f"eval {pawns_for_rook:+.2f}"]
    if prev_cp is not None:
        prev_pawns_for_rook = (prev_cp / 100.0) if env.engine_plays == "white" else -(prev_cp / 100.0)
        swing = pawns_for_rook - prev_pawns_for_rook
        if abs(swing) >= 0.3:
            parts.append(f"swung {swing:+.2f}")
    cls = entry.get("classification")
    if cls and cls not in ("best", "good"):
        parts.append(f"last move: {cls}")
    return " · ".join(parts)


def _split_last_pair(san_history: list[str], engine_plays: str) -> tuple[str | None, str | None]:
    """Return ``(user_san, engine_san)`` for the most recently played pair.

    After :class:`MoveInterceptHook` runs on a normal turn, history ends
    in ``[..., user_move, engine_reply]``. On a terminal user move
    (delivers mate / stalemate) the engine never played, so history
    ends in ``[..., user_move]`` and ``engine_san`` is ``None``.

    The pairing respects who plays which colour: when the engine plays
    white it moves first, so the last pair is ``(engine, user)``, not
    the other way around.
    """
    if not san_history:
        return (None, None)
    # Determine move parity: white moves on even indices (0, 2, 4, …).
    last_idx = len(san_history) - 1
    last_is_white = (last_idx % 2) == 0
    last_side = "white" if last_is_white else "black"
    last_engine = last_side == engine_plays

    if last_engine and len(san_history) >= 2:
        return (san_history[-2], san_history[-1])
    if not last_engine:
        # Engine never replied (terminal position after user's move).
        if len(san_history) >= 2:
            return (san_history[-1], None)
        return (san_history[-1], None)
    # History of length 1 and engine plays last — engine opened, no
    # user move yet (shouldn't happen on a post-user-move snapshot).
    return (None, san_history[-1])


def _canned_game_end(
    state: ChessState,
    env: ChessEnvironment,
    session_state: dict[str, Any],
    persona: str,
) -> str:
    """Pick a templated game-end reply for ``persona`` and outcome.

    Rotates through the available phrasings via a counter on
    ``session_state`` so back-to-back games (or the same game replayed
    in tests) don't always close the same way. Falls back to the
    casual set when ``persona`` isn't recognised.
    """
    winner = (state.winner or "").lower()
    if winner == env.engine_plays.lower():
        outcome = "won"
    elif winner == env.user_plays.lower():
        outcome = "lost"
    else:
        outcome = "draw"
    bucket = _GAME_END_LINES.get(persona) or _GAME_END_LINES["casual"]
    lines = bucket.get(outcome) or _GAME_END_LINES["casual"][outcome]
    idx = int(session_state.get("canned_game_end_index", 0)) % len(lines)
    session_state["canned_game_end_index"] = idx + 1
    return lines[idx]


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "snapshot") and hasattr(deps, "user_plays") and hasattr(deps, "engine_plays"):
        return deps  # type: ignore[return-value]
    return None


__all__ = ["CommentaryGateHook"]
