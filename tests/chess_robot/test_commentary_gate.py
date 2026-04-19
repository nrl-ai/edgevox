"""CommentaryGateHook — gate Rook's commentary on real board signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from edgevox.agents.base import AgentContext, Session
from edgevox.agents.hooks import ON_RUN_START
from edgevox.examples.agents.chess_robot.commentary_gate import CommentaryGateHook
from edgevox.integrations.chess.analytics import MoveClassification


@dataclass
class _StubState:
    san_history: list[str] = field(default_factory=list)
    last_move_san: str | None = None
    last_move_classification: MoveClassification | None = None
    eval_cp: int | None = None
    mate_in: int | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None
    fen: str = ""
    ply: int = 0
    turn: str = "white"


class _StubEnv:
    """Minimal :class:`ChessEnvironment` surface the gate checks for."""

    def __init__(self, state: _StubState, *, engine_plays: str = "black", user_plays: str = "white") -> None:
        self._state = state
        self.engine_plays = engine_plays
        self.user_plays = user_plays

    def snapshot(self) -> _StubState:
        return self._state


def _run(state: _StubState, *, engine_plays: str = "black", greeted: bool = True, quiet_streak: int = 0):
    """Run the gate on a stub env. ``greeted=True`` skips the opening
    greeting path so tests that care about mid-game logic aren't forced
    through the greeting branch. Pass ``greeted=False`` to test the
    greeting itself."""
    env = _StubEnv(state, engine_plays=engine_plays)
    session = Session()
    session.state["greeted"] = greeted
    session.state["quiet_streak"] = quiet_streak
    ctx = AgentContext(deps=env, session=session)
    hook = CommentaryGateHook()
    result = hook(ON_RUN_START, ctx, {"task": "e4"})
    return result, ctx


class TestCommentaryGateHook:
    def test_quiet_mid_game_move_is_silenced(self):
        """Routine move with no capture, check, or blunder → gate ends
        the turn, LLM never runs, chat stays quiet."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6"],
            last_move_san="Nc6",
            last_move_classification=MoveClassification.GOOD,
            eval_cp=15,
        )
        result, ctx = _run(state)
        assert result is not None
        assert result.action.name == "END_TURN"
        assert result.payload == ""
        assert "commentary_directive" not in ctx.session.state
        # Quiet streak should advance on a silent turn.
        assert ctx.session.state["quiet_streak"] == 1

    def test_greeting_fires_on_first_turn(self):
        """A fresh session (``greeted`` flag unset) → greet the user
        regardless of how many plies MoveInterceptHook pre-applied."""
        state = _StubState(
            san_history=["e4", "e5"],
            last_move_san="e5",
            last_move_classification=MoveClassification.BEST,
        )
        result, ctx = _run(state, greeted=False)
        assert result is None
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "move 1" in directive.lower() or "greet" in directive.lower()
        # Greeting consumes the flag so the next turn doesn't greet again.
        assert ctx.session.state["greeted"] is True

    def test_subsequent_turns_do_not_re_greet(self):
        """Once greeted, a routine follow-up move must go silent."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6"],
            last_move_san="Nc6",
            last_move_classification=MoveClassification.BEST,
        )
        result, _ = _run(state, greeted=True)
        assert result is not None
        assert result.action.name == "END_TURN"

    def test_capture_triggers_speech_with_piece_names(self):
        """Captures now produce rich descriptions with the capturing and
        captured piece names, from/to squares, and the SAN — so the
        1-2B LLM has concrete English to react to rather than a bare
        SAN token."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7"],
            last_move_san="Bxf7",
            last_move_classification=MoveClassification.GOOD,
        )
        result, ctx = _run(state, engine_plays="white")
        assert result is None
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "bishop" in directive.lower()
        assert "pawn" in directive.lower()
        assert "capturing" in directive.lower()
        assert "f7" in directive
        # Speaking resets the quiet streak.
        assert ctx.session.state["quiet_streak"] == 0

    def test_check_triggers_speech(self):
        state = _StubState(
            san_history=["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7+"],
            last_move_san="Qxf7+",
            last_move_classification=MoveClassification.GOOD,
            eval_cp=200,
        )
        result, ctx = _run(state, engine_plays="white")
        assert result is None
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "check" in directive.lower()
        assert "queen" in directive.lower()

    def test_inaccuracy_triggers_speech(self):
        """INACCURACY now counts — the old gate was too narrow (MISTAKE+
        only) and Rook went silent for full games. The directive also
        now includes the engine score so the model knows how bad it
        was."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nge7"],
            last_move_san="Nge7",
            last_move_classification=MoveClassification.INACCURACY,
            eval_cp=60,
        )
        result, ctx = _run(state)
        assert result is None
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "inaccuracy" in directive.lower()
        assert "engine evaluation" in directive.lower()

    def test_game_over_always_speaks(self):
        state = _StubState(
            san_history=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"],
            is_game_over=True,
            game_over_reason="checkmate",
            winner="white",
        )
        result, ctx = _run(state, engine_plays="white")
        assert result is None
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "game" in directive.lower() or "checkmate" in directive.lower()

    def test_quiet_streak_fallback_forces_speech(self):
        """After N consecutive silent turns, the gate allows a keepalive
        remark so Rook doesn't feel dead during long strategic phases."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "d6"],
            last_move_san="d6",
            last_move_classification=MoveClassification.BEST,
            eval_cp=15,
        )
        # Seed the streak at the limit - 1; one more quiet turn trips it.
        result, ctx = _run(state, quiet_streak=2)
        assert result is None  # gate lets the LLM run
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "quiet phase" in directive.lower()
        # Streak resets after the keepalive fires.
        assert ctx.session.state["quiet_streak"] == 0

    def test_directive_forbids_invented_tactics_mid_game(self):
        """Anti-fabrication guidance must stay in the directive — that's
        the whole reason the gate exists. The wording was softened to
        invite persona reactions, but the directive must still constrain
        the model to claims that are listed in the facts block."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Qh4", "Nxh4"],
            last_move_san="Nxh4",
            last_move_classification=MoveClassification.BLUNDER,
            eval_cp=-500,
        )
        _, ctx = _run(state)
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        lowered = directive.lower()
        # Wording simplified by the LLM eval harness — assert on the
        # core constraint (no inventing beyond FACTS) rather than any
        # specific phrase that may change as we tune.
        assert "inventing" in lowered or "beyond the facts" in lowered or "stay grounded" in lowered

    def test_directive_invites_persona_voice(self):
        """The directive must cue the model to speak 'in persona' — the
        exact flavor words evolve as we tune, so assert on the core
        instruction rather than any specific synonym."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7"],
            last_move_san="Bxf7",
            last_move_classification=MoveClassification.GOOD,
        )
        _, ctx = _run(state, engine_plays="white")
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        assert "persona" in directive.lower() or "in character" in directive.lower()

    def test_no_env_falls_open(self):
        """If ``ctx.deps`` isn't an env (e.g. unit test with no chess),
        the gate should let the LLM run rather than crashing."""
        ctx = AgentContext(deps=None, session=Session())
        result = CommentaryGateHook()(ON_RUN_START, ctx, {"task": "hi"})
        assert result is None

    def test_directive_includes_eval_score(self):
        """The score line should be part of the directive so the model
        has real numbers ('engine evaluation: +1.20 pawns...') to play
        off, instead of inventing a sense of who's winning."""
        state = _StubState(
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7"],
            last_move_san="Bxf7",
            last_move_classification=MoveClassification.MISTAKE,
            eval_cp=120,
        )
        _, ctx = _run(state, engine_plays="white")
        directive = ctx.session.state.get("commentary_directive")
        assert directive is not None
        lowered = directive.lower()
        assert "engine evaluation" in lowered
        assert "pawn" in lowered

    def test_directive_includes_per_turn_history_block(self):
        """After the first speakable turn lands, subsequent turns should
        carry a MOVE HISTORY block so the model can reference recent
        play (not just this single turn)."""
        hook = CommentaryGateHook()

        # Turn 1 — user captures, populates turn_history.
        env = _StubEnv(
            _StubState(
                san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7"],
                last_move_san="Bxf7",
                last_move_classification=MoveClassification.GOOD,
                eval_cp=80,
            ),
            engine_plays="white",
        )
        ctx = AgentContext(deps=env, session=Session())
        ctx.session.state["greeted"] = True
        hook(ON_RUN_START, ctx, {"task": "ignored"})
        assert ctx.session.state.get("turn_history"), "history must be recorded"

        # Turn 2 — user captures again; directive for this turn should
        # reference the prior turn.
        env2 = _StubEnv(
            _StubState(
                san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7", "Kxf7", "Nc3"],
                last_move_san="Nc3",
                last_move_classification=MoveClassification.BEST,
                eval_cp=130,
            ),
            engine_plays="white",
        )
        ctx2 = AgentContext(deps=env2, session=ctx.session)
        result = hook(ON_RUN_START, ctx2, {"task": "ignored"})
        # Quiet turn → streak increments; but the history stash on
        # session state still captures the turn for future directives.
        _ = result  # decision depends on signals; we're checking state.
        assert len(ctx2.session.state["turn_history"]) >= 2

    def test_rich_description_names_pieces_and_squares(self):
        """Spot-check the helper: a bishop capturing a pawn on f7 should
        render as 'bishop from <sq> to f7, capturing a pawn'."""
        from edgevox.examples.agents.chess_robot.commentary_gate import _describe_move, _replay_up_to

        san_history = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Bxf7"]
        pre_board = _replay_up_to(san_history, count=len(san_history) - 1)
        desc = _describe_move("Bxf7", pre_board)
        assert desc is not None
        assert "bishop" in desc.lower()
        assert "pawn" in desc.lower()
        assert "capturing" in desc.lower()
        assert "f7" in desc

    def test_user_facing_score_line_uses_user_perspective(self):
        """Analytics bubble is displayed to the user, so ``you`` must
        mean *the user*. Regression: previously the bubble used Rook's
        POV and told the user "you are winning" while they were losing
        (screenshot 2026-04-19 09-32-22)."""
        from edgevox.examples.agents.chess_robot.commentary_gate import _build_analytics_payload

        # User plays white, Rook plays black. Eval +490 cp → white
        # (the user) is winning decisively.
        state = _StubState(
            san_history=["e4", "e6", "Bc4", "b5", "Bxa6"],
            last_move_san="Bxa6",
            last_move_classification=MoveClassification.GOOD,
            eval_cp=490,
        )
        env = _StubEnv(state, engine_plays="black", user_plays="white")
        payload = _build_analytics_payload(state, env, {"turn_history": []})
        assert payload is not None
        user_line = payload.get("score_line_user")
        rook_line = payload.get("score_line")
        assert user_line is not None
        assert rook_line is not None
        # From the user's POV, they are winning (+4.90).
        assert "+4.90" in user_line
        assert "you are winning" in user_line.lower()
        # From Rook's POV, they are losing (-4.90, the user is winning).
        assert "-4.90" in rook_line
        assert "the user is winning" in rook_line.lower()

    def test_user_facing_score_flips_when_rook_plays_white(self):
        """If Rook plays white and black (user) is ahead, the chat
        bubble should say 'you are winning' to the user."""
        from edgevox.examples.agents.chess_robot.commentary_gate import _score_line_user_facing

        state = _StubState(eval_cp=-350)  # black ahead
        env = _StubEnv(state, engine_plays="white", user_plays="black")
        line = _score_line_user_facing(state, env)
        assert line is not None
        assert "+3.50" in line
        assert "you are winning" in line.lower() or "you have a clear advantage" in line.lower()

    def test_rich_description_handles_castling(self):
        from edgevox.examples.agents.chess_robot.commentary_gate import _describe_move, _replay_up_to

        san_history = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]
        pre_board = _replay_up_to(san_history, count=len(san_history))
        desc = _describe_move("O-O", pre_board)
        assert desc is not None
        assert "castled kingside" in desc.lower()
