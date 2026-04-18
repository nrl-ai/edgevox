"""Chess partner — voice-controlled chess agent with pluggable engine + persona.

Talk to a chess-playing robot opponent with natural commentary and
live analytics. Three personas shipped:

- ``grandmaster`` — terse 2600 Stockfish, cites openings
- ``casual`` — chatty Maia @ 1400, explains ideas
- ``trash_talker`` — cocky Maia @ 1800, teases blunders

Launch::

    edgevox-agent chess                                          # full TUI voice
    edgevox-agent chess --simple-ui                              # lightweight CLI voice
    edgevox-agent chess --text-mode                              # keyboard chat
    edgevox-agent chess --persona trash_talker --engine maia     # switch persona/engine
    edgevox-agent chess --user-plays black                       # play black

Requirements:

- ``python-chess`` (installed as a core EdgeVox dependency).
- Stockfish binary on ``$PATH`` for the Stockfish backend
  (``apt install stockfish`` / ``brew install stockfish``).
- For Maia: ``lc0`` on ``$PATH`` + a Maia weight file. Pass
  ``--maia-weights /path/to/maia-1500.pb.gz``.

The current board is rendered in the console / TUI automatically via the
``ChessEnvironment`` state-listener subscription (see ``_pre_run`` below).

Web UI: run the WebSocket server with the chess agent bound::

    edgevox-serve --agent edgevox.examples.agents.chess_partner:build_server_agent
    # or: EDGEVOX_CHESS_PERSONA=trash_talker edgevox-serve --agent ...

The React components under ``webui/src/components/`` (``ChessBoard.tsx``
/ ``EvalBar.tsx`` / ``MoveList.tsx``) pick up ``chess_state`` events
forwarded automatically by the server pipeline and mount a live board
panel next to the chat.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from rich.console import Console

from edgevox.agents import AgentContext, LLMAgent
from edgevox.examples.agents.framework import AgentApp
from edgevox.integrations.chess import (
    BoardStateInjectionHook,
    ChessEnvironment,
    EngineUnavailable,
    MoveCommentaryHook,
    build_engine,
    resolve_persona,
)
from edgevox.integrations.chess.tui import render_board_rich
from edgevox.llm import tool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools — each one takes ``ctx`` and operates on ``ctx.deps`` (ChessEnvironment)
# ---------------------------------------------------------------------------


@tool
def get_board_state(ctx: AgentContext) -> dict[str, Any]:
    """Return the current board: FEN, side-to-move, move history, eval."""
    return ctx.deps.snapshot().to_json()


@tool
def list_legal_moves(ctx: AgentContext) -> list[str]:
    """Return every legal move for the side to move, as UCI strings."""
    return ctx.deps.list_legal_moves()


@tool
def play_user_move(move: str, ctx: AgentContext) -> dict[str, Any]:
    """Apply the user's move to the board.

    Args:
        move: UCI (``e2e4``) or SAN (``e4``, ``Nxd5``). Raises if illegal.
    """
    try:
        state = ctx.deps.play_user_move(move)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "state": state.to_json()}


@tool
def engine_move(ctx: AgentContext) -> dict[str, Any]:
    """Ask the engine to choose and play a move for its side.

    Returns the move in SAN/UCI plus the resulting board state.
    """
    try:
        state, move = ctx.deps.engine_move()
    except (ValueError, EngineUnavailable) as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "move_san": move.san,
        "move_uci": move.uci,
        "eval_cp": move.score_from_white,
        "pv": move.pv,
        "state": state.to_json(),
    }


@tool
def analyze_position(depth: int = 12, *, ctx: AgentContext) -> dict[str, Any]:
    """Run the engine on the current position for commentary / analytics.

    Args:
        depth: search depth plies. 12 is a good default — sub-200 ms
            on CPU and much stronger than the eval the player sees on
            the bar.
    """
    try:
        analysis = ctx.deps.analyse(depth=depth)
    except EngineUnavailable as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "eval_cp": analysis.score_from_white,
        "mate_in": analysis.mate_in,
        "best_move_san": analysis.san,
        "best_move_uci": analysis.uci,
        "pv": analysis.pv,
        "depth": analysis.depth,
    }


@tool
def undo_last_move(ctx: AgentContext) -> dict[str, Any]:
    """Undo the most recent move — useful when the user says 'wait, I meant...'."""
    try:
        state = ctx.deps.undo_last_move()
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "state": state.to_json()}


@tool
def new_game(
    user_plays: str = "white",
    engine_skill: int | None = None,
    *,
    ctx: AgentContext,
) -> dict[str, Any]:
    """Start a fresh game.

    Args:
        user_plays: ``white`` or ``black``.
        engine_skill: 0-20 Stockfish skill level; ignored for Maia.
    """
    state = ctx.deps.new_game(user_plays=user_plays, engine_skill=engine_skill)
    return {"ok": True, "state": state.to_json()}


CHESS_TOOLS = [
    get_board_state,
    list_legal_moves,
    play_user_move,
    engine_move,
    analyze_position,
    undo_last_move,
    new_game,
]


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


def _pre_run(args: argparse.Namespace) -> None:
    """Build the :class:`ChessEnvironment` + attach persona + hooks.

    Runs after arg parsing but before the agent launches. We follow
    the ``robot_panda.py`` pattern: mutate ``APP.deps`` / persona
    in-place so the :class:`AgentApp` construction up top stays a
    declarative one-liner.
    """
    persona = resolve_persona(args.persona)

    # CLI overrides win over persona defaults. Persona engine_options only
    # apply when the effective engine matches the persona's default engine —
    # otherwise we'd forward Maia kwargs (``elo``) into Stockfish (or vice
    # versa) and crash on the unknown kwarg.
    engine_kind = args.engine or persona.engine_kind
    engine_kwargs: dict[str, Any] = dict(persona.engine_options) if engine_kind == persona.engine_kind else {}
    if args.stockfish_skill is not None:
        engine_kwargs["skill"] = args.stockfish_skill
    if args.stockfish_binary:
        engine_kwargs["binary"] = args.stockfish_binary
    if engine_kind == "maia":
        if not args.maia_weights:
            raise SystemExit(
                "--engine maia requires --maia-weights /path/to/maia-XXXX.pb.gz (download from https://maiachess.com/)."
            )
        engine_kwargs["weights"] = args.maia_weights
        if args.maia_elo is not None:
            engine_kwargs["elo"] = args.maia_elo
        if args.lc0_binary:
            engine_kwargs["binary"] = args.lc0_binary

    try:
        engine = build_engine(engine_kind, **engine_kwargs)
    except EngineUnavailable as e:
        raise SystemExit(f"engine unavailable: {e}") from e

    APP.deps = ChessEnvironment(
        engine,
        user_plays=args.user_plays,
        engine_skill=engine_kwargs.get("skill") if engine_kind == "stockfish" else None,
        analyse_depth=args.analyse_depth,
        analyse_time=args.analyse_time,
    )

    # Console board re-render on every state change. Only in text-mode /
    # simple-ui — the full Textual TUI owns its terminal and would clash
    # with raw rich prints. The TUI's own ``ChessPanel`` subscribes
    # through ``EdgeVoxApp._maybe_build_chess_panel``.
    if getattr(args, "text_mode", False) or getattr(args, "simple_ui", False):
        _console = Console()
        APP.deps.subscribe(lambda state: _console.print(render_board_rich(state)))
        # Paint the starting position so the user sees the board immediately
        # rather than only after the first move.
        _console.print(render_board_rich(APP.deps.snapshot()))

    # Persona → system prompt + greeting. The default LLMAgent constructed
    # by :class:`AgentApp.__post_init__` already carries the persona-shaped
    # instructions + tools + hooks we declared at module scope — no need to
    # rebuild it here. We only swap the display-level fields so the banner
    # reflects the chosen persona.
    APP.instructions = persona.system_prompt
    APP.greeting = (
        f"{persona.display_name} ready. I'm playing {APP.deps.engine_plays}; you have {APP.deps.user_plays}. Your move."
    )
    APP.name = f"Chess — {persona.display_name}"
    # Rebuild the LLMAgent so its ``instructions`` match the selected
    # persona (the placeholder one was built at import time).
    APP.agent = LLMAgent(
        name=APP.name,
        description="Voice-controlled chess partner.",
        instructions=persona.system_prompt,
        tools=CHESS_TOOLS,
        hooks=[BoardStateInjectionHook(), MoveCommentaryHook()],
    )


APP = AgentApp(
    name="Chess Partner",
    description="Voice-controlled chess with pluggable engines (Stockfish / Maia) and personas.",
    instructions="You are a chess partner.",  # replaced in _pre_run
    tools=CHESS_TOOLS,
    hooks=[BoardStateInjectionHook(), MoveCommentaryHook()],
    deps=None,  # filled in by _pre_run
    stop_words=("stop", "halt", "freeze", "abort", "resign"),
    greeting="Setting up the chess board…",
    extra_args=[
        (
            ("--persona",),
            {
                "default": "casual",
                "choices": ["grandmaster", "casual", "trash_talker"],
                "help": "Chess opponent personality + default engine (default: casual).",
            },
        ),
        (
            ("--engine",),
            {
                "default": None,
                "choices": ["stockfish", "maia"],
                "help": "Override the persona's default engine backend.",
            },
        ),
        (
            ("--user-plays",),
            {"default": "white", "choices": ["white", "black"], "help": "Which side the user plays."},
        ),
        (
            ("--stockfish-skill",),
            {"type": int, "default": None, "help": "Stockfish skill level 0-20 (overrides persona)."},
        ),
        (
            ("--stockfish-binary",),
            {"default": None, "help": "Path to stockfish executable (default: $PATH)."},
        ),
        (
            ("--maia-weights",),
            {"default": None, "help": "Path to Maia .pb.gz weights file (required for --engine maia)."},
        ),
        (("--maia-elo",), {"type": int, "default": None, "help": "Label only — weights decide real strength."}),
        (("--lc0-binary",), {"default": None, "help": "Path to lc0 executable (default: $PATH)."}),
        (
            ("--analyse-depth",),
            {"type": int, "default": 12, "help": "Search depth for analysis/classification (default: 12)."},
        ),
        (
            ("--analyse-time",),
            {"type": float, "default": 0.5, "help": "Seconds the engine spends choosing its move."},
        ),
    ],
    pre_run=_pre_run,
)


def build_server_agent(core) -> tuple[LLMAgent, ChessEnvironment]:
    """Factory for ``edgevox-serve --agent edgevox.examples.agents.chess_partner:build_server_agent``.

    Builds a chess agent + environment for the WebSocket server using
    environment variables for configuration (same knobs as the CLI):

    - ``EDGEVOX_CHESS_PERSONA`` — one of ``grandmaster`` / ``casual`` / ``trash_talker``
      (default: ``casual``).
    - ``EDGEVOX_CHESS_ENGINE`` — ``stockfish`` (default) or ``maia``.
    - ``EDGEVOX_CHESS_USER_PLAYS`` — ``white`` (default) or ``black``.
    - ``EDGEVOX_CHESS_STOCKFISH_SKILL`` — 0-20.
    - ``EDGEVOX_CHESS_MAIA_WEIGHTS`` — path, required when engine=maia.

    The returned ``LLMAgent`` is unbound; ``edgevox-serve`` binds its LLM
    to :class:`ServerCore.llm` before accepting connections.
    """
    import os

    persona_slug = os.environ.get("EDGEVOX_CHESS_PERSONA", "casual")
    engine_kind = os.environ.get("EDGEVOX_CHESS_ENGINE")
    user_plays = os.environ.get("EDGEVOX_CHESS_USER_PLAYS", "white")

    persona = resolve_persona(persona_slug)
    engine_kind = engine_kind or persona.engine_kind
    engine_kwargs: dict[str, Any] = dict(persona.engine_options) if engine_kind == persona.engine_kind else {}
    if skill := os.environ.get("EDGEVOX_CHESS_STOCKFISH_SKILL"):
        engine_kwargs["skill"] = int(skill)
    if engine_kind == "maia":
        weights = os.environ.get("EDGEVOX_CHESS_MAIA_WEIGHTS")
        if not weights:
            raise EngineUnavailable("EDGEVOX_CHESS_MAIA_WEIGHTS env var required for maia backend")
        engine_kwargs["weights"] = weights

    engine = build_engine(engine_kind, **engine_kwargs)
    env = ChessEnvironment(engine, user_plays=user_plays)
    agent = LLMAgent(
        name=f"Chess — {persona.display_name}",
        description="Voice-controlled chess partner.",
        instructions=persona.system_prompt,
        tools=CHESS_TOOLS,
        hooks=[BoardStateInjectionHook(), MoveCommentaryHook()],
    )
    del core  # explicit: factory doesn't need the core reference today
    return agent, env


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
