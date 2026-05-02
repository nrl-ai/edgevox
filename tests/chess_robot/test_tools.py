"""Tests for the conversational chess tool surface.

Covers ``build_chess_tools`` end-to-end: the closure factory wiring,
each tool's behaviour against a real :class:`ChessEnvironment` (via
the FakeEngine fixture), and the structured shapes the LLM downstream
will see in tool results.

Live LLM dispatch is not exercised here — the agent loop is covered
by harness tests; this module proves each tool is correct so the
LLM has a reliable contract to call against.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from edgevox.examples.agents.chess_robot.tools import build_chess_tools


@dataclass
class _StubConfig:
    user_plays: str = "white"
    persona: str = "casual"


class _StubBridge:
    """Minimal bridge surface for tools to close over.

    Captures ``set_persona`` calls so the test can assert dispatch
    happened. ``env`` is set externally per test (``bridge.env = env``)
    so we share the conftest ``ChessEnvironment`` fixture.
    """

    def __init__(self) -> None:
        self.env = None
        self.config = _StubConfig()
        self.persona_calls: list[str] = []

    def set_persona(self, name: str) -> None:
        self.persona_calls.append(name)
        self.config.persona = name


@pytest.fixture
def bridge(env):
    b = _StubBridge()
    b.env = env
    return b


@pytest.fixture
def tools(bridge):
    return {_tool_name(t): t for t in build_chess_tools(bridge)}


def _tool_name(t) -> str:
    schema = getattr(t, "__edgevox_tool__", None)
    return schema.name if schema else t.__name__


# --- registry shape ------------------------------------------------------


def test_factory_returns_expected_tool_set(bridge):
    names = sorted(_tool_name(t) for t in build_chess_tools(bridge))
    assert names == [
        "chitchat",
        "get_legal_moves",
        "get_position",
        "get_top_moves",
        "new_game",
        "play_move",
        "resign",
        "set_persona",
        "undo_move",
    ]


def test_every_tool_has_a_schema(bridge):
    for t in build_chess_tools(bridge):
        schema = getattr(t, "__edgevox_tool__", None)
        assert schema is not None, f"{t.__name__} missing @tool decoration"
        assert schema.description, f"{schema.name} missing description (docstring)"


# --- play_move -----------------------------------------------------------


def test_play_move_san(tools, env):
    result = tools["play_move"]("e4")
    assert "user played e4" in result
    assert "engine replied" in result
    # Engine reply should land on the board.
    assert env.snapshot().ply == 2


def test_play_move_uci(tools, env):
    result = tools["play_move"]("e2e4")
    assert "user played e4" in result
    assert env.snapshot().ply == 2


def test_play_move_illegal(tools, env):
    result = tools["play_move"]("e5")  # not a legal first move for white
    assert result.startswith("illegal:")
    assert env.snapshot().ply == 0


def test_play_move_with_no_env_returns_safe_string(bridge):
    bridge.env = None
    tools_no_env = {_tool_name(t): t for t in build_chess_tools(bridge)}
    assert tools_no_env["play_move"]("e4") == "board not loaded yet"


# --- get_position --------------------------------------------------------


def test_get_position_initial(tools):
    pos = tools["get_position"]()
    assert pos["turn"] == "white"
    assert pos["user_plays"] == "white"
    assert pos["engine_plays"] == "black"
    assert pos["ply"] == 0
    assert pos["is_game_over"] is False
    assert "fen" in pos


def test_get_position_reflects_played_move(tools):
    tools["play_move"]("e4")
    pos = tools["get_position"]()
    assert pos["ply"] == 2
    assert pos["last_move_san"] is not None


# --- get_top_moves -------------------------------------------------------


def test_get_top_moves_returns_at_least_one(tools):
    moves = tools["get_top_moves"](3)
    assert len(moves) >= 1
    assert "move_uci" in moves[0]
    assert "san" in moves[0]


def test_get_top_moves_clamps_n_to_range(tools):
    # n=99 should clamp to 5 max — but the FakeEngine's PV may be
    # shorter, so we just assert no crash and bounded length.
    moves = tools["get_top_moves"](99)
    assert 1 <= len(moves) <= 5


# --- get_legal_moves -----------------------------------------------------


def test_get_legal_moves_initial_is_twenty(tools):
    moves = tools["get_legal_moves"]()
    assert len(moves) == 20  # standard opening position


# --- undo_move -----------------------------------------------------------


def test_undo_pops_engine_then_user(tools, env):
    tools["play_move"]("e4")
    assert env.snapshot().ply == 2
    result = tools["undo_move"]()
    assert "undone" in result
    assert env.snapshot().ply == 0


def test_undo_with_nothing_to_undo(tools):
    result = tools["undo_move"]()
    assert "nothing to undo" in result


# --- new_game ------------------------------------------------------------


def test_new_game_resets_board(tools, env, bridge):
    tools["play_move"]("e4")
    assert env.snapshot().ply > 0
    result = tools["new_game"]("white")
    assert "new game" in result
    assert env.snapshot().ply == 0
    assert bridge.config.user_plays == "white"


def test_new_game_black_makes_engine_open(tools, env, bridge):
    result = tools["new_game"]("black")
    assert "engine opened" in result
    # After engine opens, ply=1 (one half-move).
    assert env.snapshot().ply == 1
    assert bridge.config.user_plays == "black"


# --- resign --------------------------------------------------------------


def test_resign_starts_a_new_game(tools, env):
    tools["play_move"]("e4")
    assert env.snapshot().ply == 2
    result = tools["resign"]()
    assert "resigned" in result
    # Resignation resets the board so the user can keep playing.
    assert env.snapshot().ply == 0


# --- set_persona ---------------------------------------------------------


def test_set_persona_dispatches_to_bridge(tools, bridge):
    result = tools["set_persona"]("grandmaster")
    assert "persona switched to grandmaster" in result
    assert bridge.persona_calls == ["grandmaster"]


# --- chitchat ------------------------------------------------------------


def test_chitchat_is_noop_returning_ok(tools):
    assert tools["chitchat"]() == "ok"
