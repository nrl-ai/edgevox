"""Qt widget smoke tests. Runs with the Qt offscreen platform so CI
doesn't need a display."""

from __future__ import annotations

import os

import pytest

# Force offscreen before importing Qt — must happen at import time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("qtawesome")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from edgevox.apps.chess_robot_qt.board import ChessBoardView
from edgevox.apps.chess_robot_qt.chat import ChatView
from edgevox.apps.chess_robot_qt.face import RobotFaceWidget


class TestRobotFaceWidget:
    def test_constructs(self, qtbot):
        w = RobotFaceWidget()
        qtbot.addWidget(w)
        assert w.minimumWidth() >= 1

    def test_mood_set_accepted(self, qtbot):
        w = RobotFaceWidget()
        qtbot.addWidget(w)
        w.set_mood("triumphant")
        w.set_tempo("speaking")
        w.set_persona("trash_talker")

    def test_invalid_mood_is_ignored(self, qtbot):
        w = RobotFaceWidget()
        qtbot.addWidget(w)
        w.set_mood("bogus")  # no crash, no change


class TestChatView:
    def test_bubbles_and_chips_render(self, qtbot):
        chat = ChatView(QColor("#34d399"))
        qtbot.addWidget(chat)
        chat.add_user("I play e4")
        chat.add_rook("Sicilian bait.")
        chat.add_move_chip("rook", "c5")
        assert chat.widget().layout().count() >= 4  # placeholder + 3 entries + stretch

    def test_clear_removes_entries(self, qtbot):
        chat = ChatView(QColor("#ffb066"))
        qtbot.addWidget(chat)
        chat.add_user("hi")
        chat.add_rook("hello")
        chat.clear()
        # After clear, only the placeholder + the trailing stretch remain.
        # (Layout entries: placeholder at index 0, stretch at end.)
        visible_widgets = []
        layout = chat.widget().layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                visible_widgets.append(w)
        assert len(visible_widgets) == 1  # just the placeholder


class TestChessBoardView:
    def test_constructs_and_sets_state(self, qtbot):
        from edgevox.integrations.chess.environment import ChessState

        view = ChessBoardView()
        qtbot.addWidget(view)
        view.resize(400, 400)
        state = ChessState(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ply=1,
            turn="black",
            last_move_uci="e2e4",
            last_move_san="e4",
            san_history=["e4"],
        )
        view.set_state(state)
        assert view._board.turn is not None

    def test_orientation_toggle(self, qtbot):
        view = ChessBoardView()
        qtbot.addWidget(view)
        view.set_orientation("black")
        assert view._orientation == "black"
        view.set_orientation("white")
        assert view._orientation == "white"


class TestAppBoots:
    """End-to-end check that the full window graph builds without
    crashing. We don't start the bridge (which would load a 1GB LLM);
    we just instantiate the widgets."""

    def test_main_window_builds(self, qtbot, monkeypatch):
        from edgevox.apps.chess_robot_qt.bridge import RookBridge, RookConfig

        # Stub out the bridge so ``window.__init__`` doesn't start a
        # background LLM load. We replace ``start`` with a no-op.
        monkeypatch.setattr(RookBridge, "start", lambda self: None)

        from edgevox.apps.chess_robot_qt.window import RookWindow

        bridge = RookBridge(RookConfig())
        window = RookWindow(bridge)
        qtbot.addWidget(window)
        assert window.windowTitle() == "RookApp"
        # Ensure the frameless hint is applied.
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
