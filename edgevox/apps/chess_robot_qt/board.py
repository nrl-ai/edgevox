"""Chess board widget — QGraphicsView with Unicode pieces.

Click-to-move: click a piece, then click a legal destination square.
Dots appear on legal squares; captures get a red ring. Moves are
applied via :meth:`RookBridge.submit_text` using UCI — the same entry
point as voice/typed moves, so MoveInterceptHook handles everything
downstream and the LLM narrates.

We keep Unicode piece glyphs rather than SVGs to stay MIT-clean; the
Cburnett set is lovely but CC-BY-SA which some users prefer to avoid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

if TYPE_CHECKING:
    from edgevox.integrations.chess.environment import ChessState


_LIGHT = QColor("#ead8ba")
_DARK = QColor("#a87d5a")
_SELECTED = QColor(255, 215, 0, 120)
_LAST_MOVE = QColor(88, 170, 255, 90)
_LEGAL_DOT = QColor(52, 211, 153, 180)
_LEGAL_CAPTURE = QColor(239, 68, 68, 180)
_CHECK = QColor(239, 68, 68, 160)
_COORD = QColor("#6a7a8d")

_UNICODE = {
    "K": "\u2654",
    "Q": "\u2655",
    "R": "\u2656",
    "B": "\u2657",
    "N": "\u2658",
    "P": "\u2659",
    "k": "\u265a",
    "q": "\u265b",
    "r": "\u265c",
    "b": "\u265d",
    "n": "\u265e",
    "p": "\u265f",
}


class ChessBoardView(QGraphicsView):
    """Renders a FEN, lets the user click moves, emits them as UCI."""

    move_requested = Signal(str)  # UCI like "e2e4" or "e7e8q"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")

        self._state: ChessState | None = None
        self._board = chess.Board()
        self._selected: str | None = None
        self._legal: dict[str, bool] = {}  # target → is_capture
        self._orientation: str = "white"

    # ----- public API -----

    def set_state(self, state: ChessState) -> None:
        """Snap the rendered position + clear any stale selection."""
        self._state = state
        try:
            self._board = chess.Board(state.fen)
        except ValueError:
            self._board = chess.Board()
        self._selected = None
        self._legal = {}
        self._redraw()

    def set_orientation(self, side: str) -> None:
        self._orientation = "black" if side.lower().startswith("b") else "white"
        self._redraw()

    def set_enabled_for_user(self, enabled: bool) -> None:
        """Toggle click acceptance (e.g. disable while Rook is thinking)."""
        self._interactive = enabled

    # ----- layout + rendering -----

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    def _square_size(self) -> float:
        # Keep the board a centred square inside the view.
        return float(min(self.viewport().width(), self.viewport().height()) / 8)

    def _sq_at_point(self, x: float, y: float) -> str | None:
        sz = self._square_size()
        file_idx = int(x // sz)
        rank_idx = int(y // sz)
        if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
            return None
        if self._orientation == "white":
            rank = 7 - rank_idx
            file = file_idx
        else:
            rank = rank_idx
            file = 7 - file_idx
        return chess.square_name(chess.square(file, rank))

    def _sq_to_rect(self, square: str) -> QRectF:
        sz = self._square_size()
        file = ord(square[0]) - ord("a")
        rank = int(square[1]) - 1
        if self._orientation == "white":
            x = file * sz
            y = (7 - rank) * sz
        else:
            x = (7 - file) * sz
            y = rank * sz
        return QRectF(x, y, sz, sz)

    def _redraw(self) -> None:
        self._scene.clear()
        sz = self._square_size()
        if sz <= 0:
            return
        self._scene.setSceneRect(0, 0, sz * 8, sz * 8)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Squares
        last_from = last_to = None
        if self._state and self._state.last_move_uci:
            uci = self._state.last_move_uci
            last_from, last_to = uci[0:2], uci[2:4]

        for rank in range(8):
            for file in range(8):
                sq = chess.square_name(chess.square(file, rank))
                rect = self._sq_to_rect(sq)
                base = _LIGHT if (file + rank) % 2 else _DARK
                self._scene.addRect(
                    rect,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(base),
                )
                if sq in (last_from, last_to):
                    self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(_LAST_MOVE))
                if self._selected == sq:
                    self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(_SELECTED))

        # Legal-move hints
        for target, is_capture in self._legal.items():
            rect = self._sq_to_rect(target)
            if is_capture:
                pen = QPen(_LEGAL_CAPTURE)
                pen.setWidthF(sz * 0.06)
                ring = rect.adjusted(sz * 0.08, sz * 0.08, -sz * 0.08, -sz * 0.08)
                self._scene.addEllipse(ring, pen, QBrush(Qt.BrushStyle.NoBrush))
            else:
                dot = rect.adjusted(sz * 0.35, sz * 0.35, -sz * 0.35, -sz * 0.35)
                self._scene.addEllipse(dot, QPen(Qt.PenStyle.NoPen), QBrush(_LEGAL_DOT))

        # Check highlight on own king
        if self._board.is_check():
            king_sq = self._board.king(self._board.turn)
            if king_sq is not None:
                king_name = chess.square_name(king_sq)
                rect = self._sq_to_rect(king_name)
                pen = QPen(_CHECK)
                pen.setWidthF(sz * 0.06)
                self._scene.addRect(rect, pen, QBrush(Qt.BrushStyle.NoBrush))

        # Pieces
        font = QFont("DejaVu Sans", int(sz * 0.68))
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            if piece is None:
                continue
            glyph = _UNICODE.get(piece.symbol(), "?")
            rect = self._sq_to_rect(chess.square_name(square))
            text = self._scene.addText(glyph, font)
            br = text.boundingRect()
            text.setPos(
                rect.x() + (rect.width() - br.width()) / 2,
                rect.y() + (rect.height() - br.height()) / 2,
            )
            text.setDefaultTextColor(QColor("#111") if piece.color else QColor("#f8f8f8"))

        # Coordinate labels — a / h files on bottom edge, 1 / 8 on sides.
        label_font = QFont("monospace", int(sz * 0.14))
        for i in range(8):
            file_label = "abcdefgh"[i] if self._orientation == "white" else "abcdefgh"[7 - i]
            t = self._scene.addText(file_label, label_font)
            t.setDefaultTextColor(_COORD)
            t.setPos(i * sz + sz * 0.08, 8 * sz - sz * 0.25)
            rank_label = str(8 - i) if self._orientation == "white" else str(i + 1)
            rt = self._scene.addText(rank_label, label_font)
            rt.setDefaultTextColor(_COORD)
            rt.setPos(sz * 0.04, i * sz + sz * 0.02)

    # ----- interaction -----

    def mousePressEvent(self, event) -> None:
        pos = self.mapToScene(event.position().toPoint())
        sq = self._sq_at_point(pos.x(), pos.y())
        if sq is None:
            return super().mousePressEvent(event)

        if self._selected is None:
            piece = self._board.piece_at(chess.parse_square(sq))
            if piece is None or piece.color != self._board.turn:
                return super().mousePressEvent(event)
            # If the engine is to move, don't allow the user to click.
            # We surface interactivity by whoever is to move == user side.
            if self._state is not None:
                user_is_white = self._state.turn == "white" and self._is_user_turn()
                user_is_black = self._state.turn == "black" and self._is_user_turn()
                if not (user_is_white or user_is_black):
                    return super().mousePressEvent(event)
            self._selected = sq
            self._legal = {}
            for move in self._board.legal_moves:
                if chess.square_name(move.from_square) == sq:
                    target = chess.square_name(move.to_square)
                    is_capture = self._board.is_capture(move)
                    self._legal[target] = is_capture
            self._redraw()
            return

        if sq == self._selected:
            self._selected = None
            self._legal = {}
            self._redraw()
            return

        if sq in self._legal:
            uci = f"{self._selected}{sq}"
            # Pawn promotion → queen by default. Users who want to
            # under-promote can speak "promote to knight" etc.
            src_piece = self._board.piece_at(chess.parse_square(self._selected))
            if (
                src_piece
                and src_piece.piece_type == chess.PAWN
                and ((src_piece.color and sq[1] == "8") or (not src_piece.color and sq[1] == "1"))
            ):
                uci += "q"
            self._selected = None
            self._legal = {}
            self._redraw()
            self.move_requested.emit(uci)
            return

        # Clicked another piece of the same colour → re-select.
        piece = self._board.piece_at(chess.parse_square(sq))
        if piece is not None and piece.color == self._board.turn:
            self._selected = sq
            self._legal = {}
            for move in self._board.legal_moves:
                if chess.square_name(move.from_square) == sq:
                    self._legal[chess.square_name(move.to_square)] = self._board.is_capture(move)
            self._redraw()
            return

        self._selected = None
        self._legal = {}
        self._redraw()

    def _is_user_turn(self) -> bool:
        """Very coarse heuristic: the view is interactive only while
        the user's side has the move. We infer it from ChessState.turn
        + the user_side prop plumbed via ``set_orientation`` for now."""
        if self._state is None:
            return True
        return self._state.turn == self._orientation


# Re-export for downstream type hints without a circular import.
_ = QPixmap


__all__ = ["ChessBoardView"]
