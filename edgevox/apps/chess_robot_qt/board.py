"""Chess board widget — QGraphicsView with MIT-licensed SVG pieces.

Click-to-move: click a piece, then click a legal destination square.
Dots appear on legal quiet squares; captures get a red ring. Moves
are emitted as UCI and forwarded to :meth:`RookBridge.submit_text` by
the parent window — identical path to voice / typed input, so the
same ``MoveInterceptHook`` applies each move to the environment.

Piece set: Maurizio Monge's "fantasy" (MIT) — see
``assets/pieces/ATTRIBUTION.md``. SVGs are rendered once each at the
current square size via ``QSvgRenderer`` and cached as ``QPixmap`` so
redraws are cheap even on busy animations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chess
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
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
_COORD = QColor("#8d7a64")

_PIECE_DIR = Path(__file__).resolve().parent / "assets" / "pieces"


def _piece_filename(symbol: str) -> str:
    """SAN symbol → lichess-style filename. ``K`` → ``wK.svg``, ``k`` → ``bK.svg``."""
    colour = "w" if symbol.isupper() else "b"
    return f"{colour}{symbol.upper()}.svg"


class ChessBoardView(QGraphicsView):
    """Renders a FEN, lets the user click moves, emits them as UCI."""

    move_requested = Signal(str)  # UCI like "e2e4" or "e7e8q"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")

        self._state: ChessState | None = None
        self._board = chess.Board()
        self._selected: str | None = None
        self._legal: dict[str, bool] = {}  # target → is_capture
        self._orientation: str = "white"

        # Preload and cache piece SVG renderers. Pixmap cache is keyed
        # by (symbol, size) so resizing regenerates once per size.
        self._renderers: dict[str, QSvgRenderer] = {}
        self._pixmaps: dict[tuple[str, int], QPixmap] = {}
        for symbol in "KQRBNPkqrbnp":
            path = _PIECE_DIR / _piece_filename(symbol)
            if path.is_file():
                self._renderers[symbol] = QSvgRenderer(str(path))

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

    # ----- layout + rendering -----

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()

    def _square_size(self) -> float:
        return float(min(self.viewport().width(), self.viewport().height()) / 8)

    def _sq_at_point(self, x: float, y: float) -> str | None:
        sz = self._square_size()
        if sz <= 0:
            return None
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

    def _piece_pixmap(self, symbol: str, size: int) -> QPixmap | None:
        """Cached pixmap of the piece's SVG at the target square size."""
        key = (symbol, size)
        cached = self._pixmaps.get(key)
        if cached is not None:
            return cached
        renderer = self._renderers.get(symbol)
        if renderer is None:
            return None
        # Draw pieces slightly inset so they don't touch the square edges.
        pm = QPixmap(QSize(size, size))
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # 6% inset for breathing room.
        inset = size * 0.06
        renderer.render(painter, QRectF(inset, inset, size - 2 * inset, size - 2 * inset))
        painter.end()
        self._pixmaps[key] = pm
        return pm

    def _redraw(self) -> None:
        self._scene.clear()
        sz = self._square_size()
        if sz <= 0:
            return
        self._scene.setSceneRect(0, 0, sz * 8, sz * 8)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        last_from = last_to = None
        if self._state and self._state.last_move_uci:
            uci = self._state.last_move_uci
            last_from, last_to = uci[0:2], uci[2:4]

        # Squares + last-move + selection tints.
        for rank in range(8):
            for file in range(8):
                sq = chess.square_name(chess.square(file, rank))
                rect = self._sq_to_rect(sq)
                base = _LIGHT if (file + rank) % 2 else _DARK
                self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(base))
                if sq in (last_from, last_to):
                    self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(_LAST_MOVE))
                if self._selected == sq:
                    self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(_SELECTED))

        # Legal-move hints.
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

        # Check highlight.
        if self._board.is_check():
            king_sq = self._board.king(self._board.turn)
            if king_sq is not None:
                rect = self._sq_to_rect(chess.square_name(king_sq))
                pen = QPen(_CHECK)
                pen.setWidthF(sz * 0.06)
                self._scene.addRect(rect, pen, QBrush(Qt.BrushStyle.NoBrush))

        # Pieces — render each SVG into a cached pixmap.
        piece_size = int(sz)
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            if piece is None:
                continue
            pm = self._piece_pixmap(piece.symbol(), piece_size)
            if pm is None:
                continue
            item = self._scene.addPixmap(pm)
            rect = self._sq_to_rect(chess.square_name(square))
            item.setPos(rect.x(), rect.y())

        # Coordinate labels — a/h files along the bottom edge, 1-8 on
        # the left. Tiny, in-board, lichess style.
        label_font = QFont("monospace", max(7, int(sz * 0.13)))
        for i in range(8):
            file_label = "abcdefgh"[i] if self._orientation == "white" else "abcdefgh"[7 - i]
            t = self._scene.addText(file_label, label_font)
            t.setDefaultTextColor(_COORD)
            t.setPos(i * sz + sz * 0.72, 8 * sz - sz * 0.28)
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
            # If it's not the user's turn (engine to move), don't let
            # them click — the bridge will reject the move anyway.
            if self._state is not None and self._state.turn != self._orientation:
                return super().mousePressEvent(event)
            self._selected = sq
            self._legal = {
                chess.square_name(m.to_square): self._board.is_capture(m)
                for m in self._board.legal_moves
                if chess.square_name(m.from_square) == sq
            }
            self._redraw()
            return

        if sq == self._selected:
            self._selected = None
            self._legal = {}
            self._redraw()
            return

        if sq in self._legal:
            uci = f"{self._selected}{sq}"
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

        piece = self._board.piece_at(chess.parse_square(sq))
        if piece is not None and piece.color == self._board.turn:
            self._selected = sq
            self._legal = {
                chess.square_name(m.to_square): self._board.is_capture(m)
                for m in self._board.legal_moves
                if chess.square_name(m.from_square) == sq
            }
            self._redraw()
            return

        self._selected = None
        self._legal = {}
        self._redraw()


__all__ = ["ChessBoardView"]
