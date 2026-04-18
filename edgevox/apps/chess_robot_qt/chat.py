"""Chat transcript widget.

Shows user turns, rook's replies, and engine-move chips in one
scrollable list. Auto-scrolls to the newest entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ChatEntry:
    role: str  # "user" | "rook" | "rook-move" | "user-move" | "system"
    text: str


class ChatView(QScrollArea):
    """Scrollable column of bubbles + move chips."""

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: rgba(10, 14, 22, 0.5); border-radius: 8px;")
        self._accent = accent

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        # Placeholder shown when empty.
        self._placeholder = QLabel("talk or type a move to start…")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #5b6a7d; font-style: italic; font-size: 12px;")
        self._layout.insertWidget(0, self._placeholder)

    # ----- API -----

    def set_accent(self, accent: QColor) -> None:
        self._accent = accent
        # Re-colour existing rook labels.
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if isinstance(widget, _Bubble):
                widget.update_accent(accent)

    def add_user(self, text: str) -> None:
        self._add(_Bubble("user", text, self._accent))

    def add_rook(self, text: str) -> None:
        self._add(_Bubble("rook", text, self._accent))

    def add_move_chip(self, who: str, san: str) -> None:
        """``who`` is ``'you'`` or ``'rook'`` — compact chip in the middle."""
        self._add(_MoveChip(who, san, self._accent))

    def clear(self) -> None:
        # Drop everything except the trailing stretch + placeholder.
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None and w is not self._placeholder:
                w.setParent(None)
        if self._placeholder.parent() is None:
            self._layout.insertWidget(0, self._placeholder)
        self._placeholder.show()

    # ----- internals -----

    def _add(self, w: QWidget) -> None:
        # Hide placeholder once the first real entry lands.
        if self._placeholder is not None:
            self._placeholder.hide()
        # Insert before the trailing stretch (last item).
        self._layout.insertWidget(self._layout.count() - 1, w)
        # Auto-scroll on next tick so the new widget has a height.
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


class _Bubble(QWidget):
    """One chat bubble (user or rook)."""

    def __init__(self, role: str, text: str, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._role = role
        self._accent = accent
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label_label = QLabel("YOU" if role == "user" else "ROOK")
        label_label.setStyleSheet(self._label_style(self._accent if role == "rook" else QColor("#9aa7b9")))
        col.addWidget(label_label, alignment=self._align())

        self._body = QLabel(text)
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._body.setMaximumWidth(420)
        self._body.setStyleSheet(self._bubble_style(role, accent))
        col.addWidget(self._body, alignment=self._align())

    def update_accent(self, accent: QColor) -> None:
        self._accent = accent
        if self._role == "rook":
            self._body.setStyleSheet(self._bubble_style(self._role, accent))

    def _align(self):
        return Qt.AlignmentFlag.AlignRight if self._role == "user" else Qt.AlignmentFlag.AlignLeft

    @staticmethod
    def _label_style(c: QColor) -> str:
        return f"color: {c.name()}; font-family: monospace; font-size: 10px; letter-spacing: 1px;"

    @staticmethod
    def _bubble_style(role: str, accent: QColor) -> str:
        if role == "user":
            return (
                "background: rgba(52, 211, 153, 0.14); "
                "border: 1px solid rgba(52, 211, 153, 0.28); "
                "color: #dbe4ef; padding: 7px 11px; border-radius: 10px; "
                "font-size: 13px;"
            )
        r, g, b = accent.red(), accent.green(), accent.blue()
        return (
            f"background: rgba({r}, {g}, {b}, 0.08); "
            f"border: 1px solid rgba({r}, {g}, {b}, 0.4); "
            "color: #dbe4ef; padding: 7px 11px; border-radius: 10px; "
            "font-size: 13px;"
        )


class _MoveChip(QWidget):
    """Compact centred chip like `rook played Nf3`."""

    def __init__(self, who: str, san: str, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)

        label = QLabel(f"{who} played  {san}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        accent_str = accent.name() if who == "rook" else "#8be2b6"
        r, g, b = accent.red(), accent.green(), accent.blue()
        bg = f"rgba({r}, {g}, {b}, 0.1)" if who == "rook" else "rgba(52, 211, 153, 0.14)"
        label.setStyleSheet(
            f"color: {accent_str}; background: {bg}; "
            f"padding: 3px 10px; border-radius: 9px; border: 1px solid {accent_str}40; "
            "font-family: monospace; font-size: 11px;"
        )
        row.addWidget(label)
        row.addStretch(1)


__all__ = ["ChatEntry", "ChatView"]


# Palette consumers are unused here but imported to satisfy linters.
_ = QPalette
