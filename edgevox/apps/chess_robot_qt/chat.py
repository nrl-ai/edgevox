"""Chat transcript widget.

Shows user turns, rook's replies, and engine-move chips in one
scrollable list. Auto-scrolls to the newest entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt, QTimer
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

        # Auto-scroll on any scroll-range change. Whenever the content
        # grows (new bubble, wrap recomputed, window resized), the
        # vertical scrollbar's range expands and we snap to the bottom.
        # This is more robust than a one-shot setValue after insertion
        # because the layout hasn't finished when we do the insert.
        self._auto_scroll = True
        self.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        # Detect manual scroll: if the user drags up, stop hijacking.
        self.verticalScrollBar().valueChanged.connect(self._on_user_scroll)
        self._last_max = 0

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

    def add_debug(self, title: str, body: str) -> None:
        """Render a debug dump as a collapsible monospace bubble.

        Bodies are often multi-kB (full messages array) — capped to a
        reasonable preview with a click-to-expand control so normal
        scrolling stays usable."""
        self._add(_DebugEntry(title, body))

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
        # Re-arm auto-scroll: any explicit add is an app-driven event,
        # so respect it even if the user had scrolled up mid-session.
        self._auto_scroll = True
        # The rangeChanged listener will fire once the layout settles;
        # this backup tick catches the case where range didn't move
        # (wrap-only change on the same row count).
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _on_range_changed(self, _min: int, maximum: int) -> None:
        if not self._auto_scroll:
            self._last_max = maximum
            return
        if maximum != self._last_max:
            self._last_max = maximum
            self.verticalScrollBar().setValue(maximum)

    def _on_user_scroll(self, value: int) -> None:
        bar = self.verticalScrollBar()
        # If the user has scrolled away from the bottom, pause auto-
        # follow; snapping back to the bottom once they catch up is
        # handled by :meth:`_add`.
        if value < bar.maximum() - 4:
            self._auto_scroll = False
        elif value >= bar.maximum() - 1:
            self._auto_scroll = True

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._last_max = bar.maximum()


class _Bubble(QWidget):
    """One chat bubble (user or rook)."""

    # Stylesheet padding on ``_bubble_style`` — kept in sync with the
    # style strings below so ``_fit`` sizes the label correctly.
    _PAD_H = 18  # 9 top + 9 bottom
    _PAD_W = 26  # 13 left + 13 right
    _MAX_CONTENT_W = 420

    def __init__(self, role: str, text: str, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._role = role
        self._accent = accent
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        label_label = QLabel("YOU" if role == "user" else "ROOK")
        label_label.setStyleSheet(self._label_style(self._accent if role == "rook" else QColor("#9aa7b9")))
        col.addWidget(label_label, alignment=self._align())

        # QLabel + ``wordWrap=True`` reports a single-word sizeHint when
        # used with ``addWidget(alignment=…)``; the layout then allocates
        # only that width and the text wraps to the minimum feasible
        # rectangle (first line visible, rest clipped). Instead of
        # fighting sizeHint, we measure the text with QFontMetrics and
        # set a concrete fixed size ourselves — cheap, deterministic,
        # and avoids the heightForWidth propagation pitfalls.
        self._body = QLabel(text)
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setStyleSheet(self._bubble_style(role, accent))
        self._body.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._fit(text)
        col.addWidget(self._body, alignment=self._align())

    def _fit(self, text: str) -> None:
        """Size the body label to exactly fit its wrapped text."""
        fm = self._body.fontMetrics()
        max_w = self._MAX_CONTENT_W - self._PAD_W
        natural_w = fm.horizontalAdvance(text)
        if natural_w <= max_w:
            width = natural_w
            height = fm.height()
        else:
            rect = fm.boundingRect(
                QRect(0, 0, max_w, 10_000_000),
                int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignLeft),
                text,
            )
            width = max_w
            height = rect.height()
        self._body.setFixedSize(width + self._PAD_W, height + self._PAD_H)

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
                "background: rgba(52, 211, 153, 0.16); "
                "color: #dbe4ef; padding: 9px 13px; border-radius: 14px; "
                "font-size: 13px;"
            )
        r, g, b = accent.red(), accent.green(), accent.blue()
        return (
            f"background: rgba({r}, {g}, {b}, 0.12); "
            "color: #dbe4ef; padding: 9px 13px; border-radius: 14px; "
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
            "padding: 4px 12px; border-radius: 10px; "
            "font-family: monospace; font-size: 11px;"
        )
        row.addWidget(label)
        row.addStretch(1)


class _DebugEntry(QWidget):
    """Debug dump: monospace, full-width, expandable.

    Collapsed by default so a long messages array doesn't push the
    conversation off-screen. Click the header to expand. The body
    uses a plain QLabel (not QTextEdit) because we only need
    selectable text, not editing — cheaper and avoids the scroll
    nesting that QTextEdit forces inside a QScrollArea.
    """

    _PREVIEW_CHARS = 400

    def __init__(self, title: str, body: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._body_text = body
        self._expanded = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self._header = QLabel(self._header_text(title, collapsed=True))
        self._header.setStyleSheet(
            "color: #ffb86c; background: rgba(255, 184, 108, 0.08); "
            "border-left: 2px solid #ffb86c; "
            "padding: 4px 8px; font-family: monospace; font-size: 10px; "
            "letter-spacing: 0.5px;"
        )
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        # Click → toggle. QLabel has no clicked signal; intercept via
        # mousePressEvent through a lambda on an internal flag.
        self._header.mousePressEvent = self._on_header_click  # type: ignore[assignment]
        col.addWidget(self._header)

        self._body = QLabel(self._render_body(expanded=False))
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setStyleSheet(
            "color: #9aa7b9; background: rgba(10, 14, 22, 0.7); "
            "padding: 8px 12px; border-radius: 6px; "
            "font-family: monospace; font-size: 11px;"
        )
        self._body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        col.addWidget(self._body)

        self._title = title

    def _header_text(self, title: str, *, collapsed: bool) -> str:
        arrow = "▸" if collapsed else "▾"
        hint = "click to expand" if collapsed else "click to collapse"
        size = len(self._body_text)
        return f"{arrow} DEBUG · {title} · {size} chars · {hint}"

    def _render_body(self, *, expanded: bool) -> str:
        if expanded or len(self._body_text) <= self._PREVIEW_CHARS:
            return self._body_text
        return self._body_text[: self._PREVIEW_CHARS] + "\n… (click header to expand) …"

    def _on_header_click(self, _event) -> None:
        self._expanded = not self._expanded
        self._header.setText(self._header_text(self._title, collapsed=not self._expanded))
        self._body.setText(self._render_body(expanded=self._expanded))


__all__ = ["ChatEntry", "ChatView"]


# Palette consumers are unused here but imported to satisfy linters.
_ = QPalette
