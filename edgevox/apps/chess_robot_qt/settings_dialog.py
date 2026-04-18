"""Settings dialog — piece set, board theme, persona, voice, SFX."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from edgevox.apps.chess_robot_qt.settings import (
    BOARD_THEME_LABELS,
    BOARD_THEMES,
    PERSONA_LABELS,
    PERSONAS,
    PIECE_SETS,
    Settings,
    available_input_devices,
    available_output_devices,
    piece_set_dir,
)


class SettingsDialog(QDialog):
    """Modal preferences panel — edit :class:`Settings` in place.

    Emits :attr:`changed` with the new :class:`Settings` instance on OK;
    the window applies what it can live (piece set, theme, SFX mute)
    and flags a restart hint for things that need a fresh agent
    (persona, voice pipeline).
    """

    changed = Signal(Settings)

    def __init__(self, current: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet(
            "QDialog { background: #0b111a; color: #dbe4ef; } "
            "QLabel { color: #dbe4ef; } "
            "QComboBox, QCheckBox { color: #dbe4ef; } "
            "QComboBox { background: #10161f; border: none; "
            "padding: 6px 10px; border-radius: 8px; } "
            "QComboBox QAbstractItemView { background: #10161f; color: #dbe4ef; "
            "border-radius: 8px; selection-background-color: rgba(255, 255, 255, 0.08); }"
        )

        self._current = current

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)

        header = QLabel("Preferences")
        header.setStyleSheet("font-size: 15px; font-weight: 600; color: #dbe4ef;")
        layout.addWidget(header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._piece_combo = _combo(PIECE_SETS, current.piece_set)
        form.addRow("Piece set", self._piece_combo)

        self._theme_combo = _combo(BOARD_THEME_LABELS, current.board_theme)
        form.addRow("Board theme", self._theme_combo)

        persona_items = {slug: PERSONA_LABELS[slug] for slug in PERSONAS}
        self._persona_combo = _combo(persona_items, current.persona)
        form.addRow("Persona", self._persona_combo)

        self._voice_checkbox = QCheckBox("Enable voice input")
        self._voice_checkbox.setChecked(current.voice_enabled)
        form.addRow("", self._voice_checkbox)

        self._sfx_checkbox = QCheckBox("Mute sound effects")
        self._sfx_checkbox.setChecked(current.sfx_muted)
        form.addRow("", self._sfx_checkbox)

        # Debug mode — when on, each turn's LLM input (system prompt,
        # briefing, memory block, messages) and raw reply are logged
        # inline to the chat as monospace bubbles. Applies live — no
        # restart needed.
        self._debug_checkbox = QCheckBox("Debug mode (log LLM input + reasoning to chat)")
        self._debug_checkbox.setChecked(current.debug_mode)
        form.addRow("", self._debug_checkbox)

        # Device pickers — blank entry == system default, so a machine
        # migration still works even if the saved numeric index no longer
        # resolves to the same device. We enumerate lazily here rather
        # than at Settings.load() so a failed PortAudio query doesn't
        # block app launch.
        self._mic_combo = _device_combo(available_input_devices(), current.input_device)
        form.addRow("Microphone", self._mic_combo)

        self._speaker_combo = _device_combo(available_output_devices(), current.output_device)
        form.addRow("Speaker", self._speaker_combo)

        layout.addLayout(form)

        # Hint: some settings require a restart (persona, voice, audio devices).
        hint = QLabel(
            "Piece set + theme + mute apply instantly. Persona / voice / audio-device changes take effect on the next launch."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8796a8; font-size: 11px; padding-top: 4px;")
        layout.addWidget(hint)

        # Live preview — shows the selected board theme AND piece set
        # together so the user can see what the board will look like
        # before applying.
        self._preview = _BoardPreview()
        self._preview.set_theme(current.board_theme)
        self._preview.set_piece_set(current.piece_set)
        self._theme_combo.currentTextChanged.connect(lambda: self._preview.set_theme(self._selected_theme()))
        self._piece_combo.currentTextChanged.connect(lambda: self._preview.set_piece_set(self._selected_piece_set()))
        layout.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setStyleSheet(
            "QPushButton { background: #1e2b3a; color: #dbe4ef; padding: 8px 18px; "
            "border: none; border-radius: 10px; } "
            "QPushButton:hover { background: #2b3c58; } "
            "QPushButton:default { background: #34d399; color: #0a0e14; font-weight: 600; }"
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _selected_theme(self) -> str:
        return _selected_key(self._theme_combo, BOARD_THEME_LABELS)

    def _selected_piece_set(self) -> str:
        return _selected_key(self._piece_combo, PIECE_SETS)

    def _on_accept(self) -> None:
        new = Settings(
            piece_set=_selected_key(self._piece_combo, PIECE_SETS),
            board_theme=_selected_key(self._theme_combo, BOARD_THEME_LABELS),
            persona=_selected_key(self._persona_combo, {s: PERSONA_LABELS[s] for s in PERSONAS}),
            voice_enabled=self._voice_checkbox.isChecked(),
            sfx_muted=self._sfx_checkbox.isChecked(),
            input_device=_selected_device(self._mic_combo),
            output_device=_selected_device(self._speaker_combo),
            debug_mode=self._debug_checkbox.isChecked(),
        )
        new.save()
        self.changed.emit(new)
        self.accept()


def _combo(items: dict[str, str], current_key: str) -> QComboBox:
    combo = QComboBox()
    for key, label in items.items():
        combo.addItem(label, key)
    # Prefer userData match; fall back to the first item.
    idx = next((i for i, k in enumerate(items) if k == current_key), 0)
    combo.setCurrentIndex(idx)
    return combo


def _selected_key(combo: QComboBox, items: dict[str, str]) -> str:
    text = combo.currentText()
    for key, label in items.items():
        if label == text:
            return key
    return next(iter(items))


def _device_combo(devices: list[tuple[int, str]], current: int | None) -> QComboBox:
    """Build the device picker. ``userData`` holds ``None`` for the
    default row and the raw PortAudio index for real devices, so
    ``_selected_device`` can round-trip without fuzzy name matching
    (device names can contain ambiguous duplicates on Linux)."""
    combo = QComboBox()
    combo.addItem("System default", None)
    for idx, name in devices:
        combo.addItem(f"{name}  [#{idx}]", idx)
    selected = 0
    if current is not None:
        for i in range(combo.count()):
            if combo.itemData(i) == current:
                selected = i
                break
    combo.setCurrentIndex(selected)
    return combo


def _selected_device(combo: QComboBox) -> int | None:
    data = combo.currentData()
    return None if data is None else int(data)


_PREVIEW_LAYOUT: tuple[tuple[str, ...], ...] = (
    # 8-wide x 2-tall sample board: black back rank on top, white on bottom.
    # Gives every piece type in both colours at a glance.
    ("r", "n", "b", "q", "k", "b", "n", "r"),
    ("R", "N", "B", "Q", "K", "B", "N", "R"),
)


class _BoardPreview(QFrame):
    """Mini 8x2 board showing the selected theme + piece set together.

    Renders both colour squares and every major piece type so the user
    can see exactly what their board will look like before hitting OK.
    SVG renderers are cached per piece-set so combo flicks stay cheap.
    """

    _PREVIEW_HEIGHT = 96

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(self._PREVIEW_HEIGHT)
        self._theme = "wood"
        self._piece_set = "fantasy"
        self._renderers: dict[str, QSvgRenderer] = {}
        self._load_pieces()

    def set_theme(self, theme: str) -> None:
        if theme == self._theme:
            return
        self._theme = theme
        self.update()

    def set_piece_set(self, key: str) -> None:
        if key == self._piece_set:
            return
        self._piece_set = key
        self._load_pieces()
        self.update()

    def _load_pieces(self) -> None:
        self._renderers = {}
        root = piece_set_dir(self._piece_set)
        for symbol in "KQRBNPkqrbnp":
            colour = "w" if symbol.isupper() else "b"
            path = root / f"{colour}{symbol.upper()}.svg"
            if path.is_file():
                self._renderers[symbol] = QSvgRenderer(str(path))

    def paintEvent(self, _event) -> None:
        light_hex, dark_hex = BOARD_THEMES.get(self._theme, BOARD_THEMES["wood"])
        light = QColor(light_hex)
        dark = QColor(dark_hex)

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            rows = len(_PREVIEW_LAYOUT)
            cols = len(_PREVIEW_LAYOUT[0])
            sz = min(self.width() / cols, self.height() / rows)
            board_w = sz * cols
            board_h = sz * rows
            # Centre the board in the frame.
            x0 = (self.width() - board_w) / 2
            y0 = (self.height() - board_h) / 2
            # Rounded clip so the preview matches the main board.
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(dark))
            p.drawRoundedRect(QRectF(x0, y0, board_w, board_h), 6, 6)
            p.restore()

            inset = sz * 0.06
            for r, row in enumerate(_PREVIEW_LAYOUT):
                for c, symbol in enumerate(row):
                    square_light = (c + r) % 2 == 0
                    # The real board uses (file + rank) % 2 == 1 → light.
                    # Flip here so a8 (top-left) is light, matching lichess.
                    colour = light if not square_light else dark
                    rect = QRectF(x0 + c * sz, y0 + r * sz, sz, sz)
                    p.fillRect(rect, QBrush(colour))
                    renderer = self._renderers.get(symbol)
                    if renderer is not None:
                        renderer.render(
                            p,
                            QRectF(rect.x() + inset, rect.y() + inset, sz - 2 * inset, sz - 2 * inset),
                        )
        finally:
            p.end()


__all__ = ["SettingsDialog"]
