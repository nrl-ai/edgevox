"""Robot face widget — minimal, mood-reactive, MIT-clean.

For v1 we render a stylised geometric face with QPainter driven by the
same mood vocabulary the :class:`RobotFaceHook` emits: calm / curious
/ amused / worried / triumphant / defeated. Blink runs on a QTimer.

Later upgrade path: swap this widget's ``paintEvent`` for ``rlottie``
output (LGPL — compatible via dynamic lib) or pre-rendered sprite
sheets if you want the animated Lottie look back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


@dataclass
class _MoodShape:
    eye_height: float
    brow_angle: float
    brow_y: float
    mouth_curve: float
    mouth_open: float
    cheek_glow: float


_MOODS: dict[str, _MoodShape] = {
    "calm": _MoodShape(0.8, 0.0, 0, 0.12, 0.05, 0.2),
    "curious": _MoodShape(1.0, 0.55, -2, 0.2, 0.1, 0.35),
    "amused": _MoodShape(0.5, 0.45, -1, 0.9, 0.18, 0.65),
    "worried": _MoodShape(1.0, -0.8, 4, -0.3, 0.08, 0.15),
    "triumphant": _MoodShape(0.35, 0.9, -3, 1.0, 0.32, 1.0),
    "defeated": _MoodShape(0.45, -1.0, 5, -0.85, 0.04, 0.1),
}


_PERSONA_ACCENT = {
    "grandmaster": QColor("#7aa8ff"),
    "casual": QColor("#ffb066"),
    "trash_talker": QColor("#ff5ad1"),
}
_DEFAULT_ACCENT = QColor("#34d399")
_SHELL = QColor("#10161f")
_SHELL_DARK = QColor("#060a12")
_SCLERA = QColor("#f5f7fa")


class RobotFaceWidget(QWidget):
    """A self-contained QPainter face. Exposes ``set_mood`` / ``set_tempo``
    / ``set_persona`` — no signals, just setters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 180)

        self._mood = "calm"
        self._tempo = "idle"
        self._persona = "casual"
        self._blink = False

        # Blink — two frames every 2-6 s.
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._on_blink)
        self._schedule_blink()

        # Thinking pulse — antenna animation.
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._on_pulse)
        self._pulse_timer.start(220)
        self._pulse_phase = 0.0

    # ----- API -----

    def set_mood(self, mood: str) -> None:
        if mood != self._mood and mood in _MOODS:
            self._mood = mood
            self.update()

    def set_tempo(self, tempo: str) -> None:
        if tempo != self._tempo:
            self._tempo = tempo
            self.update()

    def set_persona(self, persona: str) -> None:
        if persona != self._persona:
            self._persona = persona
            self.update()

    # ----- animation timers -----

    def _schedule_blink(self) -> None:
        self._blink_timer.start(random.randint(2200, 5600))

    def _on_blink(self) -> None:
        if self._blink:
            self._blink = False
            self._schedule_blink()
        else:
            self._blink = True
            # Hold the blink for ~110 ms.
            QTimer.singleShot(110, self._on_blink)
        self.update()

    def _on_pulse(self) -> None:
        self._pulse_phase = (self._pulse_phase + 0.11) % 1.0
        if self._tempo in ("thinking", "speaking"):
            self.update()

    # ----- painting -----

    def paintEvent(self, _event) -> None:
        shape = _MOODS.get(self._mood, _MOODS["calm"])
        accent = _PERSONA_ACCENT.get(self._persona, _DEFAULT_ACCENT)

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            side = min(self.width(), self.height())
            cx = self.width() / 2
            cy = self.height() / 2
            # View-box math: design at 220 units, scale to the widget.
            s = side / 220.0
            p.translate(cx, cy)
            p.scale(s, s)

            # Outer glow ring.
            ring_brightness = 0.35 + shape.cheek_glow * 0.5
            if self._tempo == "speaking":
                ring_brightness += 0.25
            elif self._tempo == "thinking":
                ring_brightness += 0.12
            ring = QPen(accent, 1.5)
            ring.setColor(_with_alpha(accent, 80 + int(ring_brightness * 120)))
            p.setPen(ring)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(-98, -98, 196, 196)

            # Head — rounded rect with gradient shading.
            head_rect = (-72, -76, 144, 152)
            p.setPen(QPen(QColor("#2b3c58"), 2))
            p.setBrush(_SHELL)
            p.drawRoundedRect(*head_rect, 40, 40)
            p.setPen(QPen(QColor("#1a2230"), 1))
            p.drawLine(-58, -60, 58, -60)  # top-plate seam

            # Antenna — pulses brighter while thinking.
            ant_pulse = 0.75 + (0.25 if self._tempo == "thinking" else 0.0) * abs(2 * self._pulse_phase - 1)
            p.setPen(QPen(accent, 2))
            p.drawLine(0, -76, 0, -92)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_with_alpha(accent, int(255 * ant_pulse)))
            p.drawEllipse(-4, -96, 8, 8)

            # Brows.
            brow_y = shape.brow_y
            pen_brow = QPen(accent, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen_brow)
            # left
            _draw_brow(p, side=-1, angle=shape.brow_angle, y=brow_y)
            _draw_brow(p, side=1, angle=shape.brow_angle, y=brow_y)

            # Eyes.
            ry = 14 * shape.eye_height
            if self._blink:
                ry = max(1.5, ry * 0.1)
            _draw_eye(p, -26, 0, 14, ry, accent)
            _draw_eye(p, 26, 0, 14, ry, accent)

            # Cheeks (glow) — brightens on positive mood.
            cheek_alpha = max(0, shape.mouth_curve) * shape.cheek_glow
            if cheek_alpha > 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_with_alpha(accent, int(cheek_alpha * 160)))
                p.drawEllipse(-54, 12, 20, 14)
                p.drawEllipse(34, 12, 20, 14)

            # Mouth — curve + open aperture + optional teeth line.
            _draw_mouth(p, accent, _SHELL_DARK, shape)

        finally:
            p.end()


# ----- painter helpers -----


def _draw_brow(p: QPainter, *, side: int, angle: float, y: float) -> None:
    inner = side * 10
    outer = side * (10 + 28)
    y0 = -34 + y
    # Signed rotation around the inner endpoint.
    import math

    # Rotate the outer endpoint up/down based on angle.
    dx = outer - inner
    theta = side * angle * 0.26
    rotated_dx = dx * math.cos(theta)
    rotated_dy = dx * math.sin(theta) + (-angle * 4)
    p.drawLine(inner, int(y0), int(inner + rotated_dx), int(y0 + rotated_dy))


def _draw_eye(p: QPainter, cx: int, cy: int, rx: float, ry: float, accent: QColor) -> None:
    # Socket shadow
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_with_alpha(QColor(10, 14, 22), 210))
    p.drawEllipse(int(cx - rx - 2), int(cy - ry - 2), int((rx + 2) * 2), int((ry + 2) * 2))
    # Sclera
    p.setBrush(_SCLERA)
    p.drawEllipse(int(cx - rx), int(cy - ry), int(rx * 2), int(ry * 2))
    # Iris + pupil
    iris_r = min(7.0, ry * 0.5)
    p.setBrush(accent)
    p.drawEllipse(int(cx - iris_r), int(cy - iris_r), int(iris_r * 2), int(iris_r * 2))
    p.setBrush(QColor("#0b1220"))
    inner_r = iris_r * 0.5
    p.drawEllipse(int(cx - inner_r), int(cy - inner_r), int(inner_r * 2), int(inner_r * 2))
    # Catchlight
    p.setBrush(_with_alpha(QColor("#ffffff"), 210))
    cr = iris_r * 0.3
    p.drawEllipse(int(cx + inner_r * 0.4), int(cy - inner_r), int(cr * 2), int(cr * 2))


def _draw_mouth(p: QPainter, accent: QColor, shell: QColor, shape: _MoodShape) -> None:
    y0 = 46
    half_width = 28 + max(0.0, shape.mouth_curve) * 2
    ctrl = 14 * shape.mouth_curve
    # Outer lip as a quadratic curve — we approximate with a poly line.
    from PySide6.QtGui import QPainterPath

    path = QPainterPath()
    path.moveTo(-half_width, y0)
    path.quadTo(0, y0 - ctrl, half_width, y0)
    p.setPen(QPen(accent, 4.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)

    aperture = max(1.0, shape.mouth_open * 16)
    if shape.mouth_open > 0.03:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shell)
        p.drawEllipse(
            int(-min(half_width - 4, 18)),
            int(y0 - aperture + 2 - ctrl * 0.22),
            int(min(half_width - 4, 18) * 2),
            int(aperture * 2),
        )
        if shape.mouth_curve > 0.4 and shape.mouth_open > 0.1:
            # Teeth highlight.
            pen_tooth = QPen(accent)
            pen_tooth.setWidthF(1.2)
            p.setPen(pen_tooth)
            p.drawLine(
                int(-min(half_width - 4, 18) + 3),
                int(y0 - aperture + 3 - ctrl * 0.22),
                int(min(half_width - 4, 18) - 3),
                int(y0 - aperture + 3 - ctrl * 0.22),
            )


def _with_alpha(c: QColor, alpha: int) -> QColor:
    out = QColor(c)
    out.setAlpha(max(0, min(255, alpha)))
    return out


__all__ = ["RobotFaceWidget"]
