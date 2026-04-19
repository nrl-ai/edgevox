"""Main window — custom titlebar + board/face/chat layout.

Matches the Tauri/React UI 1:1 where it makes sense:

    ┌─────────────────────────────────────── RookApp  — ◻ ✕ ┐
    │ ● RookApp · online     your turn   🎤  ↻  ☰        │
    ├──────────────────────────────────┬──────────────────┤
    │                                  │  [ robot face ]  │
    │      [ chess board ]             │                  │
    │                                  │  [ chat log    ] │
    │   captured pieces strip          │                  │
    │   move history                   │  [ input ▶ ]     │
    └──────────────────────────────────┴──────────────────┘

Frameless + our own title bar, same pattern as the Tauri app. No
browser, no webview — pure Qt widgets.
"""

from __future__ import annotations

import logging

import qtawesome as qta
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPainterPath, QRegion
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgevox.apps.chess_robot_qt.board import ChessBoardView
from edgevox.apps.chess_robot_qt.bridge import RookBridge
from edgevox.apps.chess_robot_qt.chat import ChatView
from edgevox.apps.chess_robot_qt.face import RobotFaceWidget
from edgevox.apps.chess_robot_qt.lottie_face import LottieFaceWidget
from edgevox.apps.chess_robot_qt.lottie_face import is_available as lottie_available
from edgevox.apps.chess_robot_qt.settings import Settings
from edgevox.apps.chess_robot_qt.settings_dialog import SettingsDialog
from edgevox.apps.chess_robot_qt.tts import TTSWorker
from edgevox.apps.chess_robot_qt.voice import VoiceWorker

log = logging.getLogger(__name__)


_PERSONA_ACCENT = {
    "grandmaster": "#7aa8ff",
    "casual": "#ffb066",
    "trash_talker": "#ff5ad1",
}
_DEFAULT_ACCENT = "#34d399"
_TOP_RESIZE_STRIP = 5  # px at the top of the AppBar that resize the window


class RookWindow(QMainWindow):
    """Frameless main window with custom titlebar."""

    def __init__(self, bridge: RookBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self._settings = Settings.load()
        # Sync the loaded settings into the bridge config when present —
        # e.g. restore the previously-chosen persona and LLM at launch.
        # These must land BEFORE ``bridge.start()`` fires the background
        # load job, otherwise the load job reads the stale defaults.
        if self._settings.persona != bridge.config.persona:
            bridge.config.persona = self._settings.persona
        if self._settings.llm_model and self._settings.llm_model != bridge.config.llm_path:
            bridge.config.llm_path = self._settings.llm_model
        self._accent = QColor(_PERSONA_ACCENT.get(bridge.config.persona, _DEFAULT_ACCENT))
        # The engine applies its reply the instant the user plays so the
        # LLM briefing has full context, but showing that reply on the
        # board at the same split-second feels robotic — the human just
        # spoke, Rook should appear to think. We hold the engine-move
        # reveal (board + history + chat chip) for this long while the
        # LLM keeps generating in the background.
        self._engine_reveal_delay_ms = 3500
        self._pending_engine_state = None
        self.setWindowTitle("RookApp")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)
        self.setStyleSheet("QLabel { color: #dbe4ef; }")

        # Translucent window + rounded-rect root lets the outer corners
        # fade to the desktop instead of showing a hard rectangle.
        root = _RoundedRoot(radius=16, fill=QColor("#060a12"))
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        # Inset children by a few pixels so the root's painted rounded
        # rect peeks through on three sides. The exposed strip doubles
        # as a resize grip — _RoundedRoot hit-tests clicks there and
        # delegates to windowHandle().startSystemResize(). The top edge
        # stays flush because the AppBar owns its own top-resize strip.
        col.setContentsMargins(5, 0, 5, 5)
        col.setSpacing(0)

        # One unified top bar: brand + drag region + status + turn pill
        # + action buttons + window controls. Replaces the old
        # titlebar/appbar pair (which duplicated the brand label).
        self._app_bar = _AppBar(self._accent, self)
        self._app_bar.new_game_clicked.connect(self._on_new_game)
        self._app_bar.mic_clicked.connect(self._on_mic_clicked)
        self._app_bar.menu_triggered.connect(self._on_menu_action)
        self._app_bar.close_clicked.connect(self.close)
        self._app_bar.minimize_clicked.connect(self.showMinimized)
        self._app_bar.toggle_maximize_clicked.connect(self._toggle_max)
        col.addWidget(self._app_bar)

        # Main area.
        main = QWidget()
        main_row = QHBoxLayout(main)
        main_row.setContentsMargins(14, 10, 14, 10)
        main_row.setSpacing(14)

        # Left column — framed board + history strip.
        left = QVBoxLayout()
        left.setSpacing(6)
        board_frame = QFrame()
        board_frame.setStyleSheet("QFrame { background: #0b111a; border-radius: 14px; padding: 8px; }")
        bf_col = QVBoxLayout(board_frame)
        bf_col.setContentsMargins(10, 10, 10, 10)
        # Stack a loading panel on top of the board; swap to the board
        # only once ``bridge.ready`` fires. Prevents the user from
        # pushing moves while the LLM is still downloading / loading —
        # previously the board was interactive during load and the
        # first click silently dropped on the floor.
        #
        # Uses QStackedWidget (not QStackedLayout) so the currently-
        # hidden widget is actually ``hide()``'d — a nested QStackedLayout
        # inside a QVBoxLayout can leave the stale widget painted
        # depending on how the parent sizes children, which manifested
        # as the loading panel never fully disappearing after ready.
        self._board_stack = QStackedWidget()
        self._board = ChessBoardView()
        self._board.move_requested.connect(self._on_board_move)
        self._board.setMinimumSize(420, 420)
        self._board.set_orientation(bridge.config.user_plays)
        self._board.set_piece_set(self._settings.piece_set)
        self._board.set_theme(self._settings.board_theme)
        self._loading_panel = _LoadingPanel(self._accent)
        # Force an arrow cursor on the loading panel so the pointing-
        # hand cursor the board sets on itself can't visually "leak"
        # onto the panel through parent-chain inheritance.
        self._loading_panel.setCursor(Qt.CursorShape.ArrowCursor)
        self._board_stack.addWidget(self._loading_panel)
        self._board_stack.addWidget(self._board)
        self._board_stack.setCurrentWidget(self._loading_panel)
        bf_col.addWidget(self._board_stack)
        left.addWidget(board_frame, stretch=1)
        self._history = QLabel("no moves yet")
        self._history.setStyleSheet("color: #9aa7b9; font-family: monospace; font-size: 11px; padding: 2px 8px;")
        self._history.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._history)
        main_row.addLayout(left, stretch=60)

        # Right column — face, chat, input.
        right = QVBoxLayout()
        right.setSpacing(10)
        face_frame = QFrame()
        face_frame.setStyleSheet("QFrame { background: #0b111a; border-radius: 14px; }")
        face_col = QVBoxLayout(face_frame)
        face_col.setContentsMargins(12, 12, 12, 12)
        face_col.setSpacing(8)
        self._face = LottieFaceWidget() if lottie_available() else RobotFaceWidget()
        self._face.setFixedHeight(220)
        self._face.set_persona(bridge.config.persona)
        face_col.addWidget(self._face, stretch=0)
        self._persona_label = QLabel(bridge.config.persona.replace("_", " ").upper())
        self._persona_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Qt stylesheets don't honour letter-spacing / text-transform; drive
        # both via QFont and an already-uppercased string.
        persona_font = self._persona_label.font()
        persona_font.setFamily("monospace")
        persona_font.setPointSize(max(8, persona_font.pointSize() - 2))
        persona_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 140)
        persona_font.setWeight(QFont.Weight.Medium)
        self._persona_label.setFont(persona_font)
        self._persona_label.setStyleSheet(f"color: {self._accent.name()};")
        face_col.addWidget(self._persona_label, stretch=0)
        right.addWidget(face_frame, stretch=0)

        self._reply_label = QLabel("")
        self._reply_label.setWordWrap(True)
        self._reply_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reply_label.setStyleSheet("color: #9aa7b9; font-size: 13px; padding: 6px 12px;")
        self._reply_label.setMinimumHeight(34)
        right.addWidget(self._reply_label, stretch=0)

        self._chat = ChatView(self._accent)
        right.addWidget(self._chat, stretch=1)

        self._input = _InputBar(self._accent)
        self._input.submitted.connect(self._on_submit)
        right.addWidget(self._input, stretch=0)

        main_row.addLayout(right, stretch=40)
        col.addWidget(main, stretch=1)

        # Keyboard shortcuts live in the Menu dropdown now — no need
        # for a duplicate footer strip here.

        # Wire the bridge → UI signals.
        b = bridge.signals
        b.state_changed.connect(self._on_state)
        b.chess_state_changed.connect(self._on_chess_state)
        b.face_changed.connect(self._on_face)
        b.reply_finalised.connect(self._on_reply)
        b.user_echo.connect(self._on_user_echo)
        b.error.connect(self._on_error)
        b.ready.connect(self._on_ready)
        b.load_progress.connect(self._on_load_progress)
        b.debug_event.connect(self._chat.add_debug)
        b.analytics_event.connect(self._chat.add_analytics)

        # Apply the persisted debug flag before the first turn so the
        # DebugTapHook starts emitting immediately when enabled.
        bridge.set_debug_mode(self._settings.debug_mode)

        # Start the bridge's background load of LLM + engine.
        self._app_bar.set_status("loading…")
        self._input.set_enabled(False)
        bridge.start()

        # TTS — kokoro loads lazily on a worker when the first reply
        # comes in, so we don't pay the ~300 MB ONNX cost at launch.
        # Mute means don't even load the model.
        self._tts: TTSWorker | None = None
        if not self._settings.sfx_muted:
            self._tts = TTSWorker(output_device=self._settings.output_device, parent=self)
            self._tts.error.connect(lambda msg: log.warning("TTS: %s", msg))
            self._tts.started.connect(self._on_tts_started)
            self._tts.finished.connect(self._on_tts_finished)
            # Warmup in the background — no UI status spam; failures
            # downgrade to text-only silently.
            self._tts.start()

    # ----- bridge signal handlers -----

    def _on_ready(self) -> None:
        self._app_bar.set_status("online")
        self._input.set_enabled(True)
        # Swap the loading panel out for the live board.
        self._board_stack.setCurrentWidget(self._board)
        # Prime the board with the starting position.
        snap = self._bridge.snapshot()
        if snap is not None:
            self._board.set_state(snap)
        # Replay the persisted chat transcript so the user sees their
        # prior exchanges after a restart. The bridge already restored
        # the session for the agent's context inside ``_build``; we do
        # it here rather than in ``_on_load_progress`` because the
        # session isn't constructed until right before ``ready.emit()``,
        # and load-progress events fire earlier with ``_ctx_session``
        # still ``None`` — which previously returned an empty list
        # silently and left the chat blank on relaunch.
        for role, text in self._bridge.session_messages():
            if role == "user":
                self._chat.add_user(text)
            elif role == "assistant":
                self._chat.add_rook(text)

    def _on_load_progress(self, text: str) -> None:
        """Mirror the bridge's load-progress into the app-bar status AND
        the board-area loading panel so the user has a clear signal
        that model / engine are still coming up."""
        self._app_bar.set_status(text)
        if hasattr(self, "_loading_panel"):
            self._loading_panel.set_stage(text)

    def _on_state(self, state: str) -> None:
        label = {
            "listening": "your turn",
            "thinking": "rook thinking…",
            "speaking": "rook replying…",
        }.get(state, state)
        self._app_bar.set_turn_label(label, highlight=state == "listening")
        self._face.set_tempo(state)
        # Block board interaction while Rook is mid-turn. The board's
        # own turn-check catches most cases, but a rapid double-click
        # during the engine-reveal delay (or a click the moment the
        # bridge transitions states) was slipping through and the move
        # either silently dropped or raced with the chess_state_changed
        # reveal — looking to the user like the move "reverted".
        self._board.setEnabled(state == "listening")
        # On return to "listening", force a re-render from the env's
        # authoritative state — but ONLY if there's no engine-reveal
        # pending. Otherwise the force-apply would pre-empt the 3.5s
        # delay and flash the engine's move on the board the moment
        # the agent turn ends. The engine-reveal timer owns the final
        # apply in that case.
        if state == "listening" and self._pending_engine_state is None:
            snap = self._bridge.snapshot()
            if snap is not None:
                self._apply_chess_state(snap)

    def _on_chess_state(self, state) -> None:
        engine_just_moved = bool(
            state.last_move_san
            and state.turn == self._bridge.config.user_plays
            and state.ply >= 2
            and state.ply % 2 == 0
        )
        if engine_just_moved:
            # Hold the reveal. Overwrites any prior pending state so the
            # most recent engine move always wins if two somehow queue
            # up, and re-arms the timer from "now" rather than stacking.
            self._pending_engine_state = state
            QTimer.singleShot(self._engine_reveal_delay_ms, self._reveal_pending_engine_state)
            return
        # Non-engine update (user's own move, reset, undo, game start):
        # drop any pending reveal so a stale engine state can't clobber
        # a fresh board after "new game" lands mid-delay.
        self._pending_engine_state = None
        self._apply_chess_state(state)

    def _reveal_pending_engine_state(self) -> None:
        state = self._pending_engine_state
        if state is None:
            return
        self._pending_engine_state = None
        self._apply_chess_state(state)

    def _apply_chess_state(self, state) -> None:
        self._board.set_state(state)
        # Move-history strip.
        moves = getattr(state, "san_history", []) or []
        if not moves:
            self._history.setText("no moves yet")
        else:
            pairs: list[str] = []
            for i in range(0, len(moves), 2):
                w = moves[i]
                b = moves[i + 1] if i + 1 < len(moves) else ""
                pairs.append(f"{i // 2 + 1}. {w} {b}".rstrip())
            self._history.setText("   ".join(pairs[-8:]))
        # Engine-move chip.
        if (
            state.last_move_san
            and state.turn == self._bridge.config.user_plays
            and state.ply >= 2
            and state.ply % 2 == 0
        ):
            self._chat.add_move_chip("rook", state.last_move_san)

    def _on_face(self, payload: dict) -> None:
        mood = payload.get("mood", "calm")
        tempo = payload.get("tempo", "idle")
        persona = payload.get("persona", self._bridge.config.persona)
        self._face.set_mood(mood)
        self._face.set_tempo(tempo)
        self._face.set_persona(persona)
        self._persona_label.setText(persona.replace("_", " ").upper())

    def _on_reply(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._reply_label.setText(text)
        self._chat.add_rook(text)
        if self._tts is not None:
            self._tts.speak(text)

    def _on_tts_started(self) -> None:
        # Echo suppression is automatic: AudioRecorder is linked to the
        # global InterruptiblePlayer, which pauses the mic queue at the
        # source and resumes after the cooldown. Any residual echo is
        # filtered by the RMS-ratio gate while interrupt detection runs.
        self._face.set_tempo("speaking")

    def _on_tts_finished(self) -> None:
        self._face.set_tempo("idle")

    def _on_barge_in(self) -> None:
        """User spoke over Rook. Cut TTS immediately and cancel the
        in-flight agent turn so the next transcription doesn't compete
        with a stale reply."""
        if self._tts is not None:
            self._tts.interrupt()
        self._bridge.cancel_turn()
        self._face.set_tempo("idle")
        self._app_bar.set_status("you interrupted")
        QTimer.singleShot(1500, lambda: self._app_bar.set_status("online"))

    def _on_user_echo(self, text: str) -> None:
        self._reply_label.setText("")

    def _on_error(self, msg: str) -> None:
        self._app_bar.set_status(msg, error=True)
        QTimer.singleShot(4500, lambda: self._app_bar.set_status("online"))
        # If the error fired before ``ready`` (startup failure), take the
        # loading panel down so the user sees a usable error state rather
        # than a perpetual "getting rook ready…" panel. The board will
        # still not accept moves because ``_loaded`` never flips, but
        # the user can see what went wrong.
        if hasattr(self, "_board_stack") and self._board_stack.currentWidget() is self._loading_panel:
            self._loading_panel.set_stage(f"failed: {msg}")

    # ----- user actions -----

    def _on_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._chat.add_user(text)
        self._bridge.submit_text(text)

    def _on_board_move(self, uci: str) -> None:
        self._chat.add_user(uci)
        self._bridge.submit_text(uci)

    def _on_new_game(self) -> None:
        # Clear the UI + wipe persisted memory/notes/chat history so
        # stale commentary from the previous game can't leak into the
        # new one. Bridge.reset_game() rewinds the board; submit_text
        # primes Rook to announce the new match.
        self._chat.clear()
        self._reply_label.setText("")
        if self._tts is not None:
            self._tts.interrupt()
        self._bridge.cancel_turn(reason="new_game")
        self._bridge.clear_memory()
        self._bridge.reset_game()
        self._bridge.submit_text("new game")

    def _on_mic_clicked(self) -> None:
        """Toggle voice input. Lazy-starts the VoiceWorker on first
        click so we don't pay Whisper's startup cost up-front — voice
        users opt in; text users never load STT."""
        if not hasattr(self, "_voice"):
            if not self._settings.voice_enabled:
                self._on_error("Voice is disabled in Settings — enable it there first.")
                return
            self._voice = VoiceWorker(
                language="en",
                input_device=self._settings.input_device,
                parent=self,
            )
            self._voice.transcript.connect(self._on_voice_transcript)
            self._voice.error.connect(self._on_error)
            self._voice.loading.connect(lambda on: self._app_bar.set_status("warming up mic…" if on else "online"))
            self._voice.ready.connect(lambda: self._voice_enable_listen())
            self._voice.barge_in.connect(self._on_barge_in)
            self._voice.start()
            self._voice_active = True
            self._app_bar.set_mic_active(True)
            return
        # Subsequent clicks toggle the listening gate.
        self._voice_active = not getattr(self, "_voice_active", False)
        self._voice.set_listening(self._voice_active)
        self._app_bar.set_mic_active(self._voice_active)

    def _voice_enable_listen(self) -> None:
        self._voice.set_listening(getattr(self, "_voice_active", True))

    def _on_voice_transcript(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._chat.add_user(text)
        self._bridge.submit_text(text)

    def _on_menu_action(self, action: str) -> None:
        if action == "new_game":
            self._on_new_game()
        elif action == "settings":
            self._open_settings()
        elif action == "about":
            self._app_bar.set_status("RookApp — voice chess on EdgeVox.", error=False)
            QTimer.singleShot(4500, lambda: self._app_bar.set_status("online"))

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        dlg.changed.connect(self._apply_settings)
        dlg.exec()

    def _apply_settings(self, new: Settings) -> None:
        """Apply what we can live; flag the rest as needing a restart."""
        self._board.set_piece_set(new.piece_set)
        self._board.set_theme(new.board_theme)
        if new.sfx_muted != self._settings.sfx_muted:
            # SFX are a future hook — flag for later.
            pass
        # Debug mode is purely observational — the tap hook is already
        # installed, so flipping the flag takes effect on the very next
        # fire point.
        if new.debug_mode != self._settings.debug_mode:
            self._bridge.set_debug_mode(new.debug_mode)
        # Persona swap — apply the visual + agent-voice parts live; the
        # engine strength still defers to the next "new game" because
        # rebuilding the chess engine clobbers the current position.
        if new.persona != self._settings.persona:
            self._apply_persona_live(new.persona)
        # Voice enable still requires the next launch (the VoiceWorker
        # is created lazily on first mic click but the kill path is
        # restart-only).
        voice_changed = new.voice_enabled != self._settings.voice_enabled
        self._settings = new
        if voice_changed:
            self._on_error("Voice change applies next launch.")

    def _apply_persona_live(self, slug: str) -> None:
        """Re-skin the persona-coupled UI surfaces and tell the bridge
        to swap its agent instructions + face hook in place."""
        self._bridge.set_persona(slug)
        self._accent = QColor(_PERSONA_ACCENT.get(slug, _DEFAULT_ACCENT))
        self._persona_label.setText(slug.replace("_", " ").upper())
        self._persona_label.setStyleSheet(f"color: {self._accent.name()};")
        self._face.set_persona(slug)
        # Push the new accent into widgets that cache it for restyling.
        self._chat.set_accent(self._accent)
        self._app_bar.set_accent(self._accent)

    # ----- window chrome -----

    def _toggle_max(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_N and event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            self._on_new_game()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        # Hide the window up-front so the close feels instant — then do
        # the slow teardown (TTS stop, voice stop, bridge close waits up
        # to 3 s for an in-flight turn) while no UI is visible.
        self.hide()
        QApplication.processEvents()
        if self._tts is not None:
            self._tts.stop()
        voice = getattr(self, "_voice", None)
        if voice is not None:
            voice.stop()
        self._bridge.close()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        # Clip child widgets to the rounded outline so the title bar /
        # buttons / panels don't bleed past the corners of the window.
        radius = 16
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), radius, radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        super().resizeEvent(event)


# ----- sub-widgets -----


class _RoundedRoot(QWidget):
    """Central widget that paints an antialiased rounded-rect fill.

    Combined with ``WA_TranslucentBackground`` on the main window, this
    gives the frameless window soft rounded corners without a jagged
    pixel mask. Layout / child widgets live on top of this as usual.

    The widget also doubles as the frameless window's resize handle:
    when the cursor enters the margin strip exposed by the outer layout
    it updates the cursor shape, and a press there calls
    ``windowHandle().startSystemResize(edges)`` — same compositor-owned
    path that the drag handler uses for move.
    """

    _RESIZE_MARGIN = 5

    def __init__(self, radius: int, fill: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius = radius
        self._fill = fill
        self.setMouseTracking(True)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._fill)
        p.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)

    def _edges_at(self, pos) -> Qt.Edge:
        m = self._RESIZE_MARGIN
        rect = self.rect()
        edges = Qt.Edge(0)
        if pos.x() <= m:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= rect.width() - m:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= m:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= rect.height() - m:
            edges |= Qt.Edge.BottomEdge
        return edges

    @staticmethod
    def _cursor_for(edges: Qt.Edge) -> Qt.CursorShape:
        left_top = Qt.Edge.LeftEdge | Qt.Edge.TopEdge
        right_bottom = Qt.Edge.RightEdge | Qt.Edge.BottomEdge
        right_top = Qt.Edge.RightEdge | Qt.Edge.TopEdge
        left_bottom = Qt.Edge.LeftEdge | Qt.Edge.BottomEdge
        if edges in (left_top, right_bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (right_top, left_bottom):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            return Qt.CursorShape.SizeHorCursor
        if edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def mouseMoveEvent(self, event) -> None:
        if not event.buttons():
            self.setCursor(self._cursor_for(self._edges_at(event.position().toPoint())))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self.unsetCursor()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._edges_at(event.position().toPoint())
            if edges:
                handle = self.window().windowHandle()
                if handle is not None and handle.startSystemResize(edges):
                    event.accept()
                    return
        super().mousePressEvent(event)


class _LoadingPanel(QWidget):
    """Blocking overlay shown in the board area while the LLM and
    engine load. Replaces the interactive :class:`ChessBoardView` via
    a :class:`QStackedLayout` so the user can't push moves that would
    silently drop on the floor during the first few seconds of launch.

    The stage text is updated from ``bridge.load_progress`` — the user
    sees "checking chess engine…", "downloading model…", "ready" as
    the bridge walks its boot steps.
    """

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(12)
        col.addStretch(1)

        self._title = QLabel("getting rook ready…")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("color: #dbe4ef; font-size: 18px; font-weight: 600;")
        col.addWidget(self._title)

        self._stage = QLabel("booting")
        self._stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stage.setWordWrap(True)
        self._stage.setStyleSheet(
            f"color: {accent.name()}; font-family: monospace; font-size: 12px; letter-spacing: 0.3px;"
        )
        col.addWidget(self._stage)

        hint = QLabel(
            "first launch downloads the LLM (~1 GB) and the chess engine.\n"
            "subsequent launches are faster — everything is cached."
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6f7e92; font-size: 11px;")
        col.addWidget(hint)
        col.addStretch(1)

    def set_stage(self, text: str) -> None:
        self._stage.setText(text)


class _AppBar(QFrame):
    """Unified title + status bar. Drag the empty area to move the
    frameless window; click the brand area for the same effect."""

    close_clicked = Signal()
    minimize_clicked = Signal()
    toggle_maximize_clicked = Signal()
    new_game_clicked = Signal()
    mic_clicked = Signal()
    menu_triggered = Signal(str)

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        # Transparent bar — the rounded root widget paints underneath
        # and we stay fully within the window's rounded-rect clip.
        self.setStyleSheet("background: transparent;")
        row = QHBoxLayout(self)
        # Pad away from the rounded window corners so the brand text
        # isn't colliding with the top-left arc.
        row.setContentsMargins(22, 6, 10, 6)
        row.setSpacing(10)

        # Brand cluster: two rows stacked in a QVBoxLayout so the
        # status caption sits under the brand instead of running
        # alongside it (which read cluttered before).
        brand_col = QVBoxLayout()
        brand_col.setSpacing(4)
        brand_col.setContentsMargins(0, 0, 0, 0)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(6)
        brand_row.setContentsMargins(0, 0, 0, 0)
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        self._status_dot.setStyleSheet(
            f"background: {accent.name()}; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;"
        )
        self._status_dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_row.addWidget(self._status_dot)
        brand_name = QLabel("RookApp")
        brand_name.setStyleSheet("color: #dbe4ef; font-weight: 600; font-size: 13px; letter-spacing: 0.4px;")
        brand_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_row.addWidget(brand_name)
        brand_row.addStretch(0)
        brand_col.addLayout(brand_row)

        self._status_text = QLabel("booting")
        self._status_text.setStyleSheet("color: #8796a8; font-family: monospace; font-size: 10px; padding-left: 14px;")
        self._status_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_col.addWidget(self._status_text)

        row.addLayout(brand_col)
        row.addSpacing(6)

        # Middle area: pure drag region.
        row.addStretch(1)

        self._turn_pill = QLabel("YOUR TURN")
        self._turn_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Let the pill hug its text — no min-width padding.
        self._turn_pill.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._turn_pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Qt stylesheets ignore letter-spacing; apply via QFont so the
        # uppercased status actually reads as a status chip.
        pill_font = self._turn_pill.font()
        pill_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 110)
        pill_font.setWeight(QFont.Weight.DemiBold)
        pill_font.setPointSize(max(8, pill_font.pointSize() - 2))
        self._turn_pill.setFont(pill_font)
        row.addWidget(self._turn_pill)

        self._mic_btn = _IconButton("fa5s.microphone", "Talk to Rook")
        self._mic_btn.clicked.connect(self.mic_clicked.emit)
        row.addWidget(self._mic_btn)

        new_btn = _IconButton("fa5s.redo-alt", "New game")
        new_btn.clicked.connect(self.new_game_clicked.emit)
        row.addWidget(new_btn)

        self._menu_btn = _IconButton("fa5s.bars", "Menu")
        self._menu_btn.clicked.connect(self._open_menu)
        row.addWidget(self._menu_btn)

        # Window controls on the far right — minimise / maximise / close.
        row.addSpacing(4)
        for icon_name, signal in [
            ("mdi.window-minimize", self.minimize_clicked),
            ("mdi.window-maximize", self.toggle_maximize_clicked),
            ("mdi.close", self.close_clicked),
        ]:
            btn = QToolButton()
            btn.setIcon(qta.icon(icon_name, color="#9aa7b9"))
            btn.setFixedSize(32, 32)
            btn.setAutoRaise(True)
            is_close = icon_name == "mdi.close"
            hover_bg = "#e53935" if is_close else "rgba(255, 255, 255, 0.08)"
            btn.setStyleSheet(
                "QToolButton { background: transparent; border: none; border-radius: 10px; } "
                f"QToolButton:hover {{ background: {hover_bg}; }}"
            )
            btn.clicked.connect(signal.emit)
            row.addWidget(btn)
        row.addSpacing(8)

        self._accent = accent
        self._drag_offset: QPoint | None = None
        # Cache the last-applied state so set_accent() can re-skin the
        # accent-coupled bits in place after a live persona swap.
        self._last_status: tuple[str, bool] = ("booting", False)
        self._last_turn: tuple[str, bool] = ("YOUR TURN", False)
        self._last_mic_active = False
        self.set_status("booting", error=False)

    # ----- drag / maximise-on-double-click -----

    def _child_at(self, event: QMouseEvent) -> QWidget | None:
        """Return the child widget directly under the cursor (or None
        if the click landed on the bare QFrame background). We use this
        so clicks on buttons / the turn pill / labels DON'T trigger a
        window drag — they handle their own clicks."""
        return self.childAt(event.position().toPoint())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # QToolButton + QPushButton handle their own clicks; dragging
        # from them should do nothing. QLabels / stretch / the bare
        # frame allow drag.
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(
            self._child_at(event), (QToolButton, QPushButton)
        ):
            top = self.window()
            handle = top.windowHandle()
            pos = event.position().toPoint()
            # The top ~5px of the AppBar is the window's top-resize strip
            # — takes priority over drag so the user can actually grab
            # the edge. Corner hits include the adjacent horizontal edge.
            if pos.y() <= _TOP_RESIZE_STRIP:
                edges = Qt.Edge.TopEdge
                if pos.x() <= _TOP_RESIZE_STRIP:
                    edges |= Qt.Edge.LeftEdge
                elif pos.x() >= self.width() - _TOP_RESIZE_STRIP:
                    edges |= Qt.Edge.RightEdge
                if handle is not None and handle.startSystemResize(edges):
                    event.accept()
                    return
            # Prefer compositor-owned system move — the only drag path
            # that works on Wayland (where QWidget.move is a no-op) and
            # keeps drag smooth on X11. Falls through to a manual
            # offset-based drag if the platform can't service it.
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_offset = event.globalPosition().toPoint() - top.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            top = self.window()
            top.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            target = self._child_at(event)
            if not isinstance(target, (QToolButton, QPushButton)):
                self.toggle_maximize_clicked.emit()

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._last_status = (text, error)
        colour = "#ef4444" if error else self._accent.name()
        text_colour = "#ef4444" if error else "#8796a8"
        self._status_dot.setStyleSheet(
            f"background: {colour}; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;"
        )
        # Elide long status messages at a reasonable width so they don't
        # push the action buttons off the bar.
        self._status_text.setText(text)
        self._status_text.setStyleSheet(
            f"color: {text_colour}; font-family: monospace; font-size: 11px; padding-left: 4px;"
        )

    def set_turn_label(self, label: str, *, highlight: bool) -> None:
        self._last_turn = (label, highlight)
        # Uppercase in Python — Qt stylesheets don't support text-transform.
        self._turn_pill.setText(label.upper())
        if highlight:
            r, g, b = self._accent.red(), self._accent.green(), self._accent.blue()
            self._turn_pill.setStyleSheet(
                f"QLabel {{ background: rgba({r}, {g}, {b}, 0.14); "
                f"color: {self._accent.name()}; "
                f"border-radius: 8px; padding: 2px 10px; }}"
            )
        else:
            self._turn_pill.setStyleSheet(
                "QLabel { background: rgba(255, 255, 255, 0.04); "
                "color: #8896a8; border-radius: 8px; padding: 2px 10px; }"
            )

    def set_mic_active(self, active: bool) -> None:
        """Highlight the mic button while recording."""
        self._last_mic_active = active
        colour = self._accent.name() if active else "#9aa7b9"
        if active:
            r, g, b = self._accent.red(), self._accent.green(), self._accent.blue()
            bg = f"rgba({r}, {g}, {b}, 0.16)"
        else:
            bg = "transparent"
        self._mic_btn.setStyleSheet(
            f"QToolButton {{ background: {bg}; border: none; border-radius: 10px; color: {colour}; }} "
            "QToolButton:hover { background: rgba(255, 255, 255, 0.06); }"
        )
        self._mic_btn.setIcon(qta.icon("fa5s.microphone", color=colour))

    def set_accent(self, accent: QColor) -> None:
        """Re-skin every accent-coupled element after a live persona swap.

        Called from the window's persona-change path. Replays the last
        status / turn-pill / mic state through the existing setters so
        the new accent colour propagates without re-laying-out the bar.
        """
        self._accent = accent
        text, error = self._last_status
        self.set_status(text, error=error)
        label, highlight = self._last_turn
        self.set_turn_label(label, highlight=highlight)
        self.set_mic_active(self._last_mic_active)

    def _open_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #10161f; color: #dbe4ef; border-radius: 10px; padding: 6px; } "
            "QMenu::item { padding: 6px 14px; border-radius: 6px; } "
            "QMenu::item:selected { background: rgba(255, 255, 255, 0.08); }"
        )
        new_action = menu.addAction(qta.icon("fa5s.redo-alt", color="#9aa7b9"), "New game")
        settings_action = menu.addAction(qta.icon("fa5s.cog", color="#9aa7b9"), "Settings…")
        menu.addSeparator()
        about_action = menu.addAction(qta.icon("fa5s.info-circle", color="#9aa7b9"), "About RookApp")
        btn_rect = self._menu_btn.rect()
        pos = self._menu_btn.mapToGlobal(btn_rect.bottomRight())
        chosen = menu.exec(pos)
        if chosen is new_action:
            self.menu_triggered.emit("new_game")
        elif chosen is settings_action:
            self.menu_triggered.emit("settings")
        elif chosen is about_action:
            self.menu_triggered.emit("about")


class _InputBar(QWidget):
    submitted = Signal(str)

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._field = QLineEdit()
        self._field.setPlaceholderText("type a move or question — e.g. e4, Nf3, castle kingside")
        self._field.setStyleSheet(
            "QLineEdit { background: #0d1520; border: none; "
            "border-radius: 12px; color: #dbe4ef; padding: 10px 14px; "
            "font-family: monospace; font-size: 13px; } "
            f"QLineEdit:focus {{ background: #0f1824; color: {accent.name()}; }}"
        )
        self._field.returnPressed.connect(self._emit)
        row.addWidget(self._field, stretch=1)

        self._send = QPushButton()
        self._send.setIcon(qta.icon("fa5s.paper-plane", color="#0a0e14"))
        self._send.setFixedSize(42, 40)
        self._send.setStyleSheet(
            f"QPushButton {{ background: {accent.name()}; border: none; border-radius: 12px; }} "
            "QPushButton:disabled { background: #1e2b3a; }"
        )
        self._send.clicked.connect(self._emit)
        row.addWidget(self._send)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def set_enabled(self, enabled: bool) -> None:
        self._field.setEnabled(enabled)
        self._send.setEnabled(enabled)

    def _emit(self) -> None:
        text = self._field.text().strip()
        if not text:
            return
        self.submitted.emit(text)
        self._field.clear()


class _IconButton(QToolButton):
    def __init__(self, icon_name: str, tip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(qta.icon(icon_name, color="#9aa7b9"))
        self.setToolTip(tip)
        self.setFixedSize(32, 32)
        self.setAutoRaise(True)
        self.setStyleSheet(
            "QToolButton { background: transparent; border: none; border-radius: 10px; } "
            "QToolButton:hover { background: rgba(255, 255, 255, 0.06); }"
        )


__all__ = ["RookWindow"]
