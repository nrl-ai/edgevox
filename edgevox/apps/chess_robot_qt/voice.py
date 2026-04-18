"""Voice pipeline — mic → VAD → Whisper → user text.

Thin Qt adapter over :class:`edgevox.audio.AudioRecorder` (mic + silero
VAD + echo-aware barge-in detection) and :class:`WhisperSTT`. The
recorder is linked to the global :class:`InterruptiblePlayer` so TTS
playback automatically pauses the mic queue and resumes after the
cooldown — we don't re-implement that gate here.

Emits:
    - ``transcript`` — finalised user utterance.
    - ``level`` — 0..1 RMS for the mic indicator.
    - ``barge_in`` — user spoke over Rook mid-reply; the bridge must
      cut TTS + cancel the in-flight agent turn.
    - ``error`` / ``loading`` / ``ready`` — lifecycle.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

if TYPE_CHECKING:
    from edgevox.audio import AudioRecorder
    from edgevox.stt.whisper import WhisperSTT

log = logging.getLogger(__name__)


class VoiceWorker(QObject):
    """Qt-friendly mic pipeline.

    Call :meth:`start` after construction; STT + mic load on a worker
    thread. The recorder links itself to the global
    :class:`InterruptiblePlayer` so :meth:`InterruptiblePlayer.play` pauses
    mic capture at the source (queue never fills with TTS echo) and
    resumes after :data:`ECHO_COOLDOWN_SECS`. :meth:`set_listening`
    is still available but only gates STT dispatch on user request —
    echo suppression is automatic.
    """

    transcript = Signal(str)
    level = Signal(float)
    error = Signal(str)
    loading = Signal(bool)
    ready = Signal()
    # User spoke over the bot mid-reply. The bridge must cut TTS and
    # cancel the in-flight agent turn.
    barge_in = Signal()

    def __init__(
        self,
        *,
        language: str = "en",
        input_device: int | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._language = language
        self._input_device = input_device
        self._recorder: AudioRecorder | None = None
        self._stt: WhisperSTT | None = None
        self._listening = False  # True while Rook is idle / expecting user input
        self._pool = QThreadPool.globalInstance()
        self._shutdown = threading.Event()
        self._load_lock = threading.Lock()

    # ----- lifecycle -----

    def start(self) -> None:
        """Kick off mic + STT load on a worker. Safe to call multiple
        times; the first call wins and later calls no-op."""
        with self._load_lock:
            if self._recorder is not None or self._shutdown.is_set():
                return
        self.loading.emit(True)
        self._pool.start(_StartupJob(self))

    def stop(self) -> None:
        """Stop mic capture and release resources. Call on app exit."""
        self._shutdown.set()
        self._listening = False
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                log.exception("recorder.stop() failed")
        self._recorder = None

    def set_listening(self, on: bool) -> None:
        """Gate whether mic-detected speech reaches Whisper. We keep
        the recorder running (for level meters) but drop transcription
        input when off — cheaper than stop/start cycles that re-open
        the audio device."""
        self._listening = on

    def is_listening(self) -> bool:
        return self._listening

    # ----- worker-thread body -----

    def _boot(self) -> None:
        """Build the recorder + STT on a background thread. Runs once."""
        try:
            # Deferred imports so ``VoiceWorker()`` in the main thread
            # stays cheap; we only pay STT init when the user actually
            # wants voice.
            from edgevox.audio import AudioRecorder, player
            from edgevox.stt.whisper import WhisperSTT

            log.info("Loading Whisper STT for voice input...")
            stt = WhisperSTT()
            if self._shutdown.is_set():
                return
            self._stt = stt

            # ``player_ref=player`` lets the recorder read the live TTS
            # output RMS to apply the 3x energy-ratio gate — the
            # real defence against self-trigger when the bot is speaking.
            recorder = AudioRecorder(
                on_speech=self._on_speech_segment,
                on_interrupt=self._on_interrupt,
                on_level=self._on_level,
                device=self._input_device,
                player_ref=player,
            )
            # Bi-directional link: ``player.play()`` calls
            # ``recorder.pause()`` at start of playback and
            # ``recorder.resume_after_cooldown()`` after, so we don't
            # queue TTS echo into the VAD stream in the first place.
            player.link_recorder(recorder)
            recorder.start()
            if self._shutdown.is_set():
                recorder.stop()
                return
            self._recorder = recorder
            self.loading.emit(False)
            self.ready.emit()
            log.info("Voice pipeline ready.")
        except Exception as e:
            log.exception("Voice startup failed")
            self.loading.emit(False)
            # Common-case guidance — mic permission is by far the most
            # frequent cause.
            msg = str(e)
            if "Invalid input device" in msg or "no default" in msg.lower():
                self.error.emit("No microphone detected. Connect one and restart the app.")
            elif "denied" in msg.lower() or "permission" in msg.lower():
                self.error.emit("Microphone permission denied — enable it in your OS settings.")
            else:
                self.error.emit(f"Voice setup failed: {msg}")

    # ----- recorder callbacks (audio thread) -----

    def _on_speech_segment(self, audio: np.ndarray) -> None:
        """A VAD-bounded speech chunk is ready. Dispatch to Whisper on
        a pool worker so the mic loop keeps draining."""
        if not self._listening or self._stt is None:
            return
        self._pool.start(_TranscribeJob(self, audio.copy()))

    def _on_interrupt(self) -> None:
        """Audio thread: user spoke over Rook mid-TTS. Re-emit as Qt
        signal so the bridge can tear down the in-flight turn and the
        TTS worker can cut playback. The recorder preserves the
        user's continuing speech for the next STT pass on its own."""
        if self._listening:
            self.barge_in.emit()

    def _on_level(self, level: float) -> None:
        """Mic RMS for the indicator."""
        # AudioRecorder reports 0..~1 already; clamp for safety.
        self.level.emit(max(0.0, min(1.0, level)))

    def _transcribe(self, audio: np.ndarray) -> None:
        if self._stt is None or self._shutdown.is_set():
            return
        try:
            text = self._stt.transcribe(audio, language=self._language).strip()
        except Exception as e:
            log.exception("Whisper transcribe failed")
            self.error.emit(f"STT error: {e}")
            return
        if text:
            self.transcript.emit(text)


class _StartupJob(QRunnable):
    def __init__(self, worker: VoiceWorker) -> None:
        super().__init__()
        self._worker = worker

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._boot()


class _TranscribeJob(QRunnable):
    def __init__(self, worker: VoiceWorker, audio: np.ndarray) -> None:
        super().__init__()
        self._worker = worker
        self._audio = audio

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._transcribe(self._audio)


__all__ = ["VoiceWorker"]
