"""Text-to-speech pipeline — Kokoro → sounddevice playback.

Thin Qt adapter over the engine's frame pipeline:

    Pipeline([SentenceSplitter, TTSProcessor, PlaybackProcessor])

running per utterance. Benefits over a single-shot synth+play:

- Sentences stream to the speaker as they leave TTS — first speech
  lands in ~100-200 ms instead of after the full reply is synthesised.
- :meth:`interrupt` cascades through :meth:`Pipeline.interrupt`, which
  calls :meth:`PlaybackProcessor.on_interrupt` → :meth:`player.interrupt`
  atomically; in-flight synth + playback both bail out.
- :class:`PlaybackProcessor` uses the global
  :class:`InterruptiblePlayer` which is already linked to our
  :class:`AudioRecorder` (see :mod:`voice`), so pause/resume of the mic
  queue is automatic — no separate gate to maintain here.

Kokoro is MIT-licensed — safe to ship inside a MIT app.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from edgevox.audio import player
from edgevox.core.frames import EndFrame, Pipeline, TextFrame
from edgevox.core.processors import PlaybackProcessor, SentenceSplitter, TTSProcessor

log = logging.getLogger(__name__)


class TTSWorker(QObject):
    """Qt-friendly Kokoro wrapper driven through the engine pipeline."""

    started = Signal()
    finished = Signal()
    error = Signal(str)
    ready = Signal()

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        lang_code: str = "a",
        output_device: int | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._voice = voice
        self._lang_code = lang_code
        self._tts = None  # lazy-loaded KokoroTTS
        self._pool = QThreadPool.globalInstance()
        self._shutdown = threading.Event()
        self._ready_lock = threading.Lock()
        self._warming = False
        self._active_lock = threading.Lock()
        self._active_pipeline: Pipeline | None = None
        self._active_pipeline_lock = threading.Lock()
        # Queue at most ONE pending utterance while the model warms up
        # so the first reply the user hears is the latest one, not some
        # stale "thinking..." line. Later replies overwrite the pending
        # slot. Once the model loads we drain the slot.
        self._pending_text: str | None = None
        self._pending_lock = threading.Lock()
        if output_device is not None:
            player.set_device(output_device)

    # ----- lifecycle -----

    def start(self) -> None:
        """Kick off Kokoro load on a background worker."""
        with self._ready_lock:
            if self._tts is not None or self._warming or self._shutdown.is_set():
                return
            self._warming = True
        self._pool.start(_WarmupJob(self))

    def stop(self) -> None:
        self._shutdown.set()
        self.interrupt()

    def set_output_device(self, device: int | None) -> None:
        """Route playback to a different output device."""
        player.set_device(device)

    # ----- public API -----

    def speak(self, text: str) -> None:
        """Schedule text → speech playback on a pool worker.

        If the model is still warming up, park the text in the single
        pending slot — the warmup path drains it when ready. Only the
        *latest* pending text is kept so a slow-booting model never
        replays stale earlier replies.
        """
        text = (text or "").strip()
        if not text or self._shutdown.is_set():
            return
        if self._tts is None:
            with self._pending_lock:
                self._pending_text = text
            return
        self._pool.start(_SpeakJob(self, text))

    def interrupt(self) -> None:
        """Cut any in-flight TTS synth + playback immediately.

        Safe to call when nothing is playing — no-ops. Used by the
        barge-in path (user spoke over Rook).
        """
        with self._active_pipeline_lock:
            pipe = self._active_pipeline
        if pipe is not None:
            pipe.interrupt()
        # Always cut the player too — Pipeline.interrupt fires
        # PlaybackProcessor.on_interrupt which does this, but a barge-in
        # can arrive in the gap between SentenceSplitter emitting and
        # PlaybackProcessor running. Belt + braces.
        player.interrupt()

    # ----- worker bodies -----

    def _warmup(self) -> None:
        try:
            from edgevox.tts.kokoro import KokoroTTS

            log.info("Loading Kokoro TTS for Rook...")
            tts = KokoroTTS(voice=self._voice, lang_code=self._lang_code)
            if self._shutdown.is_set():
                return
            self._tts = tts
            self.ready.emit()
            log.info("Kokoro TTS ready.")
            # Drain a reply that arrived during warmup so the user
            # actually hears Rook's first turn.
            with self._pending_lock:
                pending, self._pending_text = self._pending_text, None
            if pending:
                self._pool.start(_SpeakJob(self, pending))
        except Exception as e:
            log.exception("TTS warmup failed")
            self.error.emit(f"Voice output unavailable: {e}")
        finally:
            self._warming = False

    def _speak(self, text: str) -> None:
        if self._tts is None or self._shutdown.is_set():
            return
        # Serialise playback: the user hears one reply at a time. A
        # reply mid-flight is dropped rather than queued; the UI only
        # ever asks us to speak the latest reply.
        if not self._active_lock.acquire(blocking=False):
            return
        pipeline = Pipeline([SentenceSplitter(), TTSProcessor(self._tts), PlaybackProcessor()])
        with self._active_pipeline_lock:
            self._active_pipeline = pipeline
        try:
            self.started.emit()
            try:
                # ``EndFrame`` is how :class:`SentenceSplitter` knows to
                # flush its internal buffer — without it, a reply that
                # doesn't terminate in ``.!?`` (or the tail after the
                # last punctuation) would never leave the splitter and
                # the user hears nothing. Kept separate from the
                # input ``TextFrame`` to match the LLMProcessor →
                # SentenceSplitter contract used by the TUI / CLI.
                for _frame in pipeline.run([TextFrame(text=text), EndFrame()]):
                    if self._shutdown.is_set():
                        pipeline.interrupt()
                        break
            except Exception as e:
                log.exception("TTS pipeline failed")
                self.error.emit(f"TTS error: {e}")
        finally:
            with self._active_pipeline_lock:
                self._active_pipeline = None
            self.finished.emit()
            self._active_lock.release()


class _WarmupJob(QRunnable):
    def __init__(self, worker: TTSWorker) -> None:
        super().__init__()
        self._worker = worker

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._warmup()


class _SpeakJob(QRunnable):
    def __init__(self, worker: TTSWorker, text: str) -> None:
        super().__init__()
        self._worker = worker
        self._text = text

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._speak(self._text)


__all__ = ["TTSWorker"]
