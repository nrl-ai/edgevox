"""Concrete pipeline processors wrapping EdgeVox backends.

Each processor adapts an existing backend (STT, LLM, TTS, playback) to the
frame-based pipeline interface defined in ``edgevox.core.frames``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Generator

from edgevox.audio import play_audio, player
from edgevox.core.frames import (
    AudioFrame,
    EndFrame,
    Frame,
    InterruptFrame,
    MetricsFrame,
    Processor,
    SentenceFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioFrame,
)
from edgevox.core.pipeline import MAX_CHUNK_CHARS, _find_sentence_break
from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


class STTProcessor(Processor):
    """Speech-to-text: AudioFrame -> TranscriptionFrame.

    Wraps ``BaseSTT.transcribe()`` (batch — full audio segment at once).
    Note: once ``transcribe()`` is running, it cannot be interrupted; the
    interrupt will only take effect after it returns.
    """

    def __init__(self, stt, language: str = "en"):
        self.stt = stt
        self.language = language
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, AudioFrame):
            t0 = time.perf_counter()
            text = self.stt.transcribe(frame.audio, language=self.language)
            t_stt = time.perf_counter() - t0
            audio_duration = len(frame.audio) / frame.sample_rate
            if self._interrupted:
                return
            yield MetricsFrame(metrics={"stt": t_stt, "audio_duration": audio_duration})
            if text and not text.isspace():
                yield TranscriptionFrame(text=text, stt_time=t_stt, audio_duration=audio_duration)
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class LLMProcessor(Processor):
    """LLM chat: TextFrame -> many TextFrames (token stream) + EndFrame.

    Wraps ``LLM.chat_stream()`` which yields tokens as a generator.
    """

    def __init__(self, llm):
        self.llm = llm
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, (TextFrame, TranscriptionFrame)):
            # Pass the input frame through first so downstream can observe it
            # (e.g., UI displays the user's transcription).
            yield frame
            t0 = time.perf_counter()
            first = True
            stream = self.llm.chat_stream(frame.text)
            try:
                for token in stream:
                    if self._interrupted:
                        break
                    if first:
                        yield MetricsFrame(metrics={"ttft": time.perf_counter() - t0})
                        first = False
                    yield TextFrame(text=token)
            finally:
                # Close in a daemon thread — llama-cpp's stream.close() can
                # block if the model is mid-forward-pass; we don't want to
                # freeze the main pipeline waiting for it.
                def _bg_close(s=stream):
                    with contextlib.suppress(Exception):
                        s.close()

                threading.Thread(target=_bg_close, daemon=True).start()
            yield EndFrame()
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


class SentenceSplitter(Processor):
    """Accumulates LLM tokens into complete sentences.

    TextFrame tokens are buffered until a sentence boundary (``.!?``) is
    detected, then yielded as a ``SentenceFrame``.  Reuses the
    abbreviation-aware splitting logic from ``edgevox.core.pipeline``.
    """

    def __init__(self):
        self._buffer: str = ""
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, TextFrame):
            # Pass the token through for observers (e.g., ROS2 bridge)
            yield frame
            self._buffer += frame.text

            # Extract complete sentences
            while True:
                pos = _find_sentence_break(self._buffer)
                if pos is None:
                    break
                sentence = self._buffer[:pos].strip()
                if sentence:
                    yield SentenceFrame(text=sentence)
                self._buffer = self._buffer[pos:].lstrip()

            # Break very long clauses at comma/semicolon
            if len(self._buffer) > MAX_CHUNK_CHARS:
                for sep in ["; ", ", ", ": "]:
                    idx = self._buffer.rfind(sep, MAX_CHUNK_CHARS // 2)
                    if idx > 0:
                        chunk = self._buffer[: idx + len(sep)].strip()
                        if chunk:
                            yield SentenceFrame(text=chunk)
                        self._buffer = self._buffer[idx + len(sep) :]
                        break

        elif isinstance(frame, EndFrame):
            # Flush remaining text
            remaining = self._buffer.strip()
            if remaining:
                yield SentenceFrame(text=remaining)
            self._buffer = ""
            yield frame
        else:
            yield frame

    def on_interrupt(self):
        self._buffer = ""
        self._interrupted = True


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TTSProcessor(Processor):
    """Text-to-speech: SentenceFrame -> TTSAudioFrame(s).

    Uses ``synthesize_stream()`` when the backend supports true streaming
    (e.g., Kokoro), otherwise falls back to batch ``synthesize()``.
    """

    def __init__(self, tts):
        self.tts = tts
        self._supports_stream = type(tts).synthesize_stream is not BaseTTS.synthesize_stream
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, SentenceFrame):
            # Pass the sentence through so downstream can display it before
            # we start generating audio (reduces perceived latency).
            yield frame
            t0 = time.perf_counter()
            if self._supports_stream:
                for chunk in self.tts.synthesize_stream(frame.text):
                    if self._interrupted:
                        return
                    yield TTSAudioFrame(audio=chunk, sample_rate=self.tts.sample_rate, sentence=frame.text)
            else:
                audio = self.tts.synthesize(frame.text)
                if self._interrupted:
                    return
                yield TTSAudioFrame(audio=audio, sample_rate=self.tts.sample_rate, sentence=frame.text)
            yield MetricsFrame(metrics={"tts_sentence": time.perf_counter() - t0})
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


class PlaybackProcessor(Processor):
    """Plays TTS audio through speakers.  Yields ``InterruptFrame`` if
    playback is interrupted (``play_audio`` returns False).
    """

    def __init__(self):
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            yield InterruptFrame()
            return
        if isinstance(frame, TTSAudioFrame):
            completed = play_audio(frame.audio, sample_rate=frame.sample_rate)
            if not completed or self._interrupted:
                yield InterruptFrame()
                return
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True
        player.interrupt()
