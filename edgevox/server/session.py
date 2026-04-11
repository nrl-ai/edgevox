"""Per-WebSocket session state.

Each browser tab maps to one ``SessionState`` that owns:
  * its own Silero VAD instance (the model has streaming LSTM state),
  * its own conversation history (swapped into the shared LLM under the lock),
  * its own interrupt flag.

The VAD/segmentation loop mirrors ``AudioRecorder._process_loop`` in
``edgevox/audio/_original.py`` so behavior matches the TUI exactly. We do not
reuse ``AudioRecorder`` itself because it owns a sounddevice input stream — we
already have audio coming over the wire.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterator

import numpy as np

from edgevox.audio._original import (
    SILENCE_FRAMES_THRESHOLD,
    TARGET_SAMPLE_RATE,
    VAD,
    VAD_SAMPLES,
)

log = logging.getLogger(__name__)


def chunk_pcm(samples: np.ndarray, frame_size: int = VAD_SAMPLES) -> Iterator[np.ndarray]:
    """Split a flat float32 PCM buffer into fixed-size frames.

    Pads the final partial frame with zeros — VAD requires exactly ``frame_size``
    samples per call. Frames are full-size copies safe to hand to the VAD.
    """
    if samples.size == 0:
        return
    n = samples.size
    for start in range(0, n, frame_size):
        chunk = samples[start : start + frame_size]
        if chunk.size < frame_size:
            chunk = np.pad(chunk, (0, frame_size - chunk.size))
        yield chunk.astype(np.float32, copy=False)


class SessionState:
    """One conversation: VAD state machine + history + interrupt flag.

    Audio in: ``feed_audio`` is called with float32 mono @ 16 kHz buffers of
    arbitrary length. When a complete speech segment is detected (≥
    SILENCE_FRAMES_THRESHOLD silent frames after speech), the segment is yielded
    via ``drain_segments``. Callers run ``drain_segments`` after every feed.
    """

    def __init__(
        self,
        language: str,
        history: list[dict],
        session_id: str | None = None,
        vad: VAD | None = None,
    ):
        self.id = session_id or uuid.uuid4().hex[:12]
        self.language = language
        self.history: list[dict] = history
        self.created_at = time.time()
        self.last_active = self.created_at

        # ``vad`` is injectable so unit tests can run without onnxruntime/Silero.
        self._vad = vad if vad is not None else VAD()
        self._speech_buffer: list[np.ndarray] = []
        self._silence_count = 0
        self._in_speech = False
        self._pending_segments: list[np.ndarray] = []
        self._level = 0.0

        self.interrupt_event = asyncio.Event()
        self.busy = False  # True while a turn is being processed

    @property
    def level(self) -> float:
        return self._level

    def feed_audio(self, samples: np.ndarray) -> None:
        """Push a buffer of float32 PCM @ 16 kHz through VAD."""
        if samples.size == 0:
            return
        self.last_active = time.time()
        for frame in chunk_pcm(samples):
            rms = float(np.sqrt(np.mean(frame**2)))
            self._level = min(1.0, rms * 10.0)

            is_speech = self._vad.is_speech(frame)
            if is_speech:
                self._speech_buffer.append(frame)
                self._silence_count = 0
                self._in_speech = True
            elif self._in_speech:
                self._speech_buffer.append(frame)
                self._silence_count += 1
                if self._silence_count >= SILENCE_FRAMES_THRESHOLD:
                    segment = np.concatenate(self._speech_buffer)
                    self._speech_buffer.clear()
                    self._silence_count = 0
                    self._in_speech = False
                    self._vad.reset()
                    self._pending_segments.append(segment)

    def drain_segments(self) -> list[np.ndarray]:
        """Pop any complete speech segments collected by ``feed_audio``."""
        if not self._pending_segments:
            return []
        out = self._pending_segments
        self._pending_segments = []
        return out

    def reset_audio(self) -> None:
        """Drop any in-progress speech buffer (used after interrupts)."""
        self._speech_buffer.clear()
        self._silence_count = 0
        self._in_speech = False
        self._pending_segments.clear()
        self._vad.reset()

    @staticmethod
    def segment_duration(segment: np.ndarray) -> float:
        return float(segment.size) / TARGET_SAMPLE_RATE
