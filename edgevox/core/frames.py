"""Frame types, processor base, and pipeline runner.

Frames are typed data objects that flow through a pipeline of processors.
Each processor receives frames, transforms them, and yields output frames.
The pipeline chains processors via generators for zero-overhead streaming.

Interrupt propagation: when ``Pipeline.interrupt()`` is called (e.g., user
speaks over the bot), an ``InterruptFrame`` flows through the chain and each
processor's ``on_interrupt()`` is called for cleanup.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame types
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """Base frame flowing through the pipeline."""


@dataclass
class AudioFrame(Frame):
    """Raw audio from the microphone (16 kHz mono float32)."""

    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = 16_000


@dataclass
class TranscriptionFrame(Frame):
    """User speech transcription from STT."""

    text: str = ""
    stt_time: float = 0.0
    audio_duration: float = 0.0


@dataclass
class TextFrame(Frame):
    """A chunk of text (LLM token)."""

    text: str = ""


@dataclass
class SentenceFrame(Frame):
    """A complete sentence ready for TTS."""

    text: str = ""


@dataclass
class TTSAudioFrame(Frame):
    """Synthesised audio from TTS."""

    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = 24_000
    sentence: str = ""


@dataclass
class InterruptFrame(Frame):
    """Signals that the user interrupted — processors should clean up."""


@dataclass
class EndFrame(Frame):
    """Signals end of the current turn (LLM finished generating)."""


@dataclass
class MetricsFrame(Frame):
    """Carries timing / metrics data alongside the pipeline."""

    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Processor base class
# ---------------------------------------------------------------------------


class Processor:
    """Base pipeline processor.  Subclass and override ``process()``.

    Processing patterns:

    * **1:1** — yield one output per input (e.g., STT: AudioFrame -> TextFrame)
    * **1:N** — yield multiple outputs (e.g., LLM: TextFrame -> many TextFrames)
    * **N:1** — buffer internally, yield when ready (e.g., sentence splitter)
    * **passthrough** — yield the input frame unchanged
    """

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        """Receive one frame, yield zero or more output frames."""
        yield frame

    def on_interrupt(self) -> None:
        """Called when an ``InterruptFrame`` arrives.  Override to clean up."""

    def close(self) -> None:
        """Called when the pipeline shuts down."""


# ---------------------------------------------------------------------------
# Interrupt token
# ---------------------------------------------------------------------------


class InterruptToken:
    """Thread-safe interrupt flag shared between the pipeline and the caller."""

    def __init__(self):
        self._event = threading.Event()

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class Pipeline:
    """Chains processors: feeds output frames of one as input to the next.

    Usage::

        pipeline = Pipeline([
            STTProcessor(stt),
            LLMProcessor(llm),
            SentenceSplitter(),
            TTSProcessor(tts),
            PlaybackProcessor(),
        ])

        for frame in pipeline.run([AudioFrame(audio=mic_audio)]):
            if isinstance(frame, InterruptFrame):
                break
            ...
    """

    def __init__(self, processors: list[Processor]):
        self.processors = processors
        self._interrupt = InterruptToken()

    def run(self, input_frames: Iterable[Frame]) -> Generator[Frame, None, None]:
        """Run *input_frames* through all processors in sequence."""
        stream: Iterable[Frame] = input_frames
        for proc in self.processors:
            stream = self._chain(proc, stream)
        yield from stream

    def _chain(self, proc: Processor, input_stream: Iterable[Frame]) -> Generator[Frame, None, None]:
        for frame in input_stream:
            # External interrupt (e.g., user spoke)
            if self._interrupt.is_set:
                proc.on_interrupt()
                yield InterruptFrame()
                return

            # Propagate interrupt frames downstream
            if isinstance(frame, InterruptFrame):
                proc.on_interrupt()
                yield frame
                return

            # Normal processing
            for out in proc.process(frame):
                if self._interrupt.is_set:
                    proc.on_interrupt()
                    yield InterruptFrame()
                    return
                yield out

    def interrupt(self):
        """Signal an interrupt — the pipeline will stop and yield an ``InterruptFrame``.

        Immediately calls ``on_interrupt()`` on all processors so they can bail
        out of any in-progress blocking calls (e.g., stop iterating LLM tokens,
        abort TTS synthesis).  The pipeline runner will also check the flag
        between frames and yield an ``InterruptFrame``.
        """
        self._interrupt.set()
        for proc in self.processors:
            try:
                proc.on_interrupt()
            except Exception:
                log.exception("Error in on_interrupt for %s", proc)

    def close(self):
        """Shut down all processors."""
        for proc in self.processors:
            proc.close()
