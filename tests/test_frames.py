"""Tests for edgevox.core.frames -- Frame types, Processor, Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from edgevox.core.frames import (
    AudioFrame,
    EndFrame,
    Frame,
    InterruptFrame,
    InterruptToken,
    MetricsFrame,
    Pipeline,
    Processor,
    SentenceFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioFrame,
)


class TestFrames:
    def test_audio_frame(self):
        audio = np.zeros(512, dtype=np.float32)
        f = AudioFrame(audio=audio, sample_rate=16000)
        assert f.audio is audio
        assert f.sample_rate == 16000

    def test_text_frame(self):
        f = TextFrame(text="hello")
        assert f.text == "hello"

    def test_transcription_frame(self):
        f = TranscriptionFrame(text="hi", stt_time=0.5, audio_duration=1.2)
        assert f.text == "hi"
        assert f.stt_time == 0.5
        assert f.audio_duration == 1.2

    def test_sentence_frame(self):
        f = SentenceFrame(text="A sentence.")
        assert f.text == "A sentence."

    def test_tts_audio_frame(self):
        audio = np.zeros(100, dtype=np.float32)
        f = TTSAudioFrame(audio=audio, sample_rate=24000, sentence="hi")
        assert f.audio is audio
        assert f.sample_rate == 24000
        assert f.sentence == "hi"

    def test_interrupt_frame(self):
        f = InterruptFrame()
        assert isinstance(f, Frame)

    def test_end_frame(self):
        f = EndFrame()
        assert isinstance(f, Frame)

    def test_metrics_frame(self):
        f = MetricsFrame(metrics={"stt": 0.5})
        assert f.metrics == {"stt": 0.5}


class TestInterruptToken:
    def test_initial_state(self):
        t = InterruptToken()
        assert t.is_set is False

    def test_set(self):
        t = InterruptToken()
        t.set()
        assert t.is_set is True

    def test_clear(self):
        t = InterruptToken()
        t.set()
        t.clear()
        assert t.is_set is False


class TestProcessor:
    def test_default_passthrough(self):
        p = Processor()
        frame = TextFrame(text="hi")
        out = list(p.process(frame))
        assert out == [frame]

    def test_on_interrupt_is_noop(self):
        Processor().on_interrupt()  # should not raise

    def test_close_is_noop(self):
        Processor().close()


class UpperCaseProcessor(Processor):
    """Test processor: uppercases text frames."""

    def process(self, frame):
        if isinstance(frame, TextFrame):
            yield TextFrame(text=frame.text.upper())
        else:
            yield frame


class DuplicateProcessor(Processor):
    """Test processor: yields each frame twice (1:N)."""

    def process(self, frame):
        yield frame
        yield frame


class BufferProcessor(Processor):
    """Test processor: buffers N frames, yields concatenated (N:1)."""

    def __init__(self, n=3):
        self.n = n
        self._buffer = []

    def process(self, frame):
        if isinstance(frame, TextFrame):
            self._buffer.append(frame.text)
            if len(self._buffer) >= self.n:
                yield TextFrame(text="".join(self._buffer))
                self._buffer.clear()
        else:
            yield frame


class TestPipeline:
    def test_empty_pipeline(self):
        pipe = Pipeline([])
        frames = [TextFrame(text="hi")]
        out = list(pipe.run(frames))
        assert len(out) == 1
        assert out[0].text == "hi"

    def test_single_processor(self):
        pipe = Pipeline([UpperCaseProcessor()])
        out = list(pipe.run([TextFrame(text="hello")]))
        assert len(out) == 1
        assert out[0].text == "HELLO"

    def test_chained_processors(self):
        pipe = Pipeline([UpperCaseProcessor(), UpperCaseProcessor()])
        out = list(pipe.run([TextFrame(text="hi")]))
        assert out[0].text == "HI"  # idempotent

    def test_one_to_many(self):
        pipe = Pipeline([DuplicateProcessor()])
        out = list(pipe.run([TextFrame(text="hi")]))
        assert len(out) == 2

    def test_many_to_one_buffering(self):
        pipe = Pipeline([BufferProcessor(n=3)])
        frames = [TextFrame(text="a"), TextFrame(text="b"), TextFrame(text="c")]
        out = list(pipe.run(frames))
        assert len(out) == 1
        assert out[0].text == "abc"

    def test_interrupt_propagates(self):
        """When pipeline.interrupt() is called, InterruptFrame is yielded."""
        frames_processed = []

        class TrackProcessor(Processor):
            def process(self, frame):
                frames_processed.append(frame)
                yield frame

        pipe = Pipeline([TrackProcessor()])

        def gen():
            yield TextFrame(text="a")
            pipe.interrupt()
            yield TextFrame(text="b")
            yield TextFrame(text="c")

        out = list(pipe.run(gen()))
        # Should see the first frame plus an InterruptFrame
        assert any(isinstance(f, InterruptFrame) for f in out)

    def test_interrupt_calls_on_interrupt(self):
        interrupt_called = MagicMock()

        class CleanupProcessor(Processor):
            def on_interrupt(self):
                interrupt_called()

        pipe = Pipeline([CleanupProcessor()])
        pipe.interrupt()
        list(pipe.run([TextFrame(text="x")]))
        interrupt_called.assert_called()

    def test_interrupt_frame_propagates_downstream(self):
        """An InterruptFrame from upstream flows through all processors."""
        downstream_got_interrupt = MagicMock()

        class DownstreamProcessor(Processor):
            def on_interrupt(self):
                downstream_got_interrupt()

        pipe = Pipeline([Processor(), DownstreamProcessor()])
        out = list(pipe.run([InterruptFrame()]))
        assert any(isinstance(f, InterruptFrame) for f in out)
        downstream_got_interrupt.assert_called()

    def test_close_calls_all_processors(self):
        closes = []

        class CloseTracker(Processor):
            def __init__(self, name):
                self.name = name

            def close(self):
                closes.append(self.name)

        pipe = Pipeline([CloseTracker("a"), CloseTracker("b")])
        pipe.close()
        assert closes == ["a", "b"]
