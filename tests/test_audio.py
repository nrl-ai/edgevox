"""Tests for edgevox.audio — _resample, InterruptiblePlayer, AudioRecorder, and helpers.

All hardware-dependent code (sounddevice, onnxruntime) is mocked.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_rate_returns_same(self):
        from edgevox.audio._original import _resample

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resample(data, 16000, 16000)
        np.testing.assert_array_equal(result, data)

    def test_downsample(self):
        from edgevox.audio._original import _resample

        data = np.arange(48000, dtype=np.float32)
        result = _resample(data, 48000, 16000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_upsample(self):
        from edgevox.audio._original import _resample

        data = np.arange(16000, dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert len(result) == 48000

    def test_preserves_endpoints(self):
        from edgevox.audio._original import _resample

        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert abs(result[0] - 0.0) < 0.01
        assert abs(result[-1] - 1.0) < 0.01

    def test_single_sample(self):
        from edgevox.audio._original import _resample

        data = np.array([0.5], dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# InterruptiblePlayer (with mocked sounddevice)
# ---------------------------------------------------------------------------


class TestInterruptiblePlayer:
    @patch("edgevox.audio._original._sd")
    def test_init_state(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        assert p.is_playing is False

    @patch("edgevox.audio._original._sd")
    def test_interrupt_sets_stop(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        p.interrupt()
        assert p._stop.is_set()

    @patch("edgevox.audio._original._sd")
    def test_link_recorder(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        recorder = MagicMock()
        p.link_recorder(recorder)
        assert p._recorder is recorder
        p.link_recorder(None)
        assert p._recorder is None

    @patch("edgevox.audio._original._sd")
    def test_set_device_closes_stream_on_change(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        mock_stream = MagicMock()
        p._stream = mock_stream
        p._device = 0
        p.set_device(1)
        assert p._device == 1
        mock_stream.stop.assert_called_once()

    @patch("edgevox.audio._original._sd")
    def test_set_device_noop_on_same(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        p._device = 0
        p.set_device(0)
        # No stream close needed

    @patch("edgevox.audio._original._sd")
    def test_shutdown(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        mock_stream = MagicMock()
        p._stream = mock_stream
        p.shutdown()
        assert p._stream is None
        mock_stream.stop.assert_called_once()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_enqueues_and_drains(self, _mock_sd_fn, mock_sd_module):
        """play() enqueues audio for the audio-thread callback and waits for drain."""
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.latency = 0.0
        mock_sd_module.OutputStream.return_value = mock_stream
        mock_sd_module.query_devices.return_value = {"max_output_channels": 2}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1
        audio = np.ones(1200, dtype=np.float32)  # 50ms at 24kHz

        # Simulate the PortAudio callback draining the buffer in chunks.
        def drain():
            _time.sleep(0.005)
            while p._play_buf.shape[0] > 0:
                n = min(256, p._play_buf.shape[0])
                out = np.zeros((n, p._channels), dtype=np.float32)
                p._callback(out, n, None, None)
                _time.sleep(0.001)

        t = threading.Thread(target=drain)
        t.start()
        result = p.play(audio, sample_rate=24000)
        t.join()

        assert result is True
        assert p._play_buf.shape[0] == 0

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_interrupted_returns_false(self, _mock_sd_fn, mock_sd_module):
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.latency = 0.0
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1

        def interrupt_soon():
            _time.sleep(0.03)
            p._stop.set()

        # No drainer here — the buffer never empties on its own, so play()
        # only returns once interrupt_soon() flips _stop.
        t = threading.Thread(target=interrupt_soon)
        t.start()
        audio = np.zeros(24000 * 10, dtype=np.float32)  # very long audio
        result = p.play(audio, sample_rate=24000)
        t.join()
        assert result is False
        assert p._play_buf.shape[0] == 0

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_pauses_linked_recorder(self, _mock_sd_fn, mock_sd_module):
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.latency = 0.0
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        recorder = MagicMock()
        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1
        p.link_recorder(recorder)

        def drain():
            _time.sleep(0.005)
            while p._play_buf.shape[0] > 0:
                n = p._play_buf.shape[0]
                out = np.zeros((n, p._channels), dtype=np.float32)
                p._callback(out, n, None, None)

        t = threading.Thread(target=drain)
        t.start()
        audio = np.zeros(100, dtype=np.float32)
        p.play(audio, sample_rate=24000)
        t.join()

        recorder.pause.assert_called_once()
        recorder.resume_after_cooldown.assert_called_once()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_set_device_waits_for_inflight_play(self, _mock_sd_fn, mock_sd_module):
        """set_device interrupts an in-flight play() and waits for it to release the lock."""
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1
        p._device = 0

        play_returned = []

        def long_play():
            audio = np.zeros(24000 * 10, dtype=np.float32)
            play_returned.append(p.play(audio, sample_rate=24000))

        t = threading.Thread(target=long_play)
        t.start()
        _time.sleep(0.03)  # let play() get into the wait loop
        p.set_device(1)  # should interrupt the play and then close the stream
        t.join(timeout=1.0)

        assert not t.is_alive()
        assert play_returned == [False]
        assert p._device == 1
        assert p._stream is None
        assert not p._stop.is_set()  # cleared so future plays work

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_shutdown_waits_for_inflight_play(self, _mock_sd_fn, mock_sd_module):
        """shutdown interrupts an in-flight play() and waits for it to release the lock."""
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1

        play_returned = []

        def long_play():
            audio = np.zeros(24000 * 10, dtype=np.float32)
            play_returned.append(p.play(audio, sample_rate=24000))

        t = threading.Thread(target=long_play)
        t.start()
        _time.sleep(0.03)
        p.shutdown()
        t.join(timeout=1.0)

        assert not t.is_alive()
        assert play_returned == [False]
        assert p._stream is None


# ---------------------------------------------------------------------------
# _RefBuffer
# ---------------------------------------------------------------------------


class TestRefBuffer:
    def test_empty_pop_returns_zeros(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=1000)
        out = buf.pop(10)
        assert out.shape == (10,)
        np.testing.assert_array_equal(out, 0)
        assert len(buf) == 0

    def test_push_then_pop_full(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=1000)
        buf.push(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
        assert len(buf) == 5
        out = buf.pop(5)
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5])
        assert len(buf) == 0

    def test_pop_partial_then_remaining(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=1000)
        buf.push(np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(buf.pop(3), [0, 1, 2])
        assert len(buf) == 7
        np.testing.assert_array_equal(buf.pop(7), [3, 4, 5, 6, 7, 8, 9])
        assert len(buf) == 0

    def test_pop_spans_multiple_chunks(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=1000)
        buf.push(np.array([1, 2, 3], dtype=np.float32))
        buf.push(np.array([4, 5], dtype=np.float32))
        buf.push(np.array([6, 7, 8, 9], dtype=np.float32))
        assert len(buf) == 9
        np.testing.assert_array_equal(buf.pop(6), [1, 2, 3, 4, 5, 6])
        assert len(buf) == 3
        np.testing.assert_array_equal(buf.pop(3), [7, 8, 9])

    def test_pop_underflow_pads_zeros(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=1000)
        buf.push(np.array([1, 2, 3], dtype=np.float32))
        out = buf.pop(5)
        np.testing.assert_array_equal(out, [1, 2, 3, 0, 0])
        assert len(buf) == 0

    def test_drops_oldest_when_over_cap(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=10)
        buf.push(np.arange(8, dtype=np.float32))  # 0..7
        buf.push(np.arange(8, 15, dtype=np.float32))  # 8..14 → total 15, drop 5 oldest
        assert len(buf) == 10
        out = buf.pop(10)
        np.testing.assert_array_equal(out, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def test_drops_whole_chunks_when_excess_exceeds_head(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=10)
        buf.push(np.arange(5, dtype=np.float32))
        buf.push(np.arange(5, 10, dtype=np.float32))
        # Now total=10 (at cap, no drop yet), head untouched
        buf.push(np.arange(10, 18, dtype=np.float32))  # +8 → 18, drop 8
        assert len(buf) == 10
        out = buf.pop(10)
        np.testing.assert_array_equal(out, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    def test_clear(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=100)
        buf.push(np.arange(20, dtype=np.float32))
        buf.clear()
        assert len(buf) == 0
        np.testing.assert_array_equal(buf.pop(5), [0, 0, 0, 0, 0])

    def test_extend_iterable(self):
        """extend() compat helper accepts a Python list."""
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=100)
        buf.extend([0.1, 0.2, 0.3])
        assert len(buf) == 3
        np.testing.assert_allclose(buf.pop(3), [0.1, 0.2, 0.3])

    def test_push_copies_caller_buffer(self):
        """push() must detach from any caller-owned buffer."""
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=100)
        src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf.push(src)
        src[:] = 0.0  # mutate after push — should NOT affect the buffer
        np.testing.assert_array_equal(buf.pop(3), [1.0, 2.0, 3.0])

    def test_push_empty_is_noop(self):
        from edgevox.audio._original import _RefBuffer

        buf = _RefBuffer(max_samples=100)
        buf.push(np.array([], dtype=np.float32))
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# AudioRecorder (with mocked sounddevice)
# ---------------------------------------------------------------------------


_DEVICE_INFO = {"default_samplerate": 48000, "max_input_channels": 2}


class TestAudioRecorder:
    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_init(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        callback = MagicMock()
        rec = AudioRecorder(on_speech=callback)
        assert rec._on_speech is callback

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_pause_sets_suppressed(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._suppressed is True

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_force_resume(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._suppressed is True
        rec.force_resume(delay=0.0)
        import time

        time.sleep(0.05)
        assert rec._suppressed is False

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_pause_enables_interrupt_detect(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        assert rec._interrupt_detect is False
        rec.pause()
        assert rec._interrupt_detect is True
        assert rec._interrupt_speech_count == 0

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_force_resume_clears_interrupt_detect(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._interrupt_detect is True
        rec.force_resume(delay=0.0)
        import time

        time.sleep(0.05)
        assert rec._interrupt_detect is False
        assert rec._interrupt_speech_count == 0

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_audio_callback_queues_during_interrupt_detect(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._suppressed is True
        assert rec._interrupt_detect is True
        # Callback should queue audio even though suppressed (interrupt detect active)
        chunk = np.zeros((512, 1), dtype=np.float32)
        rec._audio_callback(chunk, 512, None, None)
        assert not rec._audio_q.empty()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_audio_callback_drops_when_fully_suppressed(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec._suppressed = True
        rec._interrupt_detect = False
        chunk = np.zeros((512, 1), dtype=np.float32)
        rec._audio_callback(chunk, 512, None, None)
        assert rec._audio_q.empty()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_interrupt_fires_when_voice_exceeds_echo_baseline(self, _mock_sd_fn, mock_sd_module):
        """User voice significantly louder than echo baseline triggers interrupt."""
        from edgevox.audio._original import (
            INTERRUPT_BASELINE_FRAMES,
            INTERRUPT_SPEECH_FRAMES,
            AudioRecorder,
        )

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        on_interrupt = MagicMock()
        rec = AudioRecorder(on_speech=MagicMock(), on_interrupt=on_interrupt)
        rec._device_sr = 16000

        rec.pause()
        rec._running = True

        # Phase 1: baseline frames -- quiet echo (RMS ~0.01)
        echo_level = 0.01
        for _ in range(INTERRUPT_BASELINE_FRAMES):
            chunk = np.full(512, echo_level, dtype=np.float32)
            rec._audio_q.put(chunk)

        # Phase 2: loud user voice (RMS ~0.1 -- well above 2.5x baseline)
        loud_level = 0.1
        for _ in range(INTERRUPT_SPEECH_FRAMES):
            chunk = np.full(512, loud_level, dtype=np.float32)
            rec._audio_q.put(chunk)

        import time

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()
        time.sleep(0.5)
        rec._running = False
        t.join(timeout=1)

        on_interrupt.assert_called_once()
        assert rec._interrupt_detect is False

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_interrupt_not_fired_when_only_echo(self, _mock_sd_fn, mock_sd_module):
        """Steady echo at baseline level should not trigger interrupt."""
        from edgevox.audio._original import (
            INTERRUPT_BASELINE_FRAMES,
            INTERRUPT_SPEECH_FRAMES,
            AudioRecorder,
        )

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        on_interrupt = MagicMock()
        rec = AudioRecorder(on_speech=MagicMock(), on_interrupt=on_interrupt)
        rec._device_sr = 16000

        rec.pause()
        rec._running = True

        # All frames at the same echo level -- never exceeds baseline
        echo_level = 0.02
        total_frames = INTERRUPT_BASELINE_FRAMES + INTERRUPT_SPEECH_FRAMES + 5
        for _ in range(total_frames):
            chunk = np.full(512, echo_level, dtype=np.float32)
            rec._audio_q.put(chunk)

        import time

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()
        time.sleep(0.5)
        rec._running = False
        t.join(timeout=1)

        on_interrupt.assert_not_called()
        assert rec._interrupt_detect is True

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_interrupt_not_fired_on_brief_spike(self, _mock_sd_fn, mock_sd_module):
        """A brief loud spike (fewer frames than threshold) should not trigger."""
        from edgevox.audio._original import (
            INTERRUPT_BASELINE_FRAMES,
            INTERRUPT_SPEECH_FRAMES,
            AudioRecorder,
        )

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        on_interrupt = MagicMock()
        rec = AudioRecorder(on_speech=MagicMock(), on_interrupt=on_interrupt)
        rec._device_sr = 16000

        rec.pause()
        rec._running = True

        echo_level = 0.01
        loud_level = 0.1

        # Baseline frames
        for _ in range(INTERRUPT_BASELINE_FRAMES):
            rec._audio_q.put(np.full(512, echo_level, dtype=np.float32))

        # Brief spike (less than threshold), then back to echo
        spike_frames = max(1, INTERRUPT_SPEECH_FRAMES - 2)
        for _ in range(spike_frames):
            rec._audio_q.put(np.full(512, loud_level, dtype=np.float32))
        for _ in range(5):
            rec._audio_q.put(np.full(512, echo_level, dtype=np.float32))

        import time

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()
        time.sleep(0.5)
        rec._running = False
        t.join(timeout=1)

        on_interrupt.assert_not_called()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_interrupt_captures_speech_for_next_turn(self, _mock_sd_fn, mock_sd_module):
        """Speech frames that triggered the interrupt should be injected into the next speech buffer."""
        from edgevox.audio._original import (
            INTERRUPT_BASELINE_FRAMES,
            INTERRUPT_SPEECH_FRAMES,
            SILENCE_FRAMES_THRESHOLD,
            AudioRecorder,
        )

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        captured_audio = []
        on_interrupt = MagicMock()

        def capture_on_speech(audio):
            captured_audio.append(audio.copy())

        rec = AudioRecorder(on_speech=capture_on_speech, on_interrupt=on_interrupt)
        rec._device_sr = 16000

        rec._running = True
        rec.pause()

        echo_level = 0.01
        loud_level = 0.1

        # Phase 1: baseline frames (echo)
        for _ in range(INTERRUPT_BASELINE_FRAMES):
            rec._audio_q.put(np.full(512, echo_level, dtype=np.float32))

        # Phase 2: loud speech → triggers interrupt (these frames should be captured)
        speech_value = loud_level
        for _ in range(INTERRUPT_SPEECH_FRAMES):
            rec._audio_q.put(np.full(512, speech_value, dtype=np.float32))

        import time

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()
        time.sleep(0.5)

        # Interrupt should have fired
        on_interrupt.assert_called_once()
        # Speech buffer should have the interrupt frames saved
        assert len(rec._interrupt_speech_buffer) == INTERRUPT_SPEECH_FRAMES

        # Now simulate resume (force_resume clears suppressed)
        rec._suppressed = False
        rec._interrupt_detect = False

        # Feed continuation speech + silence to trigger on_speech
        for _ in range(3):
            rec._audio_q.put(np.full(512, loud_level, dtype=np.float32))

        # Mock VAD to detect the continuation as speech then silence
        vad_results = [True] * 3 + [False] * (SILENCE_FRAMES_THRESHOLD + 1)
        rec._vad = MagicMock()
        rec._vad.is_speech.side_effect = vad_results

        for _ in range(SILENCE_FRAMES_THRESHOLD + 1):
            rec._audio_q.put(np.zeros(512, dtype=np.float32))

        time.sleep(0.5)
        rec._running = False
        t.join(timeout=2)

        # on_speech should have been called with audio that includes the interrupt frames
        assert len(captured_audio) == 1
        # 8 interrupt frames + 3 continuation speech + 23 silence frames = 34
        total_frames = INTERRUPT_SPEECH_FRAMES + 3 + SILENCE_FRAMES_THRESHOLD
        assert len(captured_audio[0]) == total_frames * 512

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_on_speech_dispatched_to_separate_thread(self, _mock_sd_fn, mock_sd_module):
        """on_speech must run in a separate thread so _process_loop stays free for interrupt detection."""
        from edgevox.audio._original import (
            INTERRUPT_BASELINE_FRAMES,
            INTERRUPT_SPEECH_FRAMES,
            AudioRecorder,
        )

        mock_sd_module.query_devices.return_value = _DEVICE_INFO

        speech_started = threading.Event()
        speech_release = threading.Event()
        on_interrupt = MagicMock()

        def blocking_on_speech(audio):
            """Simulate a slow pipeline (STT+LLM+TTS) that blocks for a while."""
            speech_started.set()
            speech_release.wait(timeout=5)

        rec = AudioRecorder(on_speech=blocking_on_speech, on_interrupt=on_interrupt)
        rec._device_sr = 16000

        # Mock VAD: first batch returns True (speech), then False (silence) to
        # trigger on_speech dispatch
        silence_frames = 25  # > SILENCE_FRAMES_THRESHOLD
        vad_results = [True] * 3 + [False] * silence_frames
        rec._vad = MagicMock()
        rec._vad.is_speech.side_effect = vad_results

        rec._running = True

        # Pre-fill the first batch (speech + silence to trigger on_speech)
        for _ in range(3 + silence_frames):
            rec._audio_q.put(np.zeros(512, dtype=np.float32))

        import time

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()

        # Wait for on_speech to be called (in its own thread)
        assert speech_started.wait(timeout=2), "on_speech was never called"

        # Now simulate interrupt detection: put the recorder in interrupt mode
        # (as player.play() would do) and feed echo baseline then loud user voice
        rec.pause()
        echo_level = 0.01
        loud_level = 0.1
        for _ in range(INTERRUPT_BASELINE_FRAMES):
            rec._audio_q.put(np.full(512, echo_level, dtype=np.float32))
        for _ in range(INTERRUPT_SPEECH_FRAMES):
            rec._audio_q.put(np.full(512, loud_level, dtype=np.float32))

        time.sleep(0.5)

        # on_interrupt should fire even though on_speech is still blocking
        on_interrupt.assert_called_once()

        # Clean up
        speech_release.set()
        rec._running = False
        t.join(timeout=2)

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_on_speech_runs_in_different_thread(self, _mock_sd_fn, mock_sd_module):
        """Verify on_speech runs in a thread different from _process_loop."""
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO

        speech_thread_id = []
        done = threading.Event()

        def capture_thread_on_speech(audio):
            speech_thread_id.append(threading.current_thread().ident)
            done.set()

        rec = AudioRecorder(on_speech=capture_thread_on_speech)
        rec._device_sr = 16000

        # VAD: speech then silence to trigger on_speech
        vad_results = [True] * 3 + [False] * 25
        rec._vad = MagicMock()
        rec._vad.is_speech.side_effect = vad_results

        rec._running = True
        for _ in range(3 + 25):
            rec._audio_q.put(np.zeros(512, dtype=np.float32))

        t = threading.Thread(target=rec._process_loop, daemon=True)
        t.start()

        assert done.wait(timeout=2), "on_speech was never called"
        rec._running = False
        t.join(timeout=2)

        assert len(speech_thread_id) == 1
        assert speech_thread_id[0] != t.ident, "on_speech should run in a different thread than _process_loop"


# ---------------------------------------------------------------------------
# WakeWordDetector (with mocked pymicro-wakeword)
# ---------------------------------------------------------------------------


class TestWakeWordDetector:
    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_init(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hey_jarvis_model"
        det = WakeWordDetector(wake_words=["hey jarvis"])
        assert det._detected_name == "hey jarvis"
        mock_mww.from_builtin.assert_called_once_with("hey_jarvis_model")

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_unknown_wakeword_raises(self, _model, _mww, _features):
        from edgevox.audio.wakeword import WakeWordDetector

        with pytest.raises(ValueError, match="Unknown wake word"):
            WakeWordDetector(wake_words=["nonexistent"])

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_detect_returns_none_on_silence(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        mock_instance = MagicMock()
        mock_instance.process_streaming.return_value = False
        mock_mww.from_builtin.return_value = mock_instance

        feat_instance = MagicMock()
        feat_instance.process_streaming.return_value = [MagicMock()]
        mock_features.return_value = feat_instance

        det = WakeWordDetector()
        result = det.detect(np.zeros(512, dtype=np.float32))
        assert result is None

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_detect_returns_name_on_detection(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        mock_instance = MagicMock()
        mock_instance.process_streaming.return_value = True
        mock_mww.from_builtin.return_value = mock_instance

        feat_instance = MagicMock()
        feat_instance.process_streaming.return_value = [MagicMock()]
        mock_features.return_value = feat_instance

        det = WakeWordDetector()
        result = det.detect(np.zeros(512, dtype=np.float32))
        assert result == "hey jarvis"

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_reset(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        det = WakeWordDetector()
        old_features = det._features
        det.reset()
        assert det._features is not old_features


# ---------------------------------------------------------------------------
# StreamingPipeline (with mocked STT/LLM/TTS)
# ---------------------------------------------------------------------------


class TestStreamingPipeline:
    def _make_pipeline(self):
        from edgevox.core.pipeline import StreamingPipeline

        stt = MagicMock()
        stt.transcribe.return_value = "hello"

        llm = MagicMock()
        llm.chat_stream.return_value = iter(["Sure", ",", " I", " can", " help", "."])

        tts = MagicMock()
        tts.synthesize.return_value = np.zeros(1000, dtype=np.float32)

        callbacks = {
            "state_changes": [],
            "user_texts": [],
            "bot_texts": [],
            "metrics_list": [],
        }

        pipeline = StreamingPipeline(
            stt=stt,
            llm=llm,
            tts=tts,
            on_state_change=lambda s: callbacks["state_changes"].append(s),
            on_user_text=lambda t, d: callbacks["user_texts"].append(t),
            on_bot_text=lambda t, d: callbacks["bot_texts"].append(t),
            on_metrics=lambda m: callbacks["metrics_list"].append(m),
        )
        return pipeline, stt, llm, tts, callbacks

    @patch("edgevox.core.pipeline.play_audio")
    def test_process_full_turn(self, mock_play):
        pipeline, stt, llm, tts, cb = self._make_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        metrics = pipeline.process(audio, language="en")

        stt.transcribe.assert_called_once()
        llm.chat_stream.assert_called_once_with("hello")
        assert tts.synthesize.called
        assert mock_play.called
        assert "stt" in metrics
        assert "llm" in metrics
        assert "tts" in metrics
        assert "total" in metrics
        assert len(cb["state_changes"]) >= 3  # transcribing, thinking/speaking, listening
        assert cb["user_texts"] == ["hello"]

    @patch("edgevox.core.pipeline.play_audio")
    def test_process_empty_transcription_skips(self, mock_play):
        pipeline, stt, llm, tts, _cb = self._make_pipeline()
        stt.transcribe.return_value = "   "
        audio = np.zeros(16000, dtype=np.float32)
        metrics = pipeline.process(audio)

        assert metrics.get("skipped") is True
        llm.chat_stream.assert_not_called()
        tts.synthesize.assert_not_called()

    @patch("edgevox.core.pipeline.play_audio")
    def test_interrupt_stops_pipeline(self, mock_play):
        pipeline, _stt, llm, _tts, _cb = self._make_pipeline()

        def slow_tokens(msg):
            yield "Hello"
            pipeline.interrupt()
            yield " world."

        llm.chat_stream.side_effect = slow_tokens

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.process(audio)
        # Should have been interrupted — play_audio may or may not be called


# ---------------------------------------------------------------------------
# SessionState.segment_duration
# ---------------------------------------------------------------------------


class TestSegmentDuration:
    def test_one_second(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(16000, dtype=np.float32)
        assert abs(SessionState.segment_duration(seg) - 1.0) < 0.001

    def test_half_second(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(8000, dtype=np.float32)
        assert abs(SessionState.segment_duration(seg) - 0.5) < 0.001

    def test_empty(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(0, dtype=np.float32)
        assert SessionState.segment_duration(seg) == 0.0


# ---------------------------------------------------------------------------
# TUI helpers (pure functions, no GUI needed)
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty_values(self):
        from edgevox.tui import _sparkline

        result = _sparkline([])
        assert len(result) == 24
        assert result == " " * 24

    def test_all_zeros(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.0] * 10, width=10)
        assert len(result) == 10
        assert all(c == " " for c in result)

    def test_all_ones(self):
        from edgevox.tui import _sparkline

        result = _sparkline([1.0] * 10, width=10)
        assert len(result) == 10
        assert all(c == "▇" for c in result)

    def test_mixed_values(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.0, 0.5, 1.0], width=3)
        assert len(result) == 3
        assert result[0] == " "
        assert result[2] == "▇"

    def test_padding_short_list(self):
        from edgevox.tui import _sparkline

        result = _sparkline([1.0], width=5)
        assert len(result) == 5
        assert result[-1] == "▇"
        assert result[0] == " "

    def test_clips_to_width(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.5] * 50, width=10)
        assert len(result) == 10


class TestDevicePrefs:
    def test_save_and_load(self, tmp_path, monkeypatch):
        from edgevox.tui import _load_device_prefs, _save_device_prefs

        prefs_file = tmp_path / "devices.json"
        monkeypatch.setattr("edgevox.tui._DEVICES_CFG", prefs_file)

        _save_device_prefs(mic=2, spk=4)
        prefs = _load_device_prefs()
        assert prefs["mic"] == 2
        assert prefs["spk"] == 4

    def test_load_missing_file(self, tmp_path, monkeypatch):
        from edgevox.tui import _load_device_prefs

        monkeypatch.setattr("edgevox.tui._DEVICES_CFG", tmp_path / "nonexistent.json")
        prefs = _load_device_prefs()
        assert prefs == {}

    def test_resolve_saved_device_found(self):
        from edgevox.tui import _resolve_saved_device

        available = [("Mic A", 0), ("Mic B", 2), ("Mic C", 5)]
        assert _resolve_saved_device(2, available) == 2

    def test_resolve_saved_device_not_found(self):
        from edgevox.tui import _resolve_saved_device

        available = [("Mic A", 0), ("Mic B", 2)]
        assert _resolve_saved_device(99, available) is None

    def test_resolve_saved_device_none(self):
        from edgevox.tui import _resolve_saved_device

        assert _resolve_saved_device(None, [("Mic A", 0)]) is None
