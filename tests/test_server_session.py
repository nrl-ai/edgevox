"""Unit tests for the WebSocket server's session-state machine.

These tests do NOT load any real models. The Silero VAD is replaced with a
scripted fake so the loop logic (silence buffering, segment dispatch, drain
idempotency, history isolation) can be exercised without onnxruntime.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from edgevox.audio._original import SILENCE_FRAMES_THRESHOLD, TARGET_SAMPLE_RATE, VAD_SAMPLES
from edgevox.server.audio_utils import float32_to_wav_bytes, int16_bytes_to_float32
from edgevox.server.session import SessionState, chunk_pcm

# ---------- audio_utils ----------


def test_int16_round_trip():
    src = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    pcm = (src * 32767.0).astype("<i2").tobytes()
    out = int16_bytes_to_float32(pcm)
    assert out.dtype == np.float32
    assert np.allclose(out, src, atol=1e-3)


def test_int16_empty():
    assert int16_bytes_to_float32(b"").shape == (0,)


def test_float32_to_wav_bytes_is_valid_wav():
    sr = 24_000
    audio = np.sin(np.linspace(0, 2 * np.pi * 5, sr)).astype(np.float32) * 0.3
    blob = float32_to_wav_bytes(audio, sr)
    with wave.open(io.BytesIO(blob), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == sr
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == sr


# ---------- chunk_pcm ----------


def test_chunk_pcm_pads_final_frame():
    samples = np.ones(VAD_SAMPLES + 100, dtype=np.float32)
    frames = list(chunk_pcm(samples))
    assert len(frames) == 2
    assert all(f.shape == (VAD_SAMPLES,) for f in frames)
    assert np.all(frames[1][:100] == 1.0)
    assert np.all(frames[1][100:] == 0.0)


# ---------- SessionState VAD loop (with a scripted fake VAD) ----------


class FakeVAD:
    """Returns a pre-scripted sequence of is_speech results, frame by frame."""

    def __init__(self, script: list[bool] | None = None, default: bool = False):
        self._script = list(script) if script else []
        self._default = default
        self.calls = 0
        self.resets = 0

    def is_speech(self, _frame) -> bool:
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        return self._default

    def reset(self):
        self.resets += 1


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SAMPLE_RATE), dtype=np.float32)


def _frames_for(seconds: float) -> int:
    return max(1, int(seconds * TARGET_SAMPLE_RATE) // VAD_SAMPLES)


def test_silence_does_not_dispatch():
    fake = FakeVAD(default=False)
    s = SessionState(language="en", history=[], vad=fake)
    s.feed_audio(_silence(1.0))
    assert s.drain_segments() == []
    assert fake.calls > 0  # we did call into VAD


def test_speech_then_silence_dispatches_once():
    """20 frames of speech + (SILENCE_FRAMES_THRESHOLD) frames of silence → one dispatch."""
    speech_frames = 20
    silence_frames = SILENCE_FRAMES_THRESHOLD
    script = [True] * speech_frames + [False] * silence_frames
    fake = FakeVAD(script=script)
    s = SessionState(language="en", history=[], vad=fake)

    total = (speech_frames + silence_frames) * VAD_SAMPLES
    s.feed_audio(np.ones(total, dtype=np.float32) * 0.1)

    segments = s.drain_segments()
    assert len(segments) == 1
    expected = (speech_frames + silence_frames) * VAD_SAMPLES
    assert segments[0].shape == (expected,)
    assert s.drain_segments() == []
    assert fake.resets >= 1


def test_speech_below_silence_threshold_stays_buffered():
    """Speech followed by *not enough* silence should NOT dispatch yet."""
    script = [True] * 10 + [False] * (SILENCE_FRAMES_THRESHOLD - 1)
    fake = FakeVAD(script=script)
    s = SessionState(language="en", history=[], vad=fake)
    s.feed_audio(np.ones(len(script) * VAD_SAMPLES, dtype=np.float32) * 0.1)
    assert s.drain_segments() == []


def test_drain_is_idempotent():
    fake = FakeVAD(default=False)
    s = SessionState(language="en", history=[], vad=fake)
    s.feed_audio(_silence(0.5))
    assert s.drain_segments() == []
    assert s.drain_segments() == []


def test_reset_audio_clears_state():
    fake = FakeVAD(script=[True, True, True])
    s = SessionState(language="en", history=[], vad=fake)
    s.feed_audio(np.ones(3 * VAD_SAMPLES, dtype=np.float32) * 0.1)
    s.reset_audio()
    assert s._speech_buffer == []
    assert s._silence_count == 0
    assert s._in_speech is False


def test_silence_threshold_constant_unchanged():
    """If the upstream constant moves, this test surfaces it loudly."""
    assert SILENCE_FRAMES_THRESHOLD == 23


# ---------- history swap ----------


class _FakeLLM:
    """Minimal stand-in for edgevox.llm.LLM exercising the history-swap pattern."""

    def __init__(self):
        self._history: list[dict] = [{"role": "system", "content": "sys"}]

    def chat_stream(self, message: str):
        self._history.append({"role": "user", "content": message})
        reply = f"reply to: {message}"
        for token in reply.split():
            yield token + " "
        self._history.append({"role": "assistant", "content": reply})


def _run_fake_turn(llm: _FakeLLM, session: SessionState, message: str) -> str:
    """Mirror the swap pattern in ws._run_turn_blocking."""
    saved = llm._history
    llm._history = session.history
    try:
        tokens = list(llm.chat_stream(message))
    finally:
        session.history = llm._history
        llm._history = saved
    return "".join(tokens).strip()


def test_history_swap_preserves_per_session_context():
    llm = _FakeLLM()
    base = list(llm._history)

    s1 = SessionState(language="en", history=[dict(m) for m in base], vad=FakeVAD())
    s2 = SessionState(language="en", history=[dict(m) for m in base], vad=FakeVAD())

    _run_fake_turn(llm, s1, "hello from one")
    _run_fake_turn(llm, s2, "hello from two")
    _run_fake_turn(llm, s1, "again from one")

    s1_users = [m["content"] for m in s1.history if m["role"] == "user"]
    s2_users = [m["content"] for m in s2.history if m["role"] == "user"]
    assert s1_users == ["hello from one", "again from one"]
    assert s2_users == ["hello from two"]

    # The shared LLM's "saved" history should have been restored after each turn.
    assert llm._history == base


@pytest.mark.parametrize("payload", [b"", b"\x00\x00", b"\x00\x40\x00\xc0"])
def test_int16_decode_handles_short_payloads(payload):
    out = int16_bytes_to_float32(payload)
    assert out.dtype == np.float32
    assert out.size == len(payload) // 2
