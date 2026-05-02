"""End-to-end smoke test for the FastAPI app and the /ws protocol.

Real STT/LLM/TTS models are too heavy for CI, so we hand-construct a
``ServerCore`` and stub the three model objects with deterministic fakes.
This still exercises the actual WebSocket endpoint, the per-turn worker,
and the per-session history swap — only the model calls themselves are
replaced.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edgevox.audio._original import VAD_SAMPLES
from edgevox.server.audio_utils import int16_bytes_to_float32  # noqa: F401  (imported for completeness)
from edgevox.server.main import create_app
from edgevox.server.session import SessionState

# ---------- fakes ----------


class FakeSTT:
    def __init__(self):
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        self.calls.append(audio.size)
        return "hello world"


class FakeLLM:
    def __init__(self):
        self._history: list[dict] = [{"role": "system", "content": "sys"}]

    def chat_stream(self, message: str):
        self._history.append({"role": "user", "content": message})
        reply = "hi there. how are you?"
        for tok in reply.split(" "):
            yield tok + " "
        self._history.append({"role": "assistant", "content": reply})


class FakeTTS:
    sample_rate = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        # Half a second of silence per sentence is enough; we just verify the WAV bytes.
        return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)


class FakeVAD:
    """Always-speech VAD: a single feed of N frames produces a dispatch on the
    next round if the silence-frame budget is exhausted. Tests will explicitly
    interleave speech and silence by calling .is_speech as the loop dictates."""

    def __init__(self, script: list[bool]):
        self._script = list(script)

    def is_speech(self, _frame) -> bool:
        if self._script:
            return self._script.pop(0)
        return False

    def reset(self):
        pass


class FakeCore:
    """Mimics ServerCore enough for create_app + handle_connection."""

    def __init__(self):
        self.language = "en"
        self.stt = FakeSTT()
        self.llm = FakeLLM()
        self.tts = FakeTTS()
        self.voice = "fake-voice"
        self.inference_lock = asyncio.Lock()
        self.sessions: dict[str, SessionState] = {}
        self._base_history = list(self.llm._history)
        # Real ServerCore has an optional agent harness attached; ws.py
        # branches on `core.agent is not None`. The non-agent path is what
        # this fake covers, so the attribute exists and is None.
        self.agent = None

    def fresh_history(self) -> list[dict]:
        return [dict(m) for m in self._base_history]

    def info(self) -> dict:
        return {
            "language": self.language,
            "languages": ["en"],
            "voice": self.voice,
            "voices": ["fake-voice"],
            "stt": "FakeSTT",
            "tts": "FakeTTS",
            "tts_sample_rate": self.tts.sample_rate,
            "active_sessions": len(self.sessions),
        }


@pytest.fixture
def client(monkeypatch):
    core = FakeCore()
    # Patch SessionState's default VAD construction so we don't need onnxruntime.
    # The fake script: enough True frames to enter speech, then SILENCE_FRAMES_THRESHOLD
    # False frames to dispatch.
    from edgevox.audio._original import SILENCE_FRAMES_THRESHOLD

    speech_then_silence = [True] * 5 + [False] * SILENCE_FRAMES_THRESHOLD

    real_init = SessionState.__init__

    def patched_init(self, language, history, session_id=None, vad=None):
        if vad is None:
            vad = FakeVAD(script=list(speech_then_silence))
        real_init(self, language=language, history=history, session_id=session_id, vad=vad)

    monkeypatch.setattr(SessionState, "__init__", patched_init)

    app = create_app(core)  # type: ignore[arg-type]
    return TestClient(app), core


# ---------- HTTP endpoints ----------


def test_health_endpoint(client):
    tc, _core = client
    r = tc.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_sessions"] == 0


def test_info_endpoint(client):
    tc, _core = client
    r = tc.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["voice"] == "fake-voice"
    assert body["tts_sample_rate"] == 24_000


def test_static_index_served(client):
    tc, _core = client
    r = tc.get("/")
    # The SPA was built earlier in this test session; if not, /api/info still works
    # and the root falls back to a 503 with a hint.
    assert r.status_code in (200, 503)


# ---------- WebSocket flow ----------


def _drain_until(ws, kind: str, timeout: float = 5.0) -> dict:
    """Receive JSON messages until we see ``type == kind``."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"never saw {kind}")
        msg = ws.receive()
        if msg.get("text"):
            payload = json.loads(msg["text"])
            if payload.get("type") == kind:
                return payload


# ``ws.receive()`` against Starlette's ``TestClient`` blocks indefinitely
# under recent anyio versions when the server-side turn worker exits
# without closing the socket — the bug isn't in the tested code but in
# the test client's threading model. The two websocket end-to-end tests
# below are therefore marked ``integration`` so the default (push / PyPI
# publish) pytest run skips them; the dedicated ``Integration Tests``
# workflow still exercises them on its own schedule.
@pytest.mark.integration
def test_websocket_full_turn(client):
    tc, core = client
    with tc.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert "session_id" in ready
        listening = ws.receive_json()
        assert listening == {"type": "state", "value": "listening"}

        # Send enough binary frames to satisfy the fake VAD's script
        # (5 speech + SILENCE_FRAMES_THRESHOLD silence = 28 frames).
        from edgevox.audio._original import SILENCE_FRAMES_THRESHOLD

        n_frames = 5 + SILENCE_FRAMES_THRESHOLD
        # Each frame is VAD_SAMPLES int16 samples = 1024 bytes.
        pcm = (np.zeros(VAD_SAMPLES, dtype=np.float32) * 0).astype("<i2").tobytes()
        for _ in range(n_frames):
            ws.send_bytes(pcm)

        # Drain messages and collect the ones we care about.
        seen_user_text = False
        seen_bot_token = False
        seen_bot_sentence = 0
        seen_audio_blobs = 0
        seen_bot_text = False
        seen_metrics = False

        # We don't know exact ordering of state/level events, so loop on receive.
        for _ in range(200):
            msg = ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            data = msg.get("bytes")
            if text:
                payload = json.loads(text)
                kind = payload.get("type")
                if kind == "user_text":
                    seen_user_text = True
                    assert payload["text"] == "hello world"
                elif kind == "bot_token":
                    seen_bot_token = True
                elif kind == "bot_sentence":
                    seen_bot_sentence += 1
                elif kind == "bot_text":
                    seen_bot_text = True
                elif kind == "metrics":
                    seen_metrics = True
                    break
            elif data:
                seen_audio_blobs += 1

        assert seen_user_text, "expected user_text JSON"
        assert seen_bot_token, "expected at least one bot_token"
        assert seen_bot_sentence >= 1, "expected at least one bot_sentence"
        assert seen_audio_blobs >= 1, "expected WAV audio after bot_sentence"
        assert seen_bot_text, "expected final bot_text"
        assert seen_metrics, "expected metrics payload"

    # The fake LLM history should have been preserved on the (now-removed) session;
    # the global LLM history should have been restored to the system-only base.
    assert core.llm._history == [{"role": "system", "content": "sys"}]


@pytest.mark.integration
def test_websocket_reset_clears_history(client):
    tc, core = client
    with tc.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.receive_json()  # listening
        ws.send_text(json.dumps({"type": "reset"}))
        # Server replies with an info message
        info = ws.receive_json()
        assert info.get("type") == "info"
    assert len(core.sessions) == 0
