"""Tests for the pluggable VAD-based barge-in watchers + VectorMemoryStore.

Heavy deps (``webrtcvad``, ``sqlite-vec``, faster-whisper's Silero ONNX)
are skipped with ``importorskip`` so these tests run in CI matrices
that don't install the ``voice-vad`` / ``memory-vec`` extras.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from edgevox.agents.interrupt import InterruptController

# ---------------------------------------------------------------------------
# Dummy controller that records triggers without touching real threads
# ---------------------------------------------------------------------------


class _SpyController(InterruptController):
    """Captures every ``trigger`` call into a list for assertions."""

    def __init__(self):
        super().__init__()
        self.triggers: list[tuple[str, str]] = []

    def trigger(self, *, reason: str = "", source: str = "watcher", **_: object) -> bool:  # type: ignore[override]
        self.triggers.append((reason, source))
        return True


def _tts_off() -> bool:
    return False


def _frames(n_speech: int, samples: int, amplitude: float = 0.3) -> list[np.ndarray]:
    """Generate ``n_speech`` high-energy frames followed by some silence.

    The watchers classify by VAD model, not by amplitude, but
    high-amplitude input still matters for the Silero backend (low
    amplitude can fail the speech-prob threshold). WebRTC VAD also
    happens to need non-trivial energy to classify as speech.
    """
    speech = [(amplitude * np.random.randn(samples)).astype(np.float32) for _ in range(n_speech)]
    silence = [np.zeros(samples, dtype=np.float32) for _ in range(5)]
    return speech + silence


# ---------------------------------------------------------------------------
# WebRTC VAD — needs ``voice-vad`` extra
# ---------------------------------------------------------------------------


class TestWebRTCVADWatcher:
    def test_importable(self):
        pytest.importorskip("webrtcvad")
        from edgevox.agents.vad_watchers import WebRTCVADWatcher

        assert WebRTCVADWatcher is not None

    def test_triggers_on_sustained_speech(self):
        pytest.importorskip("webrtcvad")
        from edgevox.agents.vad_watchers import WebRTCVADWatcher

        ctrl = _SpyController()
        watcher = WebRTCVADWatcher(
            ctrl,
            is_tts_playing=_tts_off,
            aggressiveness=0,
            frame_ms=20,
            sustained_speech_ms=40,  # low threshold → easier to trigger on synthetic noise
        )
        # WebRTC VAD classifies modulated noise as speech at mode 0;
        # we feed 15 frames of random float32 audio.
        watcher.run(iter(_frames(n_speech=15, samples=320, amplitude=0.4)))
        # Don't assert on count: WebRTC+noise is probabilistic. We
        # just want at least one trigger, confirming end-to-end wire-up.
        assert any(t[0].startswith("user_speech_webrtc") for t in ctrl.triggers)

    def test_invalid_frame_ms_rejected(self):
        pytest.importorskip("webrtcvad")
        from edgevox.agents.vad_watchers import WebRTCVADWatcher

        with pytest.raises(ValueError):
            WebRTCVADWatcher(_SpyController(), is_tts_playing=_tts_off, frame_ms=15)

    def test_invalid_aggressiveness_rejected(self):
        pytest.importorskip("webrtcvad")
        from edgevox.agents.vad_watchers import WebRTCVADWatcher

        with pytest.raises(ValueError):
            WebRTCVADWatcher(_SpyController(), is_tts_playing=_tts_off, aggressiveness=5)


# ---------------------------------------------------------------------------
# Silero VAD — uses the ONNX model bundled with faster-whisper
# ---------------------------------------------------------------------------


class TestSileroVADWatcher:
    def test_importable(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("faster_whisper")
        from edgevox.agents.vad_watchers import SileroVADWatcher

        assert SileroVADWatcher is not None

    def test_skips_wrong_size_frames(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("faster_whisper")
        from edgevox.agents.vad_watchers import SileroVADWatcher

        ctrl = _SpyController()
        watcher = SileroVADWatcher(ctrl, is_tts_playing=_tts_off)
        # Feed frames of the WRONG size (320 not 512). Must not raise.
        watcher.run(iter([np.zeros(320, dtype=np.float32) for _ in range(4)]))
        assert ctrl.triggers == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_energy_backend(self):
        from edgevox.agents.vad_watchers import create_vad_watcher

        w = create_vad_watcher("energy", _SpyController(), is_tts_playing=_tts_off)
        assert type(w).__name__ == "EnergyBargeInWatcher"

    def test_unknown_backend(self):
        from edgevox.agents.vad_watchers import create_vad_watcher

        with pytest.raises(ValueError):
            create_vad_watcher("bogus", _SpyController(), is_tts_playing=_tts_off)

    def test_protocol_conformance(self):
        """All backends should structurally satisfy BargeInVADWatcher."""
        pytest.importorskip("webrtcvad")
        from edgevox.agents.vad_watchers import BargeInVADWatcher, create_vad_watcher

        w_energy = create_vad_watcher("energy", _SpyController(), is_tts_playing=_tts_off)
        w_webrtc = create_vad_watcher("webrtc", _SpyController(), is_tts_playing=_tts_off)
        assert isinstance(w_energy, BargeInVADWatcher)
        assert isinstance(w_webrtc, BargeInVADWatcher)


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------


def _fake_embed(text: str, *, dim: int = 8) -> np.ndarray:
    """Deterministic hashing embedding — good enough for structural tests.

    Real embeddings from ``llama_embed(llm)`` are exercised by the
    integration suite; unit tests just need a stable ``str -> vec``.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)


class TestVectorMemoryStore:
    def _make(self, tmp_path: Path, *, dim: int = 8):
        pytest.importorskip("sqlite_vec")
        from edgevox.agents.memory_vec import VectorMemoryStore

        return VectorMemoryStore(
            tmp_path / "vec.db",
            embed_fn=lambda t: _fake_embed(t, dim=dim),
            embedding_dim=dim,
        )

    def test_add_and_get_fact(self, tmp_path: Path):
        s = self._make(tmp_path)
        s.add_fact("user.name", "Ada")
        assert s.get_fact("user.name") == "Ada"
        s.close()

    def test_overwrite_invalidates_prior(self, tmp_path: Path):
        s = self._make(tmp_path)
        s.add_fact("k", "v1")
        time.sleep(0.005)
        s.add_fact("k", "v2")
        assert s.get_fact("k") == "v2"
        # The invalidated vector is dropped from the embedding table,
        # so a search for ``k`` returns v2, never v1.
        hits = s.search_facts("k", k=5)
        values = {f.value for f, _d in hits}
        assert "v1" not in values
        assert "v2" in values
        s.close()

    def test_search_returns_nearest(self, tmp_path: Path):
        # With the hash-seeded fake embedder, only *identical* text
        # strings produce identical vectors. ``add_fact`` embeds
        # ``f"{key}: {value}"``, so search with that exact string to
        # get a deterministic match. Real embeddings (``llama_embed``)
        # cluster by semantic similarity; exercised by the integration
        # suite.
        s = self._make(tmp_path, dim=16)
        s.add_fact("fact.a", "peanut allergy")
        s.add_fact("fact.b", "lactose intolerance")
        s.add_fact("fact.c", "shellfish allergy")
        hits = s.search_facts("fact.a: peanut allergy", k=3)
        assert hits, "expected at least one hit"
        best, dist = hits[0]
        assert best.key == "fact.a"
        assert dist < 1e-3
        s.close()

    def test_search_respects_scope_filter(self, tmp_path: Path):
        s = self._make(tmp_path, dim=16)
        s.add_fact("k", "alpha", scope="user")
        s.add_fact("k", "alpha", scope="env:kitchen")
        hits = s.search_facts("alpha", k=5, scope="user")
        assert all(f.scope == "user" for f, _ in hits)
        s.close()

    def test_search_excludes_retired_facts(self, tmp_path: Path):
        s = self._make(tmp_path, dim=16)
        s.add_fact("k", "old")
        time.sleep(0.005)
        s.forget_fact("k")
        assert s.search_facts("old", k=5) == []
        s.close()

    def test_search_k_zero_returns_empty(self, tmp_path: Path):
        s = self._make(tmp_path, dim=8)
        s.add_fact("k", "v")
        assert s.search_facts("v", k=0) == []
        s.close()

    def test_dim_mismatch_raises_on_write(self, tmp_path: Path):
        # The store accepts ``embedding_dim`` verbatim at init (caller
        # may know it without being able to probe yet), and validates
        # each embedding against it on write. Mismatch raises on the
        # first ``add_fact``.
        pytest.importorskip("sqlite_vec")
        from edgevox.agents.memory_vec import VectorMemoryStore

        def bad_embed(_text: str) -> np.ndarray:
            return np.zeros(16, dtype=np.float32)

        store = VectorMemoryStore(tmp_path / "vec.db", embed_fn=bad_embed, embedding_dim=8)
        with pytest.raises(ValueError):
            store.add_fact("k", "v")
        store.close()

    def test_close_is_idempotent(self, tmp_path: Path):
        s = self._make(tmp_path)
        s.close()
        s.close()

    def test_preferences_and_episodes_roundtrip(self, tmp_path: Path):
        s = self._make(tmp_path)
        s.set_preference("voice", "af_heart")
        s.add_episode("grasp", {"obj": "cup"}, "ok")
        prefs = s.preferences()
        assert [p.key for p in prefs] == ["voice"]
        eps = s.recent_episodes(n=5)
        assert len(eps) == 1 and eps[0].outcome == "ok"
        s.close()

    def test_passes_memorystore_protocol(self, tmp_path: Path):
        from edgevox.agents.memory import MemoryStore

        s = self._make(tmp_path)
        assert isinstance(s, MemoryStore)
        s.close()
