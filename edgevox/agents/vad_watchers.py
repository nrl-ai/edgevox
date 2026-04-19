"""Pluggable VAD-based barge-in watchers.

The four shipping backends share the same outer contract as
:class:`edgevox.agents.interrupt.EnergyBargeInWatcher` — a ``run(frames)``
method that consumes 16 kHz float32 frames and triggers the
:class:`InterruptController` on sustained user speech — but differ in
how they classify a single frame as speech or non-speech.

Backend selection is done either directly (import the watcher class
you want) or via :func:`create_vad_watcher` which takes a string
backend name. The heavy dependencies (``webrtcvad``, ``ten-vad``) are
*lazy-imported* inside the constructor so importing this module is
always cheap — ``ModuleNotFoundError`` only surfaces if you actually
instantiate a backend whose dep isn't installed.

All Silero / TEN ONNX models prefer ``nrl-ai/edgevox-models`` on
HuggingFace (subfolder per backend) and fall back to the upstream
repo if the mirror doesn't have them yet — matches the convention
used by every other model loader in EdgeVox.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from huggingface_hub import hf_hub_download

from edgevox.agents.interrupt import EnergyBargeInWatcher

if TYPE_CHECKING:
    from edgevox.agents.interrupt import InterruptController


log = logging.getLogger(__name__)

_MODELS_REPO = "nrl-ai/edgevox-models"
_SILERO_UPSTREAM = "snakers4/silero-vad"
_TEN_UPSTREAM = "TEN-framework/ten-vad"


@runtime_checkable
class BargeInVADWatcher(Protocol):
    """Protocol every VAD-based barge-in watcher implements.

    ``run(frames)`` consumes an iterable of 16 kHz float32 numpy frames
    (typically 10-32 ms each) and triggers the controller when it
    decides the user has genuinely started speaking over the bot.
    ``stop()`` sets a flag the loop polls between frames so the watcher
    exits cleanly on shutdown.
    """

    def run(self, frames: Iterator[Any]) -> None: ...

    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Shared trigger scaffolding
# ---------------------------------------------------------------------------


class _SustainedSpeechTrigger:
    """Re-implements the echo-floor + refractory + sustained-ms logic.

    The three VAD backends below differ only in how they classify a
    frame. Sharing one trigger policy keeps barge-in behaviour
    (consecutive-speech threshold, TTS-release refractory, etc.)
    identical across backends.
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        frame_ms: int,
        sustained_speech_ms: int = 120,
        tts_release_ms: int = 180,
    ) -> None:
        self.controller = controller
        self._is_tts_playing = is_tts_playing
        self._frame_ms = frame_ms
        self._sustained_ms = sustained_speech_ms
        self._tts_release_ms = tts_release_ms
        self._stop_event = threading.Event()
        self._frame_idx = 0
        self._last_tts_active_idx = -(10**9)
        self._prev_tts_playing = False

    def stop(self) -> None:
        self._stop_event.set()

    def _post_tts_refractory_active(self) -> bool:
        frames_since_tts = self._frame_idx - self._last_tts_active_idx
        release_frames = self._tts_release_ms // max(self._frame_ms, 1)
        return frames_since_tts <= release_frames

    def process_frame(self, is_speech: bool) -> bool:
        """Advance the trigger by one frame; return True if the caller
        should fire the controller. The caller owns the actual
        ``controller.trigger`` call so it can include a reason string.
        """
        self._frame_idx += 1
        tts_playing = self._is_tts_playing()
        if tts_playing:
            self._last_tts_active_idx = self._frame_idx
            self._prev_tts_playing = True
            return False
        # Refractory just after TTS stops — suppress false positives from
        # echo tail / room reverb before AEC catches up.
        if self._prev_tts_playing and self._post_tts_refractory_active():
            if not is_speech:
                self._prev_tts_playing = False
            return False
        if not is_speech:
            self._consecutive_ms = 0
            return False
        self._consecutive_ms = getattr(self, "_consecutive_ms", 0) + self._frame_ms
        if self._consecutive_ms >= self._sustained_ms:
            self._consecutive_ms = 0
            return True
        return False


# ---------------------------------------------------------------------------
# WebRTC VAD (GMM, ~50 LOC, BSD-licensed)
# ---------------------------------------------------------------------------


class WebRTCVADWatcher:
    """Barge-in watcher using Google's WebRTC VAD (GMM baseline).

    Fixed-latency GMM classifier, accepts 10 / 20 / 30 ms frames at
    8/16/32/48 kHz. We fix the sample rate at 16 kHz to match the rest
    of the pipeline. Aggressive mode (0-3) trades false positives for
    false negatives; 2 is the sweet spot for quiet-room conversation.

    Requires ``webrtcvad`` (installed via the ``voice-vad`` extra).
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        aggressiveness: int = 2,
        frame_ms: int = 20,
        sustained_speech_ms: int = 120,
        tts_release_ms: int = 180,
    ) -> None:
        try:
            import webrtcvad  # type: ignore[import-untyped]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "WebRTCVADWatcher requires the ``voice-vad`` extra. Install with: pip install 'edgevox[voice-vad]'"
            ) from e
        if frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD only accepts 10, 20 or 30 ms frames")
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be 0..3")
        self._vad = webrtcvad.Vad(aggressiveness)
        self._trigger = _SustainedSpeechTrigger(
            controller,
            is_tts_playing=is_tts_playing,
            frame_ms=frame_ms,
            sustained_speech_ms=sustained_speech_ms,
            tts_release_ms=tts_release_ms,
        )
        self._frame_ms = frame_ms

    def stop(self) -> None:
        self._trigger.stop()

    def run(self, frames: Iterator[Any]) -> None:
        samples_per_frame = 16 * self._frame_ms  # 16 kHz
        for frame in frames:
            if self._trigger._stop_event.is_set():
                return
            arr = np.asarray(frame, dtype=np.float32)
            if arr.size < samples_per_frame:
                continue
            # WebRTC VAD wants linear int16 PCM bytes.
            pcm = np.clip(arr[:samples_per_frame] * 32768.0, -32768, 32767).astype("<i2").tobytes()
            try:
                is_speech = self._vad.is_speech(pcm, 16000)
            except Exception:
                log.exception("WebRTC VAD classify failed")
                continue
            if self._trigger.process_frame(is_speech):
                self._trigger.controller.trigger(reason="user_speech_webrtc")


# ---------------------------------------------------------------------------
# Silero VAD v6 (pure ONNX — reuses the model bundled with faster-whisper)
# ---------------------------------------------------------------------------


class SileroVADWatcher:
    """Barge-in watcher using Silero VAD v6 via onnxruntime.

    Reuses the model already bundled with ``faster-whisper`` so this
    backend adds no new download — the same VAD the production
    pipeline uses inside :class:`AudioRecorder` for utterance
    boundaries. Frames must be exactly 512 samples at 16 kHz (32 ms),
    which matches the default ``VAD_SAMPLES`` constant elsewhere in
    the codebase.
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        threshold: float = 0.4,
        sustained_speech_ms: int = 120,
        tts_release_ms: int = 180,
    ) -> None:
        try:
            from edgevox.audio._original import VAD as _SileroVAD
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "SileroVADWatcher requires onnxruntime (already a core dep) and faster-whisper's bundled model."
            ) from e
        self._vad = _SileroVAD(threshold=threshold)
        self._trigger = _SustainedSpeechTrigger(
            controller,
            is_tts_playing=is_tts_playing,
            frame_ms=32,
            sustained_speech_ms=sustained_speech_ms,
            tts_release_ms=tts_release_ms,
        )

    def stop(self) -> None:
        self._trigger.stop()

    def run(self, frames: Iterator[Any]) -> None:
        for frame in frames:
            if self._trigger._stop_event.is_set():
                return
            arr = np.asarray(frame, dtype=np.float32)
            if arr.size != 512:
                # Silero v6 is strict about frame size — skip malformed
                # frames rather than corrupt the recurrent state.
                continue
            try:
                is_speech = self._vad.is_speech(arr)
            except Exception:
                log.exception("Silero VAD classify failed")
                continue
            if self._trigger.process_frame(is_speech):
                self._trigger.controller.trigger(reason="user_speech_silero")


# ---------------------------------------------------------------------------
# TEN VAD (Tencent, Apache-2, lowest latency at 306 KB)
# ---------------------------------------------------------------------------


class TENVADWatcher:
    """Barge-in watcher using Tencent's TEN VAD.

    306 KB ONNX model, designed for on-device real-time detection with
    sub-ms latency per 10 ms frame. Accepts 10 ms or 16 ms frames at
    16 kHz — we fix at 16 ms (256 samples) to keep the timer math
    lining up with WebRTC and the overall 20 ms pipeline tick.

    The ONNX model file is resolved via ``huggingface_hub``: first try
    ``nrl-ai/edgevox-models/ten-vad/ten-vad.onnx``, fall back to
    ``TEN-framework/ten-vad`` upstream if the mirror doesn't have it.
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        threshold: float = 0.5,
        frame_ms: int = 16,
        sustained_speech_ms: int = 120,
        tts_release_ms: int = 180,
        model_path: str | None = None,
    ) -> None:
        if frame_ms not in (10, 16):
            raise ValueError("TEN VAD supports 10 or 16 ms frames at 16 kHz")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("TENVADWatcher requires onnxruntime") from e

        resolved = model_path or _resolve_ten_vad_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(
            resolved,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._threshold = threshold
        self._frame_ms = frame_ms
        self._trigger = _SustainedSpeechTrigger(
            controller,
            is_tts_playing=is_tts_playing,
            frame_ms=frame_ms,
            sustained_speech_ms=sustained_speech_ms,
            tts_release_ms=tts_release_ms,
        )
        # TEN VAD is a recurrent ONNX; track the hidden state per watcher
        # so concurrent instances don't cross-pollute.
        self._hidden: dict[str, Any] | None = None

    def stop(self) -> None:
        self._trigger.stop()

    def run(self, frames: Iterator[Any]) -> None:
        samples_per_frame = 16 * self._frame_ms  # 16 kHz

        for frame in frames:
            if self._trigger._stop_event.is_set():
                return
            arr = np.asarray(frame, dtype=np.float32)
            if arr.size < samples_per_frame:
                continue
            x = arr[:samples_per_frame].reshape(1, -1)
            try:
                inputs: dict[str, Any] = {"input": x}
                if self._hidden is not None:
                    inputs.update(self._hidden)
                outputs = self._session.run(None, inputs)
                prob = float(outputs[0].ravel()[0])
                # Propagate any additional outputs (recurrent state) back
                # as the next call's extra inputs. This is robust across
                # TEN VAD revisions that ship with vs. without state.
                if len(outputs) > 1:
                    meta_names = [o.name for o in self._session.get_outputs()[1:]]
                    self._hidden = dict(zip(meta_names, outputs[1:], strict=False))
            except Exception:
                log.exception("TEN VAD classify failed")
                continue
            is_speech = prob >= self._threshold
            if self._trigger.process_frame(is_speech):
                self._trigger.controller.trigger(reason="user_speech_ten")


def _resolve_ten_vad_model() -> str:
    """Find the TEN VAD ONNX file, preferring the consolidated mirror.

    Returns the local filesystem path to the model. Tries (in order):

    1. ``EDGEVOX_TEN_VAD_MODEL`` env var pointing at a local file.
    2. ``nrl-ai/edgevox-models`` on HF with subfolder ``ten-vad``.
    3. Upstream ``TEN-framework/ten-vad`` on HF.

    Raises :class:`FileNotFoundError` with a helpful message if every
    path fails (common case: offline box without a cached model).
    """
    env = os.environ.get("EDGEVOX_TEN_VAD_MODEL")
    if env and os.path.exists(env):
        return env

    filename = "ten-vad.onnx"
    try:
        return hf_hub_download(_MODELS_REPO, filename, subfolder="ten-vad")
    except Exception:
        log.warning("TEN VAD missing from %s; falling back to upstream", _MODELS_REPO)

    try:
        return hf_hub_download(_TEN_UPSTREAM, filename)
    except Exception as e:
        raise FileNotFoundError(
            "Could not locate ten-vad.onnx. Set EDGEVOX_TEN_VAD_MODEL=/path/to/ten-vad.onnx, "
            f"or ensure network access to {_MODELS_REPO} / {_TEN_UPSTREAM}."
        ) from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vad_watcher(
    backend: str,
    controller: InterruptController,
    *,
    is_tts_playing: Callable[[], bool],
    **kwargs: Any,
) -> BargeInVADWatcher:
    """Construct a watcher by backend name.

    ``backend`` is one of ``"energy"``, ``"webrtc"``, ``"silero"``,
    ``"ten"``. Remaining kwargs forward to the backend's constructor
    — e.g. ``aggressiveness=3`` for webrtc or ``threshold=0.3`` for
    silero.
    """
    backend = backend.lower()
    if backend == "energy":
        return EnergyBargeInWatcher(controller, is_tts_playing=is_tts_playing, **kwargs)
    if backend == "webrtc":
        return WebRTCVADWatcher(controller, is_tts_playing=is_tts_playing, **kwargs)
    if backend == "silero":
        return SileroVADWatcher(controller, is_tts_playing=is_tts_playing, **kwargs)
    if backend == "ten":
        return TENVADWatcher(controller, is_tts_playing=is_tts_playing, **kwargs)
    raise ValueError(f"unknown VAD backend {backend!r}; expected energy/webrtc/silero/ten")


__all__ = [
    "BargeInVADWatcher",
    "SileroVADWatcher",
    "TENVADWatcher",
    "WebRTCVADWatcher",
    "create_vad_watcher",
]
