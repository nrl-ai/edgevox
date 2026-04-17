"""Barge-in / interruption controller for the voice agent pipeline.

The streaming pipeline (:class:`edgevox.core.pipeline.StreamingPipeline`)
should be able to detect sustained user speech while TTS is playing and
cut everything off — TTS audio, in-flight LLM generation, and, when
configured, the currently running skill. This module provides the
coordinator that other components attach to.

Wiring (not done yet — see ``docs/plan.md``):

.. code-block:: python

    ic = InterruptController()
    ctx = AgentContext(interrupt=ic)

    # mic / VAD worker
    for frame in mic_stream():
        if vad.is_speech(frame) and tts_state.is_playing:
            ic.trigger(reason="user_barge_in")

    # TTS worker observes ic.interrupted and flushes the buffer
    # LLM worker observes ic.interrupted and stops iterating on the stream
    # Agent loop honors ctx.stop (populated from ic) between hops

The controller itself is tiny and dependency-free — all the real work
happens in the subscribers. This is the Protocol that makes it
plug-and-play.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)


InterruptReason = Literal["user_barge_in", "safety_preempt", "user_cancel", "timeout", "manual"]


@dataclass
class InterruptPolicy:
    """Tunable thresholds for barge-in behavior.

    Defaults reflect typical robot voice UX:

    - 300 ms of sustained speech energy before trigger (eliminates
      false-positives on "uh", brief throat-clears).
    - LLM generation always cancelled on interrupt.
    - Skills **not** cancelled by default: interrupting a Panda mid-grasp
      because the user said "um" is worse than letting the grasp finish.
      Opt in per-agent when appropriate.
    """

    min_duration_ms: int = 300
    energy_threshold: float = 0.02  # normalized float32 RMS
    cancel_llm: bool = True
    cancel_skills: bool = False
    # Drop the in-flight TTS sentence; if False, let the current sentence
    # finish but don't start the next one.
    cut_tts_immediately: bool = True


@dataclass
class InterruptEvent:
    reason: InterruptReason
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class InterruptController:
    """Thread-safe barge-in coordinator.

    Components participate via:

    - Producers (VAD workers, GUI buttons, safety monitors) call
      :meth:`trigger` when they observe an interrupt condition.
    - Consumers (TTS, LLM, agent loop) call :meth:`should_stop` in their
      hot loop or wait on :attr:`interrupted`.

    Every trigger is recorded in :attr:`history` for post-hoc analysis.
    """

    def __init__(self, policy: InterruptPolicy | None = None) -> None:
        self.policy = policy or InterruptPolicy()
        self.interrupted = threading.Event()
        self._lock = threading.RLock()
        self._history: list[InterruptEvent] = []
        self._subscribers: list[Callable[[InterruptEvent], None]] = []
        self._latest: InterruptEvent | None = None

    # ----- triggering -----

    def trigger(self, reason: InterruptReason = "manual", **meta: Any) -> InterruptEvent:
        """Record and broadcast an interrupt. Idempotent: multiple
        triggers while already interrupted still append to history
        but reuse the event flag."""
        event = InterruptEvent(reason=reason, meta=meta)
        with self._lock:
            self._history.append(event)
            self._latest = event
            self.interrupted.set()
        for sub in list(self._subscribers):
            try:
                sub(event)
            except Exception:
                log.exception("Interrupt subscriber raised")
        return event

    # ----- consumption -----

    def should_stop(self) -> bool:
        return self.interrupted.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until interrupt fires or ``timeout`` elapses."""
        return self.interrupted.wait(timeout=timeout)

    def reset(self) -> None:
        """Clear the interrupt flag. Call at the start of each turn
        so one interrupt doesn't poison subsequent turns."""
        with self._lock:
            self.interrupted.clear()

    # ----- observability -----

    def subscribe(self, handler: Callable[[InterruptEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subscribers:
                    self._subscribers.remove(handler)

        return unsubscribe

    @property
    def latest(self) -> InterruptEvent | None:
        with self._lock:
            return self._latest

    @property
    def history(self) -> list[InterruptEvent]:
        with self._lock:
            return list(self._history)


# ---------------------------------------------------------------------------
# VAD-energy-based watcher (utility)
# ---------------------------------------------------------------------------


class EnergyBargeInWatcher:
    """Monitors an audio stream for sustained speech energy while a
    "speaker is playing" flag is True, and triggers the controller.

    Pipeline authors can either plug this in directly or implement
    their own watcher against the same :class:`InterruptController`
    contract. Kept here so barge-in works out of the box for simple
    setups.

    Usage (pseudo)::

        watcher = EnergyBargeInWatcher(ic, is_tts_playing=tts.is_playing)
        threading.Thread(target=watcher.run, args=(mic_stream,), daemon=True).start()

    ``mic_stream`` yields float32 numpy arrays at 16 kHz.
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        frame_ms: int = 20,
    ) -> None:
        self.controller = controller
        self._is_tts_playing = is_tts_playing
        self._frame_ms = frame_ms
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, frames: Any) -> None:
        """Consume ``frames`` (iterable of float32 arrays). Triggers the
        controller when sustained speech above threshold coincides with
        TTS playback.

        Kept numpy-free in the protocol: we compute RMS with pure
        Python so this doesn't force a numpy dep even though the hot
        path uses numpy arrays. The caller is expected to pass real
        audio; this function tolerates non-numpy iterables for tests.
        """
        policy = self.controller.policy
        sustained_ms = 0
        for frame in frames:
            if self._stop.is_set():
                return
            if not self._is_tts_playing():
                sustained_ms = 0
                continue
            rms = _rms(frame)
            if rms >= policy.energy_threshold:
                sustained_ms += self._frame_ms
                if sustained_ms >= policy.min_duration_ms:
                    self.controller.trigger("user_barge_in", rms=rms)
                    # After triggering, don't re-trigger until the
                    # pipeline resets the controller.
                    sustained_ms = 0
            else:
                sustained_ms = max(0, sustained_ms - self._frame_ms)


def _rms(frame: Any) -> float:
    """Compute RMS of an audio frame. Accepts numpy arrays or python
    iterables of floats (for tests)."""
    try:
        import numpy as np

        if isinstance(frame, np.ndarray):
            if frame.size == 0:
                return 0.0
            arr = frame.astype("float32")
            return float((arr * arr).mean() ** 0.5)
    except ImportError:
        pass
    # Python fallback
    vals = list(frame)
    if not vals:
        return 0.0
    acc = 0.0
    for v in vals:
        acc += float(v) * float(v)
    return (acc / len(vals)) ** 0.5


__all__ = [
    "EnergyBargeInWatcher",
    "InterruptController",
    "InterruptEvent",
    "InterruptPolicy",
    "InterruptReason",
]
