"""Barge-in / interruption controller for the voice agent pipeline.

The streaming pipeline (:class:`edgevox.core.pipeline.StreamingPipeline`)
should be able to detect sustained user speech while TTS is playing and
cut everything off — TTS audio, in-flight LLM generation, and, when
configured, the currently running skill. This module provides the
coordinator that other components attach to.

Two signals are exposed:

- ``interrupted`` — general "stop what you're doing" event. TTS, the
  agent loop, and skill dispatch poll or wait on this.
- ``cancel_token`` — dedicated event piped into
  ``llama_cpp.Llama``'s ``stopping_criteria`` so an in-flight LLM
  generation actually aborts rather than running until ``max_tokens``
  drains. The agent loop threads this into :meth:`LLM.complete`.

Wiring:

.. code-block:: python

    ic = InterruptController()
    ctx = AgentContext(interrupt=ic)

    # mic / VAD worker
    for frame in mic_stream():
        if vad.is_speech(frame) and tts_state.is_playing:
            ic.trigger(reason="user_barge_in")

    # TTS worker observes ic.interrupted and flushes the buffer
    # LLM worker receives ic.cancel_token via LLM.complete(stop_event=…)
    # Agent loop honors ctx.should_stop() between hops and between tokens

The controller itself is tiny and dependency-free — all the real work
happens in the subscribers. This is the Protocol that makes it
plug-and-play.
"""

from __future__ import annotations

import json
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

    - 250 ms of sustained speech energy before trigger (eliminates
      false-positives on "uh", brief throat-clears).
    - LLM generation always cancelled on interrupt.
    - Skills **not** cancelled by default: interrupting a Panda mid-grasp
      because the user said "um" is worse than letting the grasp finish.
      Opt in per-agent when appropriate.

    **Echo mitigation** (when TTS audio leaks into the mic without AEC):

    - ``echo_suppression_ratio`` — when echo reference is available,
      mic energy must exceed ``ratio x echo_reference`` to count as
      user speech. Default 2.0 = "speak twice as loud as the bot".
    - ``echo_floor_window_ms`` — at the *start* of TTS playback (before
      the user could realistically have started barging in) we observe
      the mic energy as a calibration of the room's echo floor and use
      that as an additional threshold. 200 ms is enough for the AEC
      tail to settle without mistaking real early-barge-in for echo.
    - ``tts_release_ms`` — refractory period after TTS stops; the mic
      keeps picking up echo tail / room reverb for ~150-250 ms after
      the speaker goes silent. We ignore the mic during this window.

    To **wire echo mitigation** into your pipeline, pass
    ``tts_energy_provider=lambda: player.last_output_rms`` to
    :class:`EnergyBargeInWatcher` (see docstring there).
    """

    min_duration_ms: int = 250
    # Quieter speech triggers — but echo mitigations below keep it from
    # false-firing on TTS leakage. -38 dBFS on float32 [-1, 1] = quiet
    # conversational speech at typical mic gain.
    energy_threshold: float = 0.012
    cancel_llm: bool = True
    cancel_skills: bool = False
    # Drop the in-flight TTS sentence; if False, let the current sentence
    # finish but don't start the next one.
    cut_tts_immediately: bool = True
    # ----- Echo-aware tuning -----
    echo_suppression_ratio: float = 2.0
    echo_floor_window_ms: int = 200
    tts_release_ms: int = 200


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

    Two signal channels:

    - :attr:`interrupted` — general "stop what you're doing" event. Consumed
      by TTS, the agent loop, skill dispatch, etc.
    - :attr:`cancel_token` — dedicated event piped into
      ``llama_cpp.Llama``'s ``stopping_criteria`` so an in-flight LLM
      generation actually aborts rather than running to completion. Only
      set when :attr:`InterruptPolicy.cancel_llm` is True.

    Every trigger is recorded in :attr:`history` for post-hoc analysis.
    """

    def __init__(self, policy: InterruptPolicy | None = None, *, max_history: int = 500) -> None:
        self.policy = policy or InterruptPolicy()
        self.interrupted = threading.Event()
        # Fed into ``LLM.complete(stop_event=...)`` so the llama-cpp
        # sampling loop returns mid-generation. Without this the
        # ``cancel_llm`` flag is advisory — barge-in stops downstream
        # consumers but the LLM still burns through max_tokens.
        self.cancel_token = threading.Event()
        self._lock = threading.RLock()
        self._history: list[InterruptEvent] = []
        self._max_history = max_history
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
            # Ring-buffer history so a long-running voice session doesn't
            # slow-leak memory via interrupt events.
            if len(self._history) > self._max_history:
                del self._history[: len(self._history) - self._max_history]
            self._latest = event
            self.interrupted.set()
            if self.policy.cancel_llm:
                self.cancel_token.set()
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
        """Clear the interrupt flag and cancel token. Call at the start
        of each turn so one interrupt doesn't poison subsequent turns.

        History is retained for post-hoc analysis; :attr:`latest` is
        cleared so a stale event doesn't leak into the new turn.
        """
        with self._lock:
            self.interrupted.clear()
            self.cancel_token.clear()
            self._latest = None

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

    def as_tool_result(self, *, partial: str | None = None) -> dict | None:
        """Render the most recent interrupt as a synthetic tool result.

        When the agent loop cancels a skill mid-run via ``cancel_token``
        the model is left with a dangling assistant turn — it has no
        idea why control returned without a result. Injecting a
        ``role="tool"`` envelope on the next ``run()`` (with
        ``tool_name="__interrupt__"`` and a structured payload) gives
        the model a coherent recovery cue: it can apologise, summarise
        what was achieved, and ask the user how to proceed.

        Returns ``None`` when no interrupt has fired since the last
        :meth:`reset`. The returned dict is OpenAI-shaped so the loop's
        ``messages.append`` path treats it identically to a real tool
        result. Inspired by LangGraph's ``Command`` / ``interrupt()``
        pattern but scoped to single-process voice agents.
        """
        latest = self.latest
        if latest is None:
            return None
        payload: dict[str, Any] = {
            "ok": False,
            "error": "interrupted_by_user",
            "reason": latest.reason,
            "timestamp": latest.timestamp,
        }
        if partial:
            payload["partial"] = partial
        # Forward any caller-supplied meta verbatim — useful for STT
        # partial transcripts, RMS readings, etc.
        if latest.meta:
            payload["meta"] = dict(latest.meta)
        return {
            "role": "tool",
            "tool_call_id": "__interrupt__",
            "name": "__interrupt__",
            "content": json.dumps(payload, default=str),
        }


# ---------------------------------------------------------------------------
# VAD-energy-based watcher (utility)
# ---------------------------------------------------------------------------


class EnergyBargeInWatcher:
    """Monitors an audio stream for sustained speech energy while TTS
    is playing, triggering the controller when the user actually
    barges in (not when the bot's own audio leaks into the mic).

    Three layers of echo defence keep the watcher from
    self-triggering on TTS audio leaking into the mic:

    1. **Reference signal** (best). Pass ``tts_energy_provider`` — a
       callable that returns the *current* TTS output RMS observed by
       the player. The watcher requires
       ``mic_rms >= echo_suppression_ratio x tts_rms`` before counting
       a frame toward the sustained-speech timer. Wire it as e.g.
       ``tts_energy_provider=lambda: player.last_output_rms`` once
       :class:`InterruptiblePlayer` exposes that.
    2. **Echo-floor calibration**. During the first
       ``echo_floor_window_ms`` (default 200 ms) of each TTS segment,
       the watcher observes the mic energy and uses the peak as an
       additional threshold for the rest of that segment. Assumes the
       user isn't talking yet during the bot's first ~200 ms.
    3. **Release refractory**. For ``tts_release_ms`` after TTS stops
       playing the watcher ignores mic input — that window covers
       echo tail / room reverb / AEC settle time.

    The static ``energy_threshold`` is a hard floor. With echo
    mitigations active you can leave it low (~0.012 = -38 dBFS) and
    still get robust triggers even on quiet speech.

    Usage::

        watcher = EnergyBargeInWatcher(
            ic,
            is_tts_playing=player.is_playing,
            tts_energy_provider=lambda: player.last_output_rms,
        )
        threading.Thread(target=watcher.run, args=(mic_stream,), daemon=True).start()

    ``mic_stream`` yields float32 numpy arrays at 16 kHz.
    """

    def __init__(
        self,
        controller: InterruptController,
        *,
        is_tts_playing: Callable[[], bool],
        frame_ms: int = 20,
        tts_energy_provider: Callable[[], float] | None = None,
    ) -> None:
        self.controller = controller
        self._is_tts_playing = is_tts_playing
        self._tts_energy_provider = tts_energy_provider
        self._frame_ms = frame_ms
        self._stop = threading.Event()
        # Per-segment echo-floor state. Reset whenever TTS transitions
        # from off → on so each utterance's room conditions are
        # calibrated independently.
        self._tts_active_ms = 0
        self._echo_floor = 0.0
        # Wall-clock-style cursor over consumed frames. Lets the
        # refractory window survive bursty / sparse mic streams.
        self._frame_idx = 0
        self._last_tts_active_idx = -(10**9)

    def stop(self) -> None:
        self._stop.set()

    def run(self, frames: Any) -> None:
        """Consume ``frames`` (iterable of float32 arrays). Triggers
        the controller when the user genuinely speaks over the bot.

        Kept numpy-free in the protocol: RMS is computed in pure
        Python so this doesn't force a numpy dep even though the hot
        path uses numpy arrays. Tolerates non-numpy iterables for
        tests.
        """
        policy = self.controller.policy
        sustained_ms = 0
        prev_tts_playing = False

        for frame in frames:
            if self._stop.is_set():
                return
            self._frame_idx += 1
            now_idx = self._frame_idx
            tts_playing = self._is_tts_playing()
            mic_rms = _rms(frame)

            # ----- TTS off → on: start a fresh echo-floor calibration.
            if tts_playing and not prev_tts_playing:
                self._tts_active_ms = 0
                self._echo_floor = 0.0
            prev_tts_playing = tts_playing

            if tts_playing:
                self._tts_active_ms += self._frame_ms
                self._last_tts_active_idx = now_idx
                # Calibrate echo floor during the prefix window.
                if self._tts_active_ms <= policy.echo_floor_window_ms:
                    if mic_rms > self._echo_floor:
                        self._echo_floor = mic_rms
                    sustained_ms = 0  # don't count calibration frames
                    continue

                # Build the effective threshold from all sources.
                threshold = policy.energy_threshold
                if self._echo_floor > 0:
                    threshold = max(threshold, self._echo_floor * policy.echo_suppression_ratio)
                if self._tts_energy_provider is not None:
                    try:
                        tts_rms = float(self._tts_energy_provider() or 0.0)
                    except Exception:
                        tts_rms = 0.0
                    if tts_rms > 0:
                        threshold = max(threshold, tts_rms * policy.echo_suppression_ratio)

                if mic_rms >= threshold:
                    sustained_ms += self._frame_ms
                    if sustained_ms >= policy.min_duration_ms:
                        self.controller.trigger(
                            "user_barge_in",
                            rms=mic_rms,
                            threshold=threshold,
                            echo_floor=self._echo_floor,
                        )
                        sustained_ms = 0
                else:
                    sustained_ms = max(0, sustained_ms - self._frame_ms)
            else:
                # TTS not playing right now. Either we're between
                # turns, or we just stopped and are in the refractory
                # window where echo tail can still trick the watcher.
                released = (now_idx - self._last_tts_active_idx) * self._frame_ms
                if released < policy.tts_release_ms:
                    sustained_ms = 0
                    continue
                # Outside the bot's reply window altogether — nothing
                # to barge in over. Keep the state machine quiet.
                sustained_ms = 0
                self._tts_active_ms = 0


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
