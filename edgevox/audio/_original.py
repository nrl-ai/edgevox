"""Audio I/O and Voice Activity Detection with interrupt support.

Records at the device's native sample rate and resamples to 16kHz for VAD/STT.
Uses echo suppression: mic is paused while TTS plays, with cooldown after.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from edgevox.audio.aec import NoAEC, create_aec

if TYPE_CHECKING:
    import sounddevice as sd
else:
    sd: Any = None  # populated lazily by _sd()


def _sd():
    """Lazy-import sounddevice. Server-only deployments don't need a mic."""
    global sd
    if sd is None:
        import sounddevice as _real_sd

        sd = _real_sd
    return sd


log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000  # 16 kHz mono - what VAD and STT expect
CHANNELS = 1
VAD_SAMPLES = 512  # Silero VAD v6 requires exactly 512 samples at 16kHz

# How many consecutive silent VAD frames before we consider speech ended
SILENCE_FRAMES_THRESHOLD = 23  # ~736ms of silence (23 * 32ms)

# --- Interrupt detection during playback ---
# First N frames after playback starts are used to measure the echo baseline
INTERRUPT_BASELINE_FRAMES = 10  # ~320ms to measure speaker echo level
# Consecutive speech frames required after baseline is established
INTERRUPT_SPEECH_FRAMES = 8  # ~256ms of sustained loud speech
# User voice must exceed the reference (or no-AEC baseline) RMS by this
# factor before we believe the cleaned mic signal is real user speech.
# 3.0 is empirically the smallest ratio that filters out specsub-cleaned
# residual echo on typical USB mic + laptop-speaker pairs while still
# letting normal indoor-conversation volumes through.
INTERRUPT_RMS_RATIO = 3.0
# Below this reference-output RMS the speaker is effectively silent and
# the ratio gate is bypassed — otherwise quiet TTS would block real user
# speech from triggering. Calibrated so a TTS gap or end-of-sentence
# tail is treated as silent.
INTERRUPT_REF_QUIET = 0.005
# Absolute minimum RMS to consider (prevents triggering on near-silence)
INTERRUPT_MIN_RMS = 0.01

# After TTS playback stops, ignore mic for this duration to flush echo residue
ECHO_COOLDOWN_SECS = 1.5


def _get_device_sample_rate() -> int:
    """Get the default input device's native sample rate."""
    try:
        info = _sd().query_devices(kind="input")
        return int(info["default_samplerate"])
    except Exception:
        return 48000


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple resample using linear interpolation. Fast and good enough for VAD."""
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    left = np.floor(indices).astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = indices - left
    return (audio[left] * (1 - frac) + audio[right] * frac).astype(np.float32)


class VAD:
    """Silero VAD v6 via pure onnxruntime — no torch required.

    Uses the silero_vad_v6.onnx bundled with faster-whisper.
    Inputs: audio [1, 576] (64 context + 512 samples), h/c LSTM state.
    """

    def __init__(self, threshold: float = 0.4):
        import os

        import onnxruntime as ort
        from faster_whisper.utils import get_assets_path

        model_path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros(64, dtype="float32")
        self._threshold = threshold

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if a 512-sample (16kHz) chunk contains speech."""
        inp = np.concatenate([self._context, audio_chunk]).reshape(1, 576)
        out, self._h, self._c = self._session.run(
            None,
            {"input": inp, "h": self._h, "c": self._c},
        )
        self._context = audio_chunk[-64:].copy()
        return float(out[0]) >= self._threshold

    def reset(self):
        self._h[:] = 0
        self._c[:] = 0
        self._context[:] = 0


class _RefBuffer:
    """Thread-safe AEC reference ring buffer that stores numpy chunks.

    The producer is the PortAudio audio-thread callback, which must do
    O(1) work per push and never iterate samples in Python.  The consumer
    is the recorder process loop, which pops fixed-size frames.  Old
    chunks are dropped from the head once the total sample count exceeds
    ``max_samples`` (about one second by default).
    """

    def __init__(self, max_samples: int):
        self._max_samples = max_samples
        self._chunks: list[np.ndarray] = []
        self._head = 0  # samples already consumed from chunks[0]
        self._total = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._total

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        # Detach from any caller-owned buffer (e.g. PortAudio's outdata,
        # which gets reused after the callback returns).
        samples = np.array(samples, dtype=np.float32, copy=True)
        with self._lock:
            self._chunks.append(samples)
            self._total += samples.size
            while self._total > self._max_samples and self._chunks:
                head = self._chunks[0]
                head_remaining = head.size - self._head
                excess = self._total - self._max_samples
                if excess >= head_remaining:
                    self._chunks.pop(0)
                    self._head = 0
                    self._total -= head_remaining
                else:
                    self._head += excess
                    self._total -= excess

    def extend(self, samples) -> None:
        """Compat helper: accept any iterable of floats and push as one chunk."""
        arr = np.asarray(list(samples), dtype=np.float32)
        self.push(arr)

    def pop(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if n <= 0:
            return out
        with self._lock:
            written = 0
            while written < n and self._chunks:
                head = self._chunks[0]
                avail = head.size - self._head
                take = min(n - written, avail)
                out[written : written + take] = head[self._head : self._head + take]
                self._head += take
                self._total -= take
                written += take
                if self._head >= head.size:
                    self._chunks.pop(0)
                    self._head = 0
        return out

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._head = 0
            self._total = 0


class InterruptiblePlayer:
    """Audio player that can be interrupted mid-playback.

    Uses a callback-based PortAudio output stream backed by a numpy buffer.
    The audio thread always has something to emit — real samples when
    queued, silence otherwise — so ALSA never under-runs and we avoid the
    ``mmap_begin`` / ``SetUpBuffers`` failures that a blocking
    ``stream.write()`` loop hits when streaming TTS can't deliver chunks
    fast enough.

    The stream stays open across plays for the same device + sample rate;
    ``interrupt()`` flushes the queued buffer rather than tearing the
    stream down, which avoids races with in-flight callbacks.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self._device: int | None = None
        self._stream: sd.OutputStream | None = None
        self._stream_sr: int = 0
        self._stream_device: int | None = None
        self._channels: int = 1
        self._recorder: AudioRecorder | None = None  # linked recorder for echo suppression
        # Reference signal buffer for AEC (16 kHz mono float32)
        self._ref_buffer: _RefBuffer | None = None
        # Most recent output-frame RMS — sampled in the audio callback.
        # Read by the barge-in watcher's ``tts_energy_provider`` hook to
        # build an echo-relative threshold, so the watcher can tell
        # "speaker is loud" from "user is loud" without needing AEC.
        self._last_output_rms: float = 0.0

        # Pending audio for the PortAudio callback. Shape (N, channels), float32.
        self._buf_lock = threading.Lock()
        self._play_buf = np.zeros((0, 1), dtype=np.float32)

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    @property
    def last_output_rms(self) -> float:
        """Most recent audio-callback frame's RMS (mono, post-resample
        device output). 0.0 when nothing is playing.

        Pass ``tts_energy_provider=lambda: player.last_output_rms`` to
        :class:`~edgevox.agents.interrupt.EnergyBargeInWatcher` so its
        echo-suppression threshold scales with how loud the bot
        currently is. Updated on every audio callback, so it lags
        real output by at most one block (~10-20 ms typical).
        """
        return self._last_output_rms

    def link_recorder(self, recorder: AudioRecorder | None):
        """Link a recorder for automatic echo suppression (pause mic during playback)."""
        self._recorder = recorder

    def enable_ref_capture(self):
        """Enable capturing the playback reference signal for AEC.

        The buffer holds up to 1 second of 16 kHz audio.  Call this when an
        AEC-enabled recorder is linked.
        """
        self._ref_buffer = _RefBuffer(TARGET_SAMPLE_RATE)

    def get_ref_frame(self, n: int) -> np.ndarray:
        """Pop *n* samples from the reference buffer.

        Returns zeros when no playback audio is available (silence = no echo).
        """
        if self._ref_buffer is None:
            return np.zeros(n, dtype=np.float32)
        return self._ref_buffer.pop(n)

    def set_device(self, device: int | None):
        """Set the output device index. Closes current stream if device changed.

        Interrupts any in-flight ``play()`` so we don't yank the stream out
        from underneath the audio callback.
        """
        if device == self._device:
            return
        self._stop.set()
        with self._lock:
            self._stop.clear()
            self._close_stream()
            self._device = device

    def _callback(self, outdata, frames, time_info, status):
        """PortAudio audio-thread callback.

        Drains the queued buffer into ``outdata``, padding with silence on
        underrun so the device never starves.  Pushes the actually-played
        samples (downsampled to 16 kHz mono) to the AEC reference buffer.
        Must stay non-blocking — no I/O, no logging, no Python locks held
        across the resample.
        """
        with self._buf_lock:
            available = self._play_buf.shape[0]
            n = min(frames, available)
            if n > 0:
                outdata[:n] = self._play_buf[:n]
                self._play_buf = self._play_buf[n:]
            if n < frames:
                outdata[n:] = 0

        if n > 0:
            mono = outdata[:n, 0] if outdata.ndim == 2 else outdata[:n]
            arr = np.asarray(mono, dtype=np.float32)
            # Cheap RMS so the barge-in watcher has a live reference
            # signal without needing the full AEC pipeline.
            self._last_output_rms = float(np.sqrt((arr * arr).mean())) if arr.size else 0.0
            if self._ref_buffer is not None:
                ref_16k = _resample(arr, self._stream_sr, TARGET_SAMPLE_RATE)
                self._ref_buffer.push(ref_16k)
        else:
            # Silent frame — nothing playing right now.
            self._last_output_rms = 0.0

    def _get_stream(self, sample_rate: int) -> sd.OutputStream:
        """Get or create a persistent output stream."""
        _sd()
        if (
            self._stream is not None
            and self._stream.active
            and self._stream_sr == sample_rate
            and self._stream_device == self._device
        ):
            return self._stream
        self._close_stream()
        # Query the device's native channels to avoid ALSA channel mismatch
        try:
            info = sd.query_devices(self._device) if self._device is not None else sd.query_devices(kind="output")
            channels = min(int(info.get("max_output_channels", 2)), 2)
        except Exception:
            channels = 1
        channels = max(channels, 1)
        self._channels = channels
        with self._buf_lock:
            self._play_buf = np.zeros((0, channels), dtype=np.float32)
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        self._stream_sr = sample_rate
        self._stream_device = self._device
        return self._stream

    def _close_stream(self):
        """Close the persistent stream and drop any queued audio."""
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
        with self._buf_lock:
            self._play_buf = np.zeros((0, max(self._channels, 1)), dtype=np.float32)

    def _flush_buffer(self):
        """Drop any queued playback audio. Callback then emits silence."""
        with self._buf_lock:
            self._play_buf = np.zeros((0, max(self._channels, 1)), dtype=np.float32)

    def interrupt(self):
        """Stop current playback immediately.

        Flushes the queued buffer so the callback emits silence on the next
        tick. The stream itself is left running — touching it from another
        thread races with in-flight callbacks and triggers the ALSA
        ``mmap_begin`` failures we're trying to avoid.
        """
        self._stop.set()
        self._flush_buffer()

    def play(self, audio: np.ndarray, sample_rate: int = 24_000) -> bool:
        """Enqueue audio for playback. Returns True if completed, False if interrupted.

        Pauses the linked recorder's mic stream during playback to prevent
        the bot from hearing its own voice (echo suppression).
        """
        with self._lock:
            self._stop.clear()
            self._playing.set()

            if self._recorder:
                self._recorder.pause()

            try:
                self._get_stream(sample_rate)
                # Reshape mono to match stream channel count
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                if audio.shape[1] < self._channels:
                    audio = np.tile(audio, (1, self._channels))
                audio = np.ascontiguousarray(audio, dtype=np.float32)

                with self._buf_lock:
                    self._play_buf = (
                        audio.copy()
                        if self._play_buf.shape[0] == 0
                        else np.concatenate([self._play_buf, audio], axis=0)
                    )

                # Wait for the callback to drain the queued buffer. Polling
                # at 20 ms keeps interrupt latency tight without busy-waiting.
                # Returning here is "early" by one device-latency period —
                # the last samples are still in PortAudio's internal ring —
                # but the recorder's ECHO_COOLDOWN_SECS (1.5 s) already
                # absorbs that tail, and waiting it out would insert an
                # audible gap between streaming-TTS chunks.
                while True:
                    if self._stop.is_set():
                        self._flush_buffer()
                        return False
                    with self._buf_lock:
                        empty = self._play_buf.shape[0] == 0
                    if empty:
                        break
                    time.sleep(0.02)

                return True
            except Exception:
                log.exception("Playback failed")
                return False
            finally:
                self._playing.clear()
                if self._recorder:
                    self._recorder.resume_after_cooldown()

    def shutdown(self):
        """Clean up the persistent stream.

        Interrupts any in-flight ``play()`` and waits for it to release the
        lock before tearing the stream down, so the audio callback can't
        race against close.
        """
        self._stop.set()
        with self._lock:
            self._close_stream()


# Global player instance for interrupt support
player = InterruptiblePlayer()


def _get_device_native_sr(device: int | None) -> int:
    """Get a device's native sample rate."""
    try:
        _sd()
        info = sd.query_devices(device) if device is not None else sd.query_devices(kind="output")
        return int(info["default_samplerate"])
    except Exception:
        return 48000


def play_audio(audio: np.ndarray, sample_rate: int = 24_000) -> bool:
    """Play audio through speakers. Resamples to output device rate if needed."""
    output_sr = _get_device_native_sr(player._device)
    if sample_rate != output_sr:
        audio = _resample(audio, sample_rate, output_sr)
        sample_rate = output_sr
    return player.play(audio, sample_rate)


class AudioRecorder:
    """Records audio from microphone with VAD-based speech boundary detection.

    Records at the device's native sample rate, resamples to 16kHz for VAD/STT.
    Supports echo suppression: mic input is suppressed during TTS playback.
    """

    def __init__(
        self,
        on_speech: Callable[[np.ndarray], None],
        on_interrupt: Callable[[], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_audio_frame: Callable[[np.ndarray], None] | None = None,
        device: int | None = None,
        aec_backend: str = "none",
        player_ref: InterruptiblePlayer | None = None,
    ):
        self._on_speech = on_speech
        self._on_interrupt = on_interrupt or (lambda: None)
        self._on_level = on_level or (lambda _level: None)
        self._on_audio_frame = on_audio_frame  # called with every 16kHz chunk (for wakeword)
        self._device = device
        self._vad = VAD()
        self._aec = create_aec(aec_backend)
        self._use_aec = not isinstance(self._aec, NoAEC)
        self._player_ref = player_ref
        if self._use_aec and self._player_ref is not None:
            self._player_ref.enable_ref_capture()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._suppressed = False  # True while echo-suppressed (during/after playback)
        self._suppress_gen = 0  # generation counter — stale cooldown timers check this
        self._interrupt_detect = False  # True = VAD runs for interrupt detection during playback
        self._interrupt_speech_count = 0  # consecutive speech frames during interrupt detection
        self._interrupt_baseline_count = 0  # frames used for echo baseline measurement
        self._interrupt_baseline_rms = 0.0  # measured echo RMS level
        self._interrupt_speech_buffer: list[np.ndarray] = []  # speech frames captured during interrupt
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        # Get sample rate and channel count for the selected device
        _sd()
        info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
        self._device_sr = int(info["default_samplerate"])
        self._device_channels = min(info["max_input_channels"], 2)  # use stereo if mono fails
        # Block size at device rate that produces ~512 samples at 16kHz (32ms)
        self._device_block = int(self._device_sr * VAD_SAMPLES / TARGET_SAMPLE_RATE)

    def start(self):
        self._running = True
        self._suppressed = False
        self._suppress_gen = 0
        _sd()
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._device_sr,
            channels=self._device_channels,
            dtype="float32",
            blocksize=self._device_block,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=2)

    def pause(self):
        """Suppress normal speech detection but keep VAD running for interrupt detection.

        Called by the player at the start of playback. Instead of fully muting
        the mic, we enter interrupt-detection mode: VAD still processes frames
        and fires ``on_interrupt`` if the user speaks over the bot.

        Only resets interrupt-detection state on the first transition from
        unsuppressed to suppressed, so consecutive chunks (streaming TTS)
        don't reset the echo baseline measurement mid-playback.
        """
        was_suppressed = self._suppressed
        self._suppress_gen += 1
        self._suppressed = True
        self._interrupt_detect = True
        if not was_suppressed:
            self._interrupt_speech_count = 0
            self._interrupt_baseline_count = 0
            self._interrupt_baseline_rms = 0.0
            self._interrupt_speech_buffer.clear()
            if self._use_aec:
                self._aec.reset()
                self._vad.reset()
        log.debug("Mic suppressed, interrupt detection active (gen=%d)", self._suppress_gen)

    def resume_after_cooldown(self):
        """Resume audio processing after cooldown. Called by player.

        Uses a generation counter so that a stale cooldown timer (from an earlier
        sentence) does not un-suppress the mic while a later sentence is playing.
        """
        gen = self._suppress_gen

        def _delayed_resume():
            time.sleep(ECHO_COOLDOWN_SECS)
            # Only resume if no newer pause() has been issued since we started
            if self._suppress_gen != gen:
                log.debug("Skipping stale echo cooldown (gen=%d, current=%d)", gen, self._suppress_gen)
                return
            # Drain any buffered audio from during suppression
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
            self._vad.reset()
            self._interrupt_detect = False
            self._interrupt_speech_count = 0
            self._suppressed = False
            log.debug("Mic resumed after echo cooldown (gen=%d)", gen)

        threading.Thread(target=_delayed_resume, daemon=True).start()

    def force_resume(self, delay: float = 0.1):
        """Cancel echo suppression after a short delay and reset VAD.

        Called when the processing pipeline finishes so the mic doesn't stay
        deaf during the full cooldown window. A brief delay (default 100 ms)
        lets residual echo from the speakers die down.

        Drains the queued audio because, after a *normal* turn, the
        bot's tail audio captured during playback is just echo we want
        to throw away. After a *barge-in*, use
        :meth:`resume_after_interrupt` instead — same generation-counter
        protection, but the audio queue is preserved so the user's
        continuing speech lands in the next STT pass.
        """
        self._suppress_gen += 1  # invalidate any pending cooldown timers
        gen = self._suppress_gen

        def _resume():
            if delay > 0:
                time.sleep(delay)
            if self._suppress_gen != gen:
                return
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
            self._vad.reset()
            self._interrupt_detect = False
            self._interrupt_speech_count = 0
            self._suppressed = False
            log.debug("Mic force-resumed after %.2fs (gen=%d)", delay, gen)

        threading.Thread(target=_resume, daemon=True).start()

    def resume_after_interrupt(self, delay: float = 0.15):
        """Re-arm the mic after a barge-in WITHOUT draining the audio queue.

        Critical difference vs :meth:`force_resume`: the audio queued
        during the brief delay is *kept*. After a user barge-in the user
        is typically still talking; draining the queue would lose those
        samples and force the user to re-speak. The delay is just long
        enough (~150 ms) for PortAudio's output ring + room reverb to
        die down so the bot's tail audio doesn't fire a phantom user
        turn the moment we re-enter speech detection.
        """
        self._suppress_gen += 1
        gen = self._suppress_gen

        def _resume():
            if delay > 0:
                time.sleep(delay)
            if self._suppress_gen != gen:
                return
            self._vad.reset()
            self._interrupt_detect = False
            self._interrupt_speech_count = 0
            self._suppressed = False
            log.debug("Mic resumed after barge-in (delay=%.2fs, gen=%d)", delay, gen)

        threading.Thread(target=_resume, daemon=True).start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self._suppressed and not self._interrupt_detect:
            return  # Don't queue audio during full suppression
        # Extract mono (first channel) regardless of input channel count
        mono = indata[:, 0].copy()
        self._audio_q.put(mono)

    def _process_loop(self):
        speech_buffer: list[np.ndarray] = []  # stores 16kHz audio
        silence_count = 0
        in_speech = False

        while self._running:
            try:
                raw_chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Resample to 16kHz for VAD and STT
            chunk = _resample(raw_chunk, self._device_sr, TARGET_SAMPLE_RATE)

            # Ensure exactly 512 samples for VAD
            if len(chunk) != VAD_SAMPLES:
                if len(chunk) > VAD_SAMPLES:
                    chunk = chunk[:VAD_SAMPLES]
                else:
                    chunk = np.pad(chunk, (0, VAD_SAMPLES - len(chunk)))

            # Interrupt detection mode: detect user speaking over bot playback.
            if self._interrupt_detect:
                is_user_speech = False
                rms = float(np.sqrt(np.mean(chunk**2)))

                # Live reference signal: the player's most recent
                # output-frame RMS. Available regardless of AEC mode and
                # cheap to read (no buffer pop). Drives the energy-ratio
                # gate that's the actual defense against self-trigger
                # — AEC alone leaves enough residual echo to fool VAD on
                # typical mic/speaker pairs.
                ref_rms = self._player_ref.last_output_rms if self._player_ref is not None else 0.0
                if ref_rms > INTERRUPT_REF_QUIET and rms < ref_rms * INTERRUPT_RMS_RATIO:
                    # Speaker is loud and mic isn't clearly louder
                    # — treat the frame as echo no matter what VAD
                    # would say after AEC.
                    self._interrupt_speech_count = 0
                    self._interrupt_speech_buffer.clear()
                    continue

                if self._use_aec:
                    ref = self._player_ref.get_ref_frame(len(chunk)) if self._player_ref else np.zeros_like(chunk)
                    cleaned = self._aec.process(chunk, ref)
                    is_user_speech = self._vad.is_speech(cleaned) and rms >= INTERRUPT_MIN_RMS
                else:
                    if self._interrupt_baseline_count < INTERRUPT_BASELINE_FRAMES:
                        self._interrupt_baseline_count += 1
                        self._interrupt_baseline_rms += (
                            rms - self._interrupt_baseline_rms
                        ) / self._interrupt_baseline_count
                    else:
                        threshold = max(
                            self._interrupt_baseline_rms * INTERRUPT_RMS_RATIO,
                            INTERRUPT_MIN_RMS,
                        )
                        is_user_speech = rms > threshold

                if is_user_speech:
                    # Save the speech frame so it's not lost after interrupt
                    self._interrupt_speech_buffer.append(chunk.copy())
                    self._interrupt_speech_count += 1
                    if self._interrupt_speech_count >= INTERRUPT_SPEECH_FRAMES:
                        log.info(
                            "Voice interrupt (%s, %d frames, %d samples captured)",
                            self._aec.name if self._use_aec else "rms",
                            self._interrupt_speech_count,
                            sum(len(f) for f in self._interrupt_speech_buffer),
                        )
                        self._interrupt_detect = False
                        self._interrupt_speech_count = 0
                        self._on_interrupt()
                        # Self-schedule a short-delay resume so the
                        # capture buffer flushes into Turn 2's STT
                        # without depending on the consumer calling
                        # force_resume() — the consumer's discipline
                        # was the source of the "interrupt only once"
                        # bug. The 150 ms delay absorbs the PortAudio
                        # output ring + room reverb so the bot's tail
                        # audio doesn't fire a phantom turn.
                        # ``resume_after_interrupt`` preserves the
                        # queued mic audio (the user is typically
                        # still talking) instead of draining it the
                        # way ``force_resume`` does after a normal
                        # turn.
                        self.resume_after_interrupt(delay=0.15)
                else:
                    self._interrupt_speech_count = 0
                    # Trim the rolling speech buffer so a single false
                    # frame doesn't bloat memory across long bot
                    # replies. Keep the most recent ~2 frames as a
                    # short pre-roll for the actual barge-in.
                    if len(self._interrupt_speech_buffer) > 2:
                        del self._interrupt_speech_buffer[:-2]
                continue

            # If suppressed (no interrupt detection either), discard.
            # But if we have captured interrupt speech, inject it into
            # the speech buffer so it's included in the next STT pass.
            if self._suppressed:
                speech_buffer.clear()
                silence_count = 0
                in_speech = False
                continue

            # After resuming from suppression, prepend any captured
            # interrupt speech so the user doesn't have to repeat themselves.
            if self._interrupt_speech_buffer:
                speech_buffer.extend(self._interrupt_speech_buffer)
                self._interrupt_speech_buffer.clear()
                in_speech = True
                silence_count = 0
                log.debug("Injected %d interrupt speech frames into buffer", len(speech_buffer))

            # Report audio level
            rms = float(np.sqrt(np.mean(chunk**2)))
            self._on_level(min(1.0, rms * 10))

            # Forward frame to wakeword detector (runs continuously)
            if self._on_audio_frame:
                self._on_audio_frame(chunk)

            is_speech = self._vad.is_speech(chunk)

            if is_speech:
                speech_buffer.append(chunk)
                silence_count = 0
                if not in_speech:
                    in_speech = True
            elif in_speech:
                speech_buffer.append(chunk)
                silence_count += 1
                if silence_count >= SILENCE_FRAMES_THRESHOLD:
                    audio = np.concatenate(speech_buffer)
                    speech_buffer.clear()
                    silence_count = 0
                    in_speech = False
                    self._vad.reset()
                    # Run in a separate thread so _process_loop stays free
                    # for interrupt detection during playback.
                    threading.Thread(target=self._on_speech, args=(audio,), daemon=True).start()
