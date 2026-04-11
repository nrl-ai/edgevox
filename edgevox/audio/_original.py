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


class InterruptiblePlayer:
    """Audio player that can be interrupted mid-playback.

    Keeps a persistent output stream open to avoid initialization latency
    between sentences. The stream is created on first play and reused until
    the device or sample rate changes.

    Supports echo suppression by pausing a linked AudioRecorder during playback.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self._device: int | None = None
        self._stream: sd.OutputStream | None = None
        self._stream_sr: int = 0
        self._stream_device: int | None = None
        self._recorder: AudioRecorder | None = None  # linked recorder for echo suppression

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def link_recorder(self, recorder: AudioRecorder | None):
        """Link a recorder for automatic echo suppression (pause mic during playback)."""
        self._recorder = recorder

    def set_device(self, device: int | None):
        """Set the output device index. Closes current stream if device changed."""
        if device != self._device:
            self._close_stream()
            self._device = device

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
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=self._device,
        )
        self._stream.start()
        self._stream_sr = sample_rate
        self._stream_device = self._device
        return self._stream

    def _close_stream(self):
        """Close the persistent stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def interrupt(self):
        """Stop current playback immediately."""
        self._stop.set()
        # Abort the stream to stop audio instantly, then reopen on next play
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.abort()
            self._close_stream()

    def play(self, audio: np.ndarray, sample_rate: int = 24_000) -> bool:
        """Play audio. Returns True if completed, False if interrupted.

        Pauses the linked recorder's mic stream during playback to prevent
        the bot from hearing its own voice (echo suppression).
        """
        with self._lock:
            self._stop.clear()
            self._playing.set()

            # Pause mic to prevent echo
            if self._recorder:
                self._recorder.pause()

            try:
                stream = self._get_stream(sample_rate)
                # Reshape mono to match stream channel count
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                if audio.shape[1] < self._channels:
                    audio = np.tile(audio, (1, self._channels))
                chunk_samples = int(sample_rate * 0.05)  # 50ms chunks
                for i in range(0, len(audio), chunk_samples):
                    if self._stop.is_set():
                        return False
                    stream.write(audio[i : i + chunk_samples])
                return True
            except Exception:
                # Stream died — close and retry next time
                self._close_stream()
                return False
            finally:
                self._playing.clear()
                # Resume mic after playback + cooldown
                if self._recorder:
                    self._recorder.resume_after_cooldown()

    def shutdown(self):
        """Clean up the persistent stream."""
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
    ):
        self._on_speech = on_speech
        self._on_interrupt = on_interrupt or (lambda: None)
        self._on_level = on_level or (lambda _level: None)
        self._on_audio_frame = on_audio_frame  # called with every 16kHz chunk (for wakeword)
        self._device = device
        self._vad = VAD()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._suppressed = False  # True while echo-suppressed (during/after playback)
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
        """Suppress audio processing (echo suppression). Called by player."""
        self._suppressed = True
        log.debug("Mic suppressed for echo cancellation")

    def resume_after_cooldown(self):
        """Resume audio processing after cooldown. Called by player."""

        def _delayed_resume():
            time.sleep(ECHO_COOLDOWN_SECS)
            # Drain any buffered audio from during suppression
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
            self._vad.reset()
            self._suppressed = False
            log.debug("Mic resumed after echo cooldown")

        threading.Thread(target=_delayed_resume, daemon=True).start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self._suppressed:
            return  # Don't even queue audio during suppression
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

            # If suppressed, discard (shouldn't happen since callback skips, but safety)
            if self._suppressed:
                speech_buffer.clear()
                silence_count = 0
                in_speech = False
                continue

            # Resample to 16kHz for VAD and STT
            chunk = _resample(raw_chunk, self._device_sr, TARGET_SAMPLE_RATE)

            # Ensure exactly 512 samples for VAD
            if len(chunk) != VAD_SAMPLES:
                if len(chunk) > VAD_SAMPLES:
                    chunk = chunk[:VAD_SAMPLES]
                else:
                    chunk = np.pad(chunk, (0, VAD_SAMPLES - len(chunk)))

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
                    self._on_speech(audio)
