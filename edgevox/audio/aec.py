"""Acoustic Echo Cancellation backends for interrupt detection.

Provides a pluggable AEC interface: during TTS playback the mic captures both
the user's voice and the bot's echo from speakers.  An AEC backend subtracts
the echo using a reference signal (the TTS audio being played), producing a
cleaned signal that can be fed to Silero VAD for reliable interrupt detection.

Available backends:

* ``none``    -- No AEC.  Falls back to RMS-baseline interrupt detection.
* ``nlms``    -- NLMS adaptive filter.  Classical time-domain approach.
* ``specsub`` -- Spectral subtraction.  Frequency-domain, fast, modern.
* ``dtln``    -- DTLN-aec neural model (ICASSP 2021 AEC Challenge, 3rd place).
                 Requires ``ai-edge-litert`` (pip install ai-edge-litert).
"""

from __future__ import annotations

import abc
import logging
import os
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

# NLMS defaults
_NLMS_FILTER_LEN = 1024  # taps -- covers ~64ms of echo tail at 16 kHz
_NLMS_MU = 0.3  # step size (0 < mu <= 1)
_NLMS_EPS = 1e-8  # regularisation

# Spectral subtraction defaults
_SS_OVERSUBTRACT = 2.0  # how aggressively to subtract echo (>1 = more removal)
_SS_FLOOR = 0.02  # spectral floor to avoid musical noise artifacts


class AECBackend(abc.ABC):
    """Base class for Acoustic Echo Cancellation backends.

    All backends operate on 16 kHz mono float32 audio.
    """

    @abc.abstractmethod
    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        """Remove echo from *mic_frame* using *ref_frame* as the reference.

        Parameters
        ----------
        mic_frame : np.ndarray
            Near-end (microphone) audio, float32, shape ``(frame_size,)``.
        ref_frame : np.ndarray
            Far-end (speaker / TTS) audio, float32, same shape.

        Returns
        -------
        np.ndarray
            Cleaned audio with echo removed, same shape as *mic_frame*.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state (call between utterances)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""


class NoAEC(AECBackend):
    """Pass-through -- no echo cancellation."""

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        return mic_frame

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "none"


class NLMSAdaptiveAEC(AECBackend):
    """NLMS (Normalised Least-Mean-Squares) adaptive filter.

    Pure numpy, no external dependencies.  Maintains a FIR filter whose
    coefficients converge to the room impulse response so that the echo
    component in the mic signal can be predicted and subtracted.
    """

    def __init__(
        self,
        filter_len: int = _NLMS_FILTER_LEN,
        mu: float = _NLMS_MU,
    ):
        self._filter_len = filter_len
        self._mu = mu
        self._w = np.zeros(filter_len, dtype=np.float32)
        self._ref_hist = np.zeros(filter_len, dtype=np.float32)

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        out = np.empty_like(mic_frame)
        for i in range(len(mic_frame)):
            self._ref_hist = np.roll(self._ref_hist, 1)
            self._ref_hist[0] = ref_frame[i]
            echo_hat = float(np.dot(self._w, self._ref_hist))
            error = mic_frame[i] - echo_hat
            out[i] = error
            norm = float(np.dot(self._ref_hist, self._ref_hist)) + _NLMS_EPS
            self._w += (self._mu * error / norm) * self._ref_hist
        return out

    def reset(self) -> None:
        self._w[:] = 0
        self._ref_hist[:] = 0

    @property
    def name(self) -> str:
        return "nlms"


class SpectralSubtractionAEC(AECBackend):
    """Frequency-domain spectral subtraction.

    Computes the FFT of both mic and reference signals, estimates the echo
    magnitude spectrum (with an adaptive scaling factor), subtracts it from
    the mic spectrum, and reconstructs the cleaned time-domain signal via
    overlap-add.

    Faster than time-domain NLMS (single FFT pair per frame vs. N dot
    products) and handles multi-path echo well because each frequency bin
    is treated independently.
    """

    def __init__(
        self,
        frame_size: int = 512,
        oversubtract: float = _SS_OVERSUBTRACT,
        floor: float = _SS_FLOOR,
    ):
        self._frame_size = frame_size
        self._fft_size = frame_size
        self._oversubtract = oversubtract
        self._floor = floor
        # Hann window for smooth overlap-add
        self._window = np.hanning(frame_size).astype(np.float32)
        # Adaptive echo gain estimate (per-bin EMA)
        n_bins = frame_size // 2 + 1
        self._echo_gain = np.ones(n_bins, dtype=np.float32)
        self._alpha = 0.85  # EMA smoothing for echo gain estimate

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        n = len(mic_frame)

        # Windowed FFT
        mic_fft = np.fft.rfft(mic_frame * self._window[: len(mic_frame)])
        ref_fft = np.fft.rfft(ref_frame * self._window[: len(ref_frame)])

        mic_mag = np.abs(mic_fft)
        ref_mag = np.abs(ref_fft)
        mic_phase = np.angle(mic_fft)

        # Adaptive echo gain: estimate how much of the reference shows up
        # in the mic (ratio of mic to ref magnitude, smoothed over time).
        # Only update where reference has significant energy to avoid
        # dividing by near-zero.
        ref_active = ref_mag > 1e-6
        if np.any(ref_active):
            instant_gain = np.ones_like(self._echo_gain)
            instant_gain[ref_active] = mic_mag[ref_active] / ref_mag[ref_active]
            instant_gain = np.clip(instant_gain, 0.0, 10.0)
            self._echo_gain[ref_active] = (
                self._alpha * self._echo_gain[ref_active] + (1 - self._alpha) * instant_gain[ref_active]
            )

        # Subtract estimated echo spectrum
        echo_estimate = ref_mag * self._echo_gain * self._oversubtract
        cleaned_mag = np.maximum(mic_mag - echo_estimate, mic_mag * self._floor)

        # Reconstruct with original phase
        cleaned_fft = cleaned_mag * np.exp(1j * mic_phase)
        cleaned = np.fft.irfft(cleaned_fft, n=n).astype(np.float32)

        return cleaned

    def reset(self) -> None:
        self._echo_gain[:] = 1.0

    @property
    def name(self) -> str:
        return "specsub"


class DTLNAec(AECBackend):
    """DTLN-aec: Dual-signal Transformation LSTM Network for echo cancellation.

    A lightweight neural AEC model (1.8M params) that achieved 3rd place in
    Microsoft's AEC Challenge (ICASSP 2021).  Runs two TFLite models in
    sequence: stage 1 performs frequency-domain masking, stage 2 refines in
    the time domain.

    Requires ``ai-edge-litert`` (``pip install ai-edge-litert``).
    Models are downloaded automatically on first use (~1.4 MB total).
    """

    _BLOCK_LEN = 512
    _BLOCK_SHIFT = 128
    _MODEL_REPO = "https://github.com/breizhn/DTLN-aec/raw/main/pretrained_models"
    _MODEL_FILES = ("dtln_aec_128_1.tflite", "dtln_aec_128_2.tflite")

    def __init__(self, model_dir: str | None = None):
        from ai_edge_litert.interpreter import Interpreter

        self._Interpreter = Interpreter
        self._model_dir = model_dir or self._default_model_dir()
        self._ensure_models()

        # Stage 1: frequency-domain masking
        self._interp1 = self._Interpreter(
            model_path=str(self._model_path(0)),
        )
        self._interp1.allocate_tensors()
        inp1 = self._interp1.get_input_details()
        out1 = self._interp1.get_output_details()
        self._in1_idx = [d["index"] for d in inp1]
        self._out1_idx = [d["index"] for d in out1]
        self._states1 = np.zeros(inp1[1]["shape"], dtype=np.float32)

        # Stage 2: time-domain refinement
        self._interp2 = self._Interpreter(
            model_path=str(self._model_path(1)),
        )
        self._interp2.allocate_tensors()
        inp2 = self._interp2.get_input_details()
        out2 = self._interp2.get_output_details()
        self._in2_idx = [d["index"] for d in inp2]
        self._out2_idx = [d["index"] for d in out2]
        self._states2 = np.zeros(inp2[1]["shape"], dtype=np.float32)

        # Internal overlap-add buffers
        bl = self._BLOCK_LEN
        self._in_buffer = np.zeros(bl, dtype=np.float32)
        self._ref_buffer = np.zeros(bl, dtype=np.float32)
        self._out_buffer = np.zeros(bl, dtype=np.float32)
        self._output_acc = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _default_model_dir() -> str:
        return os.path.join(os.path.expanduser("~"), ".cache", "edgevox", "dtln_aec")

    def _model_path(self, idx: int) -> str:
        return os.path.join(self._model_dir, self._MODEL_FILES[idx])

    def _ensure_models(self):
        """Download model files if not already cached."""
        os.makedirs(self._model_dir, exist_ok=True)
        for fname in self._MODEL_FILES:
            path = os.path.join(self._model_dir, fname)
            if os.path.isfile(path):
                continue
            url = f"{self._MODEL_REPO}/{fname}"
            log.info("Downloading DTLN-aec model: %s", url)
            urllib.request.urlretrieve(url, path)
            log.info("Saved to %s", path)

    def _process_block(self):
        """Run one DTLN-aec inference block (512 samples, 128-sample shift)."""
        bl = self._BLOCK_LEN

        # Stage 1: frequency-domain masking
        in_fft = np.fft.rfft(self._in_buffer)
        in_mag = np.abs(in_fft).reshape(1, 1, -1).astype(np.float32)
        ref_mag = np.abs(np.fft.rfft(self._ref_buffer)).reshape(1, 1, -1).astype(np.float32)

        self._interp1.set_tensor(self._in1_idx[0], in_mag)
        self._interp1.set_tensor(self._in1_idx[1], self._states1)
        self._interp1.set_tensor(self._in1_idx[2], ref_mag)
        self._interp1.invoke()
        out_mask = self._interp1.get_tensor(self._out1_idx[0])
        self._states1 = self._interp1.get_tensor(self._out1_idx[1])

        # Apply mask in frequency domain and IFFT
        estimated = np.fft.irfft(in_fft * out_mask.squeeze()).astype(np.float32)

        # Stage 2: time-domain refinement
        estimated_in = estimated.reshape(1, 1, bl).astype(np.float32)
        ref_in = self._ref_buffer.reshape(1, 1, bl).astype(np.float32)

        self._interp2.set_tensor(self._in2_idx[0], estimated_in)
        self._interp2.set_tensor(self._in2_idx[1], self._states2)
        self._interp2.set_tensor(self._in2_idx[2], ref_in)
        self._interp2.invoke()
        out_block = self._interp2.get_tensor(self._out2_idx[0]).squeeze()
        self._states2 = self._interp2.get_tensor(self._out2_idx[1])

        return out_block

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        bs = self._BLOCK_SHIFT

        # Feed mic_frame through the overlap-add pipeline in block_shift-sized chunks
        output_chunks = []
        for i in range(0, len(mic_frame), bs):
            chunk_end = min(i + bs, len(mic_frame))
            chunk_len = chunk_end - i

            # Shift buffers left and append new samples
            self._in_buffer[:-bs] = self._in_buffer[bs:]
            self._in_buffer[-bs:] = 0
            self._in_buffer[-chunk_len:] = mic_frame[i:chunk_end]

            self._ref_buffer[:-bs] = self._ref_buffer[bs:]
            self._ref_buffer[-bs:] = 0
            self._ref_buffer[-chunk_len:] = ref_frame[i:chunk_end] if i < len(ref_frame) else 0

            # Run model
            out_block = self._process_block()

            # Overlap-add
            self._out_buffer[:-bs] = self._out_buffer[bs:]
            self._out_buffer[-bs:] = 0
            self._out_buffer += out_block
            output_chunks.append(self._out_buffer[:chunk_len].copy())

        return np.concatenate(output_chunks).astype(np.float32)

    def reset(self) -> None:
        self._states1[:] = 0
        self._states2[:] = 0
        self._in_buffer[:] = 0
        self._ref_buffer[:] = 0
        self._out_buffer[:] = 0

    @property
    def name(self) -> str:
        return "dtln"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[AECBackend]] = {
    "none": NoAEC,
    "nlms": NLMSAdaptiveAEC,
    "specsub": SpectralSubtractionAEC,
    "dtln": DTLNAec,
}

AEC_CHOICES = list(_BACKENDS.keys())


def create_aec(backend: str = "none", **kwargs) -> AECBackend:
    """Create an AEC backend by name.

    Falls back gracefully: dtln -> specsub -> none.
    """
    backend = backend.lower()
    if backend not in _BACKENDS:
        log.warning("Unknown AEC backend %r, falling back to 'none'", backend)
        return NoAEC()

    try:
        return _BACKENDS[backend](**kwargs)
    except ImportError:
        fallback = "specsub" if backend == "dtln" else "none"
        log.warning("AEC backend %r unavailable (missing dependency), falling back to %r", backend, fallback)
        return create_aec(fallback, **kwargs)
    except Exception:
        log.exception("Failed to create AEC backend %r, falling back to 'none'", backend)
        return NoAEC()
