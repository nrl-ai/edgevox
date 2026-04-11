"""Audio format helpers for the WebSocket server.

Browsers send int16 PCM @ 16 kHz; TTS produces float32 audio at 24 kHz that we
re-encode to a self-contained WAV blob the browser can drop straight into an
``Audio`` element or ``decodeAudioData``.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def int16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode a little-endian int16 PCM blob into float32 in [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(data, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).copy()


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 audio in [-1, 1] as a WAV byte string (16-bit PCM)."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()
