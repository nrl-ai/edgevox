"""Audio subsystem — recording, VAD, playback, wake word, echo cancellation."""

from edgevox.audio._original import (
    TARGET_SAMPLE_RATE,
    AudioRecorder,
    InterruptiblePlayer,
    play_audio,
    player,
)
from edgevox.audio.aec import AEC_CHOICES, AECBackend, create_aec

__all__ = [
    "AEC_CHOICES",
    "TARGET_SAMPLE_RATE",
    "AECBackend",
    "AudioRecorder",
    "InterruptiblePlayer",
    "create_aec",
    "play_audio",
    "player",
]
