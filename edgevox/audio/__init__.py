"""Audio subsystem — recording, VAD, playback, wake word."""

from edgevox.audio._original import (
    TARGET_SAMPLE_RATE,
    AudioRecorder,
    InterruptiblePlayer,
    play_audio,
    player,
)

__all__ = [
    "TARGET_SAMPLE_RATE",
    "AudioRecorder",
    "InterruptiblePlayer",
    "play_audio",
    "player",
]
