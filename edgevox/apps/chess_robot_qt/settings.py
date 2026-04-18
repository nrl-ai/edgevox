"""User preferences: piece set, board colours, persona.

Persisted via :class:`QSettings` so changes survive app restarts.
Exposed as a lightweight :class:`Settings` dataclass the UI reads and
the :class:`SettingsDialog` writes.

All shipped piece sets are MIT-licensed (Maurizio Monge's fantasy /
celtic / spatial). Board themes are pure hex colour pairs — free
to tweak, no asset licensing concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings

# ----- piece sets -----

PIECE_SETS: dict[str, str] = {
    # key → human-readable label. Asset dir is "assets/pieces-<key>".
    "fantasy": "Fantasy (Staunton)",
    "celtic": "Celtic",
    "spatial": "Spatial (line art)",
}


def piece_set_dir(key: str) -> Path:
    root = Path(__file__).resolve().parent / "assets"
    if (root / f"pieces-{key}").is_dir():
        return root / f"pieces-{key}"
    return root / "pieces-fantasy"


# ----- board themes — light / dark square colour pairs -----

BOARD_THEMES: dict[str, tuple[str, str]] = {
    "wood": ("#f0d9b5", "#b58863"),  # lichess default — softer, warmer
    "green": ("#eeeed2", "#769656"),  # chess.com green
    "blue": ("#e0e6ee", "#7396bd"),  # lichess blue
    "gray": ("#dddddd", "#888888"),  # neutral
    "brown-dark": ("#e4c194", "#7f5338"),  # darker wood
    "night": ("#4a5b6e", "#1e2b3a"),  # dark on dark for late-night play
}

BOARD_THEME_LABELS: dict[str, str] = {
    "wood": "Wood (default)",
    "green": "Green",
    "blue": "Blue",
    "gray": "Gray",
    "brown-dark": "Dark wood",
    "night": "Night",
}


# ----- persona -----

PERSONAS = ("casual", "grandmaster", "trash_talker")
PERSONA_LABELS = {
    "casual": "Casual Club Player",
    "grandmaster": "Grandmaster",
    "trash_talker": "Trash-Talking Coach",
}


@dataclass
class Settings:
    """Current persisted preferences.

    ``input_device`` / ``output_device`` hold the PortAudio device
    index; ``None`` means "system default" and is portable across
    machines. A saved index is only valid on the machine that chose it,
    so consumers must gracefully fall back to the default when a saved
    index no longer resolves (see ``available_input_devices``).
    """

    piece_set: str = "fantasy"
    board_theme: str = "wood"
    persona: str = "casual"
    voice_enabled: bool = True
    sfx_muted: bool = False
    input_device: int | None = None
    output_device: int | None = None
    debug_mode: bool = False

    @classmethod
    def load(cls) -> Settings:
        q = QSettings("EdgeVox", "RookApp")
        return cls(
            piece_set=str(q.value("piece_set", "fantasy")),
            board_theme=str(q.value("board_theme", "wood")),
            persona=str(q.value("persona", "casual")),
            voice_enabled=_bool(q.value("voice_enabled", True)),
            sfx_muted=_bool(q.value("sfx_muted", False)),
            input_device=_device(q.value("input_device", None)),
            output_device=_device(q.value("output_device", None)),
            debug_mode=_bool(q.value("debug_mode", False)),
        )

    def save(self) -> None:
        q = QSettings("EdgeVox", "RookApp")
        q.setValue("piece_set", self.piece_set)
        q.setValue("board_theme", self.board_theme)
        q.setValue("persona", self.persona)
        q.setValue("voice_enabled", self.voice_enabled)
        q.setValue("sfx_muted", self.sfx_muted)
        # QSettings backends drop ``None`` inconsistently; stash an
        # empty string for "system default" and round-trip via
        # ``_device`` on load.
        q.setValue("input_device", "" if self.input_device is None else int(self.input_device))
        q.setValue("output_device", "" if self.output_device is None else int(self.output_device))
        q.setValue("debug_mode", self.debug_mode)


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


def _device(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def available_input_devices() -> list[tuple[int, str]]:
    """Return ``(index, name)`` pairs for devices with input channels.

    Caller uses index -1 (conventionally) for "system default"; we
    surface that via ``None`` in the dialog wiring instead. Errors are
    swallowed — if PortAudio can't enumerate we just return an empty
    list and the dialog hides the picker.
    """
    return _query_devices(kind="input")


def available_output_devices() -> list[tuple[int, str]]:
    return _query_devices(kind="output")


def _query_devices(*, kind: str) -> list[tuple[int, str]]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    out: list[tuple[int, str]] = []
    for idx, dev in enumerate(devices):
        if int(dev.get(key, 0)) > 0:
            name = str(dev.get("name") or f"device {idx}")
            out.append((idx, name))
    return out


__all__ = [
    "BOARD_THEMES",
    "BOARD_THEME_LABELS",
    "PERSONAS",
    "PERSONA_LABELS",
    "PIECE_SETS",
    "Settings",
    "available_input_devices",
    "available_output_devices",
    "piece_set_dir",
]
