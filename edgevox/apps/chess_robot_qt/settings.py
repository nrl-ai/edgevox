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


# ----- LLM choices -----
#
# Three lightweight models the user can pick between in Settings. All
# three are already registered in :mod:`edgevox.llm.models` so the same
# preset-slug string feeds straight into :class:`~edgevox.llm.LLM`.
# Model swap requires a fresh agent build → next launch; the dialog
# flags that in the restart hint.

# LLM choices exposed to the user in the Settings dialog.
#
# Ranked and annotated based on the chess-commentary benchmark
# (``docs/documentation/reports/chess-commentary-benchmark.md``).
# Leading stars in the label — ⭐⭐⭐ / ⭐⭐ / ⭐ — are shown in the
# dropdown so the user sees the recommendation without reading the
# doc. Higher stars mean better quality vs speed on realistic hardware
# (CPU / small GPU). Quality was measured on 36 hand-audited chess
# scenarios; attribution correctness on mate / blunder turns weighs
# more heavily than heuristic score.
#
# Insertion order = display order, so the first entry is the default
# picker position.
LLM_CHOICES: dict[str, str] = {
    # Top pick — cleanest semantic audit, acceptable speed on CPU.
    "gemma-4-e2b": "⭐⭐⭐  Gemma 4 E2B — default, best quality (~1.8 GB)",
    # Close second — Apache-2.0, similar quality, slightly slower
    # due to thinking-mode overhead even with /no_think.
    "qwen3-1.7b": "⭐⭐  Qwen3 1.7B — Apache-2.0 alternative (~1.1 GB)",
    # Middle-ground size bump; marginal quality gain over 1B.
    "llama-3.2-3b": "⭐⭐  Llama 3.2 3B — larger, more reliable than 1B (~2.0 GB)",
    # Lightest option — attribution slips on mid-game but canned
    # game-end replies eliminate the worst failure mode. OK for
    # low-RAM hardware.
    "llama-3.2-1b": "⭐  Llama 3.2 1B — lightest / fastest, some slips (~0.8 GB)",
    # Smallest of the Qwen2.5 family; included as an Apache-2.0
    # sub-1 GB option.
    "qwen2.5-1.5b": "⭐  Qwen2.5 1.5B — Apache-2.0, tiny (~1.0 GB)",
}


# ----- Side choices -----
#
# The "Play as" picker offered at New Game. ``random`` is a request the
# side-picker dialog resolves to ``white`` / ``black`` before it ever
# hits :class:`Settings` — only resolved sides are persisted so the
# next launch opens with the same orientation the user last saw.

SIDE_CHOICES: dict[str, str] = {
    "white": "White — you move first",
    "black": "Black — Rook opens",
    "random": "Random — pick for me",
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
    # Preset slug understood by ``edgevox.llm.llamacpp._resolve_model_path``.
    # Must be one of :data:`LLM_CHOICES`. Default is Gemma 4 E2B —
    # picked by the LLM eval harness as best across the three sizes
    # (cleanest pronouns, shortest replies, lowest fabrication). Users
    # can switch to Llama 3.2 1B (fastest / lightest) or Qwen 3 1.7B
    # (Apache-2.0) via the Settings dialog.
    llm_model: str = "gemma-4-e2b"
    # Side the human plays. Only ``"white"`` / ``"black"`` are persisted —
    # ``"random"`` is resolved by the New Game dialog before it ever
    # reaches here so the next launch keeps the orientation the user
    # actually saw. Feeds ``RookConfig.user_plays`` and
    # ``ChessBoardView.set_orientation`` at startup.
    user_side: str = "white"

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
            llm_model=_llm_slug(q.value("llm_model", "gemma-4-e2b")),
            user_side=_side_slug(q.value("user_side", "white")),
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
        q.setValue("llm_model", self.llm_model)
        q.setValue("user_side", self.user_side)


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


def _llm_slug(v) -> str:
    """Pin the saved LLM slug to a known value — a stray QSettings
    blob that doesn't match the current preset registry would
    otherwise crash ``LLM()`` on launch."""
    slug = str(v) if v is not None else "gemma-4-e2b"
    return slug if slug in LLM_CHOICES else "gemma-4-e2b"


def _side_slug(v) -> str:
    """Pin the persisted side to ``white`` / ``black``. ``random`` is
    resolved by the picker before save — a stray ``random`` in
    QSettings (from a partially-migrated install or a test harness)
    would otherwise leave ``RookConfig.user_plays`` in an unexpected
    state."""
    slug = str(v).lower() if v is not None else "white"
    return slug if slug in ("white", "black") else "white"


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
    "LLM_CHOICES",
    "PERSONAS",
    "PERSONA_LABELS",
    "PIECE_SETS",
    "Settings",
    "available_input_devices",
    "available_output_devices",
    "piece_set_dir",
]
