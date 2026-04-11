"""EdgeVox TUI — Sub-second local voice AI for robots and edge devices.

Features:
- Branded ASCII splash with model info panel
- Sparkline waveform + latency history graph
- GPU/RAM usage monitor
- Animated state indicators (pulsing/spinning)
- Conversation export (Ctrl+S)
- Slash commands (/model, /voice, /lang, /reset)
- ROS2 bridge integration (--ros2)
- Streaming LLM -> sentence-split -> TTS for lowest latency
- Interrupt: speak while bot is talking to cut it off
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np
import sounddevice as sd
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Label, OptionList, RichLog, Rule, Select, Static
from textual.widgets.option_list import Option

from edgevox.audio import TARGET_SAMPLE_RATE as MIC_SAMPLE_RATE
from edgevox.audio import AudioRecorder, play_audio, player
from edgevox.audio.wakeword import WakeWordDetector
from edgevox.core.config import LANGUAGES, get_lang, needs_stt_reload
from edgevox.core.config import lang_options as get_lang_options
from edgevox.core.pipeline import stream_sentences
from edgevox.llm import LLM
from edgevox.stt import STT, create_stt
from edgevox.tts import BaseTTS, create_tts

log = logging.getLogger(__name__)

__version__ = "0.1.0"

LOGO = "\n".join(
    [
        " ███████ ██████   ██████  ███████ ██    ██  ██████  ██   ██",
        " ██      ██   ██ ██       ██      ██    ██ ██    ██  ██ ██",
        " █████   ██   ██ ██   ███ █████   ██    ██ ██    ██   ███",
        " ██      ██   ██ ██    ██ ██       ██  ██  ██    ██  ██ ██",
        " ███████ ██████   ██████  ███████   ████    ██████  ██   ██",
    ]
)

LOGO_TAGLINE = "Sub-second local voice AI"


def list_input_devices() -> list[tuple[str, int]]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            sr = int(d["default_samplerate"])
            name = f"[{i}] {d['name']} ({sr}Hz)"
            devices.append((name, i))
    return devices


def list_output_devices() -> list[tuple[str, int]]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            sr = int(d["default_samplerate"])
            name = f"[{i}] {d['name']} ({sr}Hz)"
            devices.append((name, i))
    return devices


def _get_default_output_device() -> int | None:
    """Get system default output device index."""
    try:
        info = sd.query_devices(kind="output")
        return int(info["index"]) if "index" in info else sd.default.device[1]
    except Exception:
        return None


_DEVICES_CFG = Path.home() / ".config" / "edgevox" / "devices.json"


def _save_device_prefs(mic: int | None = None, spk: int | None = None):
    """Save selected mic/spk device indices to disk."""
    prefs = _load_device_prefs()
    if mic is not None:
        prefs["mic"] = mic
    if spk is not None:
        prefs["spk"] = spk
    _DEVICES_CFG.parent.mkdir(parents=True, exist_ok=True)
    _DEVICES_CFG.write_text(json.dumps(prefs))


def _load_device_prefs() -> dict:
    """Load saved mic/spk device indices. Returns {} if none saved."""
    try:
        return json.loads(_DEVICES_CFG.read_text())
    except Exception:
        return {}


def _resolve_saved_device(saved_idx: int | None, available: list[tuple[str, int]]) -> int | None:
    """Return saved_idx only if it still exists in the current device list."""
    if saved_idx is None:
        return None
    for _, idx in available:
        if idx == saved_idx:
            return saved_idx
    return None


def _get_gpu_info() -> dict:
    """Get GPU memory info if available."""
    from edgevox.core.gpu import get_nvidia_gpu_name, get_nvidia_used_mb, get_nvidia_vram_gb, has_metal

    vram_gb = get_nvidia_vram_gb()
    if vram_gb is not None:
        used_mb = get_nvidia_used_mb() or 0
        return {
            "name": get_nvidia_gpu_name() or "NVIDIA GPU",
            "used_gb": used_mb / 1024,
            "total_gb": vram_gb,
            "backend": "CUDA",
        }
    if has_metal():
        return {"name": "Apple Silicon", "used_gb": 0, "total_gb": 0, "backend": "Metal"}
    return {"name": "CPU", "used_gb": 0, "total_gb": 0, "backend": "CPU"}


def _get_ram_info() -> tuple[float, float]:
    """Return (used_gb, total_gb)."""
    from edgevox.core.gpu import get_ram_gb

    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3)
    except ImportError:
        total = get_ram_gb()
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            available = int(lines[2].split()[1]) / (1024**2)
            return total - available, total
        except Exception:
            return 0, total


class BotState(Enum):
    LOADING = "loading"
    WAITING_WAKEWORD = "waiting_wakeword"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


STATE_DISPLAY = {
    BotState.LOADING: (" {anim} Loading models...", "#ffb020"),
    BotState.WAITING_WAKEWORD: (" {anim} Say the wake word...", "#ffb020"),
    BotState.LISTENING: (" {anim} Listening — speak now", "#00ff88"),
    BotState.TRANSCRIBING: (" {anim} Transcribing...", "#00e5ff"),
    BotState.THINKING: (" {anim} Thinking...", "#c084fc"),
    BotState.SPEAKING: (" {anim} Speaking...", "#60a5fa"),
    BotState.INTERRUPTED: (" {anim} Interrupted — listening...", "#ffb020"),
}

# Animated indicators per state
STATE_ANIMS = {
    BotState.LOADING: "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    BotState.WAITING_WAKEWORD: "◜◠◝◞◡◟",
    BotState.LISTENING: "●○",
    BotState.TRANSCRIBING: "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    BotState.THINKING: "⣾⣽⣻⢿⡿⣟⣯⣷",
    BotState.SPEAKING: "▁▂▃▄▅▆▇█▇▅▃",
    BotState.INTERRUPTED: "⚡",
}

# Sparkline characters (8 levels)
SPARK_CHARS = " ▁▂▃▄▅▆▇"


def _sparkline(values: list[float], width: int = 24) -> str:
    """Render a list of 0.0-1.0 values as a sparkline string."""
    if not values:
        return SPARK_CHARS[0] * width
    # Take last `width` values
    vals = values[-width:]
    # Pad left if needed
    if len(vals) < width:
        vals = [0.0] * (width - len(vals)) + vals
    result = []
    for v in vals:
        idx = int(min(v, 1.0) * (len(SPARK_CHARS) - 1))
        result.append(SPARK_CHARS[idx])
    return "".join(result)


CSS = """
/* EdgeVox Terminal Theme */
Screen {
    layout: vertical;
    background: #0a0e14;
}

#status-bar {
    height: 3;
    background: #111820;
    border-bottom: solid #1e3a2e;
    content-align: center middle;
}

#main-area { height: 1fr; }

#chat-panel {
    width: 3fr;
    min-width: 40;
    border: round #1e3a2e;
    padding: 0 1;
    background: #0a0e14;
    scrollbar-color: #00ff88;
    scrollbar-color-hover: #00e5ff;
}

#side-panel {
    width: 1fr;
    min-width: 34;
    max-width: 42;
    border: round #1e3a2e;
    padding: 0;
    background: #0d1117;
}

#side-scroll {
    height: 1fr;
    scrollbar-color: #1e3a2e;
}

.side-section {
    padding: 0 1;
    margin: 0;
    height: auto;
}

.side-separator {
    height: 1;
    color: #1e3a2e;
    margin: 0;
}

#audio-level { height: 2; }
#model-info { height: auto; }
#gpu-monitor { height: auto; }
#settings-section { height: auto; }

#command-input {
    height: auto;
    max-height: 5;
    background: #111820;
    padding: 1 1;
}

#cmd-input {
    width: 1fr;
    background: #0d1117;
    border: tall #1e3a2e;
    color: #c9d1d9;
}

#cmd-input:focus {
    border: tall #00ff88;
}

#cmd-input .input--suggestion {
    color: #00ff88 50%;
}

#completion-menu {
    height: auto;
    max-height: 10;
    background: #111820;
    border: tall #00ff88;
    display: none;
    padding: 0;
    margin: 0 1;
    scrollbar-size: 1 1;
}

#completion-menu.visible {
    display: block;
}

OptionList > .option-list--option-highlighted {
    background: #1e3a2e;
    color: #00ff88;
    text-style: bold;
}

OptionList > .option-list--option {
    padding: 0 1;
    color: #8b949e;
}

OptionList > .option-list--option-hover {
    background: #161b22;
}

Select {
    margin: 0;
    height: 3;
}

#settings-label {
    margin: 0;
}

Header {
    background: #111820;
    color: #00ff88;
}

Footer {
    background: #111820;
}

FooterKey {
    background: #1e3a2e;
    color: #00e5ff;
}
"""


class StatusBar(Static):
    state: reactive[BotState] = reactive(BotState.LOADING)
    language: reactive[str] = reactive("en")
    muted: reactive[bool] = reactive(False)
    _anim_frame: int = 0
    _anim_timer: Timer | None = None

    def on_mount(self) -> None:
        self._anim_timer = self.set_interval(0.15, self._tick)

    def _tick(self) -> None:
        self._anim_frame += 1
        self.refresh()

    def render(self) -> Text:
        template, color = STATE_DISPLAY[self.state]
        anim_chars = STATE_ANIMS[self.state]
        char = anim_chars[self._anim_frame % len(anim_chars)]
        label = template.format(anim=char)
        t = Text()
        t.append(label, style=f"bold {color}")
        # Show language badge
        t.append(f"  [{self.language.upper()}]", style="#8b949e")
        # Show mute indicator
        if self.muted:
            t.append("  \U0001f507 MUTED", style="bold #ff4444")
        return t


class AudioLevel(Static):
    level: reactive[float] = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: deque[float] = deque(maxlen=28)

    def watch_level(self, value: float) -> None:
        self._history.append(value)

    def render(self) -> Text:
        spark = _sparkline(list(self._history), width=22)
        color = "#ff4444" if self.level > 0.6 else ("#ffb020" if self.level > 0.3 else "#00ff88")
        return Text.assemble(
            (" \u25cf Audio ", "bold #00e5ff"),
            (spark, color),
            (f" {self.level:.0%}", "dim"),
        )


class ModelInfoPanel(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stt_model = "..."
        self._llm_model = "..."
        self._tts_model = "..."
        self._stt_device = ""
        self._llm_device = ""

    def set_info(self, stt_model: str, stt_device: str, llm_model: str, llm_device: str, tts_model: str):
        self._stt_model = stt_model
        self._stt_device = stt_device
        self._llm_model = llm_model
        self._llm_device = llm_device
        self._tts_model = tts_model
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" \u25a0 Models\n", style="bold #00ff88")
        t.append("  STT ", style="#8b949e")
        t.append(self._stt_model, style="#00e5ff")
        if self._stt_device:
            t.append(f" {self._stt_device}", style="dim")
        t.append("\n")
        t.append("  LLM ", style="#8b949e")
        t.append(self._llm_model, style="#c084fc")
        if self._llm_device:
            t.append(f" {self._llm_device}", style="dim")
        t.append("\n")
        t.append("  TTS ", style="#8b949e")
        t.append(self._tts_model, style="#60a5fa")
        return t


class GpuMonitor(Static):
    _timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(3.0, self._update)
        self._update()

    def _update(self) -> None:
        self.refresh()

    def render(self) -> Text:
        gpu = _get_gpu_info()
        ram_used, ram_total = _get_ram_info()
        t = Text()
        t.append(" \u25a0 System\n", style="bold #00ff88")

        if gpu["backend"] != "CPU" and gpu["total_gb"] > 0:
            pct = gpu["used_gb"] / gpu["total_gb"] * 100 if gpu["total_gb"] > 0 else 0
            bar_len = 14
            filled = int(pct / 100 * bar_len)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            color = "#ff4444" if pct > 80 else ("#ffb020" if pct > 50 else "#00ff88")
            t.append("  GPU ", style="#8b949e")
            t.append(f"{gpu['name']}\n", style="bold #c9d1d9")
            t.append(f"  VRAM [{bar}] ", color)
            t.append(f"{gpu['used_gb']:.1f}/{gpu['total_gb']:.1f}G\n", style="dim")
        else:
            t.append("  GPU ", style="#8b949e")
            t.append(f"{gpu['name']} ({gpu['backend']})\n", style="#c9d1d9")

        if ram_total > 0:
            pct = ram_used / ram_total * 100
            bar_len = 14
            filled = int(pct / 100 * bar_len)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            color = "#ff4444" if pct > 80 else ("#ffb020" if pct > 50 else "#00ff88")
            t.append(f"  RAM  [{bar}] ", color)
            t.append(f"{ram_used:.1f}/{ram_total:.1f}G", style="dim")
        return t


class MetricsPanel(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stt = 0.0
        self._llm = 0.0
        self._tts = 0.0
        self._ttfs = 0.0
        self._total = 0.0
        self._turn_count = 0
        self._history: deque[float] = deque(maxlen=20)

    def update_metrics(self, stt: float, llm: float, tts: float, ttfs: float, total: float):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._ttfs = ttfs
        self._total = total
        self._turn_count += 1
        self._history.append(ttfs)
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append(" \u25a0 Latency\n", style="bold #00ff88")
        t.append(f"  STT {self._stt:>5.2f}s", style="#00e5ff")
        t.append(f"  LLM {self._llm:>5.2f}s\n", style="#c084fc")
        t.append(f"  TTS {self._tts:>5.2f}s", style="#60a5fa")
        t.append(f"  Total {self._total:>5.2f}s\n", style="#c9d1d9")
        # TTFS highlight
        ttfs_color = "#00ff88" if self._ttfs < 1.0 else ("#ffb020" if self._ttfs < 2.0 else "#ff4444")
        t.append("  TTFS ", style="#8b949e")
        t.append(f"{self._ttfs:.2f}s", style=f"bold {ttfs_color}")

        # TTFS sparkline history
        if self._history:
            norm = [min(v / 5.0, 1.0) for v in self._history]
            spark = _sparkline(norm, width=20)
            avg = sum(self._history) / len(self._history)
            t.append(f"\n  {spark} ", style="#ffb020")
            t.append(f"avg {avg:.2f}s", style="dim")
        return t


COMMANDS = {
    "/help": "Show all available commands",
    "/reset": "Reset conversation history",
    "/lang ": "Switch language (en, vi, fr, ko, ...)",
    "/langs": "List all supported languages",
    "/say ": "TTS preview — speak text directly",
    "/mictest": "Record 3s + playback to test audio",
    "/model ": "Switch Whisper model (small/medium/large-v3-turbo)",
    "/voice ": "Switch TTS voice",
    "/voices": "List available TTS voices",
    "/mic ": "Switch microphone device",
    "/spk ": "Switch speaker device",
    "/devices": "List audio devices",
    "/export": "Export chat to markdown",
    "/mute": "Mute microphone",
    "/unmute": "Unmute microphone",
}


MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "chunkformer"]

KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "ff_siwis",
    "hf_alpha",
    "hm_omega",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jm_beta",
    "pf_dora",
    "pm_alex",
    "zf_xiaobei",
    "zf_xiaoni",
    "zm_yunjian",
    "vi-female",
    "vi-male",
]


def get_completions(value: str) -> list[tuple[str, str]]:
    """Return (completion_text, description) tuples for current input."""
    if not value.startswith("/"):
        return []

    # Check if we have a space after the command (entering arguments)
    has_space = " " in value
    if has_space:
        cmd, arg = value.split(maxsplit=1) if len(value.split(maxsplit=1)) > 1 else (value.rstrip(), "")
        cmd = cmd.lower()
    else:
        cmd = value.lower()
        arg = ""

    # Suggest argument values for known commands
    if cmd == "/lang" and has_space:
        return [
            (f"/lang {code}", name)
            for name, code in get_lang_options()
            if not arg or code.startswith(arg.lower()) or name.lower().startswith(arg.lower())
        ]
    if cmd == "/model" and has_space:
        return [(f"/model {m}", "") for m in MODEL_SIZES if not arg or m.startswith(arg.lower())]
    if cmd == "/voice" and has_space:
        return [(f"/voice {v}", "") for v in KOKORO_VOICES if not arg or v.startswith(arg.lower())]
    if cmd == "/mic" and has_space:
        return [
            (f"/mic {idx}", name)
            for name, idx in list_input_devices()
            if not arg or str(idx).startswith(arg) or arg.lower() in name.lower()
        ]
    if cmd == "/spk" and has_space:
        return [
            (f"/spk {idx}", name)
            for name, idx in list_output_devices()
            if not arg or str(idx).startswith(arg) or arg.lower() in name.lower()
        ]

    # Typing a command — suggest matching commands
    return [
        (c.rstrip(), desc)
        for c, desc in COMMANDS.items()
        if c.startswith(value.lower()) and c.rstrip() != value.lower()
    ]


class CommandSuggester(Suggester):
    """Inline ghost-text suggestion for commands."""

    async def get_suggestion(self, value: str) -> str | None:
        completions = get_completions(value)
        if completions:
            return completions[0][0]
        return None


class EdgeVoxApp(App):
    TITLE = "EdgeVox"
    SUB_TITLE = LOGO_TAGLINE
    CSS = CSS
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reset", "Reset Chat"),
        Binding("m", "toggle_mute", "Mute/Unmute"),
        Binding("ctrl+s", "export_chat", "Export Chat"),
        Binding("slash", "focus_command", "Command", show=False),
    ]

    def __init__(
        self,
        stt_model: str | None = None,
        stt_device: str | None = None,
        llm_model: str | None = None,
        tts_backend: str | None = None,
        voice: str | None = None,
        language: str = "en",
        wakeword: str | None = None,
        mic_device: int | None = None,
        spk_device: int | None = None,
        session_timeout: float = 30.0,
        ros2: bool = False,
    ):
        super().__init__()
        self._stt_model = stt_model
        self._stt_device = stt_device
        self._llm_model = llm_model
        self._tts_backend = tts_backend
        self._voice = voice
        self._language = language
        self._wakeword_name = wakeword
        self._session_timeout = session_timeout
        self._ros2_enabled = ros2

        # Restore saved device prefs if not explicitly provided
        saved = _load_device_prefs()
        if mic_device is None:
            mic_device = _resolve_saved_device(
                saved.get("mic"),
                list_input_devices(),
            )
        if spk_device is None:
            spk_device = _resolve_saved_device(
                saved.get("spk"),
                list_output_devices(),
            )
        self._mic_device = mic_device
        self._spk_device = spk_device
        player.set_device(spk_device)
        self._stt: STT | None = None
        self._llm: LLM | None = None
        self._tts: BaseTTS | None = None
        self._wakeword: WakeWordDetector | None = None
        self._recorder: AudioRecorder | None = None
        self._bridge = None  # ROS2 bridge
        self._muted = False
        self._activated = False
        self._session_last_activity: float = 0
        self._session_timer: threading.Timer | None = None
        self._processing = threading.Lock()
        self._interrupted = threading.Event()
        self._chat_log: list[tuple[str, str, str]] = []  # (timestamp, speaker, text)

    def compose(self) -> ComposeResult:
        mic_devices = list_input_devices()
        mic_options = [(name, idx) for name, idx in mic_devices]
        default_mic = self._mic_device if self._mic_device is not None else (mic_devices[0][1] if mic_devices else 0)

        spk_devices = list_output_devices()
        spk_options = [(name, idx) for name, idx in spk_devices]
        if self._spk_device is not None:
            default_spk = self._spk_device
        else:
            sys_default = _get_default_output_device()
            if sys_default is not None and any(idx == sys_default for _, idx in spk_devices):
                default_spk = sys_default
            else:
                default_spk = spk_devices[0][1] if spk_devices else 0

        lang_options = get_lang_options()

        yield Header(show_clock=True)
        yield StatusBar(id="status-bar")
        with Horizontal(id="main-area"):
            yield RichLog(id="chat-panel", wrap=True, markup=True, highlight=True)
            with Vertical(id="side-panel"), VerticalScroll(id="side-scroll"):
                with Vertical(classes="side-section"):
                    yield AudioLevel(id="audio-level")
                yield Rule(classes="side-separator")
                with Vertical(classes="side-section", id="model-info"):
                    yield ModelInfoPanel(id="model-info-panel")
                yield Rule(classes="side-separator")
                with Vertical(classes="side-section", id="gpu-monitor"):
                    yield GpuMonitor(id="gpu-monitor-panel")
                yield Rule(classes="side-separator")
                with Vertical(classes="side-section"):
                    yield MetricsPanel(id="metrics-box")
                yield Rule(classes="side-separator")
                with Vertical(id="settings-section", classes="side-section"):
                    yield Label(" \u25a0 Settings", id="settings-label")
                    yield Select(mic_options, value=default_mic, prompt="Mic", id="mic-select")
                    yield Select(spk_options, value=default_spk, prompt="Spk", id="spk-select")
                    yield Select(lang_options, value=self._language, prompt="Lang", id="lang-select")
        yield OptionList(id="completion-menu")
        with Horizontal(id="command-input"):
            yield Input(
                placeholder="Type / for commands, or text to chat",
                id="cmd-input",
                suggester=CommandSuggester(use_cache=False),
            )
        yield Footer()

    def on_mount(self) -> None:
        self._show_splash()
        self._load_models()

    def _show_splash(self) -> None:
        chat = self.query_one("#chat-panel", RichLog)
        for line in LOGO.strip().split("\n"):
            chat.write(Text(line, style="bold #00ff88"))
        chat.write(Text(f"  {LOGO_TAGLINE}  v{__version__}\n", style="#00e5ff"))

    def on_select_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, int | str):
            return
        if event.select.id == "mic-select":
            self._switch_mic(int(event.value))
        elif event.select.id == "spk-select":
            self._switch_speaker(int(event.value))
        elif event.select.id == "lang-select":
            self._switch_language(str(event.value))

    def _is_menu_visible(self) -> bool:
        return self.query_one("#completion-menu", OptionList).has_class("visible")

    _suppress_completion = False

    def _select_completion(self) -> None:
        """Apply the highlighted completion to the input and close menu."""
        menu = self.query_one("#completion-menu", OptionList)
        inp = self.query_one("#cmd-input", Input)
        idx = menu.highlighted
        if idx is not None:
            option = menu.get_option_at_index(idx)
            if option.id:
                self._suppress_completion = True
                inp.value = option.id
                inp.cursor_position = len(option.id)
        menu.clear_options()
        menu.remove_class("visible")
        inp.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update completion menu as user types."""
        if event.input.id != "cmd-input":
            return
        if self._suppress_completion:
            self._suppress_completion = False
            return
        menu = self.query_one("#completion-menu", OptionList)
        value = event.value

        completions = get_completions(value)
        menu.clear_options()
        if completions:
            for text, desc in completions[:10]:
                label = f"{text}  {desc}" if desc else text
                menu.add_option(Option(label, id=text))
            menu.highlighted = 0
            menu.add_class("visible")
        else:
            menu.remove_class("visible")

    def on_key(self, event) -> None:
        """Intercept keys when completion menu is visible."""
        if not self._is_menu_visible():
            return
        menu = self.query_one("#completion-menu", OptionList)

        if event.key == "up":
            event.prevent_default()
            event.stop()
            idx = menu.highlighted
            if idx is not None and idx > 0:
                menu.highlighted = idx - 1
            elif menu.option_count > 0:
                menu.highlighted = menu.option_count - 1

        elif event.key == "down":
            event.prevent_default()
            event.stop()
            idx = menu.highlighted
            if idx is not None and idx < menu.option_count - 1:
                menu.highlighted = idx + 1
            elif menu.option_count > 0:
                menu.highlighted = 0

        elif event.key == "tab":
            event.prevent_default()
            event.stop()
            self._select_completion()

        elif event.key == "enter":
            # If menu is visible and an item is highlighted, select it
            # instead of submitting the input
            if menu.highlighted is not None:
                event.prevent_default()
                event.stop()
                self._select_completion()

        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            menu.remove_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fill input with selected completion (mouse click)."""
        if event.option_list.id != "completion-menu":
            return
        inp = self.query_one("#cmd-input", Input)
        selected = event.option.id
        if selected:
            inp.value = selected
            inp.cursor_position = len(selected)
        event.option_list.remove_class("visible")
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "cmd-input":
            return
        # Hide completion menu
        self.query_one("#completion-menu", OptionList).remove_class("visible")
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._handle_text_chat(text)
        self.query_one("#chat-panel", RichLog).focus()

    def _handle_command(self, cmd: str) -> None:
        chat = self.query_one("#chat-panel", RichLog)
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower().lstrip("/")
        arg = parts[1] if len(parts) > 1 else ""

        if command == "reset":
            self.action_reset()
        elif command == "lang" and arg:
            self._switch_language(arg.strip())
        elif command == "voice" and arg:
            self._voice = arg.strip()
            self.call_from_thread(chat.write, Text(f"  Voice set to: {self._voice}", style="italic yellow"))
        elif command == "export":
            self.action_export_chat()
        elif command == "say" and arg:
            self._say_text(arg.strip())
        elif command == "mictest":
            self._mic_test()
        elif command == "model" and arg:
            self._switch_model(arg.strip())
        elif command == "voices":
            self._list_voices()
        elif command == "langs":
            self._list_langs()
        elif command == "mic" and arg:
            try:
                self._switch_mic(int(arg.strip()))
            except (ValueError, Exception) as e:
                chat.write(Text(f"  Invalid mic device: {e}", style="italic red"))
        elif command == "spk" and arg:
            try:
                self._switch_speaker(int(arg.strip()))
            except (ValueError, Exception) as e:
                chat.write(Text(f"  Invalid speaker device: {e}", style="italic red"))
        elif command == "devices":
            self._list_devices()
        elif command == "mute":
            if not self._muted:
                self.action_toggle_mute()
        elif command == "unmute":
            if self._muted:
                self.action_toggle_mute()
        elif command == "help":
            parts = [("  Commands:\n", "bold #00ff88")]
            for cmd, desc in COMMANDS.items():
                parts += [("  ", ""), (f"{cmd:<14}", "#00e5ff"), (f"{desc}\n", "#8b949e")]
            parts += [("\n  Type text without / to chat with the bot.\n", "dim")]
            chat.write(Text.assemble(*parts))
        else:
            chat.write(Text(f"  Unknown command: /{command}. Type /help", style="italic red"))

    @work(thread=True)
    def _handle_text_chat(self, text: str):
        """Handle typed text as a chat message — send through LLM + TTS."""
        if not self._llm or not self._tts:
            chat = self.query_one("#chat-panel", RichLog)
            self.call_from_thread(chat.write, Text("  Models not loaded yet.", style="italic red"))
            return
        chat = self.query_one("#chat-panel", RichLog)
        status = self.query_one("#status-bar", StatusBar)
        metrics_panel = self.query_one("#metrics-box", MetricsPanel)

        self._interrupted.clear()
        t_start = time.perf_counter()

        # Show user message
        ts = datetime.now().strftime("%H:%M:%S")
        self._chat_log.append((ts, "You", text))
        self.call_from_thread(
            chat.write,
            Text.assemble(
                ("\u2500" * 60 + "\n", "#1e3a2e"),
                (" \u25b6 You ", "bold #00ff88"),
                (f" {ts} ", "dim"),
                (" typed \n", "#1e3a2e"),
                (f"   {text}\n", "#c9d1d9"),
            ),
        )

        # LLM stream -> overlapped TTS
        self.call_from_thread(setattr, status, "state", BotState.THINKING)
        self.call_from_thread(chat.write, Text("  \u2026 thinking", style="dim italic #c084fc"))
        result = self._stream_and_speak(self._llm.chat_stream(text), chat, status, t_start)

        t_total = time.perf_counter() - t_start
        t_llm = t_total - result["tts_total"]

        ts = datetime.now().strftime("%H:%M:%S")
        full_reply = " ".join(result["parts"])
        self._chat_log.append((ts, "Bot", full_reply))

        self.call_from_thread(
            chat.write,
            Text(f"   TTFS {result['ttfs']:.2f}s | Total {t_total:.2f}s", style="dim"),
        )
        self.call_from_thread(
            metrics_panel.update_metrics,
            0.0,
            t_llm,
            result["tts_total"],
            result["ttfs"],
            t_total,
        )
        self.call_from_thread(setattr, status, "state", BotState.LISTENING)
        with contextlib.suppress(Exception):
            self.call_from_thread(self.query_one("#cmd-input", Input).focus)

    def _stream_and_speak(self, token_stream, chat, status, t_start: float) -> dict:
        """Stream LLM tokens → split sentences → overlapped TTS → play.

        Synthesizes the next sentence in a background thread while the current
        one plays, eliminating the gap between sentences.

        Returns dict with: parts, tts_total, ttfs, interrupted.
        """
        from concurrent.futures import ThreadPoolExecutor

        sentence_stream = stream_sentences(token_stream)
        tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

        full_reply_parts = []
        t_tts_total = 0.0
        t_first_speech = None
        first_sentence = True
        interrupted = False

        def _synth(s):
            t = time.perf_counter()
            a = self._tts.synthesize(s)
            return a, time.perf_counter() - t

        def _show(s):
            nonlocal first_sentence
            if first_sentence:
                self.call_from_thread(
                    chat.write,
                    Text.assemble(
                        (" \u25c0 Bot ", "bold #00e5ff"),
                        (f"\n   {s}", "#c9d1d9"),
                    ),
                )
                first_sentence = False
            else:
                self.call_from_thread(chat.write, Text(f"   {s}", style="#c9d1d9"))

        # Collect sentences into a buffer so we can look ahead
        sentence_buf = []
        future = None

        for sentence in sentence_stream:
            if self._interrupted.is_set():
                interrupted = True
                break
            sentence_buf.append(sentence)

            # Start synthesizing current sentence
            full_reply_parts.append(sentence)
            self.call_from_thread(setattr, status, "state", BotState.SPEAKING)
            _show(sentence)

            if self._interrupted.is_set():
                interrupted = True
                break

            # Use pre-synthesized audio if available, else synth now
            if future is not None:
                audio_out, t_elapsed = future.result()
                future = None
            else:
                audio_out, t_elapsed = _synth(sentence)
            t_tts_total += t_elapsed

            if t_first_speech is None:
                t_first_speech = time.perf_counter() - t_start

            if self._interrupted.is_set():
                interrupted = True
                break

            # Peek ahead: try to get the next sentence and start synthesizing it
            # while we play the current audio
            next_sent = None
            try:
                next_sent = next(sentence_stream)
                if not self._interrupted.is_set():
                    future = tts_executor.submit(_synth, next_sent)
            except StopIteration:
                pass

            # Play current (next is synthesizing in background)
            completed = play_audio(audio_out, sample_rate=self._tts.sample_rate)
            if not completed or self._interrupted.is_set():
                interrupted = True
                if future:
                    future.cancel()
                    future = None
                break

            # If we peeked a next sentence, process it on next iteration
            # by pushing it back (we re-inject it into the loop)
            if next_sent is not None:
                if self._interrupted.is_set():
                    interrupted = True
                    break

                full_reply_parts.append(next_sent)
                _show(next_sent)

                if self._interrupted.is_set():
                    interrupted = True
                    break

                if future is not None:
                    audio_out, t_elapsed = future.result()
                    future = None
                else:
                    audio_out, t_elapsed = _synth(next_sent)
                t_tts_total += t_elapsed

                if t_first_speech is None:
                    t_first_speech = time.perf_counter() - t_start

                if self._interrupted.is_set():
                    interrupted = True
                    break

                completed = play_audio(audio_out, sample_rate=self._tts.sample_rate)
                if not completed or self._interrupted.is_set():
                    interrupted = True
                    break

        tts_executor.shutdown(wait=False)

        # Drain remaining LLM tokens to keep history consistent
        if interrupted:
            for _ in sentence_stream:
                pass

        return {
            "parts": full_reply_parts,
            "tts_total": t_tts_total,
            "ttfs": t_first_speech or 0.0,
            "interrupted": interrupted,
        }

    def action_focus_command(self) -> None:
        inp = self.query_one("#cmd-input", Input)
        inp.focus()
        inp.value = "/"

    def _sync_select(self, select_id: str, value):
        """Update a Select widget's value without triggering on_select_changed."""
        try:
            sel = self.query_one(f"#{select_id}", Select)
            self._suppress_completion = True
            sel.value = value
        except Exception:
            pass

    def _switch_mic(self, device_idx: int):
        chat = self.query_one("#chat-panel", RichLog)
        self._mic_device = device_idx
        _save_device_prefs(mic=device_idx)
        device_info = sd.query_devices(device_idx)
        chat.write(Text(f"  Mic: {device_info['name']}", style="italic yellow"))
        self._sync_select("mic-select", device_idx)
        if self._recorder:
            self._recorder.stop()
            self._recorder = AudioRecorder(
                on_speech=self._on_speech,
                on_interrupt=self._on_interrupt,
                on_level=self._on_level,
                device=device_idx,
            )
            player.link_recorder(self._recorder)
            self._recorder.start()

    def _switch_speaker(self, device_idx: int):
        chat = self.query_one("#chat-panel", RichLog)
        try:
            self._spk_device = device_idx
            _save_device_prefs(spk=device_idx)
            player.set_device(device_idx)
            device_info = sd.query_devices(device_idx)
            chat.write(Text(f"  Speaker: {device_info['name']}", style="italic yellow"))
            self._sync_select("spk-select", device_idx)
        except Exception as e:
            chat.write(Text(f"  Speaker switch failed: {e}", style="bold red"))

    @work(thread=True)
    def _switch_language(self, lang: str):
        if lang == self._language:
            return
        old_lang = self._language
        cfg = get_lang(lang)
        chat = self.query_one("#chat-panel", RichLog)
        self.call_from_thread(chat.write, Text(f"  Switching to {cfg.name}...", style="italic yellow"))
        self._language = lang

        if needs_stt_reload(old_lang, lang):
            self.call_from_thread(chat.write, Text(f"  Reloading STT for {cfg.name}...", style="dim"))
            self._stt = create_stt(language=lang, device=self._stt_device)

        self._tts = create_tts(language=lang, voice=self._voice, backend=self._tts_backend)
        self._tts.synthesize(cfg.test_phrase)

        # Update LLM system prompt for the new language
        if self._llm:
            self._llm.set_language(lang)

        # Update model info panel
        model_info = self.query_one("#model-info-panel", ModelInfoPanel)
        tts_name = "Kokoro-82M" if cfg.tts_backend == "kokoro" else "Piper"
        self.call_from_thread(
            model_info.set_info,
            self._stt._model_size,
            self._stt._device,
            "Gemma 4 E2B IT Q4_K_M",
            _get_gpu_info()["backend"],
            tts_name,
        )
        self.call_from_thread(self._sync_select, "lang-select", lang)
        self.call_from_thread(setattr, self.query_one("#status-bar", StatusBar), "language", lang)
        self.call_from_thread(chat.write, Text(f"  Language: {cfg.name} ready!", style="italic #00ff88"))

    @work(thread=True)
    def _say_text(self, text: str):
        """TTS preview — synthesize and play text without going through LLM."""
        chat = self.query_one("#chat-panel", RichLog)
        status = self.query_one("#status-bar", StatusBar)
        if not self._tts:
            self.call_from_thread(chat.write, Text("  TTS not loaded yet.", style="italic red"))
            return
        self.call_from_thread(setattr, status, "state", BotState.SPEAKING)
        cfg = get_lang(self._language)
        self.call_from_thread(
            chat.write,
            Text.assemble(
                ("  TTS ", "bold blue"),
                (f"[{cfg.name}] ", "dim"),
                (text, ""),
            ),
        )
        t0 = time.perf_counter()
        audio_out = self._tts.synthesize(text)
        elapsed = time.perf_counter() - t0
        duration = len(audio_out) / self._tts.sample_rate if len(audio_out) > 0 else 0
        self.call_from_thread(chat.write, Text(f"  TTS: {elapsed:.2f}s synth -> {duration:.1f}s audio", style="dim"))
        play_audio(audio_out, sample_rate=self._tts.sample_rate)
        self.call_from_thread(setattr, status, "state", BotState.LISTENING)

    @work(thread=True)
    def _mic_test(self):
        """Record 3 seconds of audio and play it back to test mic/speaker."""
        chat = self.query_one("#chat-panel", RichLog)
        status = self.query_one("#status-bar", StatusBar)
        self.call_from_thread(chat.write, Text("  Mic test: recording 3 seconds...", style="bold yellow"))
        self.call_from_thread(setattr, status, "state", BotState.LISTENING)

        sr = MIC_SAMPLE_RATE
        duration = 3.0
        try:
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32", device=self._mic_device)
            sd.wait()
            audio = audio.flatten()
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio**2)))
            self.call_from_thread(
                chat.write, Text(f"  Recorded {duration}s — peak: {peak:.3f}, RMS: {rms:.4f}", style="dim")
            )
            self.call_from_thread(chat.write, Text("  Playing back...", style="bold yellow"))
            play_audio(audio, sample_rate=sr)
            self.call_from_thread(chat.write, Text("  Mic test complete.", style="italic green"))
        except Exception as e:
            self.call_from_thread(chat.write, Text(f"  Mic test failed: {e}", style="bold red"))
        self.call_from_thread(setattr, status, "state", BotState.LISTENING)

    @work(thread=True)
    def _switch_model(self, model_size: str):
        """Hot-swap the Whisper STT model size."""
        chat = self.query_one("#chat-panel", RichLog)
        valid = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "chunkformer"]
        if model_size not in valid:
            self.call_from_thread(chat.write, Text(f"  Valid models: {', '.join(valid)}", style="italic yellow"))
            return
        self.call_from_thread(chat.write, Text(f"  Loading STT model: {model_size}...", style="italic yellow"))
        t0 = time.perf_counter()
        self._stt = create_stt(language=self._language, model_size=model_size, device=self._stt_device)
        elapsed = time.perf_counter() - t0
        model_info = self.query_one("#model-info-panel", ModelInfoPanel)
        cfg = get_lang(self._language)
        tts_name = "Kokoro-82M" if cfg.tts_backend == "kokoro" else "Piper"
        self.call_from_thread(
            model_info.set_info,
            self._stt._model_size,
            self._stt._device,
            "Gemma 4 E2B IT Q4_K_M",
            _get_gpu_info()["backend"],
            tts_name,
        )
        self.call_from_thread(
            chat.write, Text(f"  STT model: {model_size} loaded in {elapsed:.1f}s", style="italic green")
        )

    def _list_voices(self):
        """List available TTS voices for current language."""
        chat = self.query_one("#chat-panel", RichLog)
        cfg = get_lang(self._language)
        parts = [("  Available voices", "bold"), (f" ({cfg.name}):\n", "dim")]
        if cfg.tts_backend == "kokoro":
            kokoro_voices = [
                "af_heart",
                "af_bella",
                "af_nicole",
                "af_sarah",
                "af_sky",
                "am_adam",
                "am_michael",
                "bf_emma",
                "bf_isabella",
                "bm_george",
                "bm_lewis",
                "ef_dora",
                "em_alex",
                "ff_siwis",
                "hf_alpha",
                "hm_omega",
                "if_sara",
                "im_nicola",
                "jf_alpha",
                "jm_beta",
                "pf_dora",
                "pm_alex",
                "zf_xiaobei",
                "zf_xiaoni",
                "zm_yunjian",
            ]
            prefix = cfg.kokoro_lang
            matching = [v for v in kokoro_voices if v.startswith(prefix)]
            others = [v for v in kokoro_voices if not v.startswith(prefix)]
            for v in matching:
                marker = " *" if v == cfg.default_voice else ""
                parts += [("  ", ""), (v, "bold cyan"), (f"{marker}\n", "yellow")]
            if others:
                parts += [("  Other: ", "dim"), (", ".join(others) + "\n", "dim")]
        else:
            from edgevox.tts import PIPER_VI_VOICES

            for name in PIPER_VI_VOICES:
                marker = " *" if name == cfg.default_voice else ""
                parts += [("  ", ""), (name, "bold cyan"), (f"{marker}\n", "yellow")]
        chat.write(Text.assemble(*parts))

    def _list_langs(self):
        """List all supported languages with their backends."""
        chat = self.query_one("#chat-panel", RichLog)
        parts = [("  Supported languages:\n", "bold")]
        for code, cfg in sorted(LANGUAGES.items(), key=lambda x: x[1].name):
            current = " *" if code == self._language else ""
            stt_label = cfg.stt_backend
            tts_label = cfg.tts_backend
            if cfg.tts_backend == "kokoro" and cfg.kokoro_lang == "a" and code != "en":
                tts_label = "kokoro (en fallback)"
            parts += [
                ("  ", ""),
                (f"{code:<6}", "bold cyan"),
                (f"{cfg.name:<14}", ""),
                (f"STT:{stt_label:<12}", "dim"),
                (f"TTS:{tts_label}", "dim"),
                (f"{current}\n", "yellow"),
            ]
        chat.write(Text.assemble(*parts))

    def _list_devices(self):
        """List all audio input and output devices."""
        chat = self.query_one("#chat-panel", RichLog)
        parts = [("  Input devices (microphones):\n", "bold #00ff88")]
        for name, idx in list_input_devices():
            current = " *" if idx == self._mic_device else ""
            parts += [("  ", ""), (f"{idx:<4}", "#00e5ff"), (f"{name}{current}\n", "#c9d1d9")]
        parts += [("\n  Output devices (speakers):\n", "bold #00ff88")]
        for name, idx in list_output_devices():
            current = " *" if idx == self._spk_device else ""
            parts += [("  ", ""), (f"{idx:<4}", "#00e5ff"), (f"{name}{current}\n", "#c9d1d9")]
        chat.write(Text.assemble(*parts))

    @work(thread=True)
    def _load_models(self) -> None:
        chat = self.query_one("#chat-panel", RichLog)
        status = self.query_one("#status-bar", StatusBar)
        model_info = self.query_one("#model-info-panel", ModelInfoPanel)

        t0 = time.perf_counter()

        # ROS2 bridge
        if self._ros2_enabled:
            self.call_from_thread(chat.write, Text("  Initializing ROS2 bridge...", style="dim"))
            from edgevox.ros2_bridge import NullBridge, create_bridge

            self._bridge = create_bridge(enabled=True)
            if not isinstance(self._bridge, NullBridge):
                self._bridge.set_tts_callback(self._on_ros2_tts)
                self._bridge.set_command_callback(self._on_ros2_command)
                self.call_from_thread(chat.write, Text("  ROS2 bridge active!", style="green"))
            else:
                self.call_from_thread(chat.write, Text("  ROS2 not available (rclpy not found)", style="yellow"))
                self._bridge = None

        self.call_from_thread(chat.write, Text("  [1/3] Loading STT...", style="dim"))
        t_load = time.perf_counter()
        self._stt = create_stt(
            language=self._language,
            model_size=self._stt_model,
            device=self._stt_device,
        )
        stt_name = self._stt._model_size
        stt_device = self._stt._device
        self.call_from_thread(
            chat.write,
            Text(f"        STT: {stt_name} ({stt_device}) {time.perf_counter() - t_load:.1f}s", style="#8b949e"),
        )

        self.call_from_thread(chat.write, Text("  [2/3] Loading LLM...", style="dim"))
        t_load = time.perf_counter()
        self._llm = LLM(model_path=self._llm_model, language=self._language)
        llm_name = Path(self._llm._llm.model_path).stem
        gpu_info = _get_gpu_info()
        llm_device = gpu_info["backend"]
        self.call_from_thread(
            chat.write, Text(f"        LLM: {llm_device} {time.perf_counter() - t_load:.1f}s", style="#8b949e")
        )

        self.call_from_thread(chat.write, Text("  [3/3] Loading TTS + warmup...", style="dim"))
        t_load = time.perf_counter()
        lang_cfg = get_lang(self._language)
        self._tts = create_tts(language=self._language, voice=self._voice, backend=self._tts_backend)
        tts_name = "Kokoro-82M" if lang_cfg.tts_backend == "kokoro" else "Piper"
        self._tts.synthesize(lang_cfg.test_phrase)
        self.call_from_thread(
            chat.write, Text(f"        TTS: {tts_name} {time.perf_counter() - t_load:.1f}s", style="#8b949e")
        )

        # Update model info panel
        self.call_from_thread(model_info.set_info, stt_name, stt_device, llm_name, llm_device, tts_name)

        # Wakeword
        if self._wakeword_name:
            self.call_from_thread(chat.write, Text(f"  Loading wakeword: {self._wakeword_name}...", style="dim"))
            try:
                self._wakeword = WakeWordDetector(wake_words=[self._wakeword_name], threshold=0.15)
                self._activated = False
            except Exception as e:
                self.call_from_thread(chat.write, Text(f"  Wakeword failed: {e}", style="yellow"))
                self._wakeword = None
                self._activated = True
        else:
            self._activated = True

        elapsed = time.perf_counter() - t0
        lang_label = get_lang(self._language).name
        ros2_label = " | ROS2" if self._ros2_enabled and self._bridge else ""
        self.call_from_thread(
            chat.write,
            Text.assemble(
                (f"\n  \u2714 Ready in {elapsed:.1f}s", "bold #00ff88"),
                (f" ({lang_label}{ros2_label})", "#8b949e"),
                ("\n  Speak naturally, type a message, or /help for commands.\n", "dim"),
            ),
        )
        self.call_from_thread(setattr, status, "language", self._language)
        self.call_from_thread(
            setattr,
            status,
            "state",
            BotState.WAITING_WAKEWORD if self._wakeword else BotState.LISTENING,
        )

        # Start recorder
        self._ww_buffer = np.array([], dtype=np.float32)
        self._recorder = AudioRecorder(
            on_speech=self._on_speech,
            on_interrupt=self._on_interrupt,
            on_level=self._on_level,
            on_audio_frame=self._on_audio_frame if self._wakeword else None,
            device=self._mic_device,
        )
        player.link_recorder(self._recorder)
        self._recorder.start()

    # --- ROS2 callbacks ---

    def _on_ros2_tts(self, text: str):
        """Handle incoming TTS request from ROS2."""
        if self._tts:
            audio_out = self._tts.synthesize(text)
            play_audio(audio_out, sample_rate=self._tts.sample_rate)

    def _on_ros2_command(self, command: str):
        """Handle command from ROS2."""
        self._handle_command(command)

    # --- Audio callbacks ---

    def _on_level(self, level: float):
        try:
            w = self.query_one("#audio-level", AudioLevel)
            self.call_from_thread(setattr, w, "level", level)
        except Exception:
            pass
        if self._bridge:
            self._bridge.publish_audio_level(level)

    def _on_audio_frame(self, chunk: np.ndarray):
        if not self._wakeword or self._activated:
            return
        try:
            self._ww_buffer = np.concatenate([self._ww_buffer, chunk])
            while len(self._ww_buffer) >= 1280:
                frame = self._ww_buffer[:1280]
                self._ww_buffer = self._ww_buffer[1280:]
                detected = self._wakeword.detect(frame)
                if detected:
                    log.info(f"Wake word detected: {detected}")
                    self._start_session()
                    return
        except Exception:
            log.exception("Error in wakeword detection")

    def _start_session(self):
        was_sleeping = not self._activated
        self._activated = True
        self._session_last_activity = time.perf_counter()
        if self._session_timer:
            self._session_timer.cancel()
        self._session_timer = threading.Timer(self._session_timeout, self._end_session)
        self._session_timer.daemon = True
        self._session_timer.start()
        if was_sleeping:
            log.info(f"Session started (timeout={self._session_timeout}s)")
            try:
                status = self.query_one("#status-bar", StatusBar)
                self.call_from_thread(setattr, status, "state", BotState.LISTENING)
                chat = self.query_one("#chat-panel", RichLog)
                self.call_from_thread(
                    chat.write,
                    Text(f"  Session active ({int(self._session_timeout)}s timeout)", style="bold green"),
                )
            except Exception:
                pass

    def _extend_session(self):
        if not self._wakeword:
            return
        self._session_last_activity = time.perf_counter()
        if self._session_timer:
            self._session_timer.cancel()
        self._session_timer = threading.Timer(self._session_timeout, self._end_session)
        self._session_timer.daemon = True
        self._session_timer.start()

    def _end_session(self):
        if not self._activated:
            return
        self._activated = False
        self._ww_buffer = np.array([], dtype=np.float32)
        log.info("Session timed out, going to sleep")
        try:
            status = self.query_one("#status-bar", StatusBar)
            self.call_from_thread(setattr, status, "state", BotState.WAITING_WAKEWORD)
            chat = self.query_one("#chat-panel", RichLog)
            self.call_from_thread(
                chat.write,
                Text(f"  Session ended. Say '{self._wakeword_name}' to wake up.", style="dim italic"),
            )
        except Exception:
            pass

    def _on_interrupt(self):
        self._interrupted.set()
        player.interrupt()
        chat = self.query_one("#chat-panel", RichLog)
        self.call_from_thread(chat.write, Text("   \u26a1 interrupted", style="italic #ffb020"))
        status = self.query_one("#status-bar", StatusBar)
        self.call_from_thread(setattr, status, "state", BotState.INTERRUPTED)
        if self._bridge:
            self._bridge.publish_state("interrupted")

    def _on_speech(self, audio: np.ndarray):
        if self._muted:
            return
        if not self._activated and self._wakeword:
            return
        if self._wakeword:
            self._extend_session()
        if not self._processing.acquire(blocking=False):
            return
        try:
            self._process_streaming(audio)
        except Exception as e:
            log.exception("Error in voice pipeline")
            try:
                chat = self.query_one("#chat-panel", RichLog)
                self.call_from_thread(
                    chat.write,
                    Text(f"  \u2718 Pipeline error: {type(e).__name__}: {e}", style="bold red"),
                )
            except Exception:
                pass
        finally:
            self._processing.release()
            if self._wakeword and self._activated:
                self._extend_session()

    def _process_streaming(self, audio: np.ndarray):
        chat = self.query_one("#chat-panel", RichLog)
        status = self.query_one("#status-bar", StatusBar)
        metrics_panel = self.query_one("#metrics-box", MetricsPanel)

        self._interrupted.clear()
        duration = len(audio) / MIC_SAMPLE_RATE
        t_start = time.perf_counter()

        # STT
        self.call_from_thread(setattr, status, "state", BotState.TRANSCRIBING)
        if self._bridge:
            self._bridge.publish_state("transcribing")
        t0 = time.perf_counter()
        text = self._stt.transcribe(audio, language=self._language)
        t_stt = time.perf_counter() - t0

        if not text or text.isspace():
            self.call_from_thread(setattr, status, "state", BotState.LISTENING)
            if self._bridge:
                self._bridge.publish_state("listening")
            return

        ts = datetime.now().strftime("%H:%M:%S")
        self._chat_log.append((ts, "You", text))
        # User message — green accent with speech stats
        self.call_from_thread(
            chat.write,
            Text.assemble(
                ("\u2500" * 60 + "\n", "#1e3a2e"),
                (" \u25b6 You ", "bold #00ff88"),
                (f"{ts}", "dim"),
                (f"  {duration:.1f}s speech \u2192 STT {t_stt:.2f}s\n", "#1e3a2e"),
                (f"   {text}\n", "#c9d1d9"),
            ),
        )
        if self._bridge:
            self._bridge.publish_transcription(text)

        # LLM stream -> overlapped sentence TTS
        self.call_from_thread(setattr, status, "state", BotState.THINKING)
        if self._bridge:
            self._bridge.publish_state("thinking")
        self.call_from_thread(chat.write, Text("  \u2026 thinking", style="dim italic #c084fc"))

        result = self._stream_and_speak(self._llm.chat_stream(text), chat, status, t_start)
        t_tts_total = result["tts_total"]
        t_first_speech = result["ttfs"]
        full_reply_parts = result["parts"]

        t_total = time.perf_counter() - t_start
        t_llm = t_total - t_stt - t_tts_total

        full_reply = " ".join(full_reply_parts)
        ts = datetime.now().strftime("%H:%M:%S")
        self._chat_log.append((ts, "Bot", full_reply))
        if self._bridge:
            self._bridge.publish_response(full_reply)
            self._bridge.publish_metrics(
                {
                    "stt": round(t_stt, 3),
                    "llm": round(t_llm, 3),
                    "tts": round(t_tts_total, 3),
                    "ttfs": round(t_first_speech or 0, 3),
                    "total": round(t_total, 3),
                }
            )

        # Latency summary after bot response
        self.call_from_thread(
            chat.write,
            Text(f"   TTFS {t_first_speech or 0:.2f}s | Total {t_total:.2f}s", style="dim"),
        )
        self.call_from_thread(
            metrics_panel.update_metrics,
            t_stt,
            t_llm,
            t_tts_total,
            t_first_speech or 0,
            t_total,
        )
        self.call_from_thread(setattr, status, "state", BotState.LISTENING)
        if self._bridge:
            self._bridge.publish_state("listening")
        # Auto-focus input for quick follow-up
        with contextlib.suppress(Exception):
            self.call_from_thread(self.query_one("#cmd-input", Input).focus)

    def action_reset(self) -> None:
        if self._llm:
            self._llm.reset()
        chat = self.query_one("#chat-panel", RichLog)
        chat.clear()
        self._chat_log.clear()
        self._show_splash()
        lang_label = get_lang(self._language).name
        chat.write(
            Text.assemble(
                ("  \u2714 Chat reset", "italic #00ff88"),
                (f" \u2022 {lang_label}", "#8b949e"),
                (" \u2022 Speak or type to start.\n", "dim"),
            )
        )
        self.query_one("#cmd-input", Input).focus()

    def action_toggle_mute(self) -> None:
        self._muted = not self._muted
        status = self.query_one("#status-bar", StatusBar)
        status.muted = self._muted
        chat = self.query_one("#chat-panel", RichLog)
        if self._muted:
            chat.write(Text("  \U0001f507 Microphone muted", style="italic #ff4444"))
        else:
            chat.write(Text("  \U0001f50a Microphone unmuted", style="italic #00ff88"))
            status.state = BotState.LISTENING

    def action_export_chat(self) -> None:
        if not self._chat_log:
            chat = self.query_one("#chat-panel", RichLog)
            chat.write(Text("  No conversation to export.", style="italic yellow"))
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = Path.home() / f"edgevox_chat_{ts}.md"
        lines = [f"# EdgeVox Conversation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
        for timestamp, speaker, text in self._chat_log:
            lines.append(f"**{speaker}** ({timestamp}):\n{text}\n\n")
        export_path.write_text("".join(lines))
        chat = self.query_one("#chat-panel", RichLog)
        chat.write(Text(f"  Exported to {export_path}", style="italic green"))

    def on_unmount(self) -> None:
        if self._session_timer:
            self._session_timer.cancel()
        if self._recorder:
            self._recorder.stop()
        player.shutdown()
        if self._bridge:
            self._bridge.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="EdgeVox — Sub-second local voice AI for robots and edge devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  edgevox                                          # Start with defaults (auto-detect all)
  edgevox --stt large-v3-turbo --llm /path/to.gguf # Custom STT + local LLM
  edgevox --llm hf:bartowski/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf
  edgevox --tts piper --voice vi-female            # Force Piper TTS backend
  edgevox --wakeword "hey jarvis" --ros2           # Wake word + ROS2
  edgevox --language vi                            # Vietnamese mode
""",
    )
    parser.add_argument(
        "--stt",
        type=str,
        default=None,
        help="STT model: tiny, base, small, medium, large-v3-turbo, or chunkformer (auto-detected)",
    )
    parser.add_argument("--stt-device", type=str, default=None, help="STT device: cuda, cpu (auto-detected)")
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model: local GGUF path or hf:repo/name:file.gguf (default: Gemma 4 E2B Q4_K_M)",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=None,
        choices=["kokoro", "piper"],
        help="TTS backend: kokoro or piper (auto from language)",
    )
    parser.add_argument("--voice", type=str, default=None, help="TTS voice name (default: per language)")
    parser.add_argument("--language", type=str, default="en", help="Language: en, vi, fr, es, etc.")
    parser.add_argument("--wakeword", type=str, default=None, help="Wake word (e.g., 'hey jarvis')")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device index")
    parser.add_argument("--spk", type=int, default=None, help="Speaker device index")
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=30.0,
        help="Seconds of silence before wakeword session ends (default: 30)",
    )
    parser.add_argument("--ros2", action="store_true", help="Enable ROS2 bridge (publishes to /edgevox/* topics)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        filename="/tmp/edgevox.log",
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = EdgeVoxApp(
        stt_model=args.stt,
        stt_device=args.stt_device,
        llm_model=args.llm,
        tts_backend=args.tts,
        voice=args.voice,
        language=args.language,
        wakeword=args.wakeword,
        mic_device=args.mic,
        spk_device=args.spk,
        session_timeout=args.session_timeout,
        ros2=args.ros2,
    )
    app.run()


if __name__ == "__main__":
    main()
