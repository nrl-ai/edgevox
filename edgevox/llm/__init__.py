"""LLM backends and agent primitives."""

from edgevox.llm.llamacpp import LLM
from edgevox.llm.models import (
    DEFAULT_PRESET,
    PRESETS,
    ModelPreset,
    download_preset,
    list_presets,
    resolve_preset,
)
from edgevox.llm.tools import (
    Tool,
    ToolCallResult,
    ToolRegistry,
    load_entry_point_tools,
    tool,
)

__all__ = [
    "DEFAULT_PRESET",
    "LLM",
    "PRESETS",
    "ModelPreset",
    "Tool",
    "ToolCallResult",
    "ToolRegistry",
    "download_preset",
    "list_presets",
    "load_entry_point_tools",
    "resolve_preset",
    "tool",
]
