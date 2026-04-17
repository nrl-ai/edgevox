# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang python/sglang/srt/function_call/core_types.py
# (Apache-2.0). Unchanged except for this header and the final ruff-friendly
# ordering.
"""Core types shared by all tool-call detectors."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: str | None = None
    parameters: str  # JSON string


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: list[ToolCallItem] = []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


# Helper alias of function: takes a name string and returns a StructureInfo.
_GetInfoFunc = Callable[[str], StructureInfo]
