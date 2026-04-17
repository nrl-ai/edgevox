"""xLAM / JSON-array tool-call detector.

Salesforce xLAM-2 models emit tool calls as a top-level JSON array with
no wrapper tokens::

    [{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}]

Some pythonic fine-tunes additionally wrap the array in a Markdown code
fence::

    ```[{"name": "...", "arguments": {...}}]```

This detector matches both by trying to decode the first ``[``-delimited
block as JSON (or, failing that, as a Python literal via ``ast.literal_eval``
— which covers the frequent case of unquoted keys / single-quoted strings
emitted by small models).
"""

from __future__ import annotations

import ast
import json
import logging
import re

from edgevox.llm.tool_parsers._types import Tool
from edgevox.llm.tool_parsers.base import BaseFormatDetector
from edgevox.llm.tool_parsers.core_types import StreamingParseResult, StructureInfo, _GetInfoFunc

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)


def _load_relaxed(block: str) -> object | None:
    """Try ``json.loads`` then ``ast.literal_eval``. Return ``None`` on failure."""
    try:
        return json.loads(block)
    except Exception:
        pass
    try:
        return ast.literal_eval(block)
    except Exception:
        return None


def _find_array_slice(text: str) -> str | None:
    """Return the substring from the first ``[`` to its matching ``]``."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class XLAMDetector(BaseFormatDetector):
    """Detector for xLAM and similar ``[{"name": …, "arguments": …}]`` emitters."""

    def __init__(self):
        super().__init__()
        self.bot_token = "["
        self.eot_token = "]"

    def has_tool_call(self, text: str) -> bool:
        stripped = text.lstrip()
        if stripped.startswith("["):
            return True
        # Also allow a Markdown code fence wrapper.
        match = _FENCE_RE.search(text)
        return bool(match and match.group(1).lstrip().startswith("["))

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        # Prefer the fenced block if present (Hammer-style).
        fence = _FENCE_RE.search(text)
        candidate = fence.group(1).strip() if fence else text.strip()

        block = _find_array_slice(candidate)
        if block is None:
            return StreamingParseResult(normal_text=text, calls=[])

        payload = _load_relaxed(block)
        if not isinstance(payload, list):
            return StreamingParseResult(normal_text=text, calls=[])

        calls = self.parse_base_json(payload, tools)
        # Any prose before / after the array is "normal text".
        trailing = candidate.replace(block, "").strip()
        return StreamingParseResult(normal_text=trailing, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[{"name":"' + name + '","arguments":',
            end="}]",
            trigger="[{",
        )
