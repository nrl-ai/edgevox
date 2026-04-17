"""Granite tool-call detector.

IBM Granite 4.0 Nano models emit tool calls with the same ``<tool_call>``
wrapper as Hermes / Qwen, but with **Python-dict syntax** (unquoted keys,
possible single-quoted strings) instead of strict JSON::

    <tool_call>\n{name: get_time, arguments: {"timezone": "Asia/Tokyo"}}

The Hermes detector's ``json.loads`` fails on this. We fall back to
``ast.literal_eval`` after a light key-quoting pass.
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

_TAG_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)(?:</tool_call>|$)", re.DOTALL)
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")
# Granite shape: ``name: <ident>, arguments: {json}`` with bare-identifier
# values on ``name`` that neither JSON nor literal_eval can handle.
_NAME_ARGS_RE = re.compile(
    r"\bname\s*:\s*[\"']?(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)[\"']?"
    r".*?\barguments\s*:\s*(?P<args>\{.*?\})(?=\s*[,}]|\s*$)",
    re.DOTALL,
)
# Granite (under a strong tool-use system prompt) sometimes emits a pythonic
# call *inside* the ``<tool_call>`` block: ``get_time(timezone="Asia/Tokyo")``.
_PYTHONIC_CALL_RE = re.compile(
    r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<body>[^()]*)\)",
)
_PYTHONIC_KV_RE = re.compile(
    r'(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:"(?P<s>[^"]*)"|\'(?P<s2>[^\']*)\'|'
    r"(?P<n>-?\d+(?:\.\d+)?)|(?P<b>True|False|true|false))",
)


def _parse_pythonic_body(name_args: str) -> dict[str, object] | None:
    """Parse ``k="v", k2=1, k3=true`` from a python-style call body."""
    result: dict[str, object] = {}
    for m in _PYTHONIC_KV_RE.finditer(name_args):
        key = m.group("k")
        if (val := m.group("s")) is not None or (val := m.group("s2")) is not None:
            result[key] = val
        elif (num := m.group("n")) is not None:
            result[key] = float(num) if "." in num else int(num)
        elif (b := m.group("b")) is not None:
            result[key] = b.lower() == "true"
    return result or None


def _quote_keys(s: str) -> str:
    """Wrap unquoted JSON-style keys in double-quotes so ``json.loads`` can parse."""
    return _UNQUOTED_KEY_RE.sub(r'\1"\2":', s)


def _load_relaxed(block: str) -> object | None:
    """Try, in order:

    1. ``json.loads`` as-is.
    2. ``json.loads`` after quoting bare keys.
    3. ``ast.literal_eval``.
    4. Regex extract ``name: <ident>`` + ``arguments: {json}`` — Granite's
       bare-identifier output, which none of the standard parsers accept.
    """
    block = block.strip()
    for variant in (block, _quote_keys(block), block.replace("'", '"')):
        try:
            return json.loads(variant)
        except Exception:
            continue
    try:
        return ast.literal_eval(block)
    except Exception:
        pass
    match = _NAME_ARGS_RE.search(block)
    if match:
        try:
            args = json.loads(match.group("args"))
        except Exception:
            args = _load_relaxed(match.group("args")) or {}
        return {"name": match.group("name"), "arguments": args}

    # Pythonic call inside the ``<tool_call>`` tag: ``fn(k="v", …)``.
    pymatch = _PYTHONIC_CALL_RE.search(block)
    if pymatch:
        return {
            "name": pymatch.group("name"),
            "arguments": _parse_pythonic_body(pymatch.group("body")) or {},
        }
    return None


class GraniteDetector(BaseFormatDetector):
    """Detector for Granite 4.0 Nano ``<tool_call>{name:…, arguments:…}`` output."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        calls = []
        for match in _TAG_RE.finditer(text):
            body = match.group("body")
            payload = _load_relaxed(body)
            if payload is None:
                continue
            if isinstance(payload, list | dict):
                calls.extend(self.parse_base_json(payload, tools))

        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )
