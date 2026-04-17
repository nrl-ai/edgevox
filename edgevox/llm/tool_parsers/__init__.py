"""Tool-call format detectors for llama.cpp / llama-cpp-python models.

The core classes here are vendored from SGLang's ``function_call`` package
(Apache-2.0 — see ``NOTICE``) because SGLang's detectors are the cleanest
and most battle-tested post-hoc parsers available. EdgeVox wraps them in
a tiny registry so any preset in :mod:`edgevox.llm.models` can declare
which detector(s) to try.

Public API
==========

- :class:`BaseFormatDetector` — abstract base.
- :class:`HermesDetector`, :class:`Qwen25Detector`, :class:`Llama32Detector`,
  :class:`MistralDetector`, :class:`PythonicDetector` — concrete detectors.
- :data:`DETECTORS` — mapping of detector name → class.
- :func:`register_detector` — add a custom detector under a name.
- :func:`parse_tool_calls` — try a chain of detectors against raw model output
  and return OpenAI-shaped ``tool_calls``, or ``None`` if nothing matched.
- :class:`Tool`, :func:`coerce_tools` — convert OpenAI schema dicts into the
  local ``Tool`` dataclass.
"""

from __future__ import annotations

import html
import json
import logging

from edgevox.llm.tool_parsers._types import Function, Tool, coerce_tools
from edgevox.llm.tool_parsers.base import BaseFormatDetector
from edgevox.llm.tool_parsers.core_types import StreamingParseResult, ToolCallItem
from edgevox.llm.tool_parsers.detectors import (
    GraniteDetector,
    HermesDetector,
    Llama32Detector,
    MistralDetector,
    PythonicDetector,
    Qwen25Detector,
    XLAMDetector,
)

log = logging.getLogger(__name__)


# Built-in detector registry. Keys are the names callers use in
# ``ModelPreset.tool_call_parsers`` (see :mod:`edgevox.llm.models`).
DETECTORS: dict[str, type[BaseFormatDetector]] = {
    "hermes": HermesDetector,  # <tool_call>…</tool_call> — Qwen/Hermes/chatml
    "qwen25": Qwen25Detector,  # <tool_call>\n…\n</tool_call> — Qwen2.5/Qwen3 strict
    "llama32": Llama32Detector,  # <|python_tag|>{…} + bare JSON — Llama 3.1/3.2/3.3
    "mistral": MistralDetector,  # [TOOL_CALLS] […] — Mistral Nemo / Ministral
    "pythonic": PythonicDetector,  # [fn(arg=val), …] — Llama-4 / Llama-3.2 pythonic
    "xlam": XLAMDetector,  # [{"name":…,"arguments":…}] array (xLAM) or fenced variant (Hammer)
    "granite": GraniteDetector,  # <tool_call>{name:…, arguments:…} with unquoted keys (Granite 4)
}


def register_detector(name: str, detector: type[BaseFormatDetector]) -> None:
    """Register a custom :class:`BaseFormatDetector` subclass under ``name``.

    After registration the detector can be referenced from a
    :class:`~edgevox.llm.models.ModelPreset` via
    ``tool_call_parsers=("your-name", …)``.
    """
    if not issubclass(detector, BaseFormatDetector):
        raise TypeError(f"{detector!r} must subclass BaseFormatDetector")
    DETECTORS[name] = detector


def _to_openai_tool_call(item: ToolCallItem, idx: int) -> dict:
    """Convert a vendored :class:`ToolCallItem` to an OpenAI-shaped dict."""
    # ``item.parameters`` is already a JSON string in the vendored format.
    try:
        parsed = json.loads(item.parameters) if item.parameters else {}
    except json.JSONDecodeError:
        parsed = {}
    return {
        "id": f"{item.name or 'call'}_{idx}",
        "type": "function",
        "function": {
            "name": item.name or "",
            "arguments": json.dumps(parsed, ensure_ascii=False),
        },
    }


def parse_tool_calls(
    text: str,
    tools: list | None,
    detectors: list[str] | tuple[str, ...] | None = None,
) -> list[dict] | None:
    """Try each named detector against ``text``; return OpenAI-shaped calls or ``None``.

    Args:
        text: raw assistant content emitted by the model.
        tools: OpenAI ``tools=[…]`` schema OR a list of :class:`Tool` objects.
            Used for validation — unknown tool names are dropped (strict).
        detectors: ordered names of detectors to try. ``None`` uses every
            registered detector in insertion order.

    Returns:
        List of OpenAI ``tool_calls`` dicts, or ``None`` if no detector matched.
    """
    if not text or not text.strip():
        return None

    # Some GGUF chat templates (notably IBM Granite 4.0-H) emit HTML-escaped
    # tool-call markers — ``&lt;tool_call&gt;`` instead of ``<tool_call>``.
    # Unescape once so every detector sees the canonical form.
    if "&lt;" in text or "&amp;" in text:
        text = html.unescape(text)

    names = list(detectors) if detectors else list(DETECTORS)
    typed_tools = coerce_tools(tools)

    for name in names:
        cls = DETECTORS.get(name)
        if cls is None:
            log.warning(f"Unknown tool-call detector '{name}' — skipping")
            continue
        detector = cls()
        if not detector.has_tool_call(text):
            continue
        try:
            result: StreamingParseResult = detector.detect_and_parse(text, typed_tools)
        except Exception:
            log.exception(f"Detector '{name}' raised on content")
            continue
        if result.calls:
            return [_to_openai_tool_call(call, i) for i, call in enumerate(result.calls)]

    return None


__all__ = [
    "DETECTORS",
    "BaseFormatDetector",
    "Function",
    "HermesDetector",
    "Llama32Detector",
    "MistralDetector",
    "PythonicDetector",
    "Qwen25Detector",
    "StreamingParseResult",
    "Tool",
    "ToolCallItem",
    "coerce_tools",
    "parse_tool_calls",
    "register_detector",
]
