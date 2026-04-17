"""Minimal ``Tool`` / ``Function`` dataclasses used by the vendored SGLang
detectors in place of SGLang's OpenAI-protocol Pydantic models.

We only need the attributes the detectors actually touch: ``tool.function.name``,
``tool.function.parameters``, ``tool.function.strict``. This lets us keep the
vendored code unmodified while avoiding the ``sglang`` package dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Function:
    name: str
    parameters: dict[str, Any] | None = None
    strict: bool = False


@dataclass
class Tool:
    """OpenAI-shaped tool definition — attribute-access compatible with the
    fields the vendored detectors read."""

    function: Function
    type: str = "function"

    @classmethod
    def from_openai_schema(cls, schema: dict[str, Any]) -> Tool:
        """Build a :class:`Tool` from an OpenAI ``tools=[…]`` schema entry."""
        fn = schema.get("function", schema)
        return cls(
            function=Function(
                name=fn["name"],
                parameters=fn.get("parameters"),
                strict=fn.get("strict", False),
            ),
            type=schema.get("type", "function"),
        )


def coerce_tools(tools: list[Any] | None) -> list[Tool]:
    """Accept either our ``Tool`` dataclass or raw OpenAI schema dicts."""
    if not tools:
        return []
    out: list[Tool] = []
    for t in tools:
        if isinstance(t, Tool):
            out.append(t)
        elif isinstance(t, dict):
            out.append(Tool.from_openai_schema(t))
        else:
            # Duck-typed object with ``.function.name`` — trust it.
            out.append(t)  # type: ignore[arg-type]
    return out
