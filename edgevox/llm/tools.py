"""Tool-calling primitives for building voice agents.

The ``@tool`` decorator introspects a Python function's signature and
docstring into a JSON Schema compatible with llama-cpp-python's
OpenAI-style ``tools=[...]`` parameter. A ``ToolRegistry`` holds a set of
tools, serialises them for the model, and dispatches ``tool_calls``
emitted by the model back to the underlying Python callables.

Third-party packages can ship tools by registering them under the
``edgevox.tools`` entry-point group; ``load_entry_point_tools`` discovers
and imports them at runtime.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

log = logging.getLogger(__name__)

_PRIMITIVE_SCHEMA: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _is_optional(tp: Any) -> tuple[bool, Any]:
    """If ``tp`` is ``Optional[X]`` / ``X | None``, return ``(True, X)``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON schema fragment."""
    _, inner = _is_optional(tp)
    tp = inner

    if tp in _PRIMITIVE_SCHEMA:
        return dict(_PRIMITIVE_SCHEMA[tp])

    origin = get_origin(tp)
    if origin in (list, tuple, set, frozenset) or tp in (list, tuple, set, frozenset):
        args = get_args(tp)
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin is dict or tp is dict:
        return {"type": "object"}

    if isinstance(tp, type) and issubclass(tp, str):
        return {"type": "string"}

    if typing.get_type_hints(tp, include_extras=False) if inspect.isclass(tp) else False:
        return {"type": "object"}

    return {"type": "string"}


_ARG_HEADER_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_ARG_LINE_RE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {arg_name: description}).

    Accepts Google-style ``Args:`` blocks. The summary is everything
    before the first section header. Unknown sections are ignored.
    """
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: list[str] = []
    args: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for raw in lines:
        if _ARG_HEADER_RE.match(raw):
            in_args = True
            current = None
            continue
        if in_args and raw.strip() and not raw.startswith(" ") and raw.rstrip().endswith(":"):
            in_args = False
            current = None
            continue
        if in_args:
            m = _ARG_LINE_RE.match(raw)
            if m:
                current = m.group(1)
                args[current] = m.group(2).strip()
            elif current and raw.strip():
                args[current] = f"{args[current]} {raw.strip()}".strip()
        else:
            summary_lines.append(raw)

    summary = " ".join(line.strip() for line in summary_lines if line.strip()).strip()
    return summary, args


@dataclass
class Tool:
    """A single callable tool.

    Holds the underlying function alongside its pre-computed JSON schema
    so dispatch stays hot-path cheap.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI/llama-cpp ``tools=`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: dict[str, Any]) -> Any:
        """Invoke the underlying function with the decoded arguments."""
        return self.func(**arguments)


def tool(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Mark a function as a voice-agent tool.

    The decorator builds a JSON schema from the function's type hints
    and docstring and attaches it to the function as ``__edgevox_tool__``.
    The original callable remains usable; nothing about normal Python
    invocation changes.

    Args:
        name: Override the tool name exposed to the model. Defaults to
            the function's ``__name__``.
        description: Override the top-level description. Defaults to
            the docstring summary.
    """

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        hints = get_type_hints(func)
        hints.pop("return", None)
        sig = inspect.signature(func)
        summary, arg_docs = _parse_docstring(func.__doc__)

        # Framework-injected parameters: tools may declare ``ctx`` (an
        # AgentContext) or ``handle`` (a GoalHandle) and get them at
        # dispatch time. They're stripped from the JSON schema the
        # model sees so the LLM doesn't try to fabricate them.
        _FRAMEWORK_PARAMS = {"ctx", "handle"}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name in _FRAMEWORK_PARAMS:
                continue
            annotation = hints.get(param_name, str)
            schema = _type_to_schema(annotation)
            if param_name in arg_docs:
                schema["description"] = arg_docs[param_name]
            properties[param_name] = schema

            is_opt, _ = _is_optional(annotation)
            if param.default is inspect.Parameter.empty and not is_opt:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        func.__edgevox_tool__ = Tool(  # type: ignore[attr-defined]
            name=name or func.__name__,
            description=description or summary or (func.__name__.replace("_", " ")),
            parameters=parameters,
            func=func,
        )
        return func

    if _func is not None:
        return wrap(_func)
    return wrap


def _extract(obj: Any) -> Tool:
    """Pull the :class:`Tool` descriptor off a decorated function."""
    if isinstance(obj, Tool):
        return obj
    descriptor = getattr(obj, "__edgevox_tool__", None)
    if descriptor is None:
        raise TypeError(f"{obj!r} is not an @tool-decorated function. Wrap it with @tool first.")
    return descriptor


@dataclass
class ToolCallResult:
    """Outcome of a single tool dispatch."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ToolRegistry:
    """Holds tools and dispatches model-emitted ``tool_calls``."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, *funcs: Callable[..., Any] | Tool) -> ToolRegistry:
        """Add one or more tools. Accepts functions or :class:`Tool`."""
        for f in funcs:
            descriptor = _extract(f)
            if descriptor.name in self.tools:
                log.warning("Tool %r already registered — overwriting.", descriptor.name)
            self.tools[descriptor.name] = descriptor
        return self

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return all tools as llama-cpp/OpenAI schema entries."""
        return [t.openai_schema() for t in self.tools.values()]

    def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
        *,
        ctx: Any = None,
    ) -> ToolCallResult:
        """Run a tool by name. Catches exceptions so the agent loop can
        feed the error back to the model for a retry.

        If ``ctx`` is provided and the underlying function declares a
        ``ctx`` parameter in its signature, it is injected at call time.
        """
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                return ToolCallResult(name=str(name), arguments={}, error=f"invalid JSON arguments: {e}")
        else:
            decoded = arguments or {}

        # Some llama-cpp chat handlers return ``name`` as a dict on failure;
        # coerce to a string so ``dict.get`` doesn't raise ``TypeError``.
        if not isinstance(name, str):
            return ToolCallResult(
                name=str(name),
                arguments=decoded,
                error=f"non-string tool name from model: {name!r}",
            )

        tool_obj = self.tools.get(name)
        if tool_obj is None:
            return ToolCallResult(name=name, arguments=decoded, error=f"unknown tool: {name!r}")

        # Validate arguments against the tool's declared JSON schema
        # before dispatch. Catches malformed LLM output (missing
        # required fields, wrong types, out-of-range enums) at the
        # boundary rather than deep inside user code. Structured
        # validation errors get fed back to the LLM as tool-call
        # errors, so the model can retry with corrected arguments.
        validation_error = _validate_against_schema(decoded, tool_obj.parameters)
        if validation_error is not None:
            # Keep the ``bad arguments:`` prefix so downstream matchers
            # like ``SchemaRetryHook`` (edgevox.llm._agent_harness.LOOP_BREAK_SIGNALS)
            # still fire on validation failures and synthesise a schema
            # hint back to the model.
            return ToolCallResult(name=name, arguments=decoded, error=f"bad arguments: {validation_error}")

        call_kwargs = dict(decoded)
        if ctx is not None:
            try:
                sig = inspect.signature(tool_obj.func)
                if "ctx" in sig.parameters:
                    call_kwargs["ctx"] = ctx
            except (TypeError, ValueError):
                pass

        try:
            result = tool_obj.func(**call_kwargs)
        except TypeError as e:
            return ToolCallResult(name=name, arguments=decoded, error=f"bad arguments: {e}")
        except Exception as e:
            log.exception("Tool %r raised", name)
            return ToolCallResult(name=name, arguments=decoded, error=f"{type(e).__name__}: {e}")
        return ToolCallResult(name=name, arguments=decoded, result=result)


# Minimum JSON-schema subset needed to validate ``@tool`` argument
# dicts. Implements required-field + type checks + enum checks only —
# the subset that actually catches LLM mistakes. We deliberately don't
# pull in ``jsonschema`` as a dependency; the generated schemas are
# well-constrained (``type: object`` + flat properties) and a tight
# implementation here is both faster and removes a runtime dep.
_JSON_TYPE_PY = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _validate_against_schema(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Return ``None`` if ``args`` conforms, otherwise a human-readable
    error describing the first violation encountered. Only validates
    the JSON-schema subset produced by :func:`tool`."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        # Unrecognised / custom schemas are passed through unchecked —
        # callers authoring their own ``Tool(parameters=...)`` take
        # responsibility for validation in the tool body.
        return None

    props = schema.get("properties") or {}
    required = schema.get("required") or []

    for key in required:
        if key not in args:
            return f"missing required field {key!r}"

    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            # Unknown property — tolerate by default (matches OpenAI
            # tool-call semantics; strict mode would use
            # ``additionalProperties: false`` and reject here).
            continue
        expected_type = spec.get("type")
        if expected_type is not None:
            py_type = _JSON_TYPE_PY.get(expected_type)
            if py_type is not None and not isinstance(value, py_type):
                # Pydantic-style leniency: ``int`` is OK where ``number``
                # is expected (``_JSON_TYPE_PY`` maps ``number`` to the
                # tuple already), and ``True`` counts as ``1`` for
                # ``integer`` only when the caller explicitly allowed
                # booleans — but boolean-as-int is a common LLM mistake
                # worth flagging.
                got = type(value).__name__
                return f"field {key!r}: expected {expected_type}, got {got}"
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            return f"field {key!r}: value {value!r} not in allowed enum {enum}"
    return None


def load_entry_point_tools(group: str = "edgevox.tools") -> list[Tool]:
    """Discover tools exposed by installed packages via entry points.

    Third-party packages declare tools in their ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."edgevox.tools"]
        home_assistant = "my_pkg.tools:REGISTRY"

    The entry point may resolve to a single ``@tool``-decorated function,
    a list/tuple of them, or a :class:`ToolRegistry`.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []

    found: list[Tool] = []
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover — <3.10 fallback
        eps = entry_points().get(group, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:
            log.warning("Failed to load tool entry point %r: %s", ep.name, e)
            continue
        if isinstance(obj, ToolRegistry):
            found.extend(obj.tools.values())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                try:
                    found.append(_extract(item))
                except TypeError as e:
                    log.warning("Entry point %r item skipped: %s", ep.name, e)
        else:
            try:
                found.append(_extract(obj))
            except TypeError as e:
                log.warning("Entry point %r skipped: %s", ep.name, e)
    return found


__all__ = [
    "Tool",
    "ToolCallResult",
    "ToolRegistry",
    "load_entry_point_tools",
    "tool",
]
