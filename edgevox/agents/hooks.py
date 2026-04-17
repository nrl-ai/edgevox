"""Pluggable hook system for the EdgeVox agent loop.

Hooks are the extension mechanism that keeps :class:`LLMAgent` itself
boring and small. Instead of patching the loop for each new behavior
(guardrails, audit logs, plan mode, loop detection, memory persistence,
token budgets, …), a hook subscribes to one of the six fire points and
gets to inspect, modify, short-circuit, or terminate the run.

Fire points
-----------

=================  ===================================  ==============================
Point              Payload                              Typical customers
=================  ===================================  ==============================
``on_run_start``   ``{"task": str}``                    guardrails, memory injection,
                                                        compaction trigger, loop-counter
                                                        reset
``before_llm``     ``{"messages": list, "hop": int,     token-budget truncation,
                    "tools": list | None}``             message scrubbers
``after_llm``      ``{"content": str, "tool_calls":     echoed-payload substitution,
                    list, "hop": int}``                 think-tag stripping, redaction
``before_tool``    :class:`ToolCallRequest`             loop detection, plan-mode
                                                        confirmation, arg rewriting
``after_tool``     :class:`~edgevox.llm.tools.           schema-retry enrichment,
                    ToolCallResult`                     episode logging, metrics
``on_run_end``     :class:`~edgevox.agents.base.        session persistence, audit
                    AgentResult`                        trails
=================  ===================================  ==============================

Semantics
---------

A hook callable ``(point, ctx, payload) -> HookResult | None`` returns:

- ``None`` or ``HookResult.cont()`` — keep going, no change.
- :meth:`HookResult.replace` — replace the payload going forward.
- :meth:`HookResult.end` — terminate the turn with this reply. Ignored
  from ``after_tool`` (you get ``before_llm`` on the next hop instead).

Multiple hooks run in registration order; each sees the previous hook's
modified payload. The first ``end_turn`` short-circuits.

Plug-and-play
-------------

- Per-agent: ``LLMAgent(..., hooks=[MyHook()])``.
- Per-context: ``ctx.hooks.register(MyHook())``.
- Project-wide: expose a hook via ``edgevox.hooks`` entry point and
  :func:`load_entry_point_hooks` discovers it.

Hooks can be classes (most flexible — carry state, config) or functions
wrapped with :func:`hook` (concise, one-off).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fire points
# ---------------------------------------------------------------------------

ON_RUN_START = "on_run_start"
BEFORE_LLM = "before_llm"
AFTER_LLM = "after_llm"
BEFORE_TOOL = "before_tool"
AFTER_TOOL = "after_tool"
ON_RUN_END = "on_run_end"

FIRE_POINTS: frozenset[str] = frozenset(
    {
        ON_RUN_START,
        BEFORE_LLM,
        AFTER_LLM,
        BEFORE_TOOL,
        AFTER_TOOL,
        ON_RUN_END,
    }
)


# ---------------------------------------------------------------------------
# Hook result
# ---------------------------------------------------------------------------


class HookAction(str, Enum):
    """What the hook wants the loop to do with its return value."""

    CONTINUE = "continue"
    MODIFY = "modify"
    END_TURN = "end_turn"


@dataclass
class HookResult:
    """Return value from a hook invocation.

    Use the classmethod constructors rather than instantiating directly —
    they encode the intent clearly in call sites.
    """

    action: HookAction = HookAction.CONTINUE
    payload: Any = None
    reason: str = ""

    @classmethod
    def cont(cls) -> HookResult:
        """Continue unchanged. Equivalent to returning ``None``."""
        return cls(action=HookAction.CONTINUE)

    @classmethod
    def replace(cls, payload: Any, *, reason: str = "") -> HookResult:
        """Replace the current payload going forward at this fire point."""
        return cls(action=HookAction.MODIFY, payload=payload, reason=reason)

    @classmethod
    def end(cls, reply: str, *, reason: str = "") -> HookResult:
        """Terminate the current turn. ``reply`` is returned verbatim."""
        return cls(action=HookAction.END_TURN, payload=reply, reason=reason)

    @property
    def is_continue(self) -> bool:
        return self.action is HookAction.CONTINUE

    @property
    def is_modify(self) -> bool:
        return self.action is HookAction.MODIFY

    @property
    def is_end(self) -> bool:
        return self.action is HookAction.END_TURN


# ---------------------------------------------------------------------------
# Tool call request (payload for before_tool)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRequest:
    """Mutable request envelope passed to ``before_tool`` hooks.

    Hooks can modify ``name`` / ``arguments`` to rewrite a call, or set
    ``skip_dispatch=True`` with a ``synthetic_result`` to return a
    pre-computed result without actually running the tool. The loop
    detector uses this to answer "you already called that" without a
    real tool invocation.
    """

    name: str
    arguments: str | dict[str, Any]
    hop: int
    is_skill: bool = False
    skip_dispatch: bool = False
    synthetic_result: Any = None
    # Reason kept alongside so audit logs can explain the skip.
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Hook(Protocol):
    """Anything with a ``points`` attribute and a ``__call__`` matching
    ``(point, ctx, payload) -> HookResult | None``. Duck-typed; inherit
    from this Protocol is not required."""

    points: frozenset[str]

    def __call__(
        self,
        point: str,
        ctx: AgentContext,
        payload: Any,
    ) -> HookResult | None: ...


HookCallable = Callable[["AgentContext", Any], "HookResult | None"]


# ---------------------------------------------------------------------------
# @hook decorator
# ---------------------------------------------------------------------------


def hook(*points: str) -> Callable[[Callable[..., Any]], Hook]:
    """Decorator to wrap a function into a :class:`Hook`.

    The decorated function may take either ``(point, ctx, payload)`` or
    ``(ctx, payload)``; the decorator introspects and adapts.

    Register the *returned* object (not the bare function):

    .. code-block:: python

        @hook("before_tool")
        def my_filter(ctx, payload):
            if payload.name == "dangerous":
                return HookResult.end("no.")

        agent = LLMAgent(..., hooks=[my_filter])
    """
    if not points:
        raise ValueError("hook() requires at least one fire point")
    for p in points:
        if p not in FIRE_POINTS:
            raise ValueError(f"Unknown hook point {p!r}; must be one of {sorted(FIRE_POINTS)}")

    frozen_points = frozenset(points)

    def wrap(fn: Callable[..., Any]) -> Hook:
        sig = inspect.signature(fn)
        arity = len(sig.parameters)
        if arity not in (2, 3):
            raise TypeError(
                f"@hook function {fn.__name__} must take (point, ctx, payload) or (ctx, payload), "
                f"got {arity} parameters"
            )
        takes_point = arity == 3

        class _FnHook:
            points = frozen_points

            def __init__(self, target: Callable[..., Any]) -> None:
                self._fn = target
                self.__name__ = getattr(target, "__name__", repr(target))
                self.__doc__ = target.__doc__

            def __call__(
                self,
                point: str,
                ctx: AgentContext,
                payload: Any,
            ) -> HookResult | None:
                if takes_point:
                    return self._fn(point, ctx, payload)
                return self._fn(ctx, payload)

            def __repr__(self) -> str:
                return f"<hook {self.__name__} points={sorted(self.points)}>"

        return _FnHook(fn)

    return wrap


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


class HookRegistry:
    """Ordered collection of hooks, indexed by fire point.

    Registration order within a point is preserved and determines
    execution order. Hook exceptions are logged and treated as
    ``continue`` — one misbehaving hook never breaks the loop.
    """

    def __init__(self, hooks: Iterable[Hook] | None = None) -> None:
        self._by_point: dict[str, list[Hook]] = {p: [] for p in FIRE_POINTS}
        if hooks:
            for h in hooks:
                self.register(h)

    # ----- registration -----

    def register(self, h: Hook) -> HookRegistry:
        """Register a hook. Raises if ``h`` has no ``points`` attribute."""
        points = getattr(h, "points", None)
        if not points:
            raise TypeError(f"{h!r} is not a Hook — missing .points attribute")
        for p in points:
            if p not in FIRE_POINTS:
                raise ValueError(f"Hook {h!r} declares unknown point {p!r}")
            self._by_point[p].append(h)
        return self

    def extend(self, other: HookRegistry | Iterable[Hook]) -> HookRegistry:
        """Merge another registry or iterable of hooks into this one."""
        if isinstance(other, HookRegistry):
            for p in FIRE_POINTS:
                self._by_point[p].extend(other._by_point.get(p, ()))
        else:
            for h in other:
                self.register(h)
        return self

    def copy(self) -> HookRegistry:
        """Return a shallow copy of this registry."""
        new = HookRegistry()
        for p in FIRE_POINTS:
            new._by_point[p] = list(self._by_point.get(p, ()))
        return new

    # ----- inspection -----

    def __len__(self) -> int:
        return sum(len(hs) for hs in self._by_point.values())

    def __contains__(self, point: str) -> bool:
        return bool(self._by_point.get(point))

    def __iter__(self) -> Iterator[Hook]:
        seen: set[int] = set()
        for hs in self._by_point.values():
            for h in hs:
                if id(h) not in seen:
                    seen.add(id(h))
                    yield h

    def at(self, point: str) -> list[Hook]:
        """Hooks registered at a specific point (copy)."""
        return list(self._by_point.get(point, ()))

    # ----- firing -----

    def fire(self, point: str, ctx: AgentContext, payload: Any) -> HookResult:
        """Run all hooks at ``point`` in registration order.

        Each hook receives the payload (possibly modified) from the
        previous hook. Returns the first ``end_turn`` or the final
        accumulated modify. If no hook matches or all return continue,
        returns a continue with the original payload unchanged.
        """
        if point not in FIRE_POINTS:
            raise ValueError(f"Unknown hook point {point!r}")

        hooks = self._by_point.get(point, ())
        if not hooks:
            return HookResult(action=HookAction.CONTINUE, payload=payload)

        current = payload
        modified = False
        reasons: list[str] = []

        for h in hooks:
            try:
                outcome = h(point, ctx, current)
            except Exception:
                log.exception("Hook %r raised at %s", h, point)
                continue
            if outcome is None or outcome.is_continue:
                continue
            if outcome.is_end:
                return outcome
            if outcome.is_modify:
                current = outcome.payload
                modified = True
                if outcome.reason:
                    reasons.append(outcome.reason)

        if modified:
            return HookResult(action=HookAction.MODIFY, payload=current, reason="; ".join(reasons))
        return HookResult(action=HookAction.CONTINUE, payload=current)


# ---------------------------------------------------------------------------
# Merged fire across two registries (agent + ctx)
# ---------------------------------------------------------------------------


def fire_chain(
    registries: Iterable[HookRegistry | None],
    point: str,
    ctx: AgentContext,
    payload: Any,
) -> HookResult:
    """Fire the same point across multiple registries in order.

    Used by :class:`LLMAgent` to run agent-level hooks, then ctx-level
    hooks, so both layers see modifications from earlier layers.
    """
    current = payload
    modified = False
    reasons: list[str] = []
    for reg in registries:
        if reg is None or not len(reg):
            continue
        r = reg.fire(point, ctx, current)
        if r.is_end:
            return r
        if r.is_modify:
            current = r.payload
            modified = True
            if r.reason:
                reasons.append(r.reason)
    if modified:
        return HookResult(action=HookAction.MODIFY, payload=current, reason="; ".join(reasons))
    return HookResult(action=HookAction.CONTINUE, payload=current)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


def load_entry_point_hooks(group: str = "edgevox.hooks") -> list[Hook]:
    """Discover hooks exposed by installed packages.

    Packages declare hooks under the ``edgevox.hooks`` entry-point group:

    .. code-block:: toml

        [project.entry-points."edgevox.hooks"]
        guardrail = "my_pkg.hooks:SafetyGuardrail"
        audit = "my_pkg.hooks:AUDIT_HOOKS"  # a list works too

    The entry point may resolve to a single :class:`Hook` instance or an
    iterable of them.
    """
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover — older Python fallback
        eps = entry_points().get(group, [])  # type: ignore[attr-defined]

    out: list[Hook] = []
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:  # pragma: no cover
            log.warning("Failed to load hook entry point %r: %s", ep.name, e)
            continue
        items = obj if isinstance(obj, (list, tuple)) else [obj]
        for item in items:
            if not hasattr(item, "points"):
                log.warning("Entry point %r produced non-hook %r (no .points) — skipping", ep.name, item)
                continue
            out.append(item)
    return out


__all__ = [
    "AFTER_LLM",
    "AFTER_TOOL",
    "BEFORE_LLM",
    "BEFORE_TOOL",
    "FIRE_POINTS",
    "ON_RUN_END",
    "ON_RUN_START",
    "Hook",
    "HookAction",
    "HookCallable",
    "HookRegistry",
    "HookResult",
    "ToolCallRequest",
    "fire_chain",
    "hook",
    "load_entry_point_hooks",
]
