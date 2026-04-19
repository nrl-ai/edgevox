"""OpenTelemetry bridge for :class:`TracingHook`'s span events.

Optional feature behind the ``observability`` extra. Subscribes to an
:class:`EventBus` (the agent's ``ctx.bus``) and turns every
``span_start`` / ``span_end`` :class:`AgentEvent` into a real OTel
span via the global ``opentelemetry.trace`` tracer provider. Drop-in
compatible with any OTLP / Jaeger / Tempo / honeycomb exporter the
user has configured globally.

Usage::

    # 1. Configure the global tracer provider once at app boot — any
    #    vanilla OTel setup works, e.g.:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # 2. Wire the bridge to your agent's bus.
    from edgevox.agents.hooks_builtin import TracingHook
    from edgevox.agents.tracing_otel import install_otel_bridge

    agent = LLMAgent(..., hooks=[TracingHook()])
    ctx = AgentContext()
    install_otel_bridge(ctx.bus)   # idempotent per-bus
    agent.run("what's the weather?", ctx)

The bridge deliberately holds spans by id in a module-local dict —
the ``span_start`` event records its own ``span_id`` so the bridge
can finish the right span on ``span_end`` even when multiple agents
are running in parallel on the same bus.

Span attributes: ``trace_id``, ``span_id``, ``parent_span_id``,
``service.name``, ``name``, ``duration_ns`` (span_end only), and
``reply_len`` (span_end only). Extend the hook to add more; the
bridge forwards every key that's not a standard OTel field.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgevox.agents.base import AgentEvent
    from edgevox.agents.bus import EventBus


log = logging.getLogger(__name__)


_BRIDGED_BUSES: set[int] = set()
_SPAN_REGISTRY: dict[str, Any] = {}
_REGISTRY_LOCK = threading.Lock()


def install_otel_bridge(bus: EventBus, *, service_name: str = "edgevox") -> None:
    """Subscribe an OTel span handler to ``bus``.

    Idempotent: calling twice on the same bus registers once. Requires
    the ``observability`` extra (``opentelemetry-api`` /
    ``opentelemetry-sdk``). When those aren't installed, logs a
    one-time warning and returns — ``TracingHook`` still emits events
    locally, the bridge just becomes a no-op.
    """
    try:
        from opentelemetry import trace
    except ModuleNotFoundError:
        log.warning(
            "install_otel_bridge: opentelemetry not installed; bridge is a no-op. "
            "pip install 'edgevox[observability]' to enable."
        )
        return

    bus_id = id(bus)
    if bus_id in _BRIDGED_BUSES:
        return
    _BRIDGED_BUSES.add(bus_id)

    tracer = trace.get_tracer(service_name)

    def _handle(event: AgentEvent) -> None:
        if event.kind == "span_start":
            _open_span(tracer, event)
        elif event.kind == "span_end":
            _close_span(event)

    bus.subscribe_all(_handle)


def _open_span(tracer: Any, event: AgentEvent) -> None:
    from opentelemetry import trace

    payload = event.payload or {}
    span_id = payload.get("span_id")
    if not span_id:
        return

    parent_span_id = payload.get("parent_span_id")
    parent_ctx = None
    if parent_span_id is not None:
        parent = _SPAN_REGISTRY.get(parent_span_id)
        if parent is not None:
            parent_ctx = trace.set_span_in_context(parent)

    span = tracer.start_span(
        name=payload.get("name") or event.agent_name or "agent.run",
        context=parent_ctx,
    )
    # Forward every non-standard payload key as a span attribute so
    # OTel exporters can render them without ad-hoc translation.
    for key, value in payload.items():
        if key in {"span_id", "parent_span_id", "trace_id", "name", "start_ns"}:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            span.set_attribute(key, repr(value))
    span.set_attribute("edgevox.span_id", span_id)
    span.set_attribute("edgevox.agent_name", event.agent_name)

    with _REGISTRY_LOCK:
        _SPAN_REGISTRY[span_id] = span


def _close_span(event: AgentEvent) -> None:
    payload = event.payload or {}
    span_id = payload.get("span_id")
    if not span_id:
        return
    with _REGISTRY_LOCK:
        span = _SPAN_REGISTRY.pop(span_id, None)
    if span is None:
        return
    for key, value in payload.items():
        if key in {"span_id", "parent_span_id", "trace_id", "name"}:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            span.set_attribute(key, repr(value))
    span.end()


def _reset_bridge_state() -> None:
    """Test helper — wipes the module-local bridge bookkeeping so a
    fresh test run doesn't carry over span registrations."""
    with _REGISTRY_LOCK:
        _SPAN_REGISTRY.clear()
        _BRIDGED_BUSES.clear()


__all__ = ["install_otel_bridge"]
