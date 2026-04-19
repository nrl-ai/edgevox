"""OpenTelemetry bridge for :class:`TracingHook`'s span events.

Optional feature behind the ``observability`` extra. Subscribes to an
:class:`EventBus` (the agent's ``ctx.bus``) and turns every
``span_start`` / ``span_end`` :class:`AgentEvent` into a real OTel
span via the global ``opentelemetry.trace`` tracer provider. Drop-in
compatible with any OTLP / Jaeger / Tempo / honeycomb exporter the
user has configured globally.

Usage::

    # 1. Configure the global tracer provider once at app boot. Use
    #    BatchSpanProcessor — it owns a background thread + queue so
    #    ``span.end()`` returns immediately, which matters because
    #    the EventBus dispatches synchronously on the publisher.
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

**Blocking / non-blocking dispatch.** The ``EventBus`` fires
subscribers synchronously on the publisher's thread. The default
bridge path calls ``tracer.start_span`` + ``span.end()`` inline — fast
with ``BatchSpanProcessor`` (which just enqueues), but slow with
``SimpleSpanProcessor`` since that processor exports on the caller's
thread and blocks the agent turn if the collector is slow / down.

Safer default for production is ``async_dispatch=True``:

    install_otel_bridge(ctx.bus, async_dispatch=True)

That pushes every ``span_start`` / ``span_end`` onto a bounded queue
drained by a single daemon worker thread. The agent turn never waits
on OTel — even if the exporter stalls — at the cost of span ordering
potentially differing slightly from wall-clock order under load.
Combining ``async_dispatch=True`` **with** ``BatchSpanProcessor`` is
the belt-and-braces setup most production services want.

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
import queue
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgevox.agents.base import AgentEvent
    from edgevox.agents.bus import EventBus


log = logging.getLogger(__name__)


_BRIDGED_BUSES: set[int] = set()
_SPAN_REGISTRY: dict[str, Any] = {}
_REGISTRY_LOCK = threading.Lock()

# Async-dispatch worker state. At most one worker per bus_id.
_DISPATCH_QUEUES: dict[int, queue.Queue] = {}
_DISPATCH_THREADS: dict[int, threading.Thread] = {}
_DISPATCH_LOCK = threading.Lock()


def install_otel_bridge(
    bus: EventBus,
    *,
    service_name: str = "edgevox",
    async_dispatch: bool = False,
    queue_size: int = 4096,
) -> None:
    """Subscribe an OTel span handler to ``bus``.

    Idempotent: calling twice on the same bus registers once. Requires
    the ``observability`` extra (``opentelemetry-api`` /
    ``opentelemetry-sdk``). When those aren't installed, logs a
    one-time warning and returns — ``TracingHook`` still emits events
    locally, the bridge just becomes a no-op.

    Args:
        bus: the ``EventBus`` the agent publishes to (usually
            ``ctx.bus``).
        service_name: forwarded to the OTel tracer + attached to every
            span as ``service.name``.
        async_dispatch: if ``True``, ``span_start`` / ``span_end``
            events are enqueued and drained by a daemon worker thread
            instead of running on the publisher's thread. Use this in
            production to isolate the agent turn from a slow
            collector. The queue is bounded at ``queue_size`` — over
            that, oldest entries are dropped with a warning rather
            than blocking the publisher.
        queue_size: bounded queue depth when ``async_dispatch=True``.
            Sized to comfortably absorb a burst of spans from several
            concurrent agents; tune upward for very chatty fleets.
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

    if async_dispatch:
        _handle = _make_async_handler(bus_id, tracer, queue_size=queue_size)
    else:

        def _handle(event: AgentEvent) -> None:
            if event.kind == "span_start":
                _open_span(tracer, event)
            elif event.kind == "span_end":
                _close_span(event)

    bus.subscribe_all(_handle)


def _make_async_handler(bus_id: int, tracer: Any, *, queue_size: int):
    """Build a handler that enqueues events to a per-bus worker
    thread. The worker drains synchronously and calls the same
    ``_open_span`` / ``_close_span`` helpers used by the sync path.
    """

    with _DISPATCH_LOCK:
        q: queue.Queue = queue.Queue(maxsize=queue_size)
        _DISPATCH_QUEUES[bus_id] = q

        def _worker() -> None:
            while True:
                item = q.get()
                if item is None:
                    return
                try:
                    event: AgentEvent = item
                    if event.kind == "span_start":
                        _open_span(tracer, event)
                    elif event.kind == "span_end":
                        _close_span(event)
                except Exception:
                    log.exception("OTel async-dispatch worker raised on %r", getattr(item, "kind", "?"))

        t = threading.Thread(target=_worker, name=f"otel-bridge-{bus_id:x}", daemon=True)
        t.start()
        _DISPATCH_THREADS[bus_id] = t

    def _enqueue(event: AgentEvent) -> None:
        try:
            q.put_nowait(event)
        except queue.Full:
            # Bounded backpressure: drop the oldest item and push the
            # new one. The alternative — blocking the publisher —
            # defeats the whole point of async dispatch.
            try:
                q.get_nowait()
                q.put_nowait(event)
                log.warning("otel async dispatch queue full; dropped an older span event")
            except queue.Empty:
                pass

    return _enqueue


def shutdown_async_dispatch(bus: EventBus, *, timeout: float = 5.0) -> None:
    """Stop the async-dispatch worker for ``bus`` and wait for it to
    drain. Safe to call even when no worker was started.

    Call this at app shutdown to guarantee in-flight spans reach the
    exporter before the process exits.
    """
    bus_id = id(bus)
    with _DISPATCH_LOCK:
        q = _DISPATCH_QUEUES.pop(bus_id, None)
        t = _DISPATCH_THREADS.pop(bus_id, None)
    if q is not None:
        q.put(None)
    if t is not None:
        t.join(timeout=timeout)


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
    # Tear down any running async-dispatch workers so we don't leak
    # threads across test modules.
    with _DISPATCH_LOCK:
        queues = list(_DISPATCH_QUEUES.items())
        _DISPATCH_QUEUES.clear()
        _DISPATCH_THREADS.clear()
    for _, q in queues:
        q.put(None)


__all__ = ["install_otel_bridge"]
