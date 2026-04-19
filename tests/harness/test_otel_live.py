"""Live OTel wire tests — spin up an in-process OTLP receiver and
verify that :class:`TracingHook` + :func:`install_otel_bridge` actually
push spans through the real OTLP export path.

OTel's global ``TracerProvider`` is set once per process, so we can't
swap it between tests. Instead, one provider is built at class scope
with **multiple exporters** (in-memory for assertions, console for
human debug, OTLP HTTP for the real-wire check) and every test runs
against the same provider. This is also how a production service
would configure OTel: one provider, many exporters.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from edgevox.agents import AgentContext, LLMAgent, TracingHook
from edgevox.agents.tracing_otel import _reset_bridge_state, install_otel_bridge

from .conftest import ScriptedLLM, reply

# ---------------------------------------------------------------------------
# Live OTLP HTTP collector — real wire format
# ---------------------------------------------------------------------------


class _CollectedPayload:
    """Shared state between the in-thread HTTP collector and the test."""

    def __init__(self):
        self.bodies: list[bytes] = []
        self.paths: list[str] = []
        self.lock = threading.Lock()


class _CollectorHandler(BaseHTTPRequestHandler):
    # Subclasses are instantiated per-request by BaseHTTPServer and
    # need the shared payload accessible via a class attribute.
    payload: _CollectedPayload | None = None

    def log_message(self, format, *args):
        # Silence the default access log so pytest output stays clean.
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.payload is not None:
            with self.payload.lock:
                self.payload.paths.append(self.path)
                self.payload.bodies.append(body)
        # OTLP expects an empty success response on ``/v1/traces``.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture(scope="module")
def otel_collector():
    """Starts an in-thread OTLP HTTP collector once for the whole
    module, registers a TracerProvider with OTLP + in-memory
    exporters, and cleans up at session end. Yielded value is the
    ``_CollectedPayload`` so each test can assert on what the
    collector saw during its own run.
    """
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _reset_bridge_state()
    payload = _CollectedPayload()

    class _Handler(_CollectorHandler):
        pass

    _Handler.payload = payload

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    in_memory = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "edgevox-test"}))
    provider.add_span_processor(SimpleSpanProcessor(in_memory))
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"http://127.0.0.1:{port}/v1/traces")))
    # Only the FIRST ``set_tracer_provider`` wins in OTel; subsequent
    # calls log a warning and no-op. So we set once here for the
    # module and rely on ``trace.get_tracer_provider()`` from the
    # bridge picking it up.
    trace.set_tracer_provider(provider)

    try:
        yield {"payload": payload, "provider": provider, "in_memory": in_memory}
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


class TestOTLPHTTPCollector:
    def test_tracing_hook_emits_to_live_collector(self, otel_collector):
        payload = otel_collector["payload"]
        in_memory = otel_collector["in_memory"]
        provider = otel_collector["provider"]

        # Snapshot the collector's state BEFORE our turn so assertions
        # are scoped to this test, not to state from a sibling test.
        with payload.lock:
            pre_body_count = len(payload.bodies)

        ctx = AgentContext()
        install_otel_bridge(ctx.bus, service_name="edgevox-test")

        agent = LLMAgent(
            name="live-otlp",
            description="otlp receiver test",
            instructions="Answer briefly.",
            tools=[],
            hooks=[TracingHook(service_name="edgevox-test")],
        )
        agent.bind_llm(ScriptedLLM([reply("ok")]))
        agent.run("go", ctx)

        # Force-flush so the SimpleSpanProcessor isn't sitting on
        # pending spans when we assert.
        provider.force_flush(5_000)

        # Give the server thread a moment to receive the POST.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with payload.lock:
                if len(payload.bodies) > pre_body_count:
                    break
            time.sleep(0.05)

        with payload.lock:
            new_bodies = payload.bodies[pre_body_count:]
            new_paths = payload.paths[pre_body_count:]
        assert new_bodies, "collector received no OTLP POSTs for this turn"
        assert all(p == "/v1/traces" for p in new_paths)
        combined = b"".join(new_bodies)
        assert b"live-otlp" in combined
        assert b"edgevox-test" in combined

        # Cross-check: the InMemorySpanExporter on the same provider
        # must have seen our span too — proves the bridge-built span
        # wasn't filtered out mid-pipeline.
        names = {s.name for s in in_memory.get_finished_spans()}
        assert "live-otlp" in names

    def test_async_dispatch_does_not_block_slow_exporter(self, otel_collector):
        """Prove the async-dispatch mode isolates the publisher from a
        slow exporter: we install a fake exporter that sleeps for 200 ms
        per span, run a turn, and assert the agent returned in well
        under that time. Without async dispatch the turn would block."""
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

        from edgevox.agents import AgentContext, LLMAgent, TracingHook
        from edgevox.agents.tracing_otel import install_otel_bridge, shutdown_async_dispatch

        from .conftest import ScriptedLLM, reply

        class _SlowExporter(SpanExporter):
            def __init__(self):
                self.exported: list = []

            def export(self, spans):
                time.sleep(0.2)
                self.exported.extend(spans)
                return SpanExportResult.SUCCESS

            def shutdown(self): ...

        slow = _SlowExporter()
        provider = otel_collector["provider"]
        provider.add_span_processor(SimpleSpanProcessor(slow))

        from edgevox.agents.bus import EventBus

        bus = EventBus()
        install_otel_bridge(bus, service_name="edgevox-test", async_dispatch=True)

        agent = LLMAgent(
            name="async-bridge",
            description="async dispatch test",
            instructions=".",
            tools=[],
            hooks=[TracingHook(service_name="edgevox-test")],
        )
        agent.bind_llm(ScriptedLLM([reply("ok")]))

        t0 = time.monotonic()
        agent.run("go", AgentContext(bus=bus))
        elapsed = time.monotonic() - t0

        assert elapsed < 0.1, f"async dispatch should not block the agent turn; took {elapsed:.3f} s"
        # Drain the worker so we know the slow exporter got the span
        # even though the agent didn't wait for it.
        shutdown_async_dispatch(bus, timeout=3.0)
        provider.force_flush(5_000)
        # The slow exporter received our span in the background.
        assert any(s.name == "async-bridge" for s in slow.exported)

    def test_handoff_produces_parent_child_spans_on_wire(self, otel_collector):
        """Verify the parent-child relationship survives serialisation
        all the way to the HTTP collector."""
        from edgevox.agents import agent_as_tool

        payload = otel_collector["payload"]
        in_memory = otel_collector["in_memory"]
        provider = otel_collector["provider"]

        with payload.lock:
            pre = len(payload.bodies)

        child = LLMAgent(
            name="wire-worker",
            description="delegation target",
            instructions="Do work.",
            tools=[],
            hooks=[TracingHook(service_name="edgevox-test")],
        )
        child.bind_llm(ScriptedLLM([reply("child")]))

        parent = LLMAgent(
            name="wire-lead",
            description="delegator",
            instructions="Delegate then answer.",
            tools=[agent_as_tool(child)],
            hooks=[TracingHook(service_name="edgevox-test")],
        )

        def _call_child_then_reply(messages, _tools):
            # hop 0: delegate; hop 1: final reply
            if len([m for m in messages if m.get("role") == "tool"]) == 0:
                from tests.harness.conftest import call

                return call("wire-worker", task="do it")
            return reply("done")

        parent.bind_llm(ScriptedLLM([_call_child_then_reply, _call_child_then_reply]))

        ctx = AgentContext()
        install_otel_bridge(ctx.bus, service_name="edgevox-test")
        parent.run("start", ctx)

        provider.force_flush(5_000)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with payload.lock:
                if len(payload.bodies) > pre:
                    break
            time.sleep(0.05)

        names = {s.name for s in in_memory.get_finished_spans()}
        assert "wire-lead" in names
        assert "wire-worker" in names

        # Same trace on parent + child.
        spans_by_name: dict[str, Any] = {s.name: s for s in in_memory.get_finished_spans()}
        parent_span = spans_by_name["wire-lead"]
        child_span = spans_by_name["wire-worker"]
        assert child_span.attributes.get("trace_id") == parent_span.attributes.get("trace_id")

        # Both names hit the wire.
        with payload.lock:
            wire_bytes = b"".join(payload.bodies[pre:])
        assert b"wire-lead" in wire_bytes
        assert b"wire-worker" in wire_bytes
