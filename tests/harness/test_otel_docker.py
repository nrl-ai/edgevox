"""End-to-end OTel test against a real ``otel/opentelemetry-collector-contrib``
running in Docker. This is the most realistic check the repo can run:
the agent loop → ``TracingHook`` → ``install_otel_bridge`` →
``OTLPSpanExporter`` → HTTP/gRPC → real collector → file exporter →
JSON on disk.

The test is **opt-in** (skipped unless ``EDGEVOX_OTEL_DOCKER=1`` is
set). It expects the compose stack in ``deploy/otel-test/`` to be
running; the fixture below starts it if not, and tears it down at
module end.

Start manually::

    cd deploy/otel-test && docker compose up -d
    EDGEVOX_OTEL_DOCKER=1 pytest tests/harness/test_otel_docker.py -v
    docker compose down

Or let the fixture manage it (requires Docker on PATH).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

OTEL_DOCKER_ROOT = Path(__file__).resolve().parents[2] / "deploy" / "otel-test"
OTEL_HEALTH_URL = "http://127.0.0.1:13133"
OTEL_HTTP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
OTEL_TRACES_FILE = OTEL_DOCKER_ROOT / "received.json"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _collector_up() -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(OTEL_HEALTH_URL, timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


pytestmark = pytest.mark.skipif(
    os.environ.get("EDGEVOX_OTEL_DOCKER") != "1",
    reason="Docker-backed OTel test is opt-in; set EDGEVOX_OTEL_DOCKER=1 to enable.",
)


@pytest.fixture(scope="module")
def otel_collector_container():
    """Make sure the collector is running; start it if not; clean up
    (only if we started it) at module end. The opt-in env var above
    guards this — in normal CI the whole module is skipped."""
    if not _docker_available():
        pytest.skip("docker not on PATH")

    started_by_us = False
    if not _collector_up():
        # Reset the traces file so assertions don't pick up earlier
        # test runs.
        OTEL_TRACES_FILE.write_text("")
        result = _run(["docker", "compose", "up", "-d"], cwd=OTEL_DOCKER_ROOT)
        if result.returncode != 0:
            pytest.skip(f"docker compose up failed: {result.stderr.strip()}")
        started_by_us = True
        # Wait for health.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if _collector_up():
                break
            time.sleep(0.5)
        else:
            pytest.skip("OTel collector did not become healthy within 30 s")

    try:
        yield
    finally:
        if started_by_us:
            _run(["docker", "compose", "down"], cwd=OTEL_DOCKER_ROOT)


def _tail_received_spans(since: float) -> list[dict]:
    """Read every JSON record in the file-exporter output whose
    ``resourceSpans[…].scopeSpans[…].spans[…].startTimeUnixNano`` is
    strictly greater than ``since`` (nanoseconds). Returns the
    flat list of span dicts."""
    text = OTEL_TRACES_FILE.read_text(encoding="utf-8", errors="replace")
    spans: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rs in record.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for s in ss.get("spans", []):
                    try:
                        start = int(s.get("startTimeUnixNano") or 0)
                    except (TypeError, ValueError):
                        start = 0
                    if start > since:
                        spans.append(s)
    return spans


class TestDockerOTelCollector:
    def test_tracing_hook_span_reaches_collector(self, otel_collector_container):
        pytest.importorskip("opentelemetry")
        pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from edgevox.agents import AgentContext, LLMAgent, TracingHook
        from edgevox.agents.tracing_otel import _reset_bridge_state, install_otel_bridge

        from .conftest import ScriptedLLM, reply

        _reset_bridge_state()
        since_ns = time.time_ns()

        provider = TracerProvider(resource=Resource.create({"service.name": "edgevox-docker-test"}))
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=OTEL_HTTP_ENDPOINT)))
        # OTel's global provider is single-set; first caller in this
        # process wins. For the docker test we tolerate an existing
        # provider (use it if present, add our exporter onto it).
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            existing.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=OTEL_HTTP_ENDPOINT)))
            active = existing
        else:
            trace.set_tracer_provider(provider)
            active = provider

        ctx = AgentContext()
        install_otel_bridge(ctx.bus, service_name="edgevox-docker-test")

        agent = LLMAgent(
            name="docker-span",
            description="docker OTel smoke",
            instructions="Reply briefly.",
            tools=[],
            hooks=[TracingHook(service_name="edgevox-docker-test")],
        )
        agent.bind_llm(ScriptedLLM([reply("done")]))
        agent.run("go", ctx)

        active.force_flush(5_000)

        # File-exporter uses a 100 ms batch processor inside the
        # collector — poll for up to 5 s waiting for our span to land.
        deadline = time.monotonic() + 5.0
        found: list[dict] = []
        while time.monotonic() < deadline:
            received = _tail_received_spans(since_ns)
            found = [s for s in received if s.get("name") == "docker-span"]
            if found:
                break
            time.sleep(0.1)

        assert found, f"collector didn't record our span; received {len(_tail_received_spans(since_ns))} spans total"
        # Sanity-check attributes the bridge forwards.
        span = found[0]
        attrs = {a["key"]: a["value"] for a in span.get("attributes", [])}
        assert "edgevox.span_id" in attrs
        # Service name comes from the resource, not span attributes.
        resource_attrs = {}
        raw = OTEL_TRACES_FILE.read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in record.get("resourceSpans", []):
                for ra in (rs.get("resource") or {}).get("attributes", []):
                    resource_attrs[ra["key"]] = ra["value"]
        svc = resource_attrs.get("service.name", {}).get("stringValue") if resource_attrs else ""
        assert "edgevox-docker-test" in (svc or "")
