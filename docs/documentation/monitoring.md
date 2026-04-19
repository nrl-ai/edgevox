# Monitoring & Logging

Everything an EdgeVox agent does is observable through four channels, all of which you can tap without patching the framework:

| Channel | Source | What you get |
|---|---|---|
| **Structured events** | `ctx.bus` (`EventBus`) | `AgentEvent` stream — agent start/end, tool calls, skills, handoffs, hook decisions, tracing spans |
| **OTel spans** | `TracingHook` + `install_otel_bridge` | OpenTelemetry-compatible traces sent to any OTLP exporter (Jaeger, Tempo, Honeycomb, OTel Collector, console, in-memory) |
| **Audit jsonl** | `AuditLogHook` | One JSON-per-line record per tool/skill call — drop into Vector, Filebeat, Fluent Bit |
| **Stdlib logging** | `logging.getLogger("edgevox")` and children | Framework diagnostics at DEBUG / INFO / WARNING / ERROR |

None of these are enabled by default. You compose exactly the set you need.

---

## Quick start — local debugging

See every event a turn produces, printed to the console:

```python
from edgevox.agents import AgentContext, LLMAgent, TracingHook, TimingHook, EchoingHook

agent = LLMAgent(
    ...,
    hooks=[TracingHook(), TimingHook(), EchoingHook()],
)
ctx = AgentContext()
ctx.bus.subscribe_all(lambda e: print(f"[{e.kind}] {e.agent_name}: {e.payload}"))
agent.run("hello", ctx)
```

You'll see something like:

```
[span_start] my-agent: {'service.name': 'edgevox', 'name': 'my-agent', 'trace_id': 'abc…', 'span_id': 'def…', ...}
[agent_start] my-agent: {'task': 'hello'}
[tool_call]   my-agent: ToolCallResult(name='greet', result='hi there', ...)
[agent_end]   my-agent: {'reply': 'hi there'}
[span_end]    my-agent: {'trace_id': 'abc…', 'duration_ns': 173412832, 'reply_len': 7, ...}
```

That's the whole observability surface — every richer setup below is just a fancier subscriber.

---

## Tracing with OpenTelemetry

### 1. Install the extra

```bash
pip install 'edgevox[observability]'
```

Pulls in `opentelemetry-api` + `opentelemetry-sdk` (both Apache-2).

### 2. Configure the tracer provider once at app boot

Standard OTel setup — `BatchSpanProcessor` is the right default because it queues spans in a background thread and batches exports, so `span.end()` returns immediately.

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "my-voice-agent"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")))
trace.set_tracer_provider(provider)
```

Swap `OTLPSpanExporter` for any of:

- `ConsoleSpanExporter` — stdout for local debug
- `opentelemetry.exporter.jaeger.thrift.JaegerExporter` — direct to a Jaeger instance
- `opentelemetry.exporter.zipkin.json.ZipkinExporter` — Zipkin
- `honeycomb-beeline` / vendor SDK — any vendor that speaks OTel
- `InMemorySpanExporter` from `opentelemetry.sdk.trace.export.in_memory_span_exporter` — tests

### 3. Wire the bridge

```python
from edgevox.agents import LLMAgent, AgentContext, TracingHook
from edgevox.agents.tracing_otel import install_otel_bridge

agent = LLMAgent(..., hooks=[TracingHook(service_name="my-voice-agent")])
ctx = AgentContext()
install_otel_bridge(ctx.bus, async_dispatch=True)   # recommended — see below
agent.run("hello", ctx)
```

### Synchronous vs async bridge dispatch

The `EventBus` fires subscribers **synchronously on the publisher's thread**. The bridge's default path calls `tracer.start_span` + `span.end()` inline:

- With `BatchSpanProcessor` on the provider side, this is fast — the processor just enqueues.
- With `SimpleSpanProcessor` on the provider side, export happens on the caller's thread, so a slow / unreachable collector can block the agent turn.

**`async_dispatch=True`** adds defence-in-depth: span events are enqueued on a bounded queue drained by a daemon worker thread. The publisher never waits on OTel.

```python
install_otel_bridge(
    ctx.bus,
    service_name="my-voice-agent",
    async_dispatch=True,     # off by default; on for production
    queue_size=4096,         # bounded — older entries dropped with WARNING on overflow
)
```

At app shutdown, drain the worker so in-flight spans reach the exporter:

```python
from edgevox.agents.tracing_otel import shutdown_async_dispatch
shutdown_async_dispatch(ctx.bus, timeout=5.0)
```

Production-grade: `async_dispatch=True` **plus** `BatchSpanProcessor`. The bridge doesn't block the agent; the processor batches exports over the wire.

### What's in a span

`TracingHook` emits one span per agent turn. Attributes forwarded to the span:

| attribute | meaning |
|---|---|
| `service.name` | passed to the hook constructor |
| `name` | the agent's `name` |
| `edgevox.span_id` | redundant but searchable in exporters that hide the raw id |
| `edgevox.agent_name` | the agent's name, again — for cross-cutting filters |
| `trace_id`, `parent_span_id` | carried on `AgentContext`; propagated through handoffs + `agent_as_tool` so a multi-agent turn appears as one connected tree |
| `duration_ns` (span_end) | monotonic-time measurement of the turn |
| `reply_len` (span_end) | character count of the reply — sketchy signal for "did something useful happen" |

Custom attributes: subclass `TracingHook` and override `__call__` to add anything you want on the `payload` dict passed to `ctx.emit("span_start", ...)`. The bridge forwards every non-standard key as a span attribute automatically.

### Run a local collector in Docker

Repo ships a ready-to-go compose stack at `deploy/otel-test/`:

```bash
cd deploy/otel-test
docker compose up -d    # OTel collector on :4317 (gRPC) + :4318 (HTTP), healthcheck on :13133
```

Point your app's `OTLPSpanExporter` at `http://localhost:4318/v1/traces`. The stack persists every received span to `deploy/otel-test/received.json` — handy for `cat | jq` debugging of exactly what your app sent.

### Send to Jaeger

Jaeger natively accepts OTLP since v1.35:

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
```

Point the exporter at `http://localhost:4318/v1/traces`, open http://localhost:16686 — every turn appears with the full tool-call tree.

---

## Audit logs

For compliance / security trails, wire `AuditLogHook`:

```python
from edgevox.agents.hooks_builtin import AuditLogHook

agent = LLMAgent(
    ...,
    hooks=[AuditLogHook(path="/var/log/my-agent/audit.jsonl")],
)
```

Writes one JSON-per-line record per tool / skill / event. Every line is valid JSON so downstream log shippers (Vector, Filebeat, Fluent Bit, Promtail) parse cleanly and forward to your SIEM / SIEM-adjacent store.

**Log rotation** isn't built in — use the OS's `logrotate` or ship the file to a rotating sink upstream. The hook opens the file in append mode so rotation via `truncate` works; rotation via `rename-then-recreate` requires restarting the agent.

---

## Structured framework logging

Every module logs via `logging.getLogger(__name__)`, so the root namespace `edgevox` carries the whole framework:

```python
import logging

# Silence framework chatter in production:
logging.getLogger("edgevox").setLevel(logging.WARNING)

# Route just edgevox logs to a dedicated file:
handler = logging.FileHandler("/var/log/my-agent/framework.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.getLogger("edgevox").addHandler(handler)

# Turn the hot path to DEBUG when diagnosing something:
logging.getLogger("edgevox.agents").setLevel(logging.DEBUG)
```

Important namespaces:

| logger | what it says |
|---|---|
| `edgevox.agents.base` | `LLMAgent._drive` loop — tool-call hops, hook decisions, retries |
| `edgevox.agents.memory` | memory store load / flush, compaction triggers |
| `edgevox.agents.multiagent` | blackboard watchers, BackgroundAgent lifecycle |
| `edgevox.agents.tracing_otel` | bridge wiring, async-dispatch queue pressure |
| `edgevox.llm.llamacpp` | LLM load + gpu-layer decisions |
| `edgevox.audio._original` | recorder / player / VAD transitions |
| `edgevox.stt`, `edgevox.tts` | backend load + inference timing |

The framework never `print()`s — if you see unexpected output on stdout, that's your code, not ours. There's a regression test (`tests/harness/test_logging_behavior.py::TestNoStrayPrints`) guarding this.

---

## Metrics

EdgeVox doesn't ship a metrics exporter — it ships the event stream, and you build metrics on top.

### Two patterns that land cleanly

**Subscribe a counter directly to the bus:**

```python
from prometheus_client import Counter, Histogram

tool_calls = Counter("edgevox_tool_calls_total", "Tool dispatch count", ["agent", "tool"])
turn_latency = Histogram("edgevox_turn_ms", "Turn wall-clock latency (ms)", ["agent"])

def _on_event(e):
    if e.kind == "tool_call":
        tool_calls.labels(agent=e.agent_name, tool=e.payload.name).inc()
    elif e.kind == "span_end":
        ms = e.payload.get("duration_ns", 0) / 1e6
        turn_latency.labels(agent=e.agent_name).observe(ms)

ctx.bus.subscribe_all(_on_event)
```

**Or let OTel do the work.** Install `opentelemetry-sdk-metrics`, configure a `MeterProvider` with a periodic exporter (Prometheus or OTLP), and attach a hook that records on the meter. Keeps traces + metrics in one pipeline.

---

## Debugging a single turn

**The all-in-one "what did my agent do just now" recipe** — dump everything to stdout and read chronologically:

```python
import json
from edgevox.agents import AgentContext, LLMAgent
from edgevox.agents.hooks_builtin import (
    AuditLogHook, DebugTapHook, EchoingHook, TimingHook, TracingHook,
)

ctx = AgentContext()

def _pretty(e):
    payload = e.payload
    if hasattr(payload, "__dict__"):
        payload = vars(payload)
    try:
        print(f"[{e.kind:>14}] {e.agent_name:>12}  {json.dumps(payload, default=str)[:180]}")
    except Exception:
        print(f"[{e.kind:>14}] {e.agent_name:>12}  {payload!r:.180}")

ctx.bus.subscribe_all(_pretty)

agent = LLMAgent(
    ...,
    hooks=[
        TracingHook(),
        TimingHook(),
        EchoingHook(),
        DebugTapHook(),                                        # raw messages + raw reply
        AuditLogHook(path="/tmp/agent-audit.jsonl"),           # archived for later
    ],
)
agent.run("...", ctx)
```

Run once, read the single chronological stream, and you'll see the full turn: inputs → hook decisions → LLM call → tool dispatch → LLM reply → span close.

---

## What this doesn't cover

- **Distributed tracing across process boundaries** (e.g. ROS2 nodes, WebSocket server → client). The bridge exports to your OTel collector, which handles cross-process stitching as normal. If you need to propagate context into a non-OTel subsystem, pass `ctx.trace_id` through your transport header and reconstruct the parent link on the other side.
- **Privacy-aware logging.** `AuditLogHook` logs raw tool args + results. For PII-sensitive deployments, chain `OutputValidatorHook(validators=[pii_redactor()])` at `AFTER_TOOL` priority 30 so redaction runs before the audit event is emitted.
- **Alerting.** EdgeVox doesn't ship alerting rules; add them at your OTel collector / Prometheus / Loki layer.

See also:

- [`hooks.md`](/documentation/hooks) — full fire-point matrix, how `TracingHook` / `AuditLogHook` / `TimingHook` fit in
- [`configuration.md`](/documentation/configuration) — per-component opt-in / opt-out surface
- [`interrupt.md`](/documentation/interrupt) — cancel-token plumbing (related to turn latency)
