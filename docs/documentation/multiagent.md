# Multi-agent coordination

EdgeVox ships four orthogonal multi-agent primitives. Use one or all; they compose with the `Sequence` / `Fallback` / `Loop` / `Router` / `Parallel` workflows in `edgevox.agents.workflow`.

| Primitive | Purpose |
|---|---|
| `Blackboard` | thread-safe shared key/value with watchers |
| `AgentMessage` + bus helpers | direct agent-to-agent messaging on the existing `EventBus` |
| `BackgroundAgent` | wraps any `Agent` on a daemon thread, triggered by bus events |
| `AgentPool` | owns a set of agents sharing one bus + blackboard |

The workflow combinators are for explicit orchestration; these primitives are for emergent, event-driven coordination (a supervisor watching sensor data, a background planner reacting to user utterances, etc.).

## Blackboard

A dict + watchers with thread-safe set/get/delete. Nothing fancy: if you need CAS or transactions, build on top.

```python
bb = Blackboard()
bb.set("last_sensor_reading", 37.5)
bb.watch("task_queue", lambda k, old, new: print(f"queue → {new}"))
```

Watchers fire synchronously by default after `set()` returns (after the lock is released); a slow watcher blocks the writer. For non-blocking fan-out, construct the blackboard with `Blackboard(async_watchers=True)` — watchers run on a private thread pool and `set()` returns immediately. Call `bb.close()` on shutdown to release the pool.

## AgentMessage + bus helpers

`send_message(bus, from_agent=…, to=…, content=…)` publishes an `AgentMessage` on the bus; `subscribe_inbox(bus, agent_name=…, handler=…)` filters messages addressed to that agent (or broadcast with `to="*"`). Correlation ids are generated automatically so a handler can pair replies.

## BackgroundAgent

Wraps any `Agent` in a dedicated thread that triggers when bus events match a predicate.

```python
def trigger(event: AgentEvent) -> str | None:
    if event.kind == "sensor_alert":
        return f"Investigate: {event.payload}"
    return None

bg = BackgroundAgent(
    agent=my_agent,
    trigger=trigger,
    subscribe_to="*",             # bus filter
    max_queue=32,                 # bounded queue backed by queue.Queue
    overflow="drop_oldest",       # or "drop_new"
    restart="transient",          # OTP-style supervisor policy
    max_restarts=5,
    restart_window_s=60.0,
)
bg.start(ctx, bus)
```

### OTP-style restart policies

Matches Erlang OTP `supervisor` semantics:

| `restart` | On clean exit | On crash |
|---|---|---|
| `"permanent"` | always restart | always restart |
| `"transient"` (default) | stop | restart |
| `"temporary"` | stop | stop |

`max_restarts` per `restart_window_s` caps how aggressively the supervisor loops on a crashing agent (equivalent to OTP's `MaxR / MaxT`). Exhaust the budget and the loop exits; `bg.restart_count` reports the total and `bg.dropped_events` reports queue overflow.

### Bounded queue

The queue is a stdlib `queue.Queue(maxsize=max_queue)`. `overflow="drop_oldest"` evicts FIFO under pressure; `overflow="drop_new"` preserves older events. `dropped_events` counts both cases — the most common real-world diagnostic ("why did my agent miss that alert?").

## AgentPool

A bundle of agents sharing one bus + blackboard. Foreground runs go through `pool.run(name, task)`; background agents via `pool.start_background(name, trigger=…)`.

```python
pool = AgentPool()
pool.register(router_agent)
pool.register(kitchen_agent)
pool.register(sensor_monitor)

pool.start_background("sensor_monitor", trigger=sensor_trigger)
result = pool.run("router", "What's the temperature?")
```

`pool.make_ctx(**overrides)` builds an `AgentContext` pre-wired with the shared bus + blackboard so every agent sees consistent shared state.

### Pool + Parallel — fan-out across the pool

`AgentPool` and the [`Parallel` workflow](/documentation/agents#parallel-fan-out-to-n-specialists-one-reduced-reply) compose cleanly:

```python
from edgevox.agents import AgentPool, LLMAgent, Parallel

pool = AgentPool()
pool.register(lookup_agent)
pool.register(weather_agent)
pool.register(calendar_agent)

morning = Parallel(
    name="morning",
    agents=[pool.get("lookup"), pool.get("weather"), pool.get("calendar")],
)
result = morning.run("morning briefing?", ctx=pool.make_ctx())
```

`pool.make_ctx()` ensures every sub-agent sees the same bus + blackboard, so a background agent listening on `*` will observe the children's events. Combine with `bb.post_request(...)` when you want dynamic membership — the pool's agents pick up requests by capability rather than being enumerated up front.

## Debouncing triggers

`debounce_trigger(trigger, interval_s=…)` wraps any trigger so it fires at most once per interval — useful for sub-second sensor streams:

```python
bg = BackgroundAgent(
    agent=planner,
    trigger=debounce_trigger(my_trigger, interval_s=1.0),
)
```

## Coordination patterns

### Orchestrator + workers

The router emits a `handoff_to_<worker>` call; the target runs with a fresh `Session` and shared LLM. See [`agent-loop`](/documentation/agent-loop) for the handoff mechanics.

### Supervisor + volunteers (blackboard pattern)

The supervisor posts a task on the blackboard; subordinate agents watch the key and self-select based on their capability. Use `bb.post_request(request_key, task, timeout=…)` for the common request/reply shape — it returns a `concurrent.futures.Future` that resolves the moment a volunteer calls `bb.reply_to(request, result)`:

```python
fut = bb.post_request("plan.request", {"goal": "pick cup"}, timeout=5.0)
plan = fut.result()  # blocks up to 5 s, raises TimeoutError on miss
```

A unique `reply_key` is allocated per call so concurrent requests on the same `request_key` never collide. The reply watcher is auto-unsubscribed on resolution or cancel.

### Event-driven background agent

A BackgroundAgent subscribed to `"*"` or a specific kind reacts to bus events without the caller knowing it exists. Great for continuous observers (safety monitor, conversation summariser).

## Roadmap

Already shipped (see `Supervisor`, `Orchestrator` above) plus the future direction:

- **Declarative `Handoff`** with `state_update` + `parallel_targets` — LangGraph `Command`/`Send` semantics.
- **`Supervisor` builder** — LLM-routed conditional dispatch.
- **`Orchestrator` builder** — plan-schema fanout with per-subtask tool filtering (Anthropic orchestrator-worker pattern).
- **Typed `AgentMessage`** — from/to/topic/correlation_id/payload/timestamp groundwork for a future A2A adapter.

## See also

- [`agent-loop`](/documentation/agent-loop) — handoff short-circuit.
- [`hooks`](/documentation/hooks) — ctx-level hooks propagate through subagents.
