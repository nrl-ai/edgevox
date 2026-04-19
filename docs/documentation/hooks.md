# Hooks

The hook system is how EdgeVox lets you change agent-loop behaviour without patching core. A hook is a small callable that fires at one or more of six **fire points** inside [`LLMAgent.run()`](/documentation/agent-loop); it inspects the payload, optionally modifies it, and tells the loop whether to continue, replace the payload, or end the turn.

Everything in `hooks_builtin.py` + `hooks_slm.py` is a composition of exactly this contract.

## The six fire points

| Point | When | Typical payload | Typical use |
|---|---|---|---|
| `ON_RUN_START` | before any LLM call | `{"task": str}` | reset per-turn counters, inject memory, safety input-rail |
| `BEFORE_LLM` | each hop, before `llm.complete` | `{"messages": list, "tools": list, "hop": int}` | mutate system prompt, enforce token budget |
| `AFTER_LLM` | each hop, after parsing | `{"content": str, "tool_calls": list, "hop": int}` | rewrite reply, detect echoed payloads, output-rail |
| `BEFORE_TOOL` | per tool, pre-dispatch | `ToolCallRequest` | require confirmation, loop-hint, skip dispatch |
| `AFTER_TOOL` | per tool, post-dispatch | `ToolCallResult` | truncate, log episode, schema-retry enrichment |
| `ON_RUN_END` | once, after the turn resolves | `AgentResult` | persist session, audit, emit metrics |

## Hook Protocol

```python
class Hook(Protocol):
    points: frozenset[str]                       # which fire points to receive

    def __call__(
        self,
        point: str,                              # the exact fire point name
        ctx: AgentContext,                       # run context (see below)
        payload: Any,                            # point-specific shape
    ) -> HookResult | None: ...
```

`points` is a `frozenset` of fire-point constants from `edgevox.agents.hooks`. Returning `None` means "continue, no changes" (equivalent to `HookResult.cont()`).

Any callable that matches this shape works — no inheritance required. The `@hook(...)` decorator wraps a function:

```python
from edgevox.agents.hooks import hook, BEFORE_LLM

@hook(BEFORE_LLM)
def add_system_note(ctx, payload):
    msgs = list(payload["messages"])
    msgs[0]["content"] += "\nRemember: be brief."
    payload = dict(payload)
    payload["messages"] = msgs
    return HookResult.replace(payload, reason="brief reminder")
```

## HookResult

Three constructors for the three outcomes the loop honours:

| Constructor | `action` | What the loop does |
|---|---|---|
| `HookResult.cont()` | `CONTINUE` | proceed with the original payload |
| `HookResult.replace(payload, reason=...)` | `MODIFY` | swap the payload in-flight |
| `HookResult.end(reply, reason=...)` | `END_TURN` | bail out, use `reply` as the final reply |

`reason` is a short string surfaced in `AgentEvent.payload["reason"]` and `AgentResult.hook_ended` — use it for debugging and audit trails.

## Where hooks live

Two layers, fired in order at every point:

1. **Agent-level** — `LLMAgent(..., hooks=[...])`. Shared across every `run()` of that agent.
2. **Context-level** — `ctx.hooks.register(h)`. Scoped to one `AgentContext` / conversation.

Agent-level hooks fire first; ctx-level hooks see any modifications they made. This lets you ship agent-specific defaults while letting callers layer session-specific behaviour on top.

## Hook-owned state

Hooks that need per-turn state (fingerprint counters, retry budgets) store it under `ctx.hook_state[id(self)]`. Keying by `id(self)` gives each **instance** its own bag, so two `LoopDetectorHook()` objects on one context never share counts.

```python
class MyHook:
    points = frozenset({ON_RUN_START, AFTER_TOOL})

    def __call__(self, point, ctx, payload):
        if point == ON_RUN_START:
            ctx.hook_state[id(self)] = {"seen": 0}
            return None
        bag = ctx.hook_state[id(self)]
        bag["seen"] += 1
        ...
```

This replaced the old `ctx.session.state["__xxx__"]` magic-key pattern, which leaked hook internals into the user-visible `ctx.state` dict and caused silent collisions between two instances of the same hook class.

## Typed ctx fields

Hooks reach the running tool registry and LLM via typed fields on `AgentContext`, not scratchpad keys:

| Field | Who sets it | Typical use |
|---|---|---|
| `ctx.tool_registry` | `LLMAgent.run()` | schema lookup for error-repair hooks |
| `ctx.llm` | `LLMAgent.run()` | tokenizer-exact `estimate_tokens`, compaction |
| `ctx.interrupt` | caller | read `cancel_token`, subscribe to events |
| `ctx.memory`, `ctx.artifacts`, `ctx.blackboard` | caller | long-term memory, artifact store, shared state |

`ctx.state` is now user-only scratch. Hooks must not write framework plumbing there.

## Built-in hooks (`hooks_builtin.py`)

| Hook | Fires at | Purpose |
|---|---|---|
| `SafetyGuardrailHook(blocklist=…)` | `ON_RUN_START` | block-list / allow-list input rail |
| `PlanModeHook(confirm=[…], approver=…)` | `BEFORE_TOOL` | require confirmation before sensitive tools |
| `TokenBudgetHook(max_context_tokens=…)` | `BEFORE_LLM` | hard context-window cap with tokenizer-exact count |
| `ToolOutputTruncatorHook(max_chars=…)` | `AFTER_TOOL` | truncate oversized tool results |
| `MemoryInjectionHook(memory_store)` | `BEFORE_LLM` | append facts/episodes to system prompt (idempotent per turn) |
| `NotesInjectorHook(notes)` | `BEFORE_LLM` | inject the tail of a `NotesFile` |
| `ContextCompactionHook(compactor)` | `ON_RUN_START` | LLM-summarise middle turns when over budget |
| `EpisodeLoggerHook(memory_store)` | `AFTER_TOOL` | record tool outcomes as episodes |
| `AuditLogHook(path)` | `AFTER_LLM/AFTER_TOOL/ON_RUN_END` | JSONL event log for offline replay |
| `PersistSessionHook(session_store, session_id)` | `ON_RUN_END` | save `Session` to disk |
| `TimingHook()` | before/after LLM + tool | collect wall-clock timings |
| `EchoingHook()` | all six | print every fire point — debugging |

## SLM-hardening hooks (`hooks_slm.py`)

| Hook | Fires at | Purpose |
|---|---|---|
| `LoopDetectorHook(hint_after=1, break_after=2)` | `ON_RUN_START`, `BEFORE_TOOL` | fingerprint identical `(tool, args)` calls; hint on the 2nd, end-turn on the 3rd |
| `EchoedPayloadHook(fallback=…)` | `AFTER_LLM` | substitute a human-readable fallback when the model echoes a tool-result payload (markdown-fence aware) |
| `SchemaRetryHook(max_retries_per_tool=1)` | `ON_RUN_START`, `AFTER_TOOL` | rewrite argument-shape errors into a human-readable schema hint so the next hop can retry |

Compose the bundle with `default_slm_hooks()`; it's what you want on any model <4B that hasn't been specifically tool-call-finetuned.

## Hook order

Within one fire point, hooks fire in **priority order** — higher `priority` first, ties broken by registration order. Declare priority as a class or instance attribute, or pass it to `register()`:

```python
class MySafetyHook:
    points = frozenset({ON_RUN_START})
    priority = 100  # safety tier

agent.register_hook(MySafetyHook())
# or
ctx.hooks.register(MyObserver(), priority=0)
```

Recommended scale (convention, not enforced):

| Priority | Tier | Examples |
|---|---|---|
| 100 | Safety / rails | `SafetyGuardrailHook`, `PlanModeHook`, future LlamaGuard |
| 80 | Input-shape | `MemoryInjectionHook`, `NotesInjectorHook`, `TokenBudgetHook` |
| 60 | Detection | `LoopDetectorHook`, `EchoedPayloadHook` |
| 40 | Mutation | `SchemaRetryHook`, `ToolOutputTruncatorHook` |
| 0 | Observability (default) | `AuditLogHook`, `TimingHook`, `EpisodeLoggerHook` |

The `@hook(point, priority=…)` decorator accepts the same kwarg. `HookRegistry.at(point)` returns hooks in firing order — useful for introspection.

## Testing hooks

`tests/harness/conftest.py` ships a `ScriptedLLM` that returns pre-declared responses, so hook behaviour can be exercised deterministically:

```python
from tests.harness.conftest import ScriptedLLM, reply, call

def test_my_hook_short_circuits():
    llm = ScriptedLLM([reply("should not reach")])
    agent = LLMAgent("t", "", "", llm=llm, hooks=[MyHook()])
    result = agent.run("trigger")
    assert result.hook_ended == "blocked"
```

## See also

- [`agent-loop`](/documentation/agent-loop) — the loop itself.
- [`interrupt`](/documentation/interrupt) — how hooks cooperate with barge-in.
