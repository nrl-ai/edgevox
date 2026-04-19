# Tool calling

EdgeVox has to work with GGUF models that emit tool calls in 7+ different wire formats — Hermes `<tool_call>…</tool_call>`, Qwen 2.5's stricter variant, Llama 3.2's `<|python_tag|>{…}` + bare JSON, Mistral's `[TOOL_CALLS]`, "pythonic" `[fn(arg=val), …]`, Salesforce xLAM's JSON array, Granite 4's unquoted-JSON, and Gemma's inline `<|tool_call>…<tool_call|>` markers. This page documents the parser chain and the planned shift to grammar-constrained decoding.

## The chain today

```mermaid
flowchart TD
    A[LLM response] -->|structured tool_calls?| S[Dispatch]
    A -->|free-form content| B["parse_tool_calls_from_content"]
    B --> C["Try detectors on raw content<br/>(Qwen3 tool calls live inside think blocks)"]
    C -->|match| S
    C -->|miss| D[Strip think blocks]
    D --> E[Retry detectors on stripped]
    E -->|match| S
    E -->|miss| F[chatml / bare-JSON regex]
    F -->|match| S
    F -->|miss| G[Gemma inline / plain name kwargs]
    G -->|match| S
    G -->|miss| R[Plain text reply]
```

Four things to notice:

1. **Raw-first detection.** The detector chain runs on the **raw** content before `<think>` blocks are stripped. Qwen3-Instruct emits tool calls inside `<think>` (confirmed upstream bug [llama.cpp#20837](https://github.com/ggml-org/llama.cpp/issues/20837)); the raw-first order recovers them. The user-facing reply is always the stripped text so chain-of-thought never reaches TTS.
2. **Detector chain is per-preset.** `ModelPreset.tool_call_parsers: tuple[str, ...]` names detectors in priority order; `resolve_preset()` validates every name against the registered `DETECTORS` dict, so a typo fails loudly at preset load instead of silently skipping detection.
3. **Fallback regex.** The Gemma inline / plain-call path only dispatches when a tool-name allowlist matches. Code fences (```…```) are stripped first so example code the model quotes isn't accidentally dispatched.
4. **Mistral call-ids round-trip.** `ToolCallItem.id` carries the model-emitted 9-char id through the parser chain; `LLMAgent._drive` threads it verbatim into the follow-up `role="tool"` message. Synthesised ids (`"<name>_<idx>"`) are used only when the wire format didn't surface one. Mistral requires this; other models tolerate either.

## Detector registry

Detectors are vendored from [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0, see `NOTICE`) and wrapped in a small local registry:

| Name | Format | Source models |
|---|---|---|
| `hermes` | `<tool_call>…</tool_call>` | Hermes, Qwen, generic chatml |
| `qwen25` | strict `<tool_call>\n…\n</tool_call>` | Qwen2.5 / Qwen3 |
| `llama32` | `<|python_tag|>{…}` + bare JSON | Llama 3.1/3.2/3.3 |
| `mistral` | `[TOOL_CALLS] [{…}]` | Mistral Nemo / Ministral |
| `pythonic` | `[fn(arg=val), …]` | Llama-4, Llama-3.2 pythonic |
| `xlam` | JSON array, optionally fenced | Salesforce xLAM / Hammer |
| `granite` | unquoted-JSON | Granite 4 |

Register your own via `register_detector("my-format", MyDetector)` (subclass `BaseFormatDetector`). Presets reference it by name in `tool_call_parsers=(...)`.

## Think-block handling

```python
raw = '<think>Let me check.\n<tool_call>{"name":"get_time","arguments":{}}</tool_call>\n</think>One moment.'
calls, cleaned, fallback = parse_tool_calls_from_content(raw)
# calls -> [{"id":"...","function":{"name":"get_time","arguments":"{}"}}]
# cleaned -> "One moment."
# fallback -> True (chat template didn't emit structured tool_calls)
```

## `fallback_mode`

When the chat template didn't emit structured `tool_calls` (the common SLM case), the parser returns `fallback_mode=True`. `LLMAgent._drive` then injects the tool results as a synthetic **user** message rather than the `tool` role:

> `(system: tool results — get_time -> "noon". Now answer the previous request in one short sentence.)`

This is what keeps the loop model-agnostic: GGUFs that do emit structured tool_calls get the canonical round-trip; ones that don't get the synthetic recovery path.

## Grammar-constrained decoding (`tool_choice_policy`)

llama.cpp's GBNF sampler can mask invalid next tokens at every decode step so the model is *forced* to emit a syntactically valid tool call. EdgeVox builds the grammar from `ToolRegistry.openai_schemas()` via `edgevox.llm.grammars`:

| Strategy | Helper | Use case |
|---|---|---|
| Force a tool call | `tool_call_grammar(tools)` | `tool_choice="required"` — the model must call something |
| Force a specific tool | `single_tool_grammar(tool)` | `tool_choice={"name": "X"}` |
| Reply OR tool | `reply_or_tool_grammar(tools)` | `tool_choice="auto"` with malformed-JSON elimination |

`GrammarCache` memoises compiled `LlamaGrammar` objects keyed by registry fingerprint, so the per-turn cost is microseconds.

The agent loop opts in via `LLMAgent(tool_choice_policy=…)`:

| `tool_choice_policy` | Hop 0 | Subsequent hops | When to use |
|---|---|---|---|
| `"auto"` (default) | unconstrained | unconstrained | mature 7B+ models |
| `"required_first_hop"` | `tool_choice="required"` + grammar | `"auto"` (so reply can land) | **canonical SLM loop-break** — forces the model to call something on hop 0, then releases for the answer |
| `"required_always"` | `"required"` + grammar | `"required"` + grammar | rare; mostly for benchmarking |

Under `required_first_hop` the malformed-JSON / wrong-tool-name failure modes that `hooks_slm.py` detects today drop to near-zero — the grammar makes them impossible. The hooks remain registered as a safety net for the cases grammar can't cover (semantic looping, echoed payloads, empty arguments).

```python
from edgevox.agents import LLMAgent
from edgevox.llm.hooks_slm import default_slm_hooks

agent = LLMAgent(
    name="kitchen",
    description="Home-kitchen assistant",
    instructions="Help in the kitchen.",
    tools=[get_time, set_light, set_temp],
    tool_choice_policy="required_first_hop",  # forces structured first-hop call
    hooks=default_slm_hooks(),                # safety net
)
```

`llguidance` (merged into llama.cpp upstream Feb 2025) is picked up automatically when llama.cpp was built with `-DLLAMA_LLGUIDANCE=ON` — same API surface, ~10× faster grammar masking.

## Testing

- `tests/test_llm_tool_parsers.py` — unit coverage per detector.
- `tests/tool_parsing/test_qwen3_think_block.py` — raw-first parse order.
- `tests/tool_parsing/test_mistral_ids.py` — 9-char id round-trip.
- `tests/bfcl/` — BFCL v3 AST-eq regression harness (`ast_eq.py` + `fixtures.json` + `test_parser_chain.py`).

## See also

- [`agent-loop.md`](./agent-loop.md) — where the parser chain slots into `_drive`.
- [SGLang function_call](https://docs.sglang.io/advanced_features/tool_parser.html) — upstream source for vendored detectors.
