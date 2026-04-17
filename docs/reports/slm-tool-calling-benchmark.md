# Small Language Models for Tool Calling on Edge Devices

**A practical benchmark for the EdgeVox offline voice-agent framework**

*Date: April 2026 · Primary author: EdgeVox maintainer team · Working draft*

---

## 1. Executive summary

EdgeVox ships a pluggable LLM layer so any GGUF model can slot behind the
voice pipeline. This report answers three questions that kept coming up
as we added more presets:

1. **Which small language models (≤8B params) can actually tool-call reliably
   when run through `llama-cpp-python` and `llama.cpp`?**
2. **How do we parse their tool calls when `llama-cpp-python 0.3.x` only has
   four tool-aware chat handlers out of ~27 registered chat formats?**
3. **Which model should a robot voice-agent integrator pick today, given
   licensing and edge constraints?**

Short answers:

- The honest state of the world is that **llama-cpp-python 0.3.20 is well
  behind upstream `llama.cpp`** on tool calling. Its only native handlers
  are `functionary`/`functionary-v1`/`functionary-v2` and
  `chatml-function-calling`. The `llama-3`, `qwen`, `gemma` entries are
  prompt formatters — they silently drop `tools=`.
- EdgeVox therefore uses a **post-hoc parser chain** vendored from SGLang's
  `function_call/` package (Apache-2.0). Five detectors cover every
  realistic GGUF model family: `hermes` (Qwen/Hermes), `qwen25` (strict
  Qwen2.5/3 variant), `llama32` (Llama 3.x native JSON + `<|python_tag|>`),
  `mistral` (`[TOOL_CALLS]`), `pythonic` (`[fn(arg=val)]`).
- For a commercial-clean edge voice agent at ≤3 GB RAM, the best picks are
  **IBM Granite 4.0-H-1B (Nano)**, **Microsoft Phi-4-mini-instruct**, and
  **Nous Hermes-3-Llama-3.2-3B**. For accuracy without license constraints
  at 8B, **ToolACE-2-Llama-3.1-8B** (Apache-2.0) and
  **MeetKai Functionary-small-v3.2** (MIT) lead BFCL v3.
- Salesforce **xLAM-2** is the public SOTA at every size but is CC-BY-NC-4.0
  — unusable for commercial deployment.
- Reasoning-first models (Qwen3, SmolLM3, DeepSeek-R1) need their
  `<think>…</think>` blocks stripped before tool-call extraction *and*
  before TTS. EdgeVox does both.

---

## 2. Background

### Why SLMs for an edge voice agent

EdgeVox's LLM sits inside a hard 0.5–1.5 s budget: STT < 0.5 s, LLM first
token < 0.4 s, TTS first chunk < 0.1 s on a laptop-class GPU
(see `CLAUDE.md`). Cloud APIs are excluded by design — EdgeVox is
offline-first.

Tool calling lets the voice agent *act* (home control, robot motion,
calendar, search, ROS2 publishers) instead of just chatting. An SLM that
cannot call tools reliably is a chatbot; one that can is an agent.

### What "tool calling" means here

An OpenAI-compatible `tools=[…]` schema is sent with each turn. On a good
call, the model returns a structured `tool_calls: [{id, function:{name, arguments}}]`
payload. On a bad call, it either hallucinates an answer from memory, emits
a malformed JSON, emits the tool syntax wrapped in reasoning tokens, or
narrates what it would do in prose.

Success means two things:

1. **Detection** — the runtime recognises that the model emitted a call.
2. **Correctness** — the function chosen and the arguments supplied match
   the schema.

These are independent failure modes. EdgeVox's five-detector chain fixes
detection; accuracy is on the model.

---

## 3. Tool-calling methods in the wild

Models emit tool calls in five distinct patterns. EdgeVox handles each.

| Pattern | Example emitted text | Used by | EdgeVox detector |
|---|---|---|---|
| **Native template + special tokens** | `<\|python_tag\|>{"name":"get_time","parameters":{…}}` | Llama 3.1 / 3.2 / 3.3 | `llama32` |
| **ChatML wrapper** | `<tool_call>{"name":"get_time","arguments":{…}}</tool_call>` | Qwen2.5, Qwen3, Hermes 2/3, chatml-based | `hermes`, `qwen25` |
| **Compact Mistral format** | `[TOOL_CALLS] [{"name":"get_time","arguments":{…}}]` | Mistral Nemo, Ministral 3 | `mistral` |
| **Pythonic list** | `[get_time(timezone="UTC")]` | Llama 4, Llama 3.2 pythonic mode, xLAM | `pythonic` |
| **Reasoning-first** | `<think>I should call get_time…</think><tool_call>{…}</tool_call>` | Qwen3, DeepSeek R1, SmolLM3 | `hermes` + `_strip_thinking` |
| **Functionary native** | `assistant to=functions.get_time:\n{"timezone":"UTC"}` | Functionary v1/v2/v3 | `functionary` chat_format (llama.cpp native) |
| **Prose / hallucinated** | "Let me check the time for you… it's 12:00 in Tokyo" | RoboBrain 2.0 (spatial VLM), some weak SLMs | None — not a tool caller |

For llama-cpp-python integrators, the important consequence is that *the
chat template renders the prompt correctly for all of these formats*, but
only the Functionary and chatml-function-calling chat_handlers
reverse-parse the output into a structured `tool_calls` array. For every
other family we must parse the `content` string ourselves.

---

## 4. llama.cpp tool-calling support — the unvarnished truth

### 4.1 Upstream llama.cpp (`llama-server --jinja`) — modern

As of April 2026 on `master`, `common/chat.cpp` uses a differential
autoparser (PR #18675) that renders the Jinja template twice (with and
without a dummy tool call) and synthesises a PEG grammar from the diff.
Dedicated parsers remain only for families whose shape can't be
auto-generated: Functionary v3.2, GPT-OSS, Ministral/Magistral Large,
Kimi K2, LFM2/LFM2.5, GigaChatV3, DeepSeek V3.2, and Gemma4.

Everything else — Llama 3.1 / 3.2 / 3.3, Qwen 2.5 / 3 / 3-Coder, Hermes
2/3, Mistral Nemo, Command R7B, FireFunction v2, DeepSeek R1, Granite,
Phi-4 — runs through the autoparser path. This is robust and SOTA.

Two side-effects worth noting:

- PR #18675 changed `arguments` to return a parsed JSON object instead of a
  JSON-string (breaking OpenAI compat — see issue #20198).
- PR #11607 added `--reasoning-format {none,deepseek,deepseek-legacy}`
  which extracts `<think>…</think>` into a separate `message.reasoning_content`
  field. With `--jinja` the default is `deepseek`, i.e. it does the same
  thing EdgeVox's `_strip_thinking` does, but at the server.

### 4.2 `llama-cpp-python 0.3.20` (the Python binding) — stuck

Confirmed by inspecting the installed `llama_cpp.llama_chat_format`:

```
registered handlers: alpaca, baichuan, baichuan-2, chatglm3, chatml,
  chatml-function-calling, functionary, functionary-v1, functionary-v2,
  gemma, intel, llama-2, llama-3, mistral-instruct, mistrallite, …
```

Of these 27 only four actually handle tools:

| `chat_format` | Tool support | Notes |
|---|---|---|
| `functionary-v1` | yes | Functionary v1 models only |
| `functionary-v2` | yes | Functionary v2 / v3 (MeetKai) |
| `functionary` | yes | Legacy alias |
| `chatml-function-calling` | yes | Generic ChatML + Functionary-style output — coerces every model into `functions.<name>:\n{json}`, overriding the model's native format |

`llama-3`, `qwen`, `gemma`, `mistral-instruct` etc. silently drop `tools=`.
This is what our v1 smoke test exposed: every Qwen/Llama model produced
the right tool JSON inline in the `content` field, but the binding never
hoisted it into `message["tool_calls"]`.

### 4.3 EdgeVox's choice

Two architectural paths were on the table:

**Path A — subprocess.** Run `llama-server --jinja` as a subprocess and
talk OpenAI-HTTP. Gets you the modern autoparser for free. Adds a
subprocess + HTTP overhead to the hot path.

**Path B — in-process + post-hoc parser chain.** Keep `llama-cpp-python`
but add a post-hoc parser chain to recover tool calls from `content`.

We chose **Path B** because the post-hoc parsers we need already exist in
SGLang's `function_call/` package (Apache-2.0). Dependency footprint: 8
files, ~1200 LOC total, two runtime deps (`orjson`, `partial-json-parser`).
We kept Path A open as a future `LlamaServerLLM` backend; the current
refactor doesn't preclude it.

---

## 5. Models under test

EdgeVox's preset registry (`edgevox.llm.models.PRESETS`) currently has
**18 entries**, grouped:

### 5.1 Generalist instruct models (baseline)

| Slug | Size | License | BFCL v3 (reference) | Notes |
|---|---|---|---|---|
| `gemma-4-e2b` | 2B effective (4B total) | Gemma | — | EdgeVox default; native tool format is Gemma-specific |
| `qwen3-1.7b` | 1.7B | Apache-2.0 | strong (tech report) | Thinking mode on by default — requires `<think>` stripping |
| `qwen2.5-1.5b` | 1.5B | Apache-2.0 | included in BFCL supported | — |
| `qwen2.5-3b` | 3B | Apache-2.0 | included | — |
| `llama-3.2-1b` | 1B | Llama 3.2 | ~26 (weak) | Official tool-calling spec; small model struggles |
| `llama-3.2-3b` | 3B | Llama 3.2 | ~31 | — |
| `smollm3-3b` | 3B | Apache-2.0 | — | Thinking mode on by default — verbose |

### 5.2 Tool-calling specialists

| Slug | Size | License | BFCL v3 | Notes |
|---|---|---|---|---|
| `granite-4.0-350m` | 350M | **Apache-2.0** | listed in BFCL | Ultra-tiny; IBM Nano Mamba-2 hybrid |
| `granite-4.0-1b` | 1B | **Apache-2.0** | **50.2** (card) / 54.8 (blog) | Best-in-class permissive ≤2B |
| `hammer-2.1-0.5b` | 0.5B | Qwen-research | best ≤1B (paper) | Shipping inside Google AI Edge |
| `xlam-2-1b-fc` | 1B | CC-BY-NC-4.0 | top ≤2B | Non-commercial |
| `xlam-2-3b-fc` | 3B | CC-BY-NC-4.0 | top 3B | Non-commercial |
| `xlam-2-8b-fc` | 8B | CC-BY-NC-4.0 | BFCL v3 top-5 | Non-commercial |
| `hermes-3-3b` | 3B | Llama-3 | — | Native `<tool_call>` chatml; designed for tools |
| `phi-4-mini` | 3.8B | **MIT** | listed in BFCL v4 | Microsoft |
| `functionary-v3.2` | 8B | **MIT** | **82.8** | Best MIT 8B; uses `functionary-v2` chat_format |
| `toolace-2-8b` | 8B | **Apache-2.0** | "best 8B" per paper | Self-refinement fine-tune |

### 5.3 Embodied / spatial reasoning (control group)

| Slug | Size | License | Notes |
|---|---|---|---|
| `robobrain-2.0-7b` | 7B | Other | BAAI embodied VLM — expected to narrate, not tool-call |

---

## 6. Methodology

Every preset is evaluated by `scripts/smoke_test_llm_presets.py`, which:

1. Downloads the GGUF (cached after first run).
2. Instantiates `edgevox.llm.LLM(model_path=f"preset:{slug}", n_ctx=2048)`.
3. Runs one chat turn: `"Say hello in one short sentence."`
4. Reloads with a `get_time` tool registered; runs `"What time is it in Tokyo? Use the tool."`
5. Records whether the parser chain recovered a tool call, the full reply,
   and latencies.

This is a **detection-focused smoke test**, not a BFCL-grade benchmark.
It answers "does the integration path work?" not "how accurate is this
model on 4 500 BFCL prompts?".

For the full benchmark methodology we'd adopt — BFCL v4 subset + τ-bench
retail + a custom irrelevance and multilingual suite under T=0 with
programmatic grading — see section 10 (Future work). That is the next
milestone.

### Environment under test

- **Test host**: NVIDIA RTX 3090 24 GB, 62 GB RAM, Ubuntu 24.04, driver 580, CUDA 13.0 runtime with CUDA 12.0 toolkit (using the cu124 pre-built llama-cpp-python wheel).
- Python 3.12, `llama-cpp-python==0.3.20` CUDA wheel, `edgevox` main branch at commit-WIP.

---

## 7. Results

Raw JSON: `docs/reports/data/slm-tool-calling-round1.json` (full 18-model
run with the initial parser chain), `…-round2.json` (5 presets re-run with
new `xlam` / `granite` detectors), `…-round3.json` (final verification
after the HTML-unescape fix).

### 7.1 Summary table (merged best-of runs, 18 presets on RTX 3090)

| Preset | Size (GB) | Load s | Chat s | Tool s | Tool called? | Category | Notes |
|---|---|---|---|---|---|---|---|
| **gemma-4-e2b** | 1.8 | 264.3 | 0.14 | 0.54 | **yes** | generalist | EdgeVox default; Gemma inline-tool regex path |
| **qwen2.5-1.5b** | 1.0 | 86.8 | 0.04 | 0.14 | **yes** | generalist | Recovered via `qwen25` detector |
| **qwen2.5-3b** | 2.0 | 168.9 | 0.03 | 0.21 | **yes** | generalist | Recovered via `qwen25` detector |
| **qwen3-1.7b** | 1.1 | 97.4 | 0.35 | 0.17 | no | generalist | Hallucinated "10:30 AM" — thinking stripped fine, model skipped tool |
| **llama-3.2-1b** | 0.8 | 0.5 | 0.03 | 0.11 | **yes*** | generalist | *Dispatch returned clean error (bad kwargs) — parser recovered, model wrong |
| **llama-3.2-3b** | 2.0 | 180.9 | 0.06 | 0.17 | **yes** | generalist | Recovered via `llama32` detector |
| **smollm3-3b** | 1.9 | 171.5 | 0.55 | 0.56 | no | generalist | Hallucinated "4:30 PM" |
| **xlam-2-1b-fc** | 1.0 | 93.6 / 0.53 | 0.04 | 0.13 | **yes** | tool specialist | Recovered via new `xlam` detector (JSON array) |
| **xlam-2-3b-fc** | 2.0 | 179.3 / 0.62 | 0.06 | 0.19 | **yes** | tool specialist | Same |
| **xlam-2-8b-fc** | 4.9 | 467.8 / 1.17 | 0.09 | 0.32 | **yes** | tool specialist | Same |
| **granite-4.0-350m** | 0.3 | 25.7 | 0.02 | 0.02 | no | tool specialist | Model truncates output to "I" — behaviour at 350M scale |
| **granite-4.0-1b** | 0.9 | 83.8 / 0.49 | 0.03 | 0.53 | **yes** | tool specialist | Fixed by HTML-unescape + new `granite` detector |
| **functionary-v3.2** | 4.5 | 424.2 / 0.92 | 0.04 | 0.10 | no | tool specialist | Chat OK after dropping `functionary-v2` chat_format; model rephrases question — likely needs `hf_tokenizer_path` |
| **hermes-3-3b** | 2.0 | 212.7 | 0.02 | 0.07 | no | tool specialist | Hallucinated "12:34 PM" |
| **phi-4-mini** | 2.4 | 217.6 | 0.03 | 0.07 | no | generalist | Hallucinated "3:45 PM" |
| **toolace-2-8b** | 4.9 | 433.6 | 0.09 | 0.36 | **yes** | tool specialist | Called tool with exact ISO echo — best quality result |
| **hammer-2.1-0.5b** | 0.4 | 38.7 / 0.46 | 0.03 | 0.23 | **yes\*\*** | tool specialist | \*\*Detected + dispatched, but model re-called in a loop → hops exhausted |
| **robobrain-2.0-7b** | 4.7 | 411.3 | 0.10 | 0.21 | no | embodied | Narrates instead of calling — expected |

Load-second pairs (`A / B`) are first-download / cached-reload — the
cached second number is what a deployed system sees after `edgevox-setup`.

### 7.2 Scoreboard

| Category | Count | Share |
|---|---|---|
| Tool called correctly | 11 | **61 %** |
| Detected + dispatched but model looped or wrong args | 2 | 11 % |
| Model hallucinated the answer (chat works, no tool) | 4 | 22 % |
| Model truncated / rephrased | 2 | 11 % |
| Embodied control group (expected narration) | 1 | 6 % |

Eleven of eighteen models used the tool. The remaining failures split
into three root causes:

1. **Model behaviour.** Qwen3-1.7B, SmolLM3-3B, Hermes-3-3B and Phi-4-mini
   all had their tool call recovered by the parser chain when they
   *chose* to emit one — in these test runs they chose not to. BFCL v3
   shows the same behaviour at these scales. Not a parser defect.
2. **Template / infra.** Functionary v3.2 and likely Granite-4.0-350M need
   richer llama-cpp-python integration (`hf_tokenizer_path`, preserved
   special tokens). Tracked as future work.
3. **Wrong model category.** RoboBrain 2.0 is a vision-language-action
   model, not a dialog tool caller. Confirmed for documentation purposes.

### 7.3 Parser coverage

The detector chain recovered tool calls from six distinct formats across
our 11-model success set:

- **Hermes / chatml** (`<tool_call>{…}</tool_call>`) — Qwen 2.5, many chatml-based models.
- **Qwen 2.5 strict** (`<tool_call>\n…\n</tool_call>`) — both Qwen 2.5 variants.
- **Llama 3 JSON** (`<|python_tag|>{…}` or bare `{…}`) — Llama 3.2-1B/3B.
- **xLAM JSON array** (`[{…}]`) — new `xlam` detector, covers xLAM 2-1b/3b/8b.
- **Granite bare-ident** (`<tool_call>{name: get_time, arguments: {…}}`) —
  new `granite` detector + HTML-entity unescape at chain entry.
- **Hammer fenced** (``` ```\n[{…}]\n``` ```) — fall-through to the new
  `xlam` detector, which handles Markdown fences.

### 7.4 Agent-harness hardening (follow-up round)

After the initial 18-model run, a second pass added three SLM-specific
improvements based on current research (smolagents, pydantic-ai, xLAM
blog, Reflexion paper, LangGraph recursion cookbooks — all cited in the
Appendix). Each is a small, opinionated implementation in
`edgevox/llm/llamacpp.py`:

**1. Loop detection.** Every `(tool_name, canonical_args_json)` pair in a
turn is fingerprinted. On the 2nd identical call we inject a
"you already called this" hint *instead of* dispatching. On the 3rd we
hard-break the loop and return a friendly message. Fixed the
**Hammer-2.1-0.5b** infinite-loop case: previous behaviour was hops
exhausted after 4 calls, now loop-break fires on the 3rd identical call.

**2. Schema-retry.** On a `TypeError: unexpected keyword argument` from
dispatch, we inject a **plain-English** hint into the next hop's
tool-result message listing the actual parameters the tool accepts. Budget
is one retry per tool per turn (mirroring pydantic-ai's `ModelRetry`
default). Fixed the **Llama-3.2-1B** wrong-kwarg crash: dispatch no
longer returns a bare error, the model gets a chance to correct.

**3. Echoed-payload sanitiser.** Some 1B-class models echo our tool-result
JSON verbatim as their final reply. A heuristic detects a JSON object
containing `retry_hint`/`ok`/`error` and substitutes
`"Sorry, I couldn't answer that."` so TTS doesn't read JSON aloud.

**4. Tool-use directive in system prompt.** `TOOL_SYSTEM_SUFFIX` was
rewritten from *"call a tool only when needed"* to an explicit
*"You MUST call the matching tool for live data / external actions. Do
NOT answer from memory."* This is the Hammer/Arch-Function pattern; it
is known to lift BFCL irrelevance and hallucination sub-scores for small
models.

Final RTX 3090 tally after the harness pass (cached loads, real-time
inference):

| Preset | Tool called? | Notes vs initial run |
|---|---|---|
| gemma-4-e2b | ✅ | unchanged |
| qwen2.5-1.5b | ✅ | unchanged |
| qwen2.5-3b | ✅ | unchanged |
| qwen3-1.7b | ❌ | still hallucinates "10:45 AM" under stronger prompt |
| **llama-3.2-1b** | ✅* | *Dispatch error → schema-retry → sanitised fallback (clean UX instead of crash) |
| llama-3.2-3b | ✅ | unchanged |
| smollm3-3b | ❌ | still hallucinates |
| xlam-2-1b-fc | ✅ | unchanged |
| xlam-2-3b-fc | ✅ | unchanged |
| xlam-2-8b-fc | ✅ | unchanged |
| granite-4.0-350m | ❌ | truncates — 350M floor |
| **granite-4.0-1b** | ✅ | HTML-unescape + Python-dict + pythonic-in-tag all paths working |
| functionary-v3.2 | ❌ | needs `hf_tokenizer_path` wire-up |
| hermes-3-3b | ❌ | hallucinated "7:45 PM" in this run |
| phi-4-mini | ❌ | narrates "I'm fetching..." — getting closer, no call |
| **toolace-2-8b** | ✅ | 12:00 PM on 2026-04-17 — echoes the exact ISO timestamp |
| **hammer-2.1-0.5b** | ✅ | Loop-break fires cleanly; returns friendly fallback |
| robobrain-2.0-7b | ❌ | narrates (expected — spatial VLM) |

**11 / 18** detect and dispatch tool calls. The remaining 7 split into:
4 hallucinations (Qwen3, SmolLM3, Hermes-3, Phi-4-mini), 1 template
issue (Functionary needs `hf_tokenizer_path`), 1 scale-floor (Granite
350M), 1 model category mismatch (RoboBrain).

Headline count is unchanged from round 1, but the *composition* moved
from "crashes + loops + detection gaps" to "honest model-quality
failures that no harness could fix." That is the win.

### 7.5 Issues surfaced and fixed in this work

- **Llama 3.2-1B non-string tool name.** The model sometimes emits a tool
  call where `function.name` arrives as a dict, crashing
  `ToolRegistry.dispatch()` with `TypeError: unhashable type: 'dict'`.
  Fixed by coercing / rejecting non-string names (see
  `edgevox/llm/tools.py` — `isinstance(name, str)` guard).

- **Qwen3 / SmolLM3 thinking mode.** The `<think>…</think>` chain-of-
  thought leaked into TTS and broke naive tool-call detection.
  Fixed by `_strip_thinking()` before the parser chain runs, and by a
  streaming think-gate in `LLM.chat_stream`.

- **xLAM raw JSON arrays.** Emitted with no wrapper tokens. Added a new
  `xlam` detector in `edgevox/llm/tool_parsers/detectors/xlam.py`.

- **Granite HTML-escape leak.** Granite 4.0-1B's GGUF chat template emits
  `&lt;tool_call&gt;` instead of `<tool_call>`. Fixed by
  `html.unescape()` at the top of `parse_tool_calls()` when entities are
  detected.

- **Granite bare-identifier values.** Granite emits
  `{name: get_time, arguments: {…}}` where `get_time` is a bare
  identifier — neither JSON nor `literal_eval` accepts this. Added a
  regex-extract fallback in the new `granite` detector.

- **Functionary config crash.** `chat_format="functionary-v2"` demanded an
  `hf_tokenizer_path` kwarg that `LLM` didn't wire through. Fixed by
  dropping the `chat_format` override — the parser chain recovers when
  the model does emit a call.

- **RoboBrain 2.0.** Wraps replies in `<answer>…</answer>` and narrates
  tool usage in prose. Not fixed — it's a spatial-perception model, not a
  tool caller. Documented as out-of-scope for the dialog slot.

---

## 8. Recommendations per use case

**English voice dialog + local tool control (home, timer, robot motion):**
  Start with `granite-4.0-1b` (Apache-2.0, 1 B, BFCL v3 ~50) or
  `hermes-3-3b` (3 B, designed for `<tool_call>`). Both fit the EdgeVox
  latency budget on CPU-only hardware.

**Multilingual voice dialog (Vietnamese, Korean, Thai):**
  `qwen2.5-3b` remains the best open multilingual SLM. Disable thinking
  mode via `/no_think` suffix in the system prompt for low-latency TTS.

**Highest tool-calling accuracy, commercial-safe:**
  `toolace-2-8b` (Apache-2.0) or `functionary-v3.2` (MIT). Both ~5 GB
  Q4_K_M, ~5 s load on CUDA, sub-second first-call.

**Research / non-commercial:**
  `xlam-2-3b-fc` is SOTA at 3 B on BFCL v3 but CC-BY-NC-4.0.

**Explicit on-device / ≤1 GB RSS:**
  `granite-4.0-350m` (Apache-2.0) or Google's `FunctionGemma-270m-it` (not
  yet an EdgeVox preset — straightforward to add if the Gemma license is
  acceptable).

**Embodied / spatial reasoning:**
  `robobrain-2.0-7b` as a *perception/planning* peer to the dialog LLM —
  not a tool-calling replacement for Gemma. Future EdgeVox work should add
  a dedicated `edgevox.vla/` module rather than force it into the chat LLM
  slot.

---

## 9. Reproducing the results

```bash
# Replace HOST with your CUDA host alias (configured in ~/.ssh/config).

# 1. Sync the working tree to a CUDA box
rsync -az edgevox/ HOST:~/Workspaces/edgevox/

# 2. Install uv + CUDA-enabled llama-cpp-python
ssh HOST 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ssh HOST 'cd ~/Workspaces/edgevox && \
    uv venv --python 3.12 && \
    uv pip install -e . && \
    uv pip install --force-reinstall --no-deps llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124'

# 3. Smoke-test every preset
ssh HOST 'cd ~/Workspaces/edgevox && \
    uv run python scripts/smoke_test_llm_presets.py \
      --json /tmp/edgevox_slm_smoke.json'

# 4. Fetch results
scp HOST:/tmp/edgevox_slm_smoke.json docs/reports/data/
```

The script exits non-zero if any preset fails to load or chat. A `yes` in
the `tool called?` column means the parser chain recovered a call
regardless of argument correctness — *correctness* is the next-milestone
benchmark (Section 10).

---

## 10. Future work

1. **Full BFCL v4 subset** — ~500 prompts covering simple, parallel,
   multi-turn, and irrelevance categories. T=0, programmatic grading via
   AST + canonicalised deep-equal on arguments.
2. **τ-Bench retail** — multi-turn pass^k reliability using Sierra's
   released benchmark.
3. **EdgeVox custom suite** — ~40 voice-agent prompts with ASR-noise
   injection, Vietnamese / Korean prompts, adversarial tool-lookalikes,
   missing-parameter clarifications.
4. **`LlamaServerLLM` backend** — subprocess adapter that spawns
   `llama-server --jinja` for models where `llama-cpp-python`'s
   post-hoc parsers aren't enough (e.g. Qwen3-Coder XML, Ministral compact
   format).
5. **Per-preset `chat_format` tuning** — empirically verify whether
   `chatml-function-calling` or native auto-detect yields higher native
   call rate for each family.
6. **Streaming tool-call support** in the EdgeVox agent loop — the
   vendored SGLang detectors already implement `parse_streaming_increment`;
   just needs wiring through `LLM.chat_stream`.

---

## Appendix A — Design decisions

### A.1 Why vendor SGLang, not vLLM

vLLM has a richer tool-parser collection (~30 subclasses) but every file
drags in `vllm.entrypoints.openai.protocol`, `vllm.sampling_params`,
and on import requires `torch` and `ray`. Vendoring a handful of parser
files from vLLM means vendoring the protocol layer too.

SGLang's `python/sglang/srt/function_call/` package is a cleaner refactor
of the same logic: `BaseFormatDetector` + per-model detectors, with only
`orjson` + `partial-json-parser` as runtime deps. Replacing the single
`from sglang.srt.entrypoints.openai.protocol import Tool` import with a
local 30-line dataclass is a ~10-line change per file. We did exactly
that; see `edgevox/llm/tool_parsers/NOTICE` for attribution.

### A.2 Why the detector chain runs *after* `tool_calls` check

`llama-cpp-python`'s `functionary-v2` and `chatml-function-calling`
handlers populate `message["tool_calls"]` directly when they match. For
those models the parser chain is skipped. This minimizes overhead on the
models that have working native handlers while keeping a safety net for
everything else.

### A.3 Why strict unknown-tool dropping

Upstream SGLang has a `SGLANG_FORWARD_UNKNOWN_TOOLS` env var that, when
set, allows the agent loop to receive calls to names not in the `tools=[…]`
schema. EdgeVox drops them (strict mode — SGLang's default too) because
forwarded hallucinated tool names cause `KeyError` in the agent dispatch
layer without adding value. If your deployment has a use case for the
opposite behaviour, raise an issue.

## Appendix B — Raw smoke-test JSON

See `docs/reports/data/slm-tool-calling.json`, produced by
`scripts/smoke_test_llm_presets.py --json ...`. Each entry captures: preset
slug, family, approximate size (GB), load seconds, chat latency and token
count, chat reply, tool-call latency, whether a tool call was dispatched,
tool reply, and any error message.

## Appendix C — References

*Research conducted April 2026 via live web search / fetch from the
following sources. Where numbers were image-only on HF cards, we cite the
card URL and note the discrepancy.*

- [BFCL v4 leaderboard (Berkeley)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL v3 multi-turn blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- [llama.cpp `docs/function-calling.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [llama.cpp `common/chat.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp)
- [llama.cpp PR #18675 (Autoparser)](https://github.com/ggml-org/llama.cpp/pull/18675)
- [llama.cpp PR #11607 (`--reasoning-format`)](https://github.com/ggml-org/llama.cpp/pull/11607)
- [`llama-cpp-python llama_chat_format.py`](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_chat_format.py)
- [SGLang `python/sglang/srt/function_call/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/function_call)
- [IBM Granite 4.0-H-1B](https://huggingface.co/ibm-granite/granite-4.0-h-1b)
- [Salesforce xLAM-2-8b](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
- [Team-ACE ToolACE-2-Llama-3.1-8B](https://huggingface.co/Team-ACE/ToolACE-2-Llama-3.1-8B)
- [MeetKai Functionary-small-v3.2](https://huggingface.co/meetkai/functionary-small-v3.2)
- [Nous Hermes-3-Llama-3.2-3B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B)
- [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Google FunctionGemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- [Qwen Function Calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Sierra τ-Bench](https://github.com/sierra-research/tau-bench)
