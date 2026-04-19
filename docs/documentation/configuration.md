# Configuration — Turn features on and off

**The one-sentence EdgeVox philosophy: everything is optional.** Every STT backend, TTS voice, memory store, VAD classifier, echo-cancellation strategy, hook, and workflow is a plug-in. You decide which of them load, and the framework degrades cleanly when something is missing.

This page is the single place to look when you want to **enable**, **swap**, or **disable** a feature. Each section follows the same shape: *what it is → how to turn it on → how to turn it off → alternatives*.

## Install what you need, nothing more

EdgeVox ships a minimal core (`pip install edgevox`) that boots with a working default for every layer. Optional capabilities live behind `pip install 'edgevox[extra]'` flags — none of them are auto-installed, so your environment stays lean.

```bash
# Minimal install — voice pipeline + LLM + chess + agent framework
pip install edgevox

# Opt-in extras (combine any subset — all stackable)
pip install 'edgevox[gpu]'             # onnxruntime-gpu for CUDA STT / VAD
pip install 'edgevox[dtln]'            # neural echo cancellation
pip install 'edgevox[sim]'             # 2-D IR-SIM robot sim
pip install 'edgevox[sim-mujoco]'      # 3-D MuJoCo tabletop / humanoid
pip install 'edgevox[desktop]'         # RookApp PySide6 desktop app
pip install 'edgevox[voice-vad]'       # WebRTC VAD backend
pip install 'edgevox[memory-vec]'      # VectorMemoryStore via sqlite-vec
pip install 'edgevox[dev]'             # ruff, pytest, pre-commit
```

Nothing here forces extras on anyone else — the CI publishes wheels for the minimal install, and every runtime import of an optional feature is wrapped in a `try: import …` that surfaces a clear `pip install 'edgevox[X]'` hint if the dep is missing.

---

## 1. Speech-to-text backends

Two backends ship; more via the `BaseSTT` Protocol.

| Backend | Best for | Opt-in |
|---|---|---|
| **faster-whisper** | English + 99 other languages | default, no extra install |
| **sherpa-onnx (zipformer)** | Vietnamese | default, no extra install |

**Turn on a specific backend:**

```python
from edgevox.stt import create_stt

stt = create_stt(language="vi", backend="sherpa")   # explicit
stt = create_stt(language="en")                      # let the language config pick
```

**Disable STT entirely** (text-mode agents): don't instantiate one. The `LLMAgent` has no STT dependency; text-mode examples (`edgevox-agent robot-panda --text-mode`) show the pattern.

**Write your own** — subclass `BaseSTT`, implement `transcribe(audio, language) -> str`, either inject directly into the agent or add a branch to `create_stt()`.

---

## 2. Text-to-speech backends

Four backends ship.

| Backend | Languages | Voice count |
|---|---|---|
| **Kokoro** (MIT) | 9 languages, 56 voices | default for English + 8 others |
| **Piper** (MIT) | 40+ languages | default for Vietnamese, German, Russian, Arabic, Indonesian |
| **Supertonic** (Apache-2) | Korean | default for `ko` |
| **PyThaiTTS** (Apache-2) | Thai | default for `th` |

**Swap voice / backend:**

```python
from edgevox.tts import create_tts

tts = create_tts(language="en", voice="af_bella")                  # Kokoro voice
tts = create_tts(language="en", voice="en_US-amy-medium", backend="piper")
```

**Disable TTS entirely** (headless agents): don't instantiate. The agent's reply string is still returned from `agent.run()`; you just don't synthesize.

---

## 3. LLM — any GGUF via `llama-cpp-python`

One backend, one class (`edgevox.llm.LLM`), any model. The `model_path` argument accepts a local `.gguf`, a HuggingFace-flavoured shorthand (`hf:repo:file.gguf`), or a path resolved by `edgevox-setup`.

```python
from edgevox.llm import LLM

llm = LLM(model_path="gemma-3-4b-it-E2B-q4_k_m.gguf")
llm = LLM(model_path="hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf")
llm = LLM(model_path="/abs/path/to/my-model.gguf", n_ctx=8192, n_gpu_layers=-1)
```

**Swap by model only:** change `model_path`. **Swap backend entirely:** not supported via the built-in `LLM` class — write your own that conforms to the `chat_stream(...)` / `count_tokens(...)` shape and pass it to `LLMAgent.bind_llm(...)`.

---

## 4. Memory stores — three implementations

All three implement the same `MemoryStore` Protocol, so swapping is a one-line change.

| Class | Backing | When to pick it |
|---|---|---|
| `JSONMemoryStore` | debounced JSON file | prototyping, human-readable inspection |
| `SQLiteMemoryStore` | stdlib `sqlite3` + WAL | **recommended default** — crash-safe, multi-process-safe |
| `VectorMemoryStore` | `sqlite-vec` extension + `embed_fn` | semantic retrieval over facts (`search_facts("what's safe to cook?")`) |

**Swap the default JSON store for SQLite:**

```python
# Before
from edgevox.agents.memory import JSONMemoryStore
store = JSONMemoryStore("./memory.json")

# After — crash-safe, same Protocol, no other changes
from edgevox.agents import SQLiteMemoryStore
store = SQLiteMemoryStore("./memory.db")
```

**Opt in to vector search:**

```bash
pip install 'edgevox[memory-vec]'
```

```python
from edgevox.llm import LLM
from edgevox.agents import VectorMemoryStore, llama_embed

llm = LLM(model_path="nomic-embed-text-v1.5.Q4_K_M.gguf", embedding=True)
store = VectorMemoryStore("./vec.db", embed_fn=llama_embed(llm))
hits = store.search_facts("what's safe to cook?", k=5)
```

**Disable memory entirely:** don't register `MemoryInjectionHook` on the agent. The agent runs fine without a memory store; each turn just starts fresh.

**Expose memory to the LLM itself** (memory-as-tools):

```python
from edgevox.agents.memory_tools import memory_tools

agent = LLMAgent(
    ...,
    tools=[*memory_tools(store), ...],      # remember_fact / forget_fact / recall_fact
)
```

Filter with `memory_tools(store, include=("recall_fact",))` if you only want the LLM to read, not write.

See [`memory.md`](/documentation/memory) for the full data model.

---

## 5. Barge-in VAD backends

Four backends behind one `BargeInVADWatcher` Protocol. Pick based on accuracy/latency/weight trade-offs.

| Backend | Class | Install | Accuracy |
|---|---|---|---|
| **Energy** | `EnergyBargeInWatcher` | built-in | baseline; 5-15 % false triggers in noisy rooms |
| **WebRTC** | `WebRTCVADWatcher` | `edgevox[voice-vad]` | GMM baseline; large improvement over RMS |
| **Silero v6** | `SileroVADWatcher` | no extra install (reuses faster-whisper's ONNX) | ~1-2 % false triggers |
| **TEN** | `TENVADWatcher` | onnxruntime (core) + `nrl-ai/edgevox-models` fetch | lowest latency, 306 KB model |

**Turn on via the factory:**

```python
from edgevox.agents import create_vad_watcher

watcher = create_vad_watcher(
    "silero",                              # or "energy" / "webrtc" / "ten"
    controller,
    is_tts_playing=player.is_playing,
)
threading.Thread(target=watcher.run, args=(mic_stream,), daemon=True).start()
```

**Turn off barge-in entirely:** don't attach any watcher to your `InterruptController`. Interrupts still fire via `ctx.interrupt.trigger(...)` — you just lose the mic-driven path.

---

## 6. Echo cancellation backends

The player pushes its output signal into the recorder's AEC reference so self-triggering on TTS is suppressed.

| Backend | Install | Notes |
|---|---|---|
| `none` | built-in | no processing; rely on VAD echo-floor |
| `nlms` | built-in | classic adaptive filter, ~10 LOC |
| `specsub` | built-in | spectral subtraction |
| `dtln` | `edgevox[dtln]` | neural AEC via TFLite (Apache-2 model) |

```python
from edgevox.audio.aec import create_aec

recorder.set_aec(create_aec("dtln"))
recorder.set_aec(create_aec("none"))       # or disable
```

---

## 7. Hooks — the main agent extension point

Hooks fire at six points in `LLMAgent.run()`: `on_run_start`, `before_llm`, `after_llm`, `before_tool`, `after_tool`, `on_run_end`. They're priority-ordered and composable.

**Three categories ship:**

| Category | Hooks | When to enable |
|---|---|---|
| **Always-on basics** | `MemoryInjectionHook`, `NotesInjectorHook`, `PersistSessionHook`, `TokenBudgetHook`, `ContextCompactionHook` | anything with memory / long context |
| **SLM hardening** | `default_slm_hooks()` → loop-break, repetition guard, empty-args canary, name-validate | small models (1-4 B params) |
| **Observability** | `TimingHook`, `EchoingHook`, `EpisodeLoggerHook` | debugging, metrics, audit trails |
| **Safety / guardrails** | `SafetyGuardrailHook`, `ToolOutputTruncatorHook` | user-facing / production |

**Turn on a hook:**

```python
from edgevox.agents import LLMAgent, MemoryInjectionHook, TimingHook

agent = LLMAgent(
    ...,
    hooks=[
        MemoryInjectionHook(memory_store=store),
        TimingHook(),
    ],
)
```

**Turn off a hook:** remove it from the `hooks=[...]` list. There's no global enable/disable — the list is the source of truth.

**Turn on all SLM hardening at once:**

```python
from edgevox.llm.hooks_slm import default_slm_hooks

agent = LLMAgent(..., hooks=[*default_slm_hooks(), MemoryInjectionHook(store)])
```

**Write your own** — a hook is any callable with a `points` frozenset of fire-point names and a `__call__(point, ctx, payload)` method. See [`hooks.md`](/documentation/hooks).

---

## 8. Workflows — compose multiple agents without an LLM

All workflows implement the `Agent` Protocol, so they nest and compose arbitrarily.

| Workflow | Shape | When |
|---|---|---|
| `Sequence` | A → B → C | pipeline of deterministic steps |
| `Fallback` | try A; on fail try B | best-effort paths |
| `Loop` | repeat A until predicate | polling, refinement |
| `Parallel` | fan out, fan in | concurrent tool calls |
| `Router` | pick one of N | dispatch |
| `Retry` | A with backoff | flaky tools |
| `Timeout` | A with deadline | latency ceilings |
| `Supervisor` | A oversees workers | OTP-style restart |
| `Orchestrator` | plan → dispatch → reduce | multi-step task decomposition |

**Turn on:** just construct and call.
**Turn off:** don't use; `LLMAgent` alone works standalone.

---

## 9. Multi-agent coordination

Independent from workflows — these are for **emergent** patterns (supervisor watches blackboard, background planner reacts to bus events, etc.).

| Primitive | Purpose |
|---|---|
| `Blackboard` | thread-safe shared K/V with watchers; `post_request(key, task)` for request/reply |
| `AgentMessage` + `send_message` / `subscribe_inbox` | direct agent-to-agent messages over the bus |
| `BackgroundAgent` | wraps any agent in a background thread that reacts to bus / blackboard triggers |
| `AgentPool` | starts/stops a set of agents, shares the context |

**Turn on:**

```python
from edgevox.agents import Blackboard

bb = Blackboard()                               # sync watchers
bb = Blackboard(async_watchers=True)            # non-blocking watcher dispatch
fut = bb.post_request("plan.request", {"goal": "pick cup"}, timeout=5.0)
```

**Turn off:** just don't instantiate. Single-agent `LLMAgent.run()` has no bus / blackboard dependency.

See [`multiagent.md`](/documentation/multiagent) for composition patterns.

---

## 10. Simulation tiers

Three sim environments, all conforming to `SimEnvironment`. Agents don't know which one they're running in — swap by build time or CLI flag.

| Tier | Class | Install | Good for |
|---|---|---|---|
| **Tier 0** | `ToyWorld` | stdlib | unit tests, offline CI |
| **Tier 1** | `IrSimEnvironment` | `edgevox[sim]` | 2-D mobile robot navigation |
| **Tier 2a** | `MujocoArmEnvironment` | `edgevox[sim-mujoco]` | 3-D Franka tabletop |
| **Tier 2b** | `MujocoHumanoidEnvironment` | `edgevox[sim-mujoco]` | Unitree G1 / H1 with procedural gait |

**Turn on:** `pip install 'edgevox[sim]'` (or `[sim-mujoco]`) and pass the environment instance via `ctx.deps`.

**Turn off:** use `ToyWorld` or none at all — `LLMAgent` doesn't require an environment.

---

## 11. ROS2 integration

`edgevox.integrations.ros2_*` modules bridge agent skills to ROS2 topics / services / actions. Only loads if `rclpy` is importable (ROS2 is a system package, not a PyPI dep).

```bash
source /opt/ros/jazzy/setup.bash
edgevox-agent robot-external --text-mode
```

**Turn off:** don't `source` a ROS2 workspace. The modules raise `ImportError` on load; the framework handles it and continues without ROS2.

---

## 12. Desktop apps (RookApp)

RookApp is an opt-in PySide6 app shipped as `edgevox[desktop]` plus the `edgevox-chess-robot` console script.

```bash
pip install 'edgevox[desktop]'
edgevox-chess-robot --persona trash_talker
```

Configure via CLI flags or in-app **☰ → Settings…** (persona, engine, skill, theme, voice, debug mode). Preferences persist via `QSettings`.

**Turn off:** just don't install the extra — the core `edgevox` package has no Qt dependency.

---

---

## Writing your own components

Every category above is a **Protocol** — a typed shape the framework calls. Your custom class only has to match that shape. No registration needed unless you want factory-name lookup (e.g. `create_stt("my-backend")`).

### Custom STT backend

```python
from edgevox.stt import BaseSTT

class MySTT(BaseSTT):
    display_name = "MyCustomSTT"

    def transcribe(self, audio, language: str = "en") -> str:
        # audio is a float32 numpy array @ 16 kHz
        return self._my_model(audio)

# Use it:
from edgevox.agents import LLMAgent
agent = LLMAgent(...)
agent.bind_stt(MySTT())           # or inject via your own pipeline
```

### Custom TTS backend

```python
from edgevox.tts import BaseTTS
import numpy as np

class MyTTS(BaseTTS):
    sample_rate = 24_000
    display_name = "MyCustomTTS"

    def synthesize(self, text: str) -> np.ndarray:
        return self._model.run(text)

    def synthesize_stream(self, text: str):
        # Optional — default yields one chunk. Stream for lower TTFA.
        for sentence in split(text):
            yield self._model.run(sentence)
```

### Custom memory store

Implement `MemoryStore` (see [`memory.md`](/documentation/memory) for the full method list):

```python
from edgevox.agents.memory import MemoryStore, Fact, Preference, Episode

class RedisMemoryStore:
    def add_fact(self, key, value, *, scope="global", source=""): ...
    def get_fact(self, key, *, scope="global"): ...
    def facts(self, *, scope=None): ...
    def forget_fact(self, key, *, scope="global"): ...
    def set_preference(self, key, value): ...
    def preferences(self): ...
    def add_episode(self, kind, payload, outcome, *, agent=""): ...
    def recent_episodes(self, n=5, *, kind=None): ...
    def render_for_prompt(self, *, max_facts=20, max_episodes=5): ...

# isinstance check against the runtime-checkable Protocol works:
assert isinstance(RedisMemoryStore(), MemoryStore)
```

### Custom barge-in VAD watcher

Implement `BargeInVADWatcher` — just `run(frames)` + `stop()`:

```python
from edgevox.agents import BargeInVADWatcher, InterruptController

class MyVADWatcher:
    def __init__(self, controller: InterruptController, *, is_tts_playing):
        self._controller = controller
        self._is_tts = is_tts_playing
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self, frames):
        for f in frames:
            if self._stopped:
                return
            if self._my_classifier(f):
                self._controller.trigger(reason="user_speech_custom")
```

### Custom hook

A hook is any callable with a `points` frozenset and a `__call__(point, ctx, payload)`:

```python
from edgevox.agents.hooks import BEFORE_LLM, AFTER_LLM

class LoggingHook:
    points = frozenset({BEFORE_LLM, AFTER_LLM})
    priority = 0  # observability — runs after business hooks

    def __call__(self, point, ctx, payload):
        if point == BEFORE_LLM:
            log.info("messages in: %d", len(payload["messages"]))
        elif point == AFTER_LLM:
            log.info("reply: %r", payload.get("content", "")[:80])

agent = LLMAgent(..., hooks=[LoggingHook()])
```

Priority guide: Safety=100, Business=50, Observability=0. The built-ins follow this scale so yours slots cleanly in order.

### Custom workflow

Subclass `Agent` — the same Protocol `LLMAgent` implements:

```python
from edgevox.agents import Agent, AgentContext, AgentResult

class RoundRobin:
    name = "round-robin"
    description = "Alternate between agents on successive calls"

    def __init__(self, agents: list[Agent]):
        self._agents = agents
        self._idx = 0

    def run(self, user_input: str, ctx: AgentContext) -> AgentResult:
        a = self._agents[self._idx % len(self._agents)]
        self._idx += 1
        return a.run(user_input, ctx)
```

Anywhere the framework takes an `Agent` (workflow child, handoff target, background worker) you can pass this.

### Custom LLM backend

Match the surface `LLMAgent` calls: `chat_stream(messages, *, stop_event=None, **kwargs)` yielding tokens, plus `count_tokens(text) -> int`. Wrap Anthropic / OpenAI / Ollama / vLLM / your-own-thing, then:

```python
agent = LLMAgent(...)
agent.bind_llm(MyLLM())
```

The agent loop only cares about the two-method contract — no assumptions about what's under the hood.

---

## Settings reference

Every knob, by component. Defaults are what you get from a bare `LLMAgent(...)` and `create_stt() / create_tts() / ...` calls.

### `LLMAgent`

| arg | default | meaning |
|---|---|---|
| `name` | required | human-facing identifier |
| `description` | required | advertised in agent handoffs |
| `instructions` | required | system prompt |
| `tools` | `None` | list of `@tool` callables or `Tool` objects |
| `hooks` | `[]` | list of hook objects |
| `max_tool_calls` | `8` | hops per turn before abort |
| `tool_choice_policy` | `"auto"` | `"auto"` / `"required_first_hop"` / `"required_always"` |
| `parallel_tool_dispatch` | `False` | fan out independent tool calls via thread pool |
| `handoffs` | `[]` | agents this one can hand off to |
| `stream_events` | `True` | emit per-token / per-event signals via `ctx.on_event` |

### `MemoryStore` implementations

| store | constructor args | notable defaults |
|---|---|---|
| `JSONMemoryStore` | `path`, `autoload=True` | flush debounce 2 s, episode ring 500 |
| `SQLiteMemoryStore` | `path` | WAL mode on, `synchronous=NORMAL`, episode ring 500 |
| `VectorMemoryStore` | `path`, `embed_fn`, `embedding_dim=None` | probes dim via `embed_fn("dimension probe")` when not given |

All three honour `max_facts` / `max_episodes` on `render_for_prompt`.

### `InterruptController` + watchers

| knob | class | default | meaning |
|---|---|---|---|
| `policy` | controller | `InterruptPolicy()` | whether cancels also interrupt LLM, TTS, skills |
| `cancel_llm` | policy | `True` | thread cancel into llama-cpp `stopping_criteria` |
| `cancel_tts` | policy | `True` | flush player on trigger |
| `cancel_skills` | policy | `False` | signal running skills (opt-in; not all are cancellable) |
| `frame_ms` | energy / webrtc / ten | 20 / 20 / 16 | VAD frame duration |
| `aggressiveness` | webrtc | `2` | 0-3; higher trades recall for precision |
| `threshold` | silero / ten | 0.4 / 0.5 | speech-probability cutoff |
| `sustained_speech_ms` | all VAD watchers | `120` | consecutive-speech window before trigger |
| `tts_release_ms` | all VAD watchers | `180` | refractory after TTS stops |
| `echo_suppression_ratio` | energy only | `2.0` | mic/TTS energy ratio required |
| `echo_floor_window_ms` | energy only | `200` | per-segment calibration window |

### `Blackboard`

| knob | default | meaning |
|---|---|---|
| `async_watchers` | `False` | fan out watchers via thread pool |
| `max_watcher_workers` | `4` | pool size when async |

### `Compactor`

| knob | default | meaning |
|---|---|---|
| `max_context_tokens` | `3500` | trigger summarisation above |
| `keep_last` | `6` | never summarise the most-recent N turns |
| `target_tokens` | `1500` | summary size budget |

### STT

Per language, resolved via `edgevox.core.config.get_lang(code)`. Override per-call:

```python
create_stt(language="en", model_size="large-v3", device="cuda", backend="whisper")
```

### TTS

```python
create_tts(language="en", voice="af_heart", backend="kokoro", speed=1.0)
```

### RookApp (desktop)

Env vars (also exposed as CLI flags and in-app Settings):

| env | default | meaning |
|---|---|---|
| `EDGEVOX_CHESS_PERSONA` | `casual` | `grandmaster` / `casual` / `trash_talker` |
| `EDGEVOX_CHESS_ENGINE` | persona default | `stockfish` / `maia` |
| `EDGEVOX_CHESS_USER_PLAYS` | `white` | `white` / `black` |
| `EDGEVOX_CHESS_STOCKFISH_SKILL` | persona default | 0-20 |
| `EDGEVOX_CHESS_MAIA_WEIGHTS` | — | required when `engine=maia` |
| `EDGEVOX_MEMORY_DIR` | `~/.edgevox/memory` | override for default store location |
| `EDGEVOX_TEN_VAD_MODEL` | auto-fetch | override to a local TEN VAD ONNX path |

### CLI

| flag | what it does |
|---|---|
| `edgevox --text-mode` | disables STT + TTS, terminal chat only |
| `edgevox --simple-ui` | headless rich-console voice loop |
| `edgevox --web-ui` | FastAPI + WebSocket server |
| `edgevox-agent <name>` | run one of the built-in example agents |
| `edgevox-setup` | download all default models |
| `edgevox-chess-robot` | RookApp desktop entry point |

---

## What this doesn't cover

* **Custom transports** (gRPC, MQTT, …). The `EventBus` is the canonical pub/sub; replace it with your own and the workflows + multi-agent primitives work unchanged.

When in doubt: **look at the Protocol** — that's the actual contract. The built-in classes are one implementation each; you can always write another.
