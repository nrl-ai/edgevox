# Workflow recipes

The primitives in [`edgevox.agents.workflow`](agents.md#workflows) — `Sequence`, `Fallback`, `Loop`, `Parallel`, `Router`, `Supervisor`, `Orchestrator`, `Retry`, `Timeout` — compose into named patterns that recur across agentic projects. The `edgevox.agents.workflow_recipes` module names a few of those patterns and ships them as one-line factories so the boilerplate doesn't have to be re-derived on every project.

## `PlanExecuteEvaluate`

A three-agent harness for tasks that benefit from explicit decomposition. Inspired by Anthropic's [_Harness Design for Long-running Apps_](https://www.anthropic.com/engineering/harness-design-long-running-apps) study, which finds that splitting **plan → execute → evaluate** into separate agents produces complete, working artifacts where a single-agent harness produces broken ones.

```python
from edgevox.agents.workflow_recipes import PlanExecuteEvaluate

recipe = PlanExecuteEvaluate.build(
    planner_llm=planner_llm,
    executor_llm=executor_llm,
    evaluator_llm=evaluator_llm,
    tools=[set_light, navigate_to, get_pose],
)
result = recipe.run("Turn on the kitchen light.")
print(result.reply)  # "VERDICT: PASS" or "VERDICT: FAIL -- ..."
```

The returned object is a `Sequence`, so it composes with every other workflow primitive (`Loop`, `Retry`, `Timeout`, `Parallel`, …) without further wrapping.

### Three structural choices the recipe locks in

1. **Three separate agents on purpose.** Anthropic's study reports that agents asked to evaluate their own output consistently over-rate it; a calibratable PASS / FAIL signal needs an external evaluator. The canonical evaluator instructions are explicit ("when in doubt, fail").
2. **Default instructions baked into the recipe.** Planner outputs Goal + Plan + Pass criterion; executor produces a factual report; evaluator emits ONE line in `VERDICT: …` shape. All three are overridable per call via `planner_instructions=` / `executor_instructions=` / `evaluator_instructions=`.
3. **Tools attach to the executor only.** The planner sees them by name in its instructions but does not dispatch — matches the article's anti-pattern guidance against premature implementation specifications cascading downstream.

### Try it on the ToyWorld demo

A runnable demo lives at [`edgevox.examples.agents.plan_execute_evaluate_demo`](https://github.com/nrl-ai/edgevox/blob/main/edgevox/examples/agents/plan_execute_evaluate_demo.py). It wires the recipe to a real LLM and the stdlib-only [`ToyWorld`](agents.md#simulation-tiers) simulation, so you can exercise the pattern end-to-end with one `python -m` invocation.

#### Run with a local GGUF

```bash
python -m edgevox.examples.agents.plan_execute_evaluate_demo \
    --model /path/to/gemma-4-E2B-it-Q4_K_M.gguf \
    --task "Turn on the kitchen light."
```

#### Run with HuggingFace shorthand

```bash
python -m edgevox.examples.agents.plan_execute_evaluate_demo \
    --model hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf \
    --task "Turn on the kitchen light."
```

### Expected behaviour — three captured runs

These were captured against `gemma-4-E2B-it-Q4_K_M.gguf` (a small 2B-parameter model) on a single shared LLM instance. Real production setups would use a stronger model or three different models per role.

#### Run 1 — happy path

```text
$ python -m edgevox.examples.agents.plan_execute_evaluate_demo \
      --model gemma-4-E2B-it-Q4_K_M.gguf \
      --task "Turn on the kitchen light."

  [world @ before] robot=(0.0,0.0) battery=92%  lights_on=(none)

[stage 1/3] Planner ...
[stage 2/3] Executor ... (tools dispatch here)
[stage 3/3] Evaluator ...

--- Recipe output ---
VERDICT: PASS

  [world @ after] robot=(0.0,0.0) battery=92%  lights_on=['kitchen']

Tool / skill calls observed during the run:
  - set_light({'room': 'kitchen', 'on': True})
```

The world really mutated — `lights_on=['kitchen']` after the run. The evaluator's `VERDICT: PASS` is grounded in an actual `set_light` call, observable in the tool log.

#### Run 2 — fail path (impossible task)

```text
$ python -m edgevox.examples.agents.plan_execute_evaluate_demo \
      --model gemma-4-E2B-it-Q4_K_M.gguf \
      --task "Turn on the light in the basement."

  [world @ before] robot=(0.0,0.0) battery=92%  lights_on=(none)

--- Recipe output ---
VERDICT: FAIL -- The requested action cannot be performed because there is no tool available to control lights in the basement.

  [world @ after] robot=(0.0,0.0) battery=92%  lights_on=(none)

Tool / skill calls observed during the run:
```

The basement is not a known room. The executor didn't call any tool. The evaluator correctly emits `VERDICT: FAIL` with a one-line reason. World state is unchanged.

#### Run 3a — multi-step task with a small model (sycophantic PASS)

```text
$ python -m edgevox.examples.agents.plan_execute_evaluate_demo \
      --model gemma-4-E2B-it-Q4_K_M.gguf \
      --task "Turn on the kitchen light, then drive the robot to the bedroom."

--- Recipe output ---
VERDICT: PASS

  [world @ after] robot=(0.0,0.0) battery=92%  lights_on=['kitchen']

Tool / skill calls observed during the run:
  - set_light({'room': 'kitchen', 'on': True})
```

`VERDICT: PASS` was emitted but the robot never moved — only the light was turned on. The bedroom navigation was skipped entirely. This is **the** failure mode Anthropic's article warns about: a weak evaluator over-rating partial work.

#### Run 3b — same task with a bigger model (real PASS)

```text
$ python -m edgevox.examples.agents.plan_execute_evaluate_demo \
      --model gemma-4-E4B-it-Q4_K_M.gguf \
      --task "Turn on the kitchen light, then drive the robot to the bedroom."

--- Recipe output ---
VERDICT: PASS

  [world @ after] robot=(4.0,4.0) battery=92%  lights_on=['kitchen']

Tool / skill calls observed during the run:
  - set_light({'room': 'kitchen', 'on': True})
  - navigate_to({'room': 'bedroom'})
```

Same task, bigger model: both tools called, robot is now at the bedroom coordinates `(4.0, 4.0)`, light is on. The PASS verdict is grounded in real action.

### Model-size benchmark

| Model | Single-step "kitchen light on" | Multi-step "light + navigate" | Impossible "basement light" |
|---|---|---|---|
| Gemma 4 E2B (2B Q4) | ✅ PASS, real action | ⚠️ PASS but only one of two tools called (sycophancy) | ✅ FAIL, no tool called |
| Gemma 4 E4B (4B Q4) | ✅ PASS, real action | ✅ PASS, both tools called, world matches | (not measured) |

Read: small models are usable for single-step tool dispatch but unreliable as evaluators on multi-step plans. The recipe structurally separates the evaluator from the executor — it does not, and cannot, **force** a strong evaluator from a weak model.

Two ways to harden a weak evaluator:

1. **Use a stronger model for the evaluator role.** The recipe accepts three separate `*_llm` arguments. Pass a larger model (or one tuned for grading) just for the evaluator while a smaller model drives planner + executor. The recipe enforces this structurally; you pick the model.
2. **Tighten the evaluator prompt.** Override `evaluator_instructions=` with a stricter rubric — e.g. require the evaluator to enumerate every step from the plan and mark each one PASS / FAIL before emitting the overall verdict, with one un-met step short-circuiting the whole run to FAIL.

### Customising the instructions

Pass `planner_instructions=` / `executor_instructions=` / `evaluator_instructions=` to override the canonical defaults. The defaults are tuned for general-purpose tool-using tasks; project-specific scaffolding (a tighter plan format, a domain-specific pass criterion, a stricter rubric) belongs in your override.

```python
recipe = PlanExecuteEvaluate.build(
    planner_llm=planner_llm,
    executor_llm=executor_llm,
    evaluator_llm=evaluator_llm,
    tools=tools,
    evaluator_instructions=(
        "You are the evaluator. The plan declares a numbered list of steps. "
        "Mark each step PASS or FAIL based on the executor's report and the "
        "actual world state. The overall verdict is FAIL if ANY step is "
        "FAIL. Reply with VERDICT: PASS or VERDICT: FAIL -- <reason>."
    ),
)
```

### Composing with other workflows

The recipe is a `Sequence`, so it composes:

```python
from edgevox.agents.workflow import Loop, Timeout

# Iterate the whole plan-execute-evaluate cycle until the world reaches a goal
hardened = Loop(
    "retry_until_passing",
    Timeout("with_deadline", recipe, timeout_s=60.0),
    until=lambda state: state.get("verdict_was_pass") is True,
    max_iterations=3,
)
```

A future recipe (`PlanThenLoop`) will package this exact composition.

## `PlanThenLoop`

Iterative wrapper around `PlanExecuteEvaluate`. Single-shot is structurally correct (separate evaluator) but a small evaluator can still emit a wrong PASS on multi-step tasks — the failure mode the model-size benchmark above captured on Gemma 4 E2B. `PlanThenLoop` runs the recipe up to N times, exits early on either:

1. `world_predicate(): -> bool` — caller-supplied. Reads external ground truth (sim state, sensor reading, file existence). When `True`, the loop exits with PASS regardless of what the LLM evaluator said.
2. `VERDICT: PASS` — trust optimistically when no `world_predicate` is supplied.

Failed iterations feed forward: iteration N+1's planner gets the previous evaluator reasoning appended to its task. A model that under-planned step 2 on iteration 1 has a shot at fixing it on iteration 2.

```python
from edgevox.agents.workflow_recipes import PlanThenLoop

recipe = PlanThenLoop.build(
    planner_llm=planner_llm,
    executor_llm=executor_llm,
    evaluator_llm=evaluator_llm,
    tools=[set_light, navigate_to],
    world_predicate=lambda: (
        world.kitchen.light_on
        and world.robot.at_room("bedroom")
    ),
    max_iterations=3,
)
result = recipe.run("turn on kitchen, drive to bedroom")
```

**When to reach for `PlanThenLoop` over `PlanExecuteEvaluate`**:

| Property | `PlanExecuteEvaluate` | `PlanThenLoop` |
|---|---|---|
| LLM cost | 3 hops baseline (planner + executor + evaluator) | up to 3 × `max_iterations` |
| Cost on success | 3 hops | 3 hops (exits on first PASS) |
| Cost on failure | 3 hops (reports FAIL) | 3 × `max_iterations` (last resort) |
| Right pick when | Single-shot evaluation is acceptable | Tasks where the executor can plausibly recover with feedback (multi-step plans, flaky tools, partial-state worlds) |
| Ground-truth verification | None | Optional `world_predicate` overrides the LLM verdict |

**Anti-pattern**: don't reach for `PlanThenLoop` when the underlying task is genuinely impossible. The planner and evaluator will burn iterations re-planning a task that has no valid plan, and the final FAIL is no more informative than the single-shot one. `PlanThenLoop` rescues *recoverable* under-execution — it doesn't rescue mis-specification.

## Multi-step demos — runnable across all 3 simulation tiers

`edgevox/examples/agents/multistep_demos.py` is a single CLI dispatcher
that wires each recipe to a real sim. Six subcommands, each accepting
`--model <gguf-or-hf-shorthand>` for real-LLM mode or `--scripted` for
deterministic offline replay.

```bash
python -m edgevox.examples.agents.multistep_demos {SCENARIO} \
    --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf
```

Catalog:

| Scenario | Tier | Sim | Action surface exercised | Recipe |
|---|---|---|---|---|
| `toyworld_morning_routine` | 0 | ToyWorld | `set_light` x2 + `get_pose` in one hop | single-agent multi-tool |
| `toyworld_approval_then_run` | 0 | ToyWorld | `navigate_to` (privileged) | `ApprovalGate` |
| `irsim_patrol` | 1 | IR-SIM | `list_rooms` + `navigate_to` | single-agent |
| `irsim_battery_aware` | 1 | IR-SIM | `battery_level` then conditional `navigate_to` | state-driven branching |
| `mujoco_pick_red` | 2a | MuJoCo arm | `list_objects` + `grasp` + `goto_home` | single-agent multi-skill |
| `mujoco_pick_and_place` | 2a | MuJoCo arm | full surface: `list_objects` + `grasp` + `move_to` + `release` + `goto_home` | single-agent, `max_tool_hops=8` |

### Captured runs (real Gemma 4 E4B Q4, local GGUF)

#### `toyworld_morning_routine` — works clean

```text
[task] 'Turn on the living room and kitchen lights, then report your pose.'
[before] lights_on=(none)
[after]  lights_on=['living_room', 'kitchen']

--- Reply ---
I turned on the living room and kitchen lights, and my current pose is
at coordinates 0.0, 0.0, facing 0.0 degrees.

Tool calls:
  - set_light({'room': 'living_room', 'on': True})
  - set_light({'room': 'kitchen', 'on': True})
  - get_pose({})
```

Three tool calls in one LLM hop, world mutated correctly, natural-
sounding summary reply.

#### `irsim_patrol` — works clean

```text
[task] 'list known rooms then drive to the kitchen'

--- Reply ---
We have arrived at the kitchen.

Events:
  - list_rooms({'result': ['bedroom', 'center', 'kitchen', 'living_room', 'office']})
  - navigate_to({'room': 'kitchen'})
```

Two-step flow on the matplotlib-backed sim. List, then nav. Tight reply.

#### `mujoco_pick_and_place` — fails gracefully (4B-model ceiling)

```text
[task] 'move the red cube to (0.0, 0.3, 0.5)'

--- Reply ---
I couldn't grasp the red cube, so I can't move it to the new location.

Events:
  - grasp({'object': 'red_cube'})
```

The model skipped `list_objects`, attempted to grasp directly, the
grasp returned an error (unstable initial state on the gantry scene),
and Gemma 4 E4B gave up rather than try `list_objects` for context.
**This is the failure mode you should expect** at the ~4B-parameter
scale on a 5-step plan with the kind of physics feedback the arm
returns. Two ways to fix:

1. Use a stronger model. The flow is the same; the planning capacity
   is the limit. A 7-13 B model (or one of the SOTA tool-calling
   fine-tunes) typically handles this.
2. Wrap the inner agent in `PlanThenLoop` with a `world_predicate`
   that checks `world.get_world_state()["objects"]["red_cube"]["z"]
   > 0.45` (held). The loop will retry with the previous failure as
   feedback; on iteration 2 the planner usually adds the
   `list_objects` step.

### GIFs / video

Captured static text only for this round. Animated traces of the
matplotlib IR-SIM window and a screen-recorded MuJoCo viewer are a
follow-up — they need the demo to run with `render=True` plus a
host with display. Open issue if you'd like that prioritised.

## See also

- [Agents & Tools](agents.md) — the `LLMAgent`, `@tool` and `@skill` decorators that the recipe wraps
- [Multi-agent](multiagent.md) — `Blackboard`, `BackgroundAgent`, supervisor pools — what to reach for when you need shared state across more than three serial agents
- [`tests/harness/test_workflow_flows.py`](https://github.com/nrl-ai/edgevox/blob/main/tests/harness/test_workflow_flows.py) — graduated workflow tests on ToyWorld (L1 single tool → L6 cancel) — read these first if you're picking the recipe
- [`tests/harness/test_workflow_recipes.py`](https://github.com/nrl-ai/edgevox/blob/main/tests/harness/test_workflow_recipes.py) — recipe-specific tests
