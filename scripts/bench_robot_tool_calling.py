"""Multi-model tool-calling benchmark for the four robot agents.

Runs a curated scenario set against every candidate LLM — each scenario
is a natural-language utterance the voice pipeline would hand the
agent, paired with the expected tool call (name + args). The grader
checks that the model emitted a tool call, matched the right tool, and
filled the right arguments within tolerance.

Companion to ``bench_chess_commentary.py`` but focused on function
calling in robotic applications instead of persona-bound commentary.
Both run locally — no cloud APIs.

Usage::

    python scripts/bench_robot_tool_calling.py
    python scripts/bench_robot_tool_calling.py --models gemma-4-e2b,qwen2.5-1.5b
    python scripts/bench_robot_tool_calling.py --only scout,panda

Output: ``docs/documentation/reports/robot-tool-calling-benchmark.md``
plus a JSON dump under ``reports/data/``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.agents.skills import Skill
from edgevox.examples.agents import robot_humanoid, robot_irsim, robot_panda, robot_scout
from edgevox.llm import LLM
from edgevox.llm.llamacpp import parse_tool_calls_from_content
from edgevox.llm.tools import Tool as _ToolDescriptor

# --- model pool (RTX 3080 Laptop 16 GB friendly — skip the 8B specialists by default) ---
_DEFAULT_MODELS = (
    "gemma-4-e2b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen3-1.7b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "smollm3-3b",
    "granite-4.0-350m",
    "granite-4.0-1b",
    "hammer-2.1-0.5b",
    "hermes-3-3b",
    "phi-4-mini",
)

_LARGE_MODELS = (
    "functionary-v3.2",  # 8 B
    "toolace-2-8b",  # 8 B
)


# --- tool-surface normaliser (shared with smoke_test_robot_tools.py) ---


def _agent_surface(agent_key: str) -> tuple[str, str, list[_ToolDescriptor], list[Skill]]:
    """Return (display_name, persona, tools, skills) for an agent key."""
    if agent_key == "scout":
        app = robot_scout.APP
        persona = app.agent.instructions
        tools = [d for d in app.agent.tools.tools.values() if not d.func.__name__.endswith("_fallback")]
        skills = list(app.agent.skills.values())
        return "Scout (ToyWorld)", persona, tools, skills
    if agent_key == "irsim":
        app = robot_irsim.APP
        tools = [f.__edgevox_tool__ for f in app.tools]
        return "Scout (IR-SIM)", app.instructions, tools, list(app.skills)
    if agent_key == "panda":
        app = robot_panda.APP
        tools = [f.__edgevox_tool__ for f in app.tools]
        return "Panda (MuJoCo arm)", app.instructions, tools, list(app.skills)
    if agent_key == "humanoid":
        app = robot_humanoid.APP
        tools = [f.__edgevox_tool__ for f in app.tools]
        return "G1 (MuJoCo humanoid)", app.instructions, tools, list(app.skills)
    raise ValueError(f"unknown agent key: {agent_key}")


def _openai_schemas_for(agent_key: str) -> tuple[str, str, list[dict[str, Any]], set[str]]:
    """Return (display_name, persona, openai_tool_list, set_of_tool_names)."""
    name, persona, tools, skills = _agent_surface(agent_key)
    schemas: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    for t in tools:
        schemas.append(t.openai_schema())
        tool_names.add(t.name)
    for s in skills:
        schemas.append(s.as_tool_descriptor().openai_schema())
        tool_names.add(s.name)
    return name, persona, schemas, tool_names


# --- scenarios ---


@dataclass
class Scenario:
    agent: str
    name: str
    user: str
    expected_tool: str | None
    expected_args: dict[str, Any] = field(default_factory=dict)
    # Absolute tolerance per numeric arg. Missing → exact match required.
    arg_tolerance: dict[str, float] = field(default_factory=dict)
    # Acceptable alternative tool names (some LLMs might reasonably pick
    # either, e.g. ``get_pose`` vs ``current_room`` for "where are you").
    also_ok: tuple[str, ...] = ()
    tags: list[str] = field(default_factory=list)


def scenarios() -> list[Scenario]:
    out: list[Scenario] = []

    # ==================== scout (ToyWorld) ====================
    out += [
        Scenario("scout", "nav_kitchen", "Go to the kitchen.", "navigate_to_room", {"room": "kitchen"}),
        Scenario("scout", "nav_bedroom", "Drive to the bedroom please.", "navigate_to_room", {"room": "bedroom"}),
        Scenario("scout", "light_on_kitchen", "Turn on the kitchen light.", "turn_on_light", {"room": "kitchen"}),
        Scenario("scout", "light_off_bedroom", "Turn off the bedroom lights.", "turn_off_light", {"room": "bedroom"}),
        Scenario("scout", "query_battery", "What's your battery level?", "battery_level"),
        Scenario(
            "scout",
            "query_current_room",
            "Which room are you in right now?",
            "current_room",
            also_ok=("get_pose",),
        ),
        Scenario("scout", "list_rooms", "What rooms do you know about?", "list_rooms"),
        Scenario("scout", "return_home", "Come back home.", "return_home", also_ok=("navigate_to_room",)),
        Scenario(
            "scout",
            "light_check_kitchen",
            "Is the kitchen light on?",
            "light_state",
            {"room": "kitchen"},
        ),
        Scenario("scout", "conversational", "Hi Scout, how's it going?", None, tags=["no-tool"]),
    ]

    # ==================== irsim ====================
    out += [
        Scenario("irsim", "nav_kitchen", "Drive to the kitchen.", "navigate_to_room", {"room": "kitchen"}),
        Scenario("irsim", "nav_office", "Please go to the office.", "navigate_to_room", {"room": "office"}),
        Scenario(
            "irsim",
            "nav_point_35_4",
            "Drive to position x equals 3.5, y equals 4.0.",
            "navigate_to_point",
            {"x": 3.5, "y": 4.0},
            arg_tolerance={"x": 0.1, "y": 0.1},
        ),
        Scenario(
            "irsim",
            "nav_point_55",
            "Move to coordinates (5, 5).",
            "navigate_to_point",
            {"x": 5.0, "y": 5.0},
            arg_tolerance={"x": 0.1, "y": 0.1},
        ),
        Scenario("irsim", "stop", "Stop right now.", "stop"),
        Scenario(
            "irsim",
            "describe_room_kitchen",
            "Describe the kitchen for me.",
            "describe_room",
            {"room": "kitchen"},
        ),
        Scenario(
            "irsim",
            "query_pose",
            "Where are you right now?",
            "get_pose",
            also_ok=("current_room",),
        ),
        Scenario("irsim", "return_home", "Head back to the center.", "return_home", also_ok=("navigate_to_room",)),
        Scenario("irsim", "query_battery", "What's your battery at?", "battery_level"),
        Scenario("irsim", "conversational", "Hey Scout, are you there?", None, tags=["no-tool"]),
    ]

    # ==================== panda ====================
    out += [
        Scenario("panda", "grasp_red", "Pick up the red cube.", "grasp", {"object": "red_cube"}),
        Scenario("panda", "grasp_blue", "Grab the blue block.", "grasp", {"object": "blue_cube"}),
        Scenario(
            "panda",
            "hover_green",
            "Hover the gripper above the green cube.",
            "move_above_object",
            {"object": "green_cube"},
        ),
        Scenario("panda", "release", "Let go of it.", "release"),
        Scenario("panda", "home", "Return to home position.", "goto_home"),
        Scenario(
            "panda",
            "move_point",
            "Move the arm to x=0.3, y=0.1, z=0.4.",
            "move_to_point",
            {"x": 0.3, "y": 0.1, "z": 0.4},
            arg_tolerance={"x": 0.05, "y": 0.05, "z": 0.05},
        ),
        Scenario(
            "panda",
            "query_ee",
            "Where is the gripper right now?",
            "get_ee_pose",
            also_ok=("list_objects",),
        ),
        Scenario("panda", "query_gripper", "Are you holding anything?", "get_gripper_state"),
        Scenario(
            "panda",
            "locate_green",
            "Where is the green cube on the table?",
            "locate_object",
            {"name": "green_cube"},
            also_ok=("list_objects",),
        ),
        Scenario("panda", "list_objects", "What objects are on the table?", "list_objects"),
    ]

    # ==================== humanoid ====================
    out += [
        Scenario(
            "humanoid",
            "walk_half_meter",
            "Walk forward half a meter.",
            "walk_forward",
            {"distance": 0.5},
            arg_tolerance={"distance": 0.05},
        ),
        Scenario(
            "humanoid",
            "walk_one_meter",
            "Step forward one meter.",
            "walk_forward",
            {"distance": 1.0},
            arg_tolerance={"distance": 0.05},
        ),
        Scenario(
            "humanoid",
            "walk_back_30cm",
            "Step back 30 centimeters.",
            "walk_backward",
            {"distance": 0.3},
            arg_tolerance={"distance": 0.05},
        ),
        Scenario(
            "humanoid",
            "turn_left_90",
            "Turn left ninety degrees.",
            "turn_left",
            {"degrees": 90.0},
            arg_tolerance={"degrees": 1.0},
        ),
        Scenario(
            "humanoid",
            "turn_right_45",
            "Rotate right 45 degrees.",
            "turn_right",
            {"degrees": 45.0},
            arg_tolerance={"degrees": 1.0},
        ),
        Scenario("humanoid", "stand", "Stand up straight please.", "stand"),
        Scenario("humanoid", "query_pose", "Where are you currently?", "get_pose"),
        Scenario(
            "humanoid",
            "query_standing",
            "Are you still standing?",
            "is_standing",
            also_ok=("get_pose",),
        ),
        Scenario(
            "humanoid",
            "walk_back_half",
            "Walk backward half a meter.",
            "walk_backward",
            {"distance": 0.5},
            arg_tolerance={"distance": 0.05},
        ),
        Scenario("humanoid", "conversational", "Hi G1, nice to meet you.", None, tags=["no-tool"]),
    ]

    return out


# --- grading ---


@dataclass
class GradeResult:
    score: int
    flags: list[str]
    predicted_tool: str | None
    predicted_args: dict[str, Any]
    reply_text: str


def _args_close(expected: dict[str, Any], actual: dict[str, Any], tol: dict[str, float]) -> tuple[bool, list[str]]:
    """Check every expected arg is present in ``actual`` within tolerance.
    Return (all_ok, flags_for_wrong_args). Extra keys in actual are fine.
    """
    flags: list[str] = []
    all_ok = True
    for k, want in expected.items():
        if k not in actual:
            flags.append(f"missing arg {k!r}")
            all_ok = False
            continue
        got = actual[k]
        if isinstance(want, bool) or isinstance(got, bool):
            if bool(got) != bool(want):
                flags.append(f"arg {k}: expected {want!r}, got {got!r}")
                all_ok = False
            continue
        if isinstance(want, (int, float)):
            # Coerce string-encoded numbers ("3.5", "90") before comparing —
            # Llama-3.x-1B and a few smaller models return all args as
            # strings regardless of the schema's declared ``number`` type.
            try:
                got_num = float(got) if not isinstance(got, bool) else None
            except (TypeError, ValueError):
                got_num = None
            if got_num is None:
                flags.append(f"arg {k}: expected {want}, got {got!r} (non-numeric)")
                all_ok = False
                continue
            delta = tol.get(k, 1e-9)
            if abs(got_num - float(want)) > delta:
                flags.append(f"arg {k}: expected {want}, got {got_num}")
                all_ok = False
            continue
        if str(got).strip().lower() != str(want).strip().lower():
            flags.append(f"arg {k}: expected {want!r}, got {got!r}")
            all_ok = False
    return all_ok, flags


def grade(scn: Scenario, raw: dict[str, Any], parsers: tuple[str, ...], tool_schemas: list[dict]) -> GradeResult:
    """Grade a single LLM response against one scenario.

    Scoring (0-100):
    - No tool call expected & none emitted → 100.
    - No tool call expected but one emitted → 40 (flag: "spurious").
    - Tool call expected, wrong tool → 25 (flag: "wrong tool").
    - Tool call expected, right tool, all args correct → 100.
    - Tool call expected, right tool, some args wrong → scale by arg-correct-fraction (min 60 if tool matched).
    - No tool call emitted when one was expected → 0.
    """
    choice = raw["choices"][0]["message"]
    content = choice.get("content") or ""
    structured_calls = choice.get("tool_calls") or []

    predicted_name: str | None = None
    predicted_args: dict[str, Any] = {}

    if structured_calls:
        first = structured_calls[0]
        fn = first.get("function", {})
        predicted_name = fn.get("name")
        try:
            predicted_args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            predicted_args = {}
    else:
        # Fall through to the content parser chain (covers every family
        # llama-cpp-python doesn't hoist into structured tool_calls).
        tool_names = {t["function"]["name"] for t in tool_schemas}
        calls, _, _ = parse_tool_calls_from_content(
            content,
            preset_parsers=parsers,
            known_tools=tool_names,
            tool_schemas=tool_schemas,
        )
        if calls:
            first = calls[0]
            fn = first.get("function", {})
            predicted_name = fn.get("name")
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    predicted_args = json.loads(args_raw)
                except Exception:
                    predicted_args = {}
            elif isinstance(args_raw, dict):
                predicted_args = args_raw

    if scn.expected_tool is None:
        if predicted_name is None:
            return GradeResult(100, [], None, {}, content)
        return GradeResult(40, ["spurious tool call"], predicted_name, predicted_args, content)

    if predicted_name is None:
        return GradeResult(0, ["no tool call emitted"], None, {}, content)

    accepted_names = {scn.expected_tool, *scn.also_ok}
    if predicted_name not in accepted_names:
        return GradeResult(
            25,
            [f"wrong tool: picked {predicted_name!r}, expected one of {sorted(accepted_names)}"],
            predicted_name,
            predicted_args,
            content,
        )

    # Tool matched. If we accepted an `also_ok` alias the args schema may
    # differ — only enforce arg matching against the *primary* expected
    # tool, and if the predicted name is one of the aliases we skip arg
    # checks and just give a 90 (partial credit).
    if predicted_name != scn.expected_tool:
        return GradeResult(90, [f"alias-tool: {predicted_name!r}"], predicted_name, predicted_args, content)

    ok, arg_flags = _args_close(scn.expected_args, predicted_args, scn.arg_tolerance)
    if ok:
        return GradeResult(100, [], predicted_name, predicted_args, content)

    # Partial credit: scale by fraction of expected args that match.
    correct = sum(
        1 for k in scn.expected_args if f"arg {k}:" not in " ".join(arg_flags) and f"missing arg {k!r}" not in arg_flags
    )
    total = max(1, len(scn.expected_args))
    score = 60 + int(30 * correct / total)
    return GradeResult(score, arg_flags, predicted_name, predicted_args, content)


# --- runner ---


@dataclass
class ModelResult:
    slug: str
    per_scenario: dict[str, dict[str, Any]] = field(default_factory=dict)
    load_s: float = 0.0
    total_s: float = 0.0
    error: str | None = None

    @property
    def avg_score(self) -> float:
        if not self.per_scenario:
            return 0.0
        return sum(r["score"] for r in self.per_scenario.values()) / len(self.per_scenario)

    @property
    def tool_call_rate(self) -> float:
        """Fraction of scenarios where the model emitted a tool call at all."""
        total = [r for r in self.per_scenario.values() if r["expected_tool"] is not None]
        if not total:
            return 0.0
        emitted = sum(1 for r in total if r["predicted_tool"] is not None)
        return emitted / len(total)


def run_model(slug: str, scns: list[Scenario], *, n_ctx: int, temperature: float, max_tokens: int) -> ModelResult:
    result = ModelResult(slug=slug)
    try:
        t0 = time.perf_counter()
        llm = LLM(model_path=f"preset:{slug}", n_ctx=n_ctx)
        result.load_s = time.perf_counter() - t0
    except Exception as e:
        result.error = f"load failed: {e!r}"
        return result

    parsers = tuple(llm._tool_call_parsers)

    # Cache per-agent payload so we only build schemas once per model.
    agent_payloads: dict[str, tuple[str, list[dict], set[str]]] = {}

    for agent_key in ("scout", "irsim", "panda", "humanoid"):
        _, persona, schemas, names = _openai_schemas_for(agent_key)
        agent_payloads[agent_key] = (persona, schemas, names)

    for scn in scns:
        persona, schemas, _ = agent_payloads[scn.agent]
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": scn.user},
        ]
        t0 = time.perf_counter()
        try:
            raw = llm.complete(
                messages,
                tools=schemas,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            elapsed = time.perf_counter() - t0
            gr = grade(scn, raw, parsers, schemas)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            gr = GradeResult(0, [f"exception: {type(e).__name__}: {e}"], None, {}, "")
        result.per_scenario[f"{scn.agent}/{scn.name}"] = {
            "agent": scn.agent,
            "scenario": scn.name,
            "user": scn.user,
            "expected_tool": scn.expected_tool,
            "expected_args": scn.expected_args,
            "predicted_tool": gr.predicted_tool,
            "predicted_args": gr.predicted_args,
            "reply_text": gr.reply_text,
            "score": gr.score,
            "flags": gr.flags,
            "elapsed_s": elapsed,
            "tags": scn.tags,
        }
        result.total_s += elapsed

    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    return result


def render_report(results: list[ModelResult], scns: list[Scenario], config: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Robot tool-calling benchmark",
        "",
        "Multi-model comparison of LLM tool-call emissions across four robot agent surfaces "
        "(scout, irsim, panda, humanoid). Generated by `scripts/bench_robot_tool_calling.py`.",
        "",
        f"Run config: temperature={config['temperature']} · max_tokens={config['max_tokens']} · "
        f"n_ctx={config['n_ctx']} · scenarios={len(scns)}",
        "",
        "## Scoreboard",
        "",
        "| Model | Avg score | Tool-call rate | Load (s) | Per-reply (s) | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.avg_score if r.error is None else 1):
        if r.error:
            lines.append(f"| `{r.slug}` | — | — | — | — | {r.error} |")
            continue
        per = r.total_s / max(1, len(r.per_scenario))
        lines.append(
            f"| `{r.slug}` | **{r.avg_score:.1f}** | {r.tool_call_rate * 100:.0f}% | {r.load_s:.1f} | {per:.2f} | — |"
        )
    lines.append("")

    # Per-agent breakdown.
    agents = ["scout", "irsim", "panda", "humanoid"]
    lines.append("## Per-agent average")
    lines.append("")
    lines.append("| Model | " + " | ".join(agents) + " |")
    lines.append("|" + "|".join(["---"] * (len(agents) + 1)) + "|")
    for r in sorted(results, key=lambda r: -r.avg_score if r.error is None else 1):
        if r.error:
            continue
        row = [f"`{r.slug}`"]
        for a in agents:
            scores = [v["score"] for v in r.per_scenario.values() if v["agent"] == a]
            row.append(f"{sum(scores) / len(scores):.0f}" if scores else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-scenario table.
    lines.append("## Per-scenario scores")
    lines.append("")
    header = ["Model"] + [f"{s.agent}/{s.name}" for s in scns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sorted(results, key=lambda r: -r.avg_score if r.error is None else 1):
        if r.error:
            continue
        row = [f"`{r.slug}`"]
        for s in scns:
            v = r.per_scenario.get(f"{s.agent}/{s.name}")
            row.append(f"{v['score']:d}" if v else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Scenario-by-scenario sample predictions (first model's reply per scenario).
    lines.append("## Sample predictions")
    lines.append("")
    for s in scns:
        lines.append(f"### `{s.agent}/{s.name}` — {s.user!r}")
        lines.append("")
        lines.append(
            f"*Expected:* `{s.expected_tool}({s.expected_args})`" if s.expected_tool else "*Expected:* (no tool call)"
        )
        lines.append("")
        for r in results:
            if r.error:
                continue
            v = r.per_scenario.get(f"{s.agent}/{s.name}")
            if not v:
                continue
            predicted = f"{v['predicted_tool']}({v['predicted_args']})" if v["predicted_tool"] else "(no tool call)"
            flags = f"  ⚠ {'; '.join(v['flags'])}" if v["flags"] else ""
            lines.append(f"- **`{r.slug}`** → `{predicted}` · score {v['score']}{flags}")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Each scenario is a natural-language utterance the voice pipeline would hand the agent, "
        "paired with the expected tool call. The full agent persona + OpenAI-format tool schemas "
        "are fed to the LLM via `LLM.complete(messages, tools=...)`. When the binding doesn't "
        "hoist structured `tool_calls`, EdgeVox's post-hoc parser chain runs against the raw "
        "content (same chain the agent loop uses)."
    )
    lines.append("")
    lines.append(
        "Scoring: 100 for exact tool + exact args, 90 for an accepted alias, 60-89 for right "
        "tool with some-correct args, 25 for wrong tool, 40 for a spurious call, 0 for no call "
        "when one was expected."
    )
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append(
        "- [`chess-commentary-benchmark.md`](./chess-commentary-benchmark) — commentary-quality benchmark on the same model pool."
    )
    lines.append(
        "- [`slm-tool-calling-benchmark.md`](./slm-tool-calling-benchmark) — single-tool detection smoke test."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Robot tool-calling benchmark.")
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS))
    parser.add_argument("--include-large", action="store_true", help=f"Also run: {', '.join(_LARGE_MODELS)}")
    parser.add_argument("--only", default=None, help="Comma-separated agent keys to bench (default: all four).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="docs/documentation/reports/robot-tool-calling-benchmark.md",
    )
    parser.add_argument(
        "--json-output",
        default="docs/documentation/reports/data/robot-tool-calling-benchmark.json",
    )
    args = parser.parse_args()

    slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    if args.include_large:
        slugs += list(_LARGE_MODELS)

    scns = scenarios()
    if args.only:
        wanted = {a.strip() for a in args.only.split(",") if a.strip()}
        scns = [s for s in scns if s.agent in wanted]
        if not scns:
            print(f"No scenarios match --only={args.only!r}", file=sys.stderr)
            return 1

    results: list[ModelResult] = []
    for slug in slugs:
        print(f"\n{'=' * 78}\n{slug}\n{'=' * 78}", flush=True)
        r = run_model(slug, scns, n_ctx=args.n_ctx, temperature=args.temperature, max_tokens=args.max_tokens)
        if r.error:
            print(f"  {r.error}", flush=True)
        else:
            print(
                f"  avg score: {r.avg_score:.1f} / 100 over {len(r.per_scenario)} scenarios ({r.total_s:.1f}s total)",
                flush=True,
            )
            for key, v in r.per_scenario.items():
                pred = f"{v['predicted_tool']}({v['predicted_args']})" if v["predicted_tool"] else "(no call)"
                flags = f"  ⚠ {'; '.join(v['flags'])}" if v["flags"] else ""
                print(f"  [{key}] score={v['score']}  pred={pred}{flags}", flush=True)
        results.append(r)

    cfg = {"temperature": args.temperature, "max_tokens": args.max_tokens, "n_ctx": args.n_ctx}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(render_report(results, scns, cfg))
    print(f"\nWrote report → {args.output}")

    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "scenarios": [
            {
                "agent": s.agent,
                "name": s.name,
                "user": s.user,
                "expected_tool": s.expected_tool,
                "expected_args": s.expected_args,
                "tags": s.tags,
            }
            for s in scns
        ],
        "models": [
            {
                "slug": r.slug,
                "load_s": r.load_s,
                "error": r.error,
                "avg_score": r.avg_score,
                "tool_call_rate": r.tool_call_rate,
                "scenarios": r.per_scenario,
            }
            for r in results
        ],
    }
    Path(args.json_output).write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote JSON  → {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
