"""Higher-level workflow recipes assembled from the primitives.

The primitives in :mod:`edgevox.agents.workflow` -- ``Sequence``,
``Fallback``, ``Loop``, ``Parallel``, ``Router``, ``Supervisor``,
``Orchestrator``, ``Retry``, ``Timeout`` -- compose into named patterns
that recur across agentic projects. This module names a few of those
patterns and ships them as one-line factories so the boilerplate
("instantiate three LLMAgents with the right instructions, wrap in a
Sequence, return") doesn't have to be re-derived on every project.

Currently shipped:

- :class:`PlanExecuteEvaluate` -- Anthropic's three-agent harness
  pattern. Planner expands the user request into a plan + pass
  criterion, executor runs the plan via tools, evaluator checks the
  executor's report against the criterion. The evaluator is a
  *separate* agent on purpose: agents asked to evaluate their own
  work consistently over-rate it.

Future recipes (not yet shipped):

- ``PlanThenLoop`` -- planner + executor wrapped in a Loop until a
  goal predicate fires (combines the planner-executor decomposition
  with sensor-grounded retry).
- ``RetryWithBackoff`` -- ``Retry`` with exponential delay and
  jittered sleep between attempts.
- ``ParallelWithReducer`` -- ``Parallel`` plus a combine function so
  the caller doesn't have to walk per-agent replies by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgevox.agents.base import LLMAgent
from edgevox.agents.workflow import Sequence

if TYPE_CHECKING:
    from edgevox.agents.skills import Skill
    from edgevox.llm.tools import Tool

# Default instruction blocks tuned for the recipe. Override per-call
# only when the canonical wording doesn't fit the task.

_DEFAULT_PLANNER_INSTRUCTIONS = """\
You are the planner in a three-stage agentic harness (plan -> execute -> evaluate).
Read the user request, then output:

1. A short Goal line restating what success looks like.
2. A numbered Plan: a list of concrete tool-call-shaped steps an
   executor agent can follow. Use the actual tool names the executor
   has. Don't include implementation hints the executor can derive.
3. A Pass criterion: one or two checks that decide whether the plan
   succeeded. The criterion must be objectively testable -- name a
   world-state predicate or an exact tool result, not a vibe.

Do NOT call any tools yourself. Output only the plan as plain text.
Keep the plan minimal -- granular implementation guidance cascades
errors downstream.
"""

_DEFAULT_EXECUTOR_INSTRUCTIONS = """\
You are the executor in a three-stage harness. Read the plan you have
been handed, execute it via the tools available to you, and then write
a one-paragraph report summarising what each step did and what the
final tool results were. Be factual; do not claim a step succeeded
unless the tool returned a success result.
"""

_DEFAULT_EVALUATOR_INSTRUCTIONS = """\
You are the evaluator in a three-stage harness. You receive the
executor's report. Compare it against the Pass criterion that was
declared in the plan. Reply with EXACTLY one line in this shape:

    VERDICT: PASS

or

    VERDICT: FAIL -- <one-sentence reason>

You must NOT call tools. You must NOT rephrase or expand the
evaluation. Be strict: when in doubt, fail. A self-rated success is
not the same as a verified one.
"""


class PlanExecuteEvaluate:
    """Three-agent recipe: plan -> execute -> evaluate.

    Wraps the three-step Sequence in a single factory so projects can
    write::

        from edgevox.agents.workflow_recipes import PlanExecuteEvaluate

        recipe = PlanExecuteEvaluate.build(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            evaluator_llm=evaluator_llm,
            tools=[set_light, navigate_to],
        )
        result = recipe.run("turn on the kitchen light")
        # result.reply starts with "VERDICT: "

    The returned object is a :class:`~edgevox.agents.workflow.Sequence`
    -- it composes with every other workflow primitive (Loop, Retry,
    Timeout, Parallel, etc.) without further wrapping.

    The recipe deliberately uses three *separate* LLMs / agents.
    Anthropic's "harness design for long-running apps" study reports
    that agents asked to evaluate their own output consistently
    over-rate the work; using a separate evaluator with strict
    instructions gives a calibratable PASS / FAIL signal.
    """

    @staticmethod
    def build(
        *,
        planner_llm: object,
        executor_llm: object,
        evaluator_llm: object,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        planner_instructions: str | None = None,
        executor_instructions: str | None = None,
        evaluator_instructions: str | None = None,
        name: str = "plan_execute_evaluate",
        max_tool_hops: int = 5,
    ) -> Sequence:
        """Construct a Sequence(planner, executor, evaluator).

        Args:
            planner_llm: LLM-shaped object passed to the planner agent.
                Anything with a ``.complete(messages, ...)`` method works
                -- including ``ScriptedLLM`` for tests.
            executor_llm: same shape, drives the executor.
            evaluator_llm: same shape, drives the evaluator.
            tools: tools made available to the executor only. The
                planner sees them by *name* in its instructions, but
                does not dispatch them itself.
            skills: cancellable skills made available to the executor.
            planner_instructions: optional override of the canonical
                planner instructions. Pass when you need a different
                planner voice or a project-specific plan format.
            executor_instructions: same, for the executor.
            evaluator_instructions: same, for the evaluator.
            name: workflow name (shown in event bus + traces).
            max_tool_hops: hop budget for the executor (planner +
                evaluator make 0 tool calls so this only matters
                downstream).

        Returns:
            A Sequence whose ``run(task)`` returns an ``AgentResult``
            with the evaluator's verdict line as ``.reply``.
        """
        planner = LLMAgent(
            name="Planner",
            description="produces a plan + pass criterion for the executor",
            instructions=planner_instructions or _DEFAULT_PLANNER_INSTRUCTIONS,
            llm=planner_llm,  # type: ignore[arg-type]
        )
        executor = LLMAgent(
            name="Executor",
            description="executes the planner's plan via tools",
            instructions=executor_instructions or _DEFAULT_EXECUTOR_INSTRUCTIONS,
            tools=tools,
            skills=skills,
            llm=executor_llm,  # type: ignore[arg-type]
            max_tool_hops=max_tool_hops,
        )
        evaluator = LLMAgent(
            name="Evaluator",
            description="evaluates the executor's report against the plan",
            instructions=evaluator_instructions or _DEFAULT_EVALUATOR_INSTRUCTIONS,
            llm=evaluator_llm,  # type: ignore[arg-type]
        )
        return Sequence(name, [planner, executor, evaluator])


class PlanThenLoop:
    """Iterative plan-execute-evaluate with a hard goal predicate.

    The single-shot :class:`PlanExecuteEvaluate` recipe is structurally
    correct -- the evaluator is separate from the executor -- but it
    can still emit a wrong PASS when the underlying model is small
    (the failure mode Anthropic's article explicitly names). Picking
    a stronger model for the evaluator is one fix; the orthogonal fix
    that this recipe ships is to **loop** the recipe and verify
    against an external predicate, so a wrong PASS doesn't terminate
    the run.

    Usage::

        recipe = PlanThenLoop.build(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            evaluator_llm=evaluator_llm,
            tools=[set_light, navigate_to],
            world_predicate=lambda: world.kitchen.light_on
                                    and world.robot.at_room("bedroom"),
            max_iterations=3,
        )
        result = recipe.run("kitchen on, robot in bedroom")

    Two predicates short-circuit the loop:

    1. ``world_predicate(): -> bool`` -- caller-supplied. Reads
       external ground truth (sim state, sensor reading, file
       existence). When this returns ``True`` the loop exits with a
       PASS verdict regardless of what the LLM evaluator said.
    2. The evaluator's own ``VERDICT: PASS`` -- exits the loop
       optimistically, on the assumption that when the model + the
       world both agree, that's good enough.

    Failed attempts feed forward: the planner on iteration N+1
    receives the previous evaluator's reasoning as part of its task
    string, so a model that under-planned step 2 on iteration 1 has
    a shot at fixing it on iteration 2.
    """

    @staticmethod
    def build(
        *,
        planner_llm: object,
        executor_llm: object,
        evaluator_llm: object,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        world_predicate: object | None = None,
        max_iterations: int = 3,
        planner_instructions: str | None = None,
        executor_instructions: str | None = None,
        evaluator_instructions: str | None = None,
        name: str = "plan_then_loop",
        max_tool_hops: int = 5,
    ) -> _PlanThenLoopRunner:
        """Build the iterative recipe.

        Args:
            planner_llm / executor_llm / evaluator_llm: LLM-shaped
                objects, same contract as :class:`PlanExecuteEvaluate`.
            tools / skills: attach to the executor only.
            world_predicate: optional callable (no args) returning
                ``True`` when the goal is reached as observed in
                external state. When set, this overrides the LLM
                evaluator's verdict for early termination.
            max_iterations: hard cap on how many plan-execute-evaluate
                rounds run before the recipe gives up and returns
                the last verdict.
            planner_instructions / executor_instructions /
                evaluator_instructions: forwarded to the inner recipe.
            name: workflow name (shown in event bus + traces).
            max_tool_hops: forwarded.

        Returns:
            A workflow object exposing ``.run(task, ctx) -> AgentResult``.
        """
        inner = PlanExecuteEvaluate.build(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            evaluator_llm=evaluator_llm,
            tools=tools,
            skills=skills,
            planner_instructions=planner_instructions,
            executor_instructions=executor_instructions,
            evaluator_instructions=evaluator_instructions,
            name=f"{name}_inner",
            max_tool_hops=max_tool_hops,
        )
        return _PlanThenLoopRunner(
            name=name,
            inner=inner,
            world_predicate=world_predicate,
            max_iterations=max_iterations,
        )


class _PlanThenLoopRunner:
    """Internal runner -- iterative driver around PlanExecuteEvaluate.

    Implements the workflow Agent protocol (``run(task, ctx)``) so it
    composes with Sequence / Retry / Timeout / Parallel just like the
    underlying Sequence does.
    """

    def __init__(
        self,
        *,
        name: str,
        inner: Sequence,
        world_predicate: object | None,
        max_iterations: int,
    ) -> None:
        self.name = name
        self.description = f"PlanThenLoop({inner.description})"
        self._inner = inner
        self._world_predicate = world_predicate
        self._max_iterations = max_iterations

    def run(self, task: str, ctx: object | None = None):
        """Run plan-execute-evaluate up to max_iterations, exiting early
        on PASS or world-predicate satisfaction. Returns the last
        ``AgentResult``.
        """
        from edgevox.agents.base import AgentContext, AgentResult

        ctx = ctx or AgentContext()
        current_task = task
        last_result: AgentResult | None = None
        history: list[str] = []

        for i in range(self._max_iterations):
            result = self._inner.run(current_task, ctx)
            last_result = result
            verdict = result.reply

            # World predicate is ground truth -- if it passes, we're
            # done regardless of what the LLM evaluator said.
            world_ok = False
            if self._world_predicate is not None:
                try:
                    world_ok = bool(self._world_predicate())
                except Exception:
                    world_ok = False

            if world_ok:
                # Override the verdict to make it explicit that the
                # world predicate was the source of truth.
                final = AgentResult(
                    reply=(
                        f"VERDICT: PASS -- world predicate satisfied on iteration {i + 1}; "
                        f"LLM said: {verdict.strip()[:140]}"
                    ),
                    agent_name=self.name,
                    elapsed=getattr(result, "elapsed", 0.0),
                )
                return final

            # LLM PASS without a world predicate -- trust optimistically.
            if self._world_predicate is None and verdict.strip().upper().startswith("VERDICT: PASS"):
                return result

            # Otherwise: feed the failure forward into the next iteration.
            history.append(f"Iteration {i + 1} verdict: {verdict.strip()[:200]}")
            current_task = (
                f"{task}\n\nPrevious attempts (most recent last):\n"
                + "\n".join(history)
                + "\n\nTry again, addressing the issue called out above."
            )

        return last_result or AgentResult(reply="VERDICT: FAIL -- no iterations ran", agent_name=self.name)


__all__ = ["PlanExecuteEvaluate", "PlanThenLoop"]
