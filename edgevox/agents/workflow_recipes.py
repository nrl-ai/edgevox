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


__all__ = ["PlanExecuteEvaluate"]
