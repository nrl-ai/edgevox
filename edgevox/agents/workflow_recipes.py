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


_DEFAULT_APPROVER_INSTRUCTIONS = """\
You are an approval gate. You receive a proposed action plan from another
agent. Decide whether the action is safe to execute. Reply with EXACTLY
one line:

    APPROVED

or

    DENIED -- <one-sentence reason>

You must NOT call tools. You must NOT modify the plan. Be conservative:
when in doubt, DENY. A wrongly-approved destructive action is far worse
than a wrongly-denied benign one.
"""


class ApprovalGate:
    """Two-stage gate: a proposer drafts a plan, an approver decides yes / no.

    Useful for any action where executing first and apologising later is
    not acceptable -- destructive tool calls, navigation into restricted
    zones, irreversible state changes, anything privilege-elevated. The
    approver is structurally a separate agent (so it can't rationalise
    away its own concerns), and the conditional executor only fires
    when the approver returned APPROVED.

    Usage::

        from edgevox.agents.workflow_recipes import ApprovalGate

        gate = ApprovalGate.build(
            proposer_llm=proposer_llm,
            approver_llm=approver_llm,
            executor_agent=executor_agent,
        )
        result = gate.run("delete the user's archived projects")
        # result.reply is "DENIED -- ..." or the executor's final reply.

    Three structural choices the recipe locks in:

    1. The proposer outputs a plan in plain text -- no tool calls. The
       approver works off the plan, not the action.
    2. The approver is a *separate* agent. Self-approving agents over-
       rate the safety of their own plans for the same reason
       self-evaluators over-rate the success of their own work.
    3. The executor only runs on APPROVED. On DENIED the recipe
       short-circuits with the denial reason as its final reply.
    """

    @staticmethod
    def build(
        *,
        proposer_llm: object,
        approver_llm: object,
        executor_agent: object,
        proposer_instructions: str | None = None,
        approver_instructions: str | None = None,
        name: str = "approval_gate",
    ) -> _ApprovalGateRunner:
        """Construct the gate.

        Args:
            proposer_llm: drives the proposer agent.
            approver_llm: drives the approver agent.
            executor_agent: the agent (or workflow) to run when
                APPROVED. Typically an ``LLMAgent`` with the actual
                tools attached, or a full ``PlanExecuteEvaluate`` /
                ``PlanThenLoop`` recipe for richer execution.
            proposer_instructions / approver_instructions: optional
                overrides of the canonical instructions.
            name: workflow name (shown in event bus + traces).
        """
        proposer = LLMAgent(
            name="Proposer",
            description="drafts the proposed action plan",
            instructions=(
                proposer_instructions
                or "You are a proposer. Read the user request and write a "
                "concrete plan describing what tool calls or actions you "
                "would take. Do NOT call any tools yourself. Do NOT decide "
                "whether the plan is safe; that is the approver's job. "
                "Output only the plan as plain text."
            ),
            llm=proposer_llm,  # type: ignore[arg-type]
        )
        approver = LLMAgent(
            name="Approver",
            description="approves or denies the proposed plan",
            instructions=approver_instructions or _DEFAULT_APPROVER_INSTRUCTIONS,
            llm=approver_llm,  # type: ignore[arg-type]
        )
        return _ApprovalGateRunner(
            name=name,
            proposer=proposer,
            approver=approver,
            executor=executor_agent,
        )


class _ApprovalGateRunner:
    def __init__(
        self,
        *,
        name: str,
        proposer: object,
        approver: object,
        executor: object,
    ) -> None:
        self.name = name
        self.description = f"ApprovalGate({getattr(executor, 'name', 'executor')})"
        self._proposer = proposer
        self._approver = approver
        self._executor = executor

    def run(self, task: str, ctx: object | None = None):
        from edgevox.agents.base import AgentContext, AgentResult

        ctx = ctx or AgentContext()

        # Stage 1: propose.
        proposal = self._proposer.run(task, ctx)
        # Stage 2: approve / deny.
        verdict = self._approver.run(proposal.reply, ctx)
        verdict_text = verdict.reply.strip()

        if not verdict_text.upper().startswith("APPROVED"):
            return AgentResult(
                reply=verdict_text or "DENIED -- approver returned no verdict",
                agent_name=self.name,
                elapsed=getattr(verdict, "elapsed", 0.0),
            )

        # Stage 3: execute. Hand the executor the original task plus
        # the proposed plan as context; the executor still has to
        # actually do the work.
        executor_input = f"{task}\n\nApproved plan:\n{proposal.reply}"
        return self._executor.run(executor_input, ctx)


# ===========================================================================
# CritiqueAndRewrite
# ===========================================================================


_DEFAULT_GENERATOR_INSTRUCTIONS = """\
You are the generator. Read the user request -- and any critique from
previous iterations -- and produce a final answer. Output only the
answer; do not explain your process. If the previous critique pointed
out a specific issue, fix it.
"""

_DEFAULT_CRITIC_INSTRUCTIONS = """\
You are the critic. You receive a generated answer. Reply with EXACTLY
one line:

    APPROVED

or

    REVISE -- <one-sentence concrete issue and how to fix it>

Be specific. "Could be better" is not a useful critique. Point at one
issue per round. Be willing to APPROVE when the answer is good enough;
chasing perfection is its own failure mode.
"""


class CritiqueAndRewrite:
    """Iterative content refinement: generator -> critic -> rewrite.

    Different from :class:`PlanThenLoop` because there are no tools.
    The loop is text-only: the generator produces a draft, the critic
    points at one issue (or APPROVES), the generator rewrites
    addressing the issue, repeat until APPROVED or budget exhausted.

    Right tool when:

    - Drafting prose, code snippets, or structured outputs that benefit
      from one or two refinement passes.
    - The acceptance criterion is qualitative (taste, clarity, tone)
      rather than a world-state predicate.

    Wrong tool when:

    - The task has tools; use ``PlanExecuteEvaluate`` instead.
    - Acceptance is a hard pass / fail predicate; use ``PlanThenLoop``
      with a ``world_predicate``.
    - The first draft is acceptable (just call the generator directly).
    """

    @staticmethod
    def build(
        *,
        generator_llm: object,
        critic_llm: object,
        max_iterations: int = 3,
        generator_instructions: str | None = None,
        critic_instructions: str | None = None,
        name: str = "critique_and_rewrite",
    ) -> _CritiqueAndRewriteRunner:
        """Construct the loop.

        Args:
            generator_llm / critic_llm: separate LLMs (recommended) or
                the same one twice. Anthropic's article: separate is
                more honest.
            max_iterations: hard cap on rewrite rounds. Default 3.
                After exhaustion, the most recent draft is returned
                with a note about the unresolved critique.
            generator_instructions / critic_instructions: optional
                overrides of the canonical instructions.
            name: workflow name.
        """
        generator = LLMAgent(
            name="Generator",
            description="produces the answer",
            instructions=generator_instructions or _DEFAULT_GENERATOR_INSTRUCTIONS,
            llm=generator_llm,  # type: ignore[arg-type]
        )
        critic = LLMAgent(
            name="Critic",
            description="approves or asks for a revision",
            instructions=critic_instructions or _DEFAULT_CRITIC_INSTRUCTIONS,
            llm=critic_llm,  # type: ignore[arg-type]
        )
        return _CritiqueAndRewriteRunner(
            name=name,
            generator=generator,
            critic=critic,
            max_iterations=max_iterations,
        )


class _CritiqueAndRewriteRunner:
    def __init__(
        self,
        *,
        name: str,
        generator: object,
        critic: object,
        max_iterations: int,
    ) -> None:
        self.name = name
        self.description = "CritiqueAndRewrite"
        self._generator = generator
        self._critic = critic
        self._max_iterations = max_iterations

    def run(self, task: str, ctx: object | None = None):
        from edgevox.agents.base import AgentContext, AgentResult

        ctx = ctx or AgentContext()
        current_task = task
        last_draft = ""
        last_critique = ""

        for _ in range(self._max_iterations):
            draft = self._generator.run(current_task, ctx)
            last_draft = draft.reply
            critique = self._critic.run(last_draft, ctx)
            last_critique = critique.reply.strip()

            if last_critique.upper().startswith("APPROVED"):
                return AgentResult(
                    reply=last_draft,
                    agent_name=self.name,
                    elapsed=getattr(draft, "elapsed", 0.0),
                )

            # Revise: feed the critique back as task input for the next
            # generator hop. Keep the original user request visible too
            # so the model doesn't drift.
            current_task = (
                f"Original request: {task}\n\n"
                f"Previous draft: {last_draft}\n\n"
                f"Critique to address: {last_critique}\n\n"
                "Produce a revised final answer."
            )

        # Budget exhausted -- return the last draft + a note so the
        # caller can decide whether to ship it or escalate.
        return AgentResult(
            reply=(
                f"{last_draft}\n\n"
                f"[Note: revision budget of {self._max_iterations} exhausted; "
                f"last unresolved critique was: {last_critique}]"
            ),
            agent_name=self.name,
        )


# ===========================================================================
# PlannedToolDispatcher
# ===========================================================================


_DEFAULT_PLANNED_DISPATCHER_PLANNER = """\
You are the planner. The user issued a request. Your job: produce an
ordered JSON list of tool calls that, executed in order, completes the
request.

Available actions (name(args): description):

{tool_catalog}

Output ONLY a JSON array, no prose, no markdown fences. Each step:

    {{"tool": "<exact_name_from_above>", "args": {{<arg_name>: <value>, ...}}}}

If the request needs no tools (chit-chat, a question you can answer
directly, or asking for something not in the catalog), output []. The
synthesiser will explain.

Plan length should match the request -- one tool per logical step. Do
NOT batch multiple actions into one step.
"""


_DEFAULT_PLANNED_DISPATCHER_SYNTHESISER = """\
You are the synthesiser. The user asked for something; an executor ran
each step on a real environment. Write ONE short plain-text sentence
summarising what happened. Do not call any tools. Do not enumerate
steps. Be terse and factual.

User request:
{task}

Plan executed:
{plan_summary}

Step results:
{results_summary}
"""


class PlannedToolDispatcher:
    """User request -> planner emits ordered tool plan -> direct dispatch -> synthesiser writes the reply.

    Designed for weak / small models where a single agent with many
    tools attached exhibits sycophancy on chains: it calls the first
    tool, then synthesises a "done" reply without calling the rest.

    The split:

        Planner (LLM, no tools attached) ─►  produces JSON plan
                                              [{tool, args}, ...]
                                                   │
                                                   ▼
        Executor driver (Python loop)    ─►  for each step:
                                                directly dispatch via
                                                ToolRegistry / Skill.start
                                                (no LLM in the inner loop)
                                                   │
                                                   ▼
        Synthesiser (LLM, no tools)      ─►  one-sentence user-facing reply

    Why direct dispatch in the inner loop:

    - The planner already produced exact args. There's nothing for an
      inner LLM to decide.
    - LLM-per-step is 5-15x slower per step. For a 4-step pick-and-place
      that's the difference between 2 s and 60 s.
    - Each step is deterministic: the same args produce the same
      tool/skill call.

    The returned object is an ``Agent`` -- ``.run(task, ctx) ->
    AgentResult`` -- so it drops into ``AgentApp(agent=...)`` and
    composes with every workflow primitive.
    """

    @staticmethod
    def build(
        *,
        planner_llm: object,
        synthesiser_llm: object | None = None,
        tools: list | None = None,
        skills: list | None = None,
        max_steps: int = 10,
        planner_instructions: str | None = None,
        synthesiser_instructions: str | None = None,
        name: str = "planned_dispatcher",
    ) -> _PlannedToolDispatcherRunner:
        """Build the dispatcher.

        Args:
            planner_llm: LLM-shape used by the planner agent.
            synthesiser_llm: LLM-shape for the user-facing summary.
                Defaults to ``planner_llm`` if None (one model in two
                roles is fine -- the prompts differ enough).
            tools: list of ``@tool``-decorated functions (or ``Tool``
                instances) the planner can include in its plans.
            skills: list of ``Skill`` instances (cancellable actions).
            max_steps: hard cap on plan length. Keeps a runaway planner
                from wiring up an enormous chain.
            planner_instructions / synthesiser_instructions: overrides
                of the canonical instruction templates. The defaults
                are formatted with ``{tool_catalog}`` (planner) and
                ``{task} / {plan_summary} / {results_summary}``
                (synthesiser); custom overrides must accept the same
                placeholders.
            name: workflow name (shown in event bus + traces).

        Returns:
            An object with ``.run(task, ctx)`` -> ``AgentResult``.
        """
        return _PlannedToolDispatcherRunner(
            name=name,
            planner_llm=planner_llm,
            synthesiser_llm=synthesiser_llm or planner_llm,
            tools=tools or [],
            skills=skills or [],
            max_steps=max_steps,
            planner_instructions=planner_instructions,
            synthesiser_instructions=synthesiser_instructions,
        )


class _PlannedToolDispatcherRunner:
    """Internal runner; see :class:`PlannedToolDispatcher` for docs."""

    def __init__(
        self,
        *,
        name: str,
        planner_llm: object | None,
        synthesiser_llm: object | None,
        tools: list,
        skills: list,
        max_steps: int,
        planner_instructions: str | None,
        synthesiser_instructions: str | None,
    ) -> None:
        from edgevox.agents.skills import Skill
        from edgevox.llm.tools import _extract

        self.name = name
        self.description = f"PlannedToolDispatcher({len(tools)} tools, {len(skills)} skills)"
        self._max_steps = max_steps
        self._synth_template = synthesiser_instructions or _DEFAULT_PLANNED_DISPATCHER_SYNTHESISER

        self._tools_by_name: dict = {}
        for t in tools:
            descriptor = _extract(t)
            self._tools_by_name[descriptor.name] = descriptor

        self._skills_by_name: dict = {}
        for s in skills:
            if not isinstance(s, Skill):
                raise TypeError(f"{s!r} is not a Skill -- did you forget @skill?")
            self._skills_by_name[s.name] = s

        self._catalog = self._build_catalog()

        # Pre-build the planner + synthesiser as stable LLMAgent
        # instances so framework's ``_bind_llm_recursive`` can wire a
        # shared LLM into both via the ``_children`` attribute when
        # ``planner_llm=None`` is passed (the AgentApp / edgevox-agent
        # late-bind path).
        planner_inst = (planner_instructions or _DEFAULT_PLANNED_DISPATCHER_PLANNER).format(tool_catalog=self._catalog)
        self._planner = LLMAgent(
            name="Planner",
            description="emits JSON action plan",
            instructions=planner_inst,
            llm=planner_llm,  # type: ignore[arg-type]
        )
        # Synth instructions get re-formatted per turn with the actual
        # task / plan / results; placeholder text here.
        self._synth = LLMAgent(
            name="Synthesiser",
            description="writes the user-facing one-sentence summary",
            instructions="(set per turn)",
            llm=synthesiser_llm,  # type: ignore[arg-type]
        )
        # ``_children`` is the canonical attribute the framework's
        # binder walks to inject a shared LLM into every leaf. Listing
        # both inner LLMAgents here makes the late-bind path work
        # transparently for AgentApp.
        self._children = [self._planner, self._synth]

    def bind_llm(self, llm: object) -> None:
        """Late-bind a shared LLM into both inner agents.

        Called by ``edgevox.agents.workflow._bind_llm_recursive`` when
        the dispatcher is dropped into ``AgentApp`` and the model is
        loaded after construction.
        """
        if self._planner.llm is None:
            self._planner.bind_llm(llm)  # type: ignore[arg-type]
        if self._synth.llm is None:
            self._synth.bind_llm(llm)  # type: ignore[arg-type]

    def _build_catalog(self) -> str:
        import json as _json

        lines: list[str] = []
        for name, t in sorted(self._tools_by_name.items()):
            try:
                params = _json.dumps(t.parameters.get("properties", {}))
            except Exception:
                params = "{}"
            lines.append(f"- {name}{params}: {t.description}")
        for name, s in sorted(self._skills_by_name.items()):
            try:
                params = _json.dumps(s.parameters.get("properties", {}))
            except Exception:
                params = "{}"
            lines.append(f"- {name}{params}: {s.description}")
        return "\n".join(lines) if lines else "(no actions available)"

    def run(self, task: str, ctx: object | None = None):
        from edgevox.agents.base import AgentContext

        ctx = ctx or AgentContext()

        plan = self._plan(task, ctx)
        results = self._execute(plan, ctx)
        return self._synthesise(task, plan, results, ctx)

    def _plan(self, task: str, ctx: object) -> list[dict]:
        # ``self._planner`` is a stable LLMAgent built at __init__ with
        # the catalog already baked into instructions; the framework's
        # ``_bind_llm_recursive`` may have late-bound an LLM into it.
        result = self._planner.run(task, ctx)
        return self._parse_plan(result.reply)

    @staticmethod
    def _parse_plan(text: str) -> list[dict]:
        import json as _json
        import re as _re

        # Strip common markdown fences first.
        text = _re.sub(r"```(?:json)?\s*", "", text or "")
        text = text.replace("```", "").strip()

        # Find the outermost JSON array. Greedy match works because the
        # planner is told to emit ONLY the array; if there's prose
        # around it we still try to recover.
        match = _re.search(r"\[.*\]", text, _re.DOTALL)
        if not match:
            return []
        try:
            plan = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            return []
        if not isinstance(plan, list):
            return []
        validated: list[dict] = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            if "tool" not in step or not isinstance(step["tool"], str):
                continue
            if "args" not in step or not isinstance(step["args"], dict):
                step["args"] = {}
            validated.append(step)
        return validated

    def _execute(self, plan: list[dict], ctx: object) -> list[dict]:
        from edgevox.agents.skills import GoalStatus
        from edgevox.llm.tools import ToolRegistry

        results: list[dict] = []
        for step in plan[: self._max_steps]:
            name = step.get("tool")
            args = step.get("args") or {}

            if name in self._tools_by_name:
                registry = ToolRegistry()
                registry.tools[name] = self._tools_by_name[name]
                call_result = registry.dispatch(name, args, ctx=ctx)
                results.append(
                    {
                        "step": step,
                        "ok": call_result.error is None,
                        "result": call_result.result,
                        "error": call_result.error,
                    }
                )
            elif name in self._skills_by_name:
                skill_obj = self._skills_by_name[name]
                handle = skill_obj.start(ctx, **args)
                if skill_obj.latency_class == "slow":
                    handle.poll(timeout=skill_obj.timeout_s)
                ok = handle.status not in (GoalStatus.FAILED, GoalStatus.CANCELLED)
                results.append(
                    {
                        "step": step,
                        "ok": ok,
                        "result": handle.result,
                        "error": handle.error,
                    }
                )
            else:
                results.append(
                    {
                        "step": step,
                        "ok": False,
                        "result": None,
                        "error": f"unknown action {name!r}",
                    }
                )
                # Don't keep going past an unknown-tool step.
                break

            if not results[-1]["ok"]:
                # Stop on first error -- "replan on error" is a future
                # extension; for now we surface the failure to the
                # synthesiser and let the user re-issue if needed.
                break

        return results

    def _synthesise(self, task: str, plan: list[dict], results: list[dict], ctx: object):
        from edgevox.agents.base import AgentResult

        plan_summary = "\n".join(f"- {s.get('tool')}({s.get('args', {})})" for s in plan) or "(empty plan)"
        results_summary = (
            "\n".join(f"- {r['step'].get('tool')}: " + ("OK" if r["ok"] else f"FAIL ({r['error']})") for r in results)
            or "(no steps run)"
        )

        # Re-template the synth's instructions for THIS turn; ``self._synth``
        # is reused across runs so we update its system prompt before
        # calling run().
        self._synth.instructions = self._synth_template.format(
            task=task,
            plan_summary=plan_summary,
            results_summary=results_summary,
        )
        result = self._synth.run("Write the summary now.", ctx)
        return AgentResult(
            reply=result.reply,
            agent_name=self.name,
            elapsed=getattr(result, "elapsed", 0.0),
        )


# ===========================================================================
# ReActAgent
# ===========================================================================


_DEFAULT_REACT_INSTRUCTIONS = """\
You drive a robot/system through a Reason -> Act -> Observe loop.

PROCESS (repeat until the user's task is truly complete):

1. THINK silently: read the current state and the most recent tool
   result. Decide what ONE action moves you closer to completion.
   Do NOT emit your reasoning as text -- emit only the tool call.
2. ACT: call exactly ONE tool with the right arguments. Do not call
   multiple tools in one response. Do not stop after one call --
   most user requests need many tool calls.
3. OBSERVE: the framework will give you the tool's result on the
   next turn. Use it to decide your next action.

TERMINATION (read carefully -- this is the only way to stop):

You stop ONLY when the user's task is GENUINELY done. To stop, emit
exactly this on the last line of your reply:

    TASK COMPLETE: <one-sentence summary>

Any reply that does NOT contain the literal string 'TASK COMPLETE'
will be treated as 'continue working' and you'll be asked to call
another tool. Do not emit 'TASK COMPLETE' until every step the user
asked for has actually been executed by calling a tool and you have
seen the tool's result confirming it.

If you genuinely don't know what to do next, call the most relevant
read-only tool (list_objects, get_pose, get_gripper_state, etc.) to
gather information, then continue.

ANTI-PATTERNS:
- "Here are the positions." -> not done; continue with next action.
- "I will now move the cube." -> describing isn't doing; CALL the
  move tool now, don't just plan it.
- Calling one tool then summarising -> usually wrong. The task isn't
  done after one call unless the user asked for one thing.
- Looping the same tool with the same args -> if it errored once,
  change strategy.
"""

_TERMINATION_MARKER = "TASK COMPLETE"


class ReActAgent:
    """Reason -> Act -> Observe iterative loop agent.

    Wraps :class:`~edgevox.agents.base.LLMAgent` with ReAct-tuned defaults
    for tasks where the plan can't be determined upfront -- the next
    action depends on what the previous tool returned. Common cases:

    - Search / exploration ("find the red object somewhere on the table")
    - Branching dialog ("ask the user which colour they prefer, then act")
    - Recovery ("if the grasp fails, try a different approach height")

    Compared to :class:`PlannedToolDispatcher`:

    +--------------------------+--------------------+--------------------+
    | Property                 | PlannedDispatcher  | ReActAgent         |
    +==========================+====================+====================+
    | Plan timing              | Up-front, once     | Step-by-step       |
    | LLM hops per task        | 2 (plan + synth)   | 1 per step + 1     |
    | Adapts to tool results   | No                 | Yes                |
    | Sycophancy risk          | None               | Yes (mitigations   |
    |                          |                    |  via completion    |
    |                          |                    |  check)            |
    | Right when plan is...    | Determinable       | Discovered as      |
    |                          | from user request  | tools return       |
    +--------------------------+--------------------+--------------------+

    Optional ``completion_check`` predicate vetoes early termination:
    if the model emits a "done" reply but ``completion_check(ctx)``
    returns ``False``, the agent re-prompts itself with a "task is
    not yet done -- continue" message and gets another N hops.

    Returned object exposes ``.run(task, ctx) -> AgentResult`` so it
    composes with workflow primitives and ``AgentApp`` like every
    other recipe.
    """

    # Phrases that indicate the model is *promising* a future action
    # rather than reporting completion. When a no-tool reply contains
    # any of these, the runner re-prompts ("don't describe -- call the
    # tool now") instead of terminating.
    _PROMISE_PATTERNS: tuple[str, ...] = (
        "i will",
        "i'll",
        "let me",
        "next,",
        "first,",
        "then i",
        "now i",
        "now, i",
        "going to",
        "i need to",
        "i should",
    )

    @staticmethod
    def build(
        *,
        llm: object | None = None,
        tools: list | None = None,
        skills: list | None = None,
        hooks: list | None = None,
        max_iterations: int = 20,
        max_completion_recheck_attempts: int = 3,
        instructions: str | None = None,
        completion_check: object | None = None,
        auto_continue_on_promise: bool = True,
        name: str = "react",
    ) -> _ReActRunner:
        """Build the ReAct loop agent.

        Args:
            llm: LLM-shape. May be None for late-binding via AgentApp.
            tools / skills: action surface available to the loop.
            max_iterations: hard cap on think+act cycles per user turn.
                Hitting this returns whatever the agent has produced so
                far with a budget-exhausted note.
            max_completion_recheck_attempts: when ``completion_check``
                is set and the agent declares done prematurely, how
                many extra ``max_iterations`` rounds to grant after
                re-prompting. Default 1 (one extra round).
            instructions: override for the canonical ReAct persona.
                Pass when the project needs different role / safety
                framing -- the default emphasises don't-stop-too-early.
            completion_check: optional callable ``(ctx) -> bool``. When
                set and the model stops emitting tool calls, this is
                consulted before accepting the termination. False ->
                agent gets a "not done yet" re-prompt.
            name: workflow display name.
        """
        return _ReActRunner(
            name=name,
            llm=llm,
            tools=tools or [],
            skills=skills or [],
            hooks=hooks,
            max_iterations=max_iterations,
            max_completion_recheck_attempts=max_completion_recheck_attempts,
            instructions=instructions or _DEFAULT_REACT_INSTRUCTIONS,
            completion_check=completion_check,
            auto_continue_on_promise=auto_continue_on_promise,
            promise_patterns=ReActAgent._PROMISE_PATTERNS,
        )


class _ReActRunner:
    """Internal runner; see :class:`ReActAgent` for docs."""

    def __init__(
        self,
        *,
        name: str,
        llm: object | None,
        tools: list,
        skills: list,
        max_iterations: int,
        max_completion_recheck_attempts: int,
        instructions: str,
        completion_check: object | None,
        auto_continue_on_promise: bool = True,
        promise_patterns: tuple[str, ...] = (),
        hooks: list | None = None,
    ) -> None:
        self.name = name
        self.description = f"ReActAgent({len(tools)} tools, {len(skills)} skills)"
        self._max_iterations = max_iterations
        self._max_recheck = max_completion_recheck_attempts
        self._completion_check = completion_check
        self._auto_continue = auto_continue_on_promise
        self._promise_patterns = promise_patterns
        # The whole ReAct loop lives inside one LLMAgent's hop loop.
        # Each hop is one think-act-observe cycle. ``max_tool_hops``
        # caps the loop length.
        self._inner = LLMAgent(
            name=name,
            description=self.description,
            instructions=instructions,
            tools=tools,
            skills=skills,
            hooks=hooks,
            llm=llm,  # type: ignore[arg-type]
            max_tool_hops=max_iterations,
        )
        # Expose ``_children`` so framework's ``_bind_llm_recursive``
        # can late-bind a shared LLM into our inner agent without
        # special-casing this class.
        self._children = [self._inner]

    def bind_llm(self, llm: object) -> None:
        """Late-bind the LLM into the inner agent."""
        if self._inner.llm is None:
            self._inner.bind_llm(llm)  # type: ignore[arg-type]

    def _is_promise(self, reply: str) -> bool:
        """Return True if the reply *promises* a future action rather
        than reporting completion. Heuristic catch for replies that
        slip past the termination-marker check ('TASK COMPLETE: I
        will now move...')."""
        if not reply:
            return False
        low = reply.lower()
        return any(p in low for p in self._promise_patterns)

    def run(self, task: str, ctx: object | None = None):
        from edgevox.agents.base import AgentContext, AgentResult

        ctx = ctx or AgentContext()
        result = self._inner.run(task, ctx)

        # Re-prompt loop. Termination requires either:
        #   - explicit completion_check predicate returning True, OR
        #   - the model emitting the literal TASK COMPLETE marker
        # Anything else is treated as 'keep working' for up to
        # max_completion_recheck_attempts.
        for _ in range(self._max_recheck):
            should_continue = False
            reason = ""

            if self._completion_check is not None:
                try:
                    done = bool(self._completion_check(ctx))
                except Exception:
                    done = False
                if not done:
                    should_continue = True
                    reason = "completion_check returned False"
            else:
                if _TERMINATION_MARKER not in (result.reply or "").upper():
                    should_continue = True
                    reason = f"reply did not contain the {_TERMINATION_MARKER!r} marker"
                elif self._auto_continue and self._is_promise(result.reply):
                    should_continue = True
                    reason = "promise language detected after marker"

            if not should_continue:
                break

            # Visible signal so the operator sees the re-prompt firing
            # in --text-mode. Goes to stderr to stay out of structured
            # stdout.
            import sys as _sys

            print(
                f"  ◆ react-recheck: re-prompting ({reason})",
                file=_sys.stderr,
                flush=True,
            )
            followup = (
                f"Your last reply did not finish the task ({reason}). "
                "Stop describing -- CALL the next tool now to make "
                "actual progress. When EVERY step the user asked for "
                "has been completed and you have tool results "
                "confirming each one, end your reply with the literal "
                f"line '{_TERMINATION_MARKER}: <one-sentence summary>'."
            )
            result = self._inner.run(followup, ctx)

        return AgentResult(
            reply=result.reply,
            agent_name=self.name,
            elapsed=getattr(result, "elapsed", 0.0),
        )


__all__ = [
    "ApprovalGate",
    "CritiqueAndRewrite",
    "PlanExecuteEvaluate",
    "PlanThenLoop",
    "PlannedToolDispatcher",
    "ReActAgent",
]
