"""Scenario tests -- four real-world situations where agentic workflows
solve a concrete problem.

Each scenario maps to a different pattern:

  S1  ApprovalGate before a destructive / privileged action.
  S2  CritiqueAndRewrite for a robot announcement.
  S3  Parallel exploration of N strategies + aggregator.
  S4  Sensor-error recovery via Retry + Fallback.

All deterministic via ScriptedLLM; all run on ToyWorld; all <1 s total.
"""

from __future__ import annotations

import pytest

from edgevox.agents.base import LLMAgent
from edgevox.agents.sim import ToyWorld
from edgevox.agents.workflow import Fallback, Parallel, Retry
from edgevox.agents.workflow_recipes import ApprovalGate, CritiqueAndRewrite
from edgevox.llm import tool

from .conftest import ScriptedLLM, calls, reply


@pytest.fixture
def world():
    return ToyWorld()


# ---------------------------------------------------------------------------
# S1 -- approval gate before a destructive action
# ---------------------------------------------------------------------------


class TestS1_ApprovalGate:
    """Real situation: a robot is asked to factory-reset its world.

    The reset action is destructive (rooms re-created, lights cleared,
    robot returned to origin). Before doing it, the recipe routes
    through an approver. Two test paths -- one APPROVED, one DENIED --
    plus the side-effect check that the world only mutates on APPROVED.
    """

    def _build_executor_and_world(self, world):
        @tool
        def factory_reset() -> str:
            """Reset the world to defaults. Destructive: clears all lights,
            returns the robot to origin, cancels any in-flight skills."""
            world.reset()
            return "world reset to defaults"

        return LLMAgent(
            name="Resetter",
            description="executes factory_reset",
            instructions="Call factory_reset to do the requested reset.",
            tools=[factory_reset],
            llm=ScriptedLLM(
                [
                    calls(("factory_reset", {})),
                    reply("World reset done."),
                ]
            ),
        )

    def test_approved_path_executes(self, world):
        # Pre-mutate world so we can confirm the reset actually ran.
        with world._lock:
            world._rooms["kitchen"].light_on = True

        executor = self._build_executor_and_world(world)
        gate = ApprovalGate.build(
            proposer_llm=ScriptedLLM([reply("Plan: call factory_reset(). The world will be cleared.")]),
            approver_llm=ScriptedLLM([reply("APPROVED")]),
            executor_agent=executor,
        )
        result = gate.run("please factory-reset the world")

        assert "reset done" in result.reply.lower() or "reset" in result.reply.lower()
        # Reset actually happened -- kitchen light is off again.
        assert world.get_world_state()["rooms"]["kitchen"]["light_on"] is False

    def test_denied_path_does_not_execute(self, world):
        # Pre-mutate the world. The reset must NOT run.
        with world._lock:
            world._rooms["kitchen"].light_on = True

        executor = self._build_executor_and_world(world)
        gate = ApprovalGate.build(
            proposer_llm=ScriptedLLM([reply("Plan: call factory_reset(). The world will be cleared.")]),
            approver_llm=ScriptedLLM(
                [reply("DENIED -- factory reset would lose user state; require explicit confirmation.")]
            ),
            executor_agent=executor,
        )
        result = gate.run("please factory-reset the world")

        assert result.reply.startswith("DENIED")
        # World untouched -- the kitchen light we set above is still on.
        assert world.get_world_state()["rooms"]["kitchen"]["light_on"] is True


# ---------------------------------------------------------------------------
# S2 -- critique-and-rewrite for a robot announcement
# ---------------------------------------------------------------------------


class TestS2_CritiqueAndRewrite:
    """Real situation: a robot is asked to draft a one-sentence
    announcement before it starts a noisy task. The first draft is
    too verbose, the critic asks for one revision, the generator
    delivers a tighter version.
    """

    def test_one_revision_pass(self):
        recipe = CritiqueAndRewrite.build(
            generator_llm=ScriptedLLM(
                [
                    reply(
                        "I am about to begin a possibly noisy cleaning task that may include various sounds, "
                        "and I appreciate your patience as I move through the rooms."
                    ),
                    reply("Starting cleaning now -- expect some noise."),
                ]
            ),
            critic_llm=ScriptedLLM(
                [
                    reply("REVISE -- announcement is two sentences and too long; aim for under 12 words."),
                    reply("APPROVED"),
                ]
            ),
            max_iterations=3,
        )
        result = recipe.run("draft a 1-sentence announcement before cleaning")

        assert "starting cleaning" in result.reply.lower()
        assert "noise" in result.reply.lower()
        # No "revision budget exhausted" note because the second draft was approved.
        assert "exhausted" not in result.reply.lower()

    def test_first_draft_approved_returns_immediately(self):
        recipe = CritiqueAndRewrite.build(
            generator_llm=ScriptedLLM([reply("Cleaning starting -- back in 5 min.")]),
            critic_llm=ScriptedLLM([reply("APPROVED")]),
            max_iterations=3,
        )
        result = recipe.run("announce")

        assert "cleaning starting" in result.reply.lower()

    def test_budget_exhausted_returns_last_draft_with_note(self):
        recipe = CritiqueAndRewrite.build(
            generator_llm=ScriptedLLM([reply(f"draft {i}") for i in range(3)]),
            critic_llm=ScriptedLLM([reply(f"REVISE -- still off on round {i}") for i in range(3)]),
            max_iterations=3,
        )
        result = recipe.run("anything")

        assert "draft 2" in result.reply  # last draft
        assert "exhausted" in result.reply.lower()


# ---------------------------------------------------------------------------
# S3 -- parallel exploration of N strategies + aggregator
# ---------------------------------------------------------------------------


class TestS3_ParallelExploration:
    """Real situation: a robot is asked "what's the fastest way to the
    bedroom?". Three planning agents propose different routes in
    parallel; an aggregator picks the shortest.

    Probes the existing ``Parallel`` primitive in a realistic shape --
    fan-out N specialists, each with their own scripted reply, then a
    reducer over the per-agent replies.
    """

    def test_fanout_then_reduce(self, world):
        planner_a = LLMAgent(
            name="planner_a",
            description="route via living_room",
            instructions="Propose a route.",
            llm=ScriptedLLM([reply("Route A: living_room -> kitchen -> bedroom (3 hops)")]),
        )
        planner_b = LLMAgent(
            name="planner_b",
            description="route direct",
            instructions="Propose a route.",
            llm=ScriptedLLM([reply("Route B: direct diagonal (1 hop)")]),
        )
        planner_c = LLMAgent(
            name="planner_c",
            description="route via office",
            instructions="Propose a route.",
            llm=ScriptedLLM([reply("Route C: living_room -> office -> bedroom (3 hops)")]),
        )

        # Reducer: pick the route with the smallest hop count.
        def shortest_route(replies: list) -> str:
            import re

            best = None
            best_hops = float("inf")
            for r in replies:
                m = re.search(r"\((\d+)\s*hops?\)", r.reply)
                if m and int(m.group(1)) < best_hops:
                    best = r.reply
                    best_hops = int(m.group(1))
            return best or "no route found"

        explorer = Parallel(
            "explore_routes",
            [planner_a, planner_b, planner_c],
            reduce=shortest_route,
        )
        result = explorer.run("fastest route to bedroom")

        assert "Route B" in result.reply
        assert "diagonal" in result.reply


# ---------------------------------------------------------------------------
# S4 -- sensor-error recovery via Retry + Fallback
# ---------------------------------------------------------------------------


class TestS4_SensorErrorRecovery:
    """Real situation: a robot tries to read its battery level but the
    sensor is flaky. Retry up to 3 times; if all fail, fall back to a
    safe default response (so the voice pipeline doesn't stall).
    """

    def test_retry_succeeds_after_two_empty_replies(self):
        """Retry's contract: re-run while the wrapped agent returns
        an empty / whitespace reply, up to max_attempts.

        We script the LLM to emit empty content on the first two
        attempts, then a real reply on the third. ``Retry`` should
        bubble the third attempt's reply up.
        """
        primary = LLMAgent(
            name="Primary",
            description="reads sensor",
            instructions="Read the sensor.",
            llm=ScriptedLLM(
                [
                    reply(""),  # attempt 1: sensor read returned nothing
                    reply(""),  # attempt 2: still empty
                    reply("Battery level is 87%."),  # attempt 3: success
                ]
            ),
        )
        retried = Retry(primary, max_attempts=3, name="with_retry")
        result = retried.run("what's the battery level?")

        assert "87" in result.reply

    def test_fallback_when_primary_keeps_failing(self):
        primary_llm = ScriptedLLM([reply("could not read sensor")])
        primary = LLMAgent(
            name="Primary",
            description="reads sensor",
            instructions="Read the sensor.",
            llm=primary_llm,
        )
        fallback = LLMAgent(
            name="Fallback",
            description="provides safe default",
            instructions="Respond with a safe default.",
            llm=ScriptedLLM([reply("Battery level is unavailable; please check manually.")]),
        )
        # Fallback runs the second agent only if the first returned an empty
        # reply. Use a wrapper that converts "could not" replies to empty so
        # the fallback fires.
        from edgevox.agents.base import AgentContext, AgentResult

        class _EmptyOnFailure:
            name = "empty_on_failure"
            description = "wraps an agent so failures present as empty replies"

            def __init__(self, inner):
                self._inner = inner

            def run(self, task: str, ctx: AgentContext | None = None):
                r = self._inner.run(task, ctx)
                if "could not" in r.reply.lower() or "fail" in r.reply.lower():
                    return AgentResult(reply="", agent_name=self.name)
                return r

            def run_stream(self, task, ctx=None):
                yield self.run(task, ctx).reply

        wrapped = _EmptyOnFailure(primary)
        chain = Fallback("primary_then_default", [wrapped, fallback])
        result = chain.run("what's the battery level?")

        assert "unavailable" in result.reply.lower()
