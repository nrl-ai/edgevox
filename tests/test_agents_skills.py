"""Unit tests for @skill, GoalHandle, and the skill dispatch loop."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from edgevox.agents.base import AgentContext, LLMAgent
from edgevox.agents.sim import ToyWorld
from edgevox.agents.skills import GoalHandle, GoalStatus, Skill, skill

# --------------- @skill decorator ---------------


class TestSkillDecorator:
    def test_basic_skill_is_skill_instance(self):
        @skill
        def ping() -> str:
            """Ping."""
            return "pong"

        assert isinstance(ping, Skill)
        assert ping.name == "ping"

    def test_strips_ctx_and_handle_from_schema(self):
        @skill
        def navigate(room: str, ctx: AgentContext, handle: GoalHandle) -> str:
            """Navigate to a room.

            Args:
                room: the room name
            """
            return f"at {room}"

        params = navigate.parameters
        assert "ctx" not in params["properties"]
        assert "handle" not in params["properties"]
        assert "room" in params["properties"]

    def test_fast_latency_defaults(self):
        @skill(latency_class="fast")
        def now() -> float:
            """Time."""
            return time.time()

        assert now.latency_class == "fast"

    def test_slow_default(self):
        @skill
        def whatever() -> str:
            """whatever."""
            return "ok"

        assert whatever.latency_class == "slow"


# --------------- GoalHandle lifecycle ---------------


class TestGoalHandle:
    def test_succeed_marks_terminal(self):
        h = GoalHandle()
        h.succeed({"result": 42})
        assert h.status is GoalStatus.SUCCEEDED
        assert h.result == {"result": 42}
        assert h.poll(timeout=0.01) is GoalStatus.SUCCEEDED

    def test_fail_records_error(self):
        h = GoalHandle()
        h.fail("boom")
        assert h.status is GoalStatus.FAILED
        assert h.error == "boom"

    def test_cancel_sets_event(self):
        h = GoalHandle()
        assert not h.should_cancel()
        h.cancel()
        assert h.should_cancel()

    def test_feedback_non_blocking(self):
        h = GoalHandle()
        h.set_feedback({"progress": 0.3})
        h.set_feedback({"progress": 0.6})
        drained = list(h.feedback())
        assert [d["progress"] for d in drained] == [0.3, 0.6]
        # Second drain is empty
        assert list(h.feedback()) == []


# --------------- Skill dispatch: fast + slow + cancellation ---------------


class TestSkillDispatch:
    def test_fast_skill_runs_inline(self):
        @skill(latency_class="fast")
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        ctx = AgentContext()
        handle = add.start(ctx, a=2, b=3)
        assert handle.status is GoalStatus.SUCCEEDED
        assert handle.result == 5

    def test_slow_skill_delegating_to_sim_adopts_handle(self):
        """When a slow skill body returns a GoalHandle (from a sim
        action), the skill decorator should adopt that handle directly
        rather than wrapping it."""
        world = ToyWorld(navigate_speed=2.0)

        @skill(latency_class="slow", timeout_s=5.0)
        def go_to(room: str, ctx: AgentContext) -> GoalHandle:
            """Go."""
            return ctx.deps.apply_action("navigate_to", room=room)

        ctx = AgentContext(deps=world)
        handle = go_to.start(ctx, room="kitchen")
        assert handle.status is GoalStatus.RUNNING
        # Wait for it to complete
        status = handle.poll(timeout=5.0)
        assert status is GoalStatus.SUCCEEDED
        assert "kitchen" in str(handle.result)

    def test_slow_skill_mid_flight_cancellation(self):
        world = ToyWorld(navigate_speed=0.3)  # slow so we can cancel

        @skill(latency_class="slow", timeout_s=30.0)
        def go_to(room: str, ctx: AgentContext) -> GoalHandle:
            """Go."""
            return ctx.deps.apply_action("navigate_to", room=room)

        ctx = AgentContext(deps=world)
        handle = go_to.start(ctx, room="bedroom")

        # Let it run briefly, then cancel
        time.sleep(0.5)
        assert handle.status is GoalStatus.RUNNING
        handle.cancel()

        status = handle.poll(timeout=2.0)
        assert status is GoalStatus.CANCELLED
        pose = world.get_world_state()["robot"]
        # bedroom is (4, 4); robot shouldn't have reached it
        assert abs(pose["x"] - 4.0) > 0.1 or abs(pose["y"] - 4.0) > 0.1

    def test_skill_timeout_fires(self):
        done = threading.Event()

        @skill(latency_class="slow", timeout_s=0.3)
        def slow_fn(ctx: AgentContext) -> GoalHandle:
            """Slow."""
            handle = GoalHandle()
            handle.status = GoalStatus.RUNNING

            def worker():
                for _ in range(100):
                    if handle.should_cancel():
                        handle.mark_cancelled()
                        done.set()
                        return
                    time.sleep(0.1)
                handle.succeed("done")
                done.set()

            threading.Thread(target=worker, daemon=True).start()
            return handle

        ctx = AgentContext()
        handle = slow_fn.start(ctx)
        # Wait longer than the timeout
        time.sleep(0.8)
        assert handle.status in (GoalStatus.CANCELLED, GoalStatus.FAILED)


# --------------- Skill dispatch inside LLMAgent ---------------


@patch("llama_cpp.Llama")
@patch("edgevox.core.gpu.has_metal", return_value=False)
@patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
class TestLLMAgentSkillDispatch:
    def _make(self, mock_llama_cls, skills):
        mock_llm = MagicMock()
        mock_llama_cls.return_value = mock_llm
        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/fake.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/fake.gguf")
        agent = LLMAgent(
            name="tester",
            description="t",
            instructions="You are Tester.",
            skills=skills,
            llm=llm,
        )
        return agent, mock_llm

    def test_skill_appears_as_tool_to_the_model(self, _v, _m, mock_llama_cls):
        @skill(latency_class="fast")
        def ping() -> str:
            """Ping."""
            return "pong"

        agent, _llm = self._make(mock_llama_cls, [ping])
        assert "ping" in agent.tools.tools
        assert "ping" in agent.skills

    def test_skill_dispatch_returns_result(self, _v, _m, mock_llama_cls):
        @skill(latency_class="fast")
        def ping() -> str:
            """Ping."""
            return "pong"

        agent, llm = self._make(mock_llama_cls, [ping])
        llm.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "c1", "function": {"name": "ping", "arguments": "{}"}}],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "pong received", "tool_calls": None}}]},
        ]
        result = agent.run("ping")
        assert result.reply == "pong received"

    def test_skill_cancelled_by_stop_event(self, _v, _m, mock_llama_cls):
        """If ctx.stop fires during a slow skill poll loop, the skill
        is cancelled and the agent returns preempted."""
        world = ToyWorld(navigate_speed=0.3)

        @skill(latency_class="slow", timeout_s=10.0)
        def go_to(room: str, ctx: AgentContext) -> GoalHandle:
            """Go."""
            return ctx.deps.apply_action("navigate_to", room=room)

        agent, llm = self._make(mock_llama_cls, [go_to])
        llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "go_to",
                                    "arguments": '{"room":"bedroom"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        ctx = AgentContext(deps=world)

        def race_stop():
            time.sleep(0.4)
            ctx.stop.set()

        threading.Thread(target=race_stop, daemon=True).start()

        agent.run("drive to bedroom", ctx)
        # Robot should not have reached the bedroom
        pose = world.get_world_state()["robot"]
        reached = abs(pose["x"] - 4.0) < 0.1 and abs(pose["y"] - 4.0) < 0.1
        assert not reached
        assert ctx.stop.is_set()
