"""Unit tests for the agent framework foundations."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from edgevox.agents.base import (
    Agent,
    AgentContext,
    AgentEvent,
    Handoff,
    LLMAgent,
    Session,
)
from edgevox.llm import tool

# --------------- Session ---------------


class TestSession:
    def test_empty_defaults(self):
        s = Session()
        assert s.messages == []
        assert s.state == {}

    def test_reset_clears_both(self):
        s = Session(messages=[{"role": "user", "content": "hi"}], state={"k": 1})
        s.reset()
        assert s.messages == []
        assert s.state == {}

    def test_state_is_mutable(self):
        s = Session()
        s.state["progress"] = 0.5
        assert s.state["progress"] == 0.5


# --------------- AgentContext ---------------


class TestAgentContext:
    def test_emit_none_callback_noop(self):
        ctx = AgentContext()
        ctx.emit("tool_call", "x", {"k": 1})  # should not raise

    def test_emit_dispatches_event(self):
        seen: list[AgentEvent] = []
        ctx = AgentContext(on_event=seen.append)
        ctx.emit("handoff", "router", {"target": "leaf"})
        assert len(seen) == 1
        assert seen[0].kind == "handoff"
        assert seen[0].agent_name == "router"

    def test_emit_swallows_callback_exceptions(self):
        def bad(_):
            raise RuntimeError("boom")

        ctx = AgentContext(on_event=bad)
        ctx.emit("tool_call", "x", {})  # must not propagate

    def test_stop_event_defaults_unset(self):
        ctx = AgentContext()
        assert not ctx.stop.is_set()

    def test_stop_event_can_be_shared(self):
        shared = threading.Event()
        ctx_a = AgentContext(stop=shared)
        ctx_b = AgentContext(stop=shared)
        shared.set()
        assert ctx_a.stop.is_set()
        assert ctx_b.stop.is_set()


# --------------- Handoff sentinel ---------------


class TestHandoff:
    def test_handoff_fields(self):
        fake = MagicMock(name="fake_agent")
        fake.name = "weather"
        h = Handoff(target=fake, task="what's it like in Paris?", reason="routing")
        assert h.target is fake
        assert h.task == "what's it like in Paris?"
        assert h.reason == "routing"

    def test_handoff_default_task_is_none(self):
        fake = MagicMock()
        h = Handoff(target=fake)
        assert h.task is None


# --------------- LLMAgent with mocked LLM ---------------


def _make_agent(
    mock_llama_cls,
    *,
    tools=None,
    skills=None,
    handoffs=None,
    instructions="You are Test.",
    name="tester",
):
    mock_llm = MagicMock()
    mock_llama_cls.return_value = mock_llm
    with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/fake.gguf"):
        from edgevox.llm.llamacpp import LLM

        llm = LLM(model_path="/tmp/fake.gguf")
    agent = LLMAgent(
        name=name,
        description="test agent",
        instructions=instructions,
        tools=tools,
        skills=skills,
        handoffs=handoffs,
        llm=llm,
    )
    return agent, mock_llm


@patch("llama_cpp.Llama")
@patch("edgevox.core.gpu.has_metal", return_value=False)
@patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
class TestLLMAgentBasic:
    def test_chitchat_turn_single_llm_call(self, _vram, _metal, mock_llama_cls):
        agent, llm = _make_agent(mock_llama_cls)
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello there.", "tool_calls": None}}]
        }
        result = agent.run("hi")
        assert result.reply == "Hello there."
        assert result.preempted is False
        assert llm.create_chat_completion.call_count == 1

    def test_persona_installed_in_system_prompt(self, _vram, _metal, mock_llama_cls):
        agent, llm = _make_agent(mock_llama_cls, instructions="You are Scout, a concise robot.")
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok", "tool_calls": None}}]}
        agent.run("hi")
        kwargs = llm.create_chat_completion.call_args.kwargs
        system_msg = kwargs["messages"][0]["content"]
        assert "Scout" in system_msg
        assert "concise robot" in system_msg

    def test_agent_start_and_end_events(self, _vram, _metal, mock_llama_cls):
        agent, llm = _make_agent(mock_llama_cls)
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "reply", "tool_calls": None}}]}
        events = []
        ctx = AgentContext(on_event=lambda e: events.append(e.kind))
        agent.run("hi", ctx)
        assert events[0] == "agent_start"
        assert events[-1] == "agent_end"

    def test_tool_call_dispatch(self, _vram, _metal, mock_llama_cls):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"hello {name}"

        agent, llm = _make_agent(mock_llama_cls, tools=[greet])
        llm.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": '{"name":"world"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Said hi.", "tool_calls": None}}]},
        ]
        events = []
        result = agent.run("greet world", AgentContext(on_event=lambda e: events.append(e.kind)))
        assert result.reply == "Said hi."
        assert llm.create_chat_completion.call_count == 2
        assert "tool_call" in events

    def test_ctx_injection_into_tool(self, _vram, _metal, mock_llama_cls):
        observed = []

        @tool
        def peek(ctx: AgentContext) -> str:
            """Return something from ctx.deps."""
            observed.append(ctx.deps)
            return f"got {ctx.deps}"

        agent, llm = _make_agent(mock_llama_cls, tools=[peek])
        llm.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "c1", "function": {"name": "peek", "arguments": "{}"}}],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "done", "tool_calls": None}}]},
        ]
        agent.run("peek", AgentContext(deps="SENTINEL"))
        assert observed == ["SENTINEL"]

    def test_ctx_stripped_from_tool_schema(self, _vram, _metal, mock_llama_cls):
        @tool
        def needs_ctx(room: str, ctx: AgentContext) -> str:
            """Do a thing."""
            return f"{room}-{ctx.deps}"

        agent, _ = _make_agent(mock_llama_cls, tools=[needs_ctx])
        schema = agent.tools.tools["needs_ctx"].parameters
        props = schema["properties"]
        assert "ctx" not in props
        assert "room" in props


@patch("llama_cpp.Llama")
@patch("edgevox.core.gpu.has_metal", return_value=False)
@patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
class TestLLMAgentHandoff:
    def test_handoff_short_circuit_costs_two_hops(self, _vram, _metal, mock_llama_cls):
        """Router picking a leaf and the leaf replying directly = 2 LLM
        calls. Regression guard against the smolagents-style 3-hop
        alternative."""

        @tool
        def get_weather(city: str) -> str:
            """Weather."""
            return f"{city}: clear"

        # Both agents share the same mocked LLM.
        mock_llm = MagicMock()
        mock_llama_cls.return_value = mock_llm
        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/fake.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/fake.gguf")

        leaf = LLMAgent(
            name="weather",
            description="weather",
            instructions="You are Skye.",
            tools=[get_weather],
            llm=llm,
        )
        router = LLMAgent(
            name="router",
            description="route",
            instructions="You are Casa.",
            handoffs=[leaf],
            llm=llm,
        )

        mock_llm.create_chat_completion.side_effect = [
            {  # router decides to handoff
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "handoff_to_weather",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {  # leaf answers directly without a tool call
                "choices": [{"message": {"content": "It's clear in Paris.", "tool_calls": None}}]
            },
        ]

        events = []
        ctx = AgentContext(on_event=lambda e: events.append((e.kind, e.agent_name)))
        result = router.run("weather in Paris?", ctx)
        assert result.reply == "It's clear in Paris."
        assert result.handed_off_to == "weather"
        assert mock_llm.create_chat_completion.call_count == 2
        kinds = [k for k, _ in events]
        assert "handoff" in kinds
        assert ("agent_start", "weather") in events

    def test_handoff_leaf_with_tool_costs_three_hops(self, _vram, _metal, mock_llama_cls):
        """Router → leaf-that-calls-a-tool → leaf-final = 3 LLM calls.
        Still cheaper than smolagents' sub-as-tool (which is 4)."""

        @tool
        def get_weather(city: str) -> str:
            """Weather."""
            return f"{city}: clear"

        mock_llm = MagicMock()
        mock_llama_cls.return_value = mock_llm
        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/fake.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/fake.gguf")

        leaf = LLMAgent(
            name="weather",
            description="weather",
            instructions="You are Skye.",
            tools=[get_weather],
            llm=llm,
        )
        router = LLMAgent(
            name="router",
            description="route",
            instructions="You are Casa.",
            handoffs=[leaf],
            llm=llm,
        )

        mock_llm.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "handoff_to_weather",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Paris is clear.", "tool_calls": None}}]},
        ]

        result = router.run("weather in Paris?")
        assert result.reply == "Paris is clear."
        assert mock_llm.create_chat_completion.call_count == 3


@patch("llama_cpp.Llama")
@patch("edgevox.core.gpu.has_metal", return_value=False)
@patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
class TestLLMAgentStopEvent:
    def test_stop_event_set_before_run_preempts_immediately(self, _vram, _metal, mock_llama_cls):
        agent, llm = _make_agent(mock_llama_cls)
        ctx = AgentContext()
        ctx.stop.set()
        result = agent.run("hi", ctx)
        assert result.preempted is True
        assert result.reply == "Stopped."
        assert llm.create_chat_completion.call_count == 0

    def test_stop_event_cleared_does_not_preempt(self, _vram, _metal, mock_llama_cls):
        agent, llm = _make_agent(mock_llama_cls)
        llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
        ctx = AgentContext()
        result = agent.run("hi", ctx)
        assert result.preempted is False


def test_agent_protocol_runtime_checkable():
    mock = MagicMock()
    mock.name = "m"
    mock.description = "d"
    mock.run = lambda *a, **k: None
    mock.run_stream = lambda *a, **k: iter([])
    assert isinstance(mock, Agent)
