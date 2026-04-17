"""Tests for @tool decorator, ToolRegistry, and the LLM agent loop."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from edgevox.llm.tools import (
    Tool,
    ToolCallResult,
    ToolRegistry,
    tool,
)

# --------------- @tool decorator ---------------


class TestToolDecorator:
    def test_basic_signature(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: first addend
                b: second addend
            """
            return a + b

        descriptor = add.__edgevox_tool__
        assert isinstance(descriptor, Tool)
        assert descriptor.name == "add"
        assert descriptor.description == "Add two numbers."
        params = descriptor.parameters
        assert params["properties"]["a"] == {"type": "integer", "description": "first addend"}
        assert params["properties"]["b"] == {"type": "integer", "description": "second addend"}
        assert set(params["required"]) == {"a", "b"}

    def test_optional_params_not_required(self):
        @tool
        def greet(name: str, loudly: bool = False) -> str:
            """Say hi."""
            return f"hi {name}"

        descriptor = greet.__edgevox_tool__
        assert descriptor.parameters["required"] == ["name"]
        assert descriptor.parameters["properties"]["loudly"] == {"type": "boolean"}

    def test_optional_union_syntax(self):
        @tool
        def search(query: str, limit: int | None = None) -> list[str]:
            """Run a search."""
            return []

        descriptor = search.__edgevox_tool__
        assert descriptor.parameters["required"] == ["query"]
        assert descriptor.parameters["properties"]["limit"] == {"type": "integer"}

    def test_list_and_dict_types(self):
        @tool
        def batch(items: list[str], opts: dict) -> int:
            """Batch."""
            return len(items)

        props = batch.__edgevox_tool__.parameters["properties"]
        assert props["items"] == {"type": "array", "items": {"type": "string"}}
        assert props["opts"] == {"type": "object"}

    def test_name_and_description_override(self):
        @tool(name="do_thing", description="Does a thing.")
        def internal_name(x: str) -> str:
            """Actual docstring ignored."""
            return x

        descriptor = internal_name.__edgevox_tool__
        assert descriptor.name == "do_thing"
        assert descriptor.description == "Does a thing."

    def test_call_still_works_as_normal_function(self):
        @tool
        def double(x: int) -> int:
            """Double."""
            return x * 2

        assert double(21) == 42

    def test_openai_schema_shape(self):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        schema = ping.__edgevox_tool__.openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "ping"
        assert schema["function"]["description"] == "Ping."
        assert schema["function"]["parameters"]["type"] == "object"


# --------------- ToolRegistry ---------------


class TestToolRegistry:
    def _registry(self) -> ToolRegistry:
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        @tool
        def failing() -> None:
            """Boom."""
            raise RuntimeError("kaboom")

        reg = ToolRegistry()
        reg.register(add, failing)
        return reg

    def test_register_and_contains(self):
        reg = self._registry()
        assert "add" in reg
        assert "failing" in reg
        assert len(reg) == 2

    def test_openai_schemas(self):
        reg = self._registry()
        schemas = reg.openai_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"add", "failing"}

    def test_dispatch_success(self):
        reg = self._registry()
        outcome = reg.dispatch("add", {"a": 2, "b": 3})
        assert outcome.ok
        assert outcome.result == 5

    def test_dispatch_json_string(self):
        reg = self._registry()
        outcome = reg.dispatch("add", '{"a": 10, "b": 5}')
        assert outcome.ok
        assert outcome.result == 15

    def test_dispatch_unknown_tool(self):
        reg = self._registry()
        outcome = reg.dispatch("nope", {})
        assert not outcome.ok
        assert "unknown" in outcome.error

    def test_dispatch_bad_args(self):
        reg = self._registry()
        outcome = reg.dispatch("add", {"a": 1})  # missing b
        assert not outcome.ok
        assert "bad arguments" in outcome.error

    def test_dispatch_invalid_json(self):
        reg = self._registry()
        outcome = reg.dispatch("add", "{not json")
        assert not outcome.ok
        assert "invalid JSON" in outcome.error

    def test_dispatch_catches_user_exception(self):
        reg = self._registry()
        outcome = reg.dispatch("failing", {})
        assert not outcome.ok
        assert "RuntimeError" in outcome.error
        assert "kaboom" in outcome.error

    def test_register_plain_function_raises(self):
        reg = ToolRegistry()

        def plain(x: int) -> int:
            return x

        with pytest.raises(TypeError):
            reg.register(plain)


# --------------- LLM agent loop ---------------


class TestGemmaInlineFallback:
    def test_parses_string_arg(self):
        from edgevox.llm.llamacpp import _parse_gemma_inline_tool_calls

        content = '<|tool_call>call:get_current_temperature{city:<|"|>Paris<|"|>}<tool_call|>'
        calls = _parse_gemma_inline_tool_calls(content)
        assert calls is not None and len(calls) == 1
        call = calls[0]
        assert call["function"]["name"] == "get_current_temperature"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}

    def test_parses_numeric_and_bool(self):
        from edgevox.llm.llamacpp import _parse_gemma_inline_tool_calls

        content = '<|tool_call>call:set_light{room:<|"|>kitchen<|"|>, on:true}<tool_call|>'
        calls = _parse_gemma_inline_tool_calls(content)
        assert calls is not None
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"room": "kitchen", "on": True}

    def test_parses_float(self):
        from edgevox.llm.llamacpp import _parse_gemma_inline_tool_calls

        content = "<|tool_call>call:set_thermostat{celsius: 21.5}<tool_call|>"
        calls = _parse_gemma_inline_tool_calls(content)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"celsius": 21.5}

    def test_no_pattern_returns_none(self):
        from edgevox.llm.llamacpp import _parse_gemma_inline_tool_calls

        assert _parse_gemma_inline_tool_calls("Just a normal reply.") is None


class TestStripThinking:
    def test_strips_single_block(self):
        from edgevox.llm.llamacpp import _strip_thinking

        assert _strip_thinking("<think>reasoning here</think>Hello.") == "Hello."

    def test_strips_multiline_block(self):
        from edgevox.llm.llamacpp import _strip_thinking

        raw = "<think>\nline 1\nline 2\n</think>\nFinal answer."
        assert _strip_thinking(raw) == "Final answer."

    def test_no_block_is_passthrough(self):
        from edgevox.llm.llamacpp import _strip_thinking

        assert _strip_thinking("just a reply") == "just a reply"


class TestParseChatmlToolCalls:
    def test_recovers_tool_call(self):
        from edgevox.llm.llamacpp import _parse_chatml_tool_calls

        content = '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}\n</tool_call>'
        calls = _parse_chatml_tool_calls(content)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_time"
        assert json.loads(calls[0]["function"]["arguments"]) == {"timezone": "Asia/Tokyo"}

    def test_no_match_returns_none(self):
        from edgevox.llm.llamacpp import _parse_chatml_tool_calls

        assert _parse_chatml_tool_calls("just text") is None

    def test_invalid_json_skipped(self):
        from edgevox.llm.llamacpp import _parse_chatml_tool_calls

        assert _parse_chatml_tool_calls("<tool_call>{bogus}</tool_call>") is None

    def test_bare_llama_json_parameters(self):
        from edgevox.llm.llamacpp import _parse_chatml_tool_calls

        # Llama 3.x native shape: top-level JSON with "function" + "parameters".
        content = '{"function": "get_time", "parameters": {"timezone": "Asia/Tokyo"}}'
        calls = _parse_chatml_tool_calls(content)
        assert calls is not None
        assert calls[0]["function"]["name"] == "get_time"
        assert json.loads(calls[0]["function"]["arguments"]) == {"timezone": "Asia/Tokyo"}

    def test_bare_json_name_arguments(self):
        from edgevox.llm.llamacpp import _parse_chatml_tool_calls

        content = '{"name": "get_time", "arguments": {"timezone": "UTC"}}'
        calls = _parse_chatml_tool_calls(content)
        assert calls is not None
        assert calls[0]["function"]["name"] == "get_time"


class TestLLMFallbackAgentLoop:
    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_inline_tool_call_is_recovered(self, _vram, _metal, mock_llama_cls):
        mock_llama = MagicMock()
        mock_llama_cls.return_value = mock_llama

        @tool
        def get_current_temperature(city: str) -> dict:
            """Get temp."""
            return {"city": city, "temp_c": 17}

        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/m.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/m.gguf", tools=[get_current_temperature])

        inline_leak = (
            '<|tool_call>call:get_current_temperature{city:<|"|>Paris<|"|>}<tool_call|> thinking about weather...'
        )
        mock_llama.create_chat_completion.side_effect = [
            {"choices": [{"message": {"content": inline_leak, "tool_calls": None}}]},
            {"choices": [{"message": {"content": "It's 17 in Paris.", "tool_calls": None}}]},
        ]

        reply = llm.chat("weather?")
        assert reply == "It's 17 in Paris."
        # Fallback path pushes results as a synthetic user message, not a tool role
        feedbacks = [m for m in llm._history if m["role"] == "user" and "tool results" in m["content"]]
        assert len(feedbacks) == 1
        assert "Paris" in feedbacks[0]["content"]
        assert "17" in feedbacks[0]["content"]


@tool
def get_time(timezone: str = "UTC") -> str:
    """Return the current time.

    Args:
        timezone: tz name, e.g. "UTC"
    """
    return f"12:00 {timezone}"


class TestLLMAgent:
    def _make_llm(self, mock_llama_cls, tools=None):
        mock_llama = MagicMock()
        mock_llama_cls.return_value = mock_llama

        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/model.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/model.gguf", tools=tools)
        return llm, mock_llama

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_tools_flag_enables_tool_system_prompt(self, _vram, _metal, mock_llama_cls):
        llm, _ = self._make_llm(mock_llama_cls, tools=[get_time])
        assert "tools available" in llm._history[0]["content"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_no_tools_keeps_plain_system_prompt(self, _vram, _metal, mock_llama_cls):
        llm, _ = self._make_llm(mock_llama_cls)
        assert "tools available" not in llm._history[0]["content"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_chat_with_tool_call_hop(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])

        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "get_time",
                                    "arguments": json.dumps({"timezone": "America/Los_Angeles"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        final_response = {"choices": [{"message": {"content": "It's 12:00 PM in LA.", "tool_calls": None}}]}
        mock_llama.create_chat_completion.side_effect = [tool_call_response, final_response]

        reply = llm.chat("what time is it in LA?")
        assert reply == "It's 12:00 PM in LA."
        assert mock_llama.create_chat_completion.call_count == 2

        # history should have: system, user, assistant(tool_calls), tool(result), assistant(final)
        tool_msgs = [m for m in llm._history if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0]["content"])
        assert payload["ok"] is True
        assert "America/Los_Angeles" in payload["result"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_chat_no_tool_call_single_completion(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hi there.", "tool_calls": None}}]
        }
        reply = llm.chat("Hi")
        assert reply == "Hi there."
        assert mock_llama.create_chat_completion.call_count == 1

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_chat_stream_with_tools_yields_single_chunk(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])

        mock_llama.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_x",
                                    "function": {"name": "get_time", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "It is 12:00 UTC.", "tool_calls": None}}]},
        ]

        chunks = list(llm.chat_stream("what time is it"))
        assert chunks == ["It is 12:00 UTC."]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_stream_path_still_streams_without_tools(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls)
        mock_llama.create_chat_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "one"}}]},
                {"choices": [{"delta": {"content": " two"}}]},
                {"choices": [{"delta": {}}]},
            ]
        )
        tokens = list(llm.chat_stream("hi"))
        assert tokens == ["one", " two"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_on_tool_call_callback_fires(self, _vram, _metal, mock_llama_cls):
        seen: list[ToolCallResult] = []

        mock_llama = MagicMock()
        mock_llama_cls.return_value = mock_llama
        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/model.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/model.gguf", tools=[get_time], on_tool_call=seen.append)

        mock_llama.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "1", "function": {"name": "get_time", "arguments": "{}"}}],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "done", "tool_calls": None}}]},
        ]

        llm.chat("hey")
        assert len(seen) == 1
        assert seen[0].name == "get_time"
        assert seen[0].ok

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_budget_exhausted_returns_fallback(self, _vram, _metal, mock_llama_cls):
        """Budget exhaustion fires when the model keeps tool-calling with *different* args.

        Same-args loops are caught earlier by loop detection — see
        ``test_loop_detection_breaks_on_repeated_call``.
        """
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])
        # Each hop emits a unique ``timezone`` so the fingerprint differs,
        # exercising the max_tool_hops path rather than loop detection.
        mock_llama.create_chat_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{i}",
                                    "function": {
                                        "name": "get_time",
                                        "arguments": '{"timezone": "tz_' + str(i) + '"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            for i in range(llm._max_tool_hops + 1)
        ]

        reply = llm.chat("loop")
        assert reply != ""
        # max_tool_hops + 1 calls before the budget path returns a fallback.
        assert mock_llama.create_chat_completion.call_count == llm._max_tool_hops + 1

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_loop_detection_breaks_on_repeated_call(self, _vram, _metal, mock_llama_cls):
        """Three identical ``(name, args)`` calls trip the loop-break."""
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])
        identical = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "loop",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"timezone": "UTC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_llama.create_chat_completion.return_value = identical

        reply = llm.chat("what time is it")
        assert reply != ""
        # Loop detector fires on the third identical call and short-circuits
        # before the model's fourth turn — strictly fewer calls than budget.
        assert mock_llama.create_chat_completion.call_count <= llm._max_tool_hops

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_schema_retry_hint_injected_on_wrong_kwargs(self, _vram, _metal, mock_llama_cls):
        """Dispatch failure on bad kwargs injects a schema hint for one retry."""
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])
        wrong_args_call = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "get_time",
                                    # ``location`` is not in the tool signature.
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        final = {"choices": [{"message": {"content": "Done.", "tool_calls": None}}]}
        mock_llama.create_chat_completion.side_effect = [wrong_args_call, final]
        llm.chat("what time is it")

        # The tool-result message injected for the retry should carry the
        # schema hint in plain English so the model can self-correct.
        tool_msgs = [m for m in llm._history if m.get("role") == "tool"]
        assert tool_msgs, "no tool-result injected for retry"
        import json as _json

        payload = _json.loads(tool_msgs[-1]["content"])
        assert payload["ok"] is False
        assert "retry_hint" in payload
        hint = payload["retry_hint"]
        assert "timezone" in hint  # the correct parameter is named
        assert "Retry" in hint

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_completion_kwargs_include_tools_when_registered(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls, tools=[get_time])
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok", "tool_calls": None}}]
        }
        llm.chat("hi")
        kwargs = mock_llama.create_chat_completion.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tool_choice"] == "auto"
        names = {t["function"]["name"] for t in kwargs["tools"]}
        assert names == {"get_time"}
