"""Unit tests for the vendored SGLang tool-call detectors.

Covers:
- Each detector on its canonical format.
- The ``parse_tool_calls`` chain orchestrator — detection order, unknown-tool
  rejection, and the custom ``register_detector`` API.
"""

from __future__ import annotations

import json

import pytest

from edgevox.llm.tool_parsers import (
    DETECTORS,
    BaseFormatDetector,
    HermesDetector,
    Llama32Detector,
    MistralDetector,
    PythonicDetector,
    Qwen25Detector,
    coerce_tools,
    parse_tool_calls,
    register_detector,
)
from edgevox.llm.tool_parsers._types import Function, Tool


@pytest.fixture
def get_time_tool() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        }
    ]


# --------------------------------------------------------------------- types


class TestCoerceTools:
    def test_dict_schema_becomes_tool(self, get_time_tool):
        tools = coerce_tools(get_time_tool)
        assert isinstance(tools[0], Tool)
        assert tools[0].function.name == "get_time"
        assert tools[0].function.parameters is not None

    def test_existing_tool_passes_through(self):
        t = Tool(function=Function(name="x"))
        assert coerce_tools([t])[0] is t

    def test_none_is_empty_list(self):
        assert coerce_tools(None) == []


# ------------------------------------------------------------- per-detector


class TestHermesDetector:
    def test_matches_tool_call_wrapper(self, get_time_tool):
        text = '<tool_call>{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}</tool_call>'
        d = HermesDetector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {"timezone": "Asia/Tokyo"}

    def test_no_wrapper_returns_no_calls(self, get_time_tool):
        text = "just some prose"
        result = HermesDetector().detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls == []

    def test_unknown_tool_dropped(self, get_time_tool):
        text = '<tool_call>{"name": "not_a_tool", "arguments": {}}</tool_call>'
        result = HermesDetector().detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls == []


class TestQwen25Detector:
    def test_newline_wrapped_format(self, get_time_tool):
        text = '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
        d = Qwen25Detector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls[0].name == "get_time"


class TestLlama32Detector:
    def test_python_tag_json(self, get_time_tool):
        text = '<|python_tag|>{"name": "get_time", "parameters": {"timezone": "UTC"}}'
        d = Llama32Detector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {"timezone": "UTC"}

    def test_bare_json_without_tag(self, get_time_tool):
        text = '{"name": "get_time", "parameters": {"timezone": "UTC"}}'
        d = Llama32Detector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls[0].name == "get_time"


class TestMistralDetector:
    def test_tool_calls_bracket(self, get_time_tool):
        text = '[TOOL_CALLS] [{"name": "get_time", "arguments": {"timezone": "UTC"}}]'
        d = MistralDetector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls[0].name == "get_time"


class TestPythonicDetector:
    def test_python_call_syntax(self, get_time_tool):
        text = '[get_time(timezone="UTC")]'
        d = PythonicDetector()
        assert d.has_tool_call(text)
        result = d.detect_and_parse(text, coerce_tools(get_time_tool))
        assert result.calls[0].name == "get_time"
        assert json.loads(result.calls[0].parameters) == {"timezone": "UTC"}

    def test_multiple_calls(self, get_time_tool):
        text = '[get_time(timezone="UTC"), get_time(timezone="JST")]'
        result = PythonicDetector().detect_and_parse(text, coerce_tools(get_time_tool))
        assert len(result.calls) == 2


# --------------------------------------------------- parse_tool_calls chain


class TestParseToolCallsChain:
    def test_all_formats_round_trip(self, get_time_tool):
        samples = {
            "hermes": '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>',
            "qwen25": '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>',
            "llama32-tag": '<|python_tag|>{"name": "get_time", "parameters": {"timezone": "UTC"}}',
            "bare-json": '{"name": "get_time", "parameters": {"timezone": "UTC"}}',
            "mistral": '[TOOL_CALLS] [{"name": "get_time", "arguments": {"timezone": "UTC"}}]',
            "pythonic": '[get_time(timezone="UTC")]',
        }
        for label, text in samples.items():
            calls = parse_tool_calls(text, get_time_tool)
            assert calls is not None, f"failed to detect: {label}"
            assert calls[0]["function"]["name"] == "get_time"

    def test_empty_text_returns_none(self, get_time_tool):
        assert parse_tool_calls("", get_time_tool) is None
        assert parse_tool_calls("   \n  ", get_time_tool) is None

    def test_plain_prose_returns_none(self, get_time_tool):
        assert parse_tool_calls("Hello there!", get_time_tool) is None

    def test_respects_detector_order(self, get_time_tool):
        text = '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
        # Constrained to mistral-only → no match, since text is chatml.
        assert parse_tool_calls(text, get_time_tool, detectors=["mistral"]) is None
        # With hermes in the chain, matches.
        assert parse_tool_calls(text, get_time_tool, detectors=["hermes"]) is not None

    def test_unknown_detector_skipped(self, get_time_tool):
        text = '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
        # Unknown detector name doesn't crash; real one still matches.
        calls = parse_tool_calls(text, get_time_tool, detectors=["nonsense", "hermes"])
        assert calls is not None

    def test_openai_shape_is_valid(self, get_time_tool):
        text = '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
        calls = parse_tool_calls(text, get_time_tool)
        call = calls[0]
        assert call["type"] == "function"
        assert "id" in call
        assert call["function"]["name"] == "get_time"
        assert isinstance(call["function"]["arguments"], str)
        json.loads(call["function"]["arguments"])  # valid JSON


# ------------------------------------------------------ custom registration


class TestRegisterDetector:
    def test_custom_detector_plugs_in(self, get_time_tool):
        class Bracket(BaseFormatDetector):
            def __init__(self):
                super().__init__()
                self.bot_token = "<<call>>"

            def has_tool_call(self, text: str) -> bool:
                return "<<call>>" in text

            def detect_and_parse(self, text, tools):
                from edgevox.llm.tool_parsers.core_types import StreamingParseResult

                raw = text.split("<<call>>", 1)[1]
                payload = json.loads(raw)
                return StreamingParseResult(calls=self.parse_base_json(payload, tools))

            def structure_info(self):
                raise NotImplementedError

        register_detector("bracket", Bracket)
        try:
            text = '<<call>>{"name": "get_time", "arguments": {"timezone": "UTC"}}'
            calls = parse_tool_calls(text, get_time_tool, detectors=["bracket"])
            assert calls is not None
            assert calls[0]["function"]["name"] == "get_time"
        finally:
            DETECTORS.pop("bracket", None)

    def test_non_subclass_raises(self):
        with pytest.raises(TypeError):
            register_detector("nope", dict)  # type: ignore[arg-type]
