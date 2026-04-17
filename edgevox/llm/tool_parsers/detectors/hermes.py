# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-project/sglang python/sglang/srt/function_call/hermes_detector.py
# (Apache-2.0). Only the import paths have changed — detection logic is unchanged.
"""Hermes / Nous tool-call format: ``<tool_call>…</tool_call>`` wrapping JSON.

Also matches the format emitted by most Qwen/Hermes chat templates in
llama-cpp-python, which is why it's the default fallback detector for
models that don't have a dedicated entry.
"""

import json
import logging
import re

from edgevox.llm.tool_parsers._types import Tool
from edgevox.llm.tool_parsers.base import BaseFormatDetector
from edgevox.llm.tool_parsers.core_types import StreamingParseResult, StructureInfo, _GetInfoFunc

logger = logging.getLogger(__name__)


class HermesDetector(BaseFormatDetector):
    """Detector for Hermes tool call format.

    Format::

        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            for match in self.tool_call_regex.findall(text):
                raw = match[0] or match[1]
                if not raw:
                    continue
                parsed = json.loads(raw.strip())
                if isinstance(parsed, list):
                    calls.extend(self.parse_base_json(parsed, tools))
                else:
                    calls.extend(self.parse_base_json(parsed, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            # Not-this-format is an expected outcome when several detectors
            # share the same ``bot_token`` (Hermes / Granite / Qwen25 all use
            # ``<tool_call>``). Log at DEBUG so the chain stays quiet.
            logger.debug(f"hermes detect_and_parse skipped: {e}")
            return StreamingParseResult(normal_text=text)

    def _clean_normal_text(self, text: str) -> str:
        if not text:
            return text

        self._normal_text_buffer += text

        if self.eot_token in self._normal_text_buffer:
            cleaned = self._normal_text_buffer.replace(self.eot_token, "")
            self._normal_text_buffer = ""
            return cleaned

        partial_len = self._ends_with_partial_token(self._normal_text_buffer, self.eot_token)
        if partial_len:
            safe_text = self._normal_text_buffer[:-partial_len]
            self._normal_text_buffer = self._normal_text_buffer[-partial_len:]
            return safe_text

        cleaned = self._normal_text_buffer
        self._normal_text_buffer = ""
        return cleaned

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            partial_len = self._ends_with_partial_token(current_text, self.bot_token)
            if partial_len:
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                safe_text = current_text
                self._buffer = ""
            return StreamingParseResult(normal_text=self._clean_normal_text(safe_text))

        bot_pos = current_text.find(self.bot_token)
        if bot_pos > 0:
            normal_text = current_text[:bot_pos]
            self._buffer = current_text[bot_pos:]
            return StreamingParseResult(normal_text=normal_text)

        result = super().parse_streaming_increment(new_text="", tools=tools)
        if result.normal_text:
            result.normal_text = self._clean_normal_text(result.normal_text)
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )
