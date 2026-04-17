"""Test harness: ScriptedLLM + helpers for exercising the agent loop.

Design goals:

- **Deterministic.** A test declares a list of LLM responses and gets
  exactly those back, in order. No chance of flakiness from real model
  variance.
- **Readable.** Helpers (:func:`reply`, :func:`call`, :func:`calls`,
  :func:`echo`) read as what they are — "reply with this text", "call
  this tool with these args".
- **Cover the full harness surface.** :class:`ScriptedLLM` implements
  both ``complete`` (the path used by the new :class:`LLMAgent._drive`)
  and ``_history`` / ``create_chat_completion`` (the path used by the
  old :class:`LLM._run_agent`).
- **Mirror real LLM semantics.** Threading lock, ``_language`` attribute,
  tokenize fallback — enough for compaction hooks to run.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# ScriptedLLM
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Fake LLM that returns a pre-declared sequence of chat messages.

    Each script item is either:

    - a ``dict`` — the ``"message"`` payload inside ``choices[0]``
      (use :func:`reply`, :func:`call`, :func:`calls` to build these).
    - an ``Exception`` — raised at that call.
    - a ``callable`` — invoked with ``(messages, tools)`` and must
      return a message dict; lets tests introspect context dynamically.
    """

    def __init__(self, script: Iterable[Any] | None = None, *, language: str = "en") -> None:
        self._script: list[Any] = list(script or [])
        self._language = language
        self.calls: list[dict] = []
        self._inference_lock = threading.RLock()

    def extend(self, items: Iterable[Any]) -> None:
        self._script.extend(items)

    def remaining(self) -> int:
        return len(self._script)

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
        )
        if not self._script:
            raise RuntimeError(
                f"ScriptedLLM exhausted after {len(self.calls)} calls; last messages=" + str(messages[-2:])
            )
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, dict):
            item = item(messages, tools)
        if not isinstance(item, dict):
            raise TypeError(f"ScriptedLLM script item must be dict/exception/callable, got {type(item)}")
        return {"choices": [{"message": item}]}


# ---------------------------------------------------------------------------
# Script builders
# ---------------------------------------------------------------------------


def reply(content: str) -> dict:
    """LLM response: plain text, no tool calls."""
    return {"content": content, "tool_calls": None}


def call(name: str, **args: Any) -> dict:
    """LLM response: a single tool call."""
    return {
        "content": None,
        "tool_calls": [
            {
                "id": f"c_{name}",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def calls(*specs: tuple[str, dict]) -> dict:
    """LLM response: multiple parallel tool calls.

    Usage: ``calls(("a", {}), ("b", {"x": 1}))``.
    """
    tcs = []
    for i, (name, args) in enumerate(specs):
        tcs.append(
            {
                "id": f"c_{i}_{name}",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return {"content": None, "tool_calls": tcs}


def echo_json() -> dict:
    """LLM response that looks like an echoed tool-result payload."""
    return {
        "content": '{"ok": true, "result": "42", "retry_hint": null}',
        "tool_calls": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scripted_llm_factory():
    """Returns a builder: ``llm = scripted_llm_factory([reply('hi'), ...])``."""

    def _make(script=None, *, language="en"):
        return ScriptedLLM(script, language=language)

    return _make


@pytest.fixture
def tmp_memory_store(tmp_path):
    from edgevox.agents.memory import JSONMemoryStore

    return JSONMemoryStore(tmp_path / "memory.json")


@pytest.fixture
def tmp_session_store(tmp_path):
    from edgevox.agents.memory import JSONSessionStore

    return JSONSessionStore(tmp_path / "sessions")


@pytest.fixture
def tmp_notes(tmp_path):
    from edgevox.agents.memory import NotesFile

    return NotesFile(tmp_path / "notes.md")


@pytest.fixture
def tmp_artifact_store(tmp_path):
    from edgevox.agents.artifacts import FileArtifactStore

    return FileArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def mem_artifact_store():
    from edgevox.agents.artifacts import InMemoryArtifactStore

    return InMemoryArtifactStore()
