"""Utilities for keeping the LLM prompt stable across hops / turns.

llama.cpp and compatible backends reuse the KV cache when the token
prefix matches what was previously evaluated. In a typical multi-hop
agent turn (tool call → tool result → follow-up), the system prompt
and tool schemas are identical on every hop, so a cache-aware loop
can skip re-evaluating them and ship to first-token in a fraction of
the time.

What defeats the cache:

1. **Dict key reordering in JSON schemas.** ``{"type":"object","properties":{...}}``
   vs ``{"properties":{...},"type":"object"}`` tokenise differently.
2. **Whitespace drift in content strings.** A trailing space on the
   system prompt bumps every subsequent byte.
3. **Message rebuilding.** If each hop rebuilds the system message
   dict with a fresh object, the content *string* is probably the
   same — but if upstream code concatenates a timestamp or a counter
   into the prompt "to help the model", the prefix is different and
   the cache misses.

This module ships three small utilities — none of them touch the
agent loop; they're additive tools downstream code can adopt when it
wants the cache win:

* :func:`canonicalise_messages` — re-serialise a message list with
  sorted dict keys everywhere, so semantic-identical messages have
  byte-identical representations.
* :func:`stable_tool_schemas` — ``json.dumps`` each tool schema with
  sorted keys, so ``ToolRegistry.openai_schemas()`` can be passed to
  the LLM exactly the same way on every call.
* :func:`tool_schema_fingerprint` — SHA-256 of the canonicalised tool
  schema list. Use it to detect no-change across hops so a
  cache-aware frontend can skip re-sending the schemas.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with dict keys sorted recursively.

    Does **not** mutate ``messages``. Safe to call every hop.
    """
    return [_canon(m) for m in messages]


def stable_tool_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool schemas re-serialised with sorted keys.

    Passing this stable view on every hop means the tokenised tool
    schema block is byte-identical, maximising llama.cpp's KV-cache
    hit rate on the tools section of the prompt.
    """
    return [_canon(s) for s in schemas]


def tool_schema_fingerprint(schemas: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 hex digest of ``schemas``.

    A cache-aware frontend can store the prior fingerprint on the
    agent context, compare it to the current one, and skip re-sending
    tool schemas to the LLM when they haven't changed. (The backend
    still needs the schemas to parse tool calls, so this is a
    frontend-side optimisation — useful for agents that ship full
    tool manifests to a remote LLM server.)
    """
    payload = json.dumps([_canon(s) for s in schemas], separators=(",", ":"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canon(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    return obj


__all__ = [
    "canonicalise_messages",
    "stable_tool_schemas",
    "tool_schema_fingerprint",
]
