"""SLM agent-loop hardening helpers.

Small language models (1B-8B) fail at tool calling in specific ways that
a naive ``call → dispatch → repeat`` loop can't recover from:

* **Looping** — calling the same tool with the same args until the budget
  is exhausted.
* **Wrong argument shapes** — emitting a call with kwargs that aren't in
  the tool signature.
* **Payload echo** — parroting the tool-result JSON back as the final
  assistant reply.

The helpers below back the three mitigations wired into
:class:`edgevox.llm.LLM._run_agent`. Research sources are cited in
``docs/reports/slm-tool-calling-benchmark.md`` §7.4.
"""

from __future__ import annotations

import json

# Fire the "you already called this" hint on the 2nd identical call.
# ``seen > LOOP_HINT_AFTER`` is the truth check; LOOP_HINT_AFTER=1 fires on seen=2.
LOOP_HINT_AFTER = 1

# Hard-break the loop on the 3rd identical call.
LOOP_BREAK_AFTER = 2

# Retry budget per tool per turn. Mirrors pydantic-ai ``ModelRetry`` / instructor's
# ``max_retries``. One retry is usually enough for a 1-3B model to self-correct
# wrong kwarg names; two retries rarely helps and doubles cost.
MAX_SCHEMA_RETRIES = 1


def fingerprint_call(name: str, arguments: str | dict) -> str:
    """Stable fingerprint of a ``(name, args)`` pair for loop detection.

    ``arguments`` may be a JSON string (what OpenAI-style ``tool_calls``
    carry) or an already-decoded dict. We canonicalise to sorted-key JSON
    so semantically-identical calls collide regardless of field order or
    whitespace.
    """
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            decoded = {"__raw__": arguments}
    else:
        decoded = arguments or {}
    canon = json.dumps(decoded, sort_keys=True, separators=(",", ":"), default=str)
    return f"{name}:{canon}"


def is_argument_shape_error(error: str | None) -> bool:
    """Heuristic: does a dispatch error come from a wrong argument shape?

    Matches the common ``TypeError`` / ``ToolRegistry.dispatch`` signatures
    that surface when a model uses the wrong kwarg names.
    """
    if not error:
        return False
    signals = (
        "unexpected keyword argument",
        "missing 1 required positional argument",
        "missing required positional argument",
        "takes no arguments",
        "takes at most",
        "bad arguments:",
    )
    return any(s in error for s in signals)


def looks_like_echoed_payload(text: str) -> bool:
    """Detect when a small model has echoed our tool-result JSON as its reply.

    Symptoms: the reply starts with a JSON object and mentions any of our
    internal keys (``retry_hint``, ``ok``, ``error``, ``you_sent``,
    ``expected_schema``). Used to swap a leaked payload for a
    human-friendly message before the reply reaches TTS.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    markers = ('"retry_hint"', '"ok":', '"error":', '"you_sent"', '"expected_schema"')
    return any(m in stripped for m in markers)


def build_schema_retry_hint(name: str, error: str, parameters: dict | None) -> str:
    """Plain-English re-prompt listing the tool's real parameter schema.

    Small models treat a JSON error payload as text to echo rather than as
    an instruction. Prose works better. ``parameters`` is the JSON-schema
    dict from :attr:`edgevox.llm.tools.Tool.parameters`.
    """
    param_lines: list[str] = []
    if parameters and isinstance(parameters, dict):
        props = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        for key, schema in props.items():
            req = "required" if key in required else "optional"
            typ = schema.get("type", "string") if isinstance(schema, dict) else "string"
            desc = schema.get("description") if isinstance(schema, dict) else None
            suffix = f" - {desc}" if desc else ""
            param_lines.append(f"  - {key} ({typ}, {req}){suffix}")
    params_text = "\n".join(param_lines) or "  (no parameters)"
    return (
        f"Error: your call to {name} failed because the arguments are wrong. "
        f"{error}. "
        f"The tool {name} accepts ONLY these parameters:\n{params_text}\n"
        f"Retry ONCE with the correct parameter names, then stop."
    )


def build_loop_hint_payload(name: str) -> dict:
    """Tool-result payload injected on the 2nd identical call."""
    return {
        "ok": False,
        "error": (
            f"You already called {name} with these exact arguments. "
            f"Pick different arguments, call a different tool, or answer the user directly."
        ),
    }


def build_loop_break_payload(name: str) -> dict:
    """Tool-result payload injected on the 3rd identical call (before hard break)."""
    return {
        "ok": False,
        "error": f"repeated call to {name} with same arguments - hard break",
    }


FALLBACK_LOOP_BREAK = "Sorry, I couldn't complete that request."
FALLBACK_BUDGET_EXHAUSTED = "Sorry, I couldn't finish that request."
FALLBACK_ECHOED_PAYLOAD = "Sorry, I couldn't answer that."
