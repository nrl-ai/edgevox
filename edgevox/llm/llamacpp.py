"""LLM chat via llama-cpp-python.

Defaults to the ``gemma-4-e2b`` preset but accepts any GGUF model from the
:mod:`edgevox.llm.models` catalog, a bare HuggingFace shorthand, or a local
path. Auto-detects GPU layers based on available VRAM.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

from edgevox.llm.models import DEFAULT_PRESET, PRESETS, resolve_preset
from edgevox.llm.tools import Tool, ToolCallResult, ToolRegistry

log = logging.getLogger(__name__)

# Gemma's chat template sometimes leaks its raw tool-call syntax into
# ``content`` instead of the structured ``tool_calls`` list. We recover
# it here so the agent loop stays reliable regardless of which chat
# format llama-cpp-python picks for the GGUF it happens to load.
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>\s*call:\s*(?P<name>\w+)\s*\{(?P<body>.*?)\}\s*<tool_call\|>",
    re.DOTALL,
)
_GEMMA_QUOTE_RE = re.compile(r"<\|\"\|>")
_KV_PAIR_RE = re.compile(
    r"(?P<k>\w+)\s*[:=]\s*"
    r'(?:"(?P<s>[^"]*)"|'
    r"(?P<n>-?\d+(?:\.\d+)?)|"
    r"(?P<b>true|false|True|False))"
)

# Plain Python-style function call that Gemma sometimes emits instead
# of the templated ``<|tool_call>`` markers. Matches ``name(key="val")``
# or ``name(key=1.2, other="s")``. Deliberately strict (identifier,
# then parenthesised kwargs only) so we don't false-positive on regular
# prose that happens to mention a function.
_PLAIN_CALL_RE = re.compile(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<body>[^()]*)\)")

# ``<think>…</think>`` blocks emitted by Qwen3 / DeepSeek-R1 / other
# "thinking" models. Stripped from user-facing replies so TTS doesn't read
# out the chain-of-thought, and also before inline tool-call parsing so
# embedded ``<tool_call>`` JSON blobs are still recovered.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# Chatml-style tool call that Qwen3 emits inside its thinking block.
_CHATML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _strip_thinking(content: str) -> str:
    """Remove ``<think>…</think>`` blocks. Returns ``content`` unchanged if none."""
    if "<think>" not in content.lower():
        return content
    return _THINK_BLOCK_RE.sub("", content).strip()


def _payload_to_call(payload: dict, idx: int, prefix: str) -> dict | None:
    """Turn a decoded JSON payload into an OpenAI-shaped tool-call dict.

    Accepts both common shapes:
      - ``{"name": "<fn>", "arguments": {...}}`` (Qwen, chatml, OpenAI)
      - ``{"function": "<fn>", "parameters": {...}}`` (Llama 3.x native)
    """
    name = payload.get("name") or payload.get("function")
    if not name:
        return None
    args = payload.get("arguments")
    if args is None:
        args = payload.get("parameters", {})
    return {
        "id": f"{prefix}_{idx}",
        "function": {
            "name": name,
            "arguments": json.dumps(args) if not isinstance(args, str) else args,
        },
    }


def _parse_chatml_tool_calls(content: str) -> list[dict] | None:
    """Recover ``<tool_call>…</tool_call>`` blocks (Qwen / chatml format)
    and bare top-level JSON tool-call objects (Llama 3.x native format).
    """
    calls: list[dict] = []
    # Wrapped ``<tool_call>…</tool_call>`` blocks.
    for idx, match in enumerate(_CHATML_TOOL_CALL_RE.finditer(content)):
        try:
            payload = json.loads(match.group("json"))
        except json.JSONDecodeError:
            continue
        call = _payload_to_call(payload, idx, "chatml_inline")
        if call:
            calls.append(call)

    if calls:
        return calls

    # Bare JSON object (Llama 3.x native): the whole scrubbed message body is
    # a ``{"function": ..., "parameters": ...}`` or ``{"name": ..., "arguments": ...}``.
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            call = _payload_to_call(payload, 0, "json_inline")
            if call:
                return [call]
    return None


def _parse_plain_kv_body(body: str) -> dict[str, Any]:
    """Parse ``key="val", other=1, flag=true`` into a dict."""
    args: dict[str, Any] = {}
    for kv in _KV_PAIR_RE.finditer(body):
        key = kv.group("k")
        if kv.group("s") is not None:
            args[key] = kv.group("s")
        elif kv.group("n") is not None:
            num = kv.group("n")
            args[key] = float(num) if "." in num else int(num)
        elif kv.group("b") is not None:
            args[key] = kv.group("b").lower() == "true"
    return args


def _parse_gemma_inline_tool_calls(content: str, known_tools: set[str] | None = None) -> list[dict] | None:
    """Return synthetic ``tool_calls`` entries parsed from raw Gemma
    template markers OR plain ``name(kwargs)`` text, or ``None``.

    Args:
        content: the assistant message body from llama-cpp.
        known_tools: if provided, the plain-call fallback only matches
            names in this set — prevents spurious matches on English
            phrases that happen to look like function calls.
    """
    if not content:
        return None

    calls: list[dict] = []

    # First pass: templated markers
    for idx, match in enumerate(_GEMMA_TOOL_CALL_RE.finditer(content)):
        body = _GEMMA_QUOTE_RE.sub('"', match.group("body"))
        calls.append(
            {
                "id": f"gemma_inline_{idx}",
                "function": {
                    "name": match.group("name"),
                    "arguments": json.dumps(_parse_plain_kv_body(body)),
                },
            }
        )

    if calls:
        return calls

    # Second pass: plain name(args) text — only when we have a tool
    # allowlist to constrain matches.
    if not known_tools:
        return None
    for idx, match in enumerate(_PLAIN_CALL_RE.finditer(content)):
        name = match.group("name")
        if name not in known_tools:
            continue
        body = match.group("body")
        calls.append(
            {
                "id": f"plain_inline_{idx}",
                "function": {
                    "name": name,
                    "arguments": json.dumps(_parse_plain_kv_body(body)),
                },
            }
        )
    return calls or None


# Exposed for back-compat — both are derived from the default preset.
DEFAULT_HF_REPO = PRESETS[DEFAULT_PRESET].repo
DEFAULT_HF_FILE = PRESETS[DEFAULT_PRESET].filename

DEFAULT_PERSONA = "You are Vox, an AI assistant built with EdgeVox."

BASE_VOICE_INSTRUCTIONS = (
    "Keep your responses concise and conversational — aim for 1-3 sentences. "
    "You are talking to the user in real time via voice."
)

# Retained for backwards compatibility with code that imports SYSTEM_PROMPT.
SYSTEM_PROMPT = f"{DEFAULT_PERSONA} {BASE_VOICE_INSTRUCTIONS}"

TOOL_SYSTEM_SUFFIX = (
    " You have tools available. You MUST call the matching tool when the user asks for live "
    "data (time, weather, calendar, system state) or an external action (home control, robot "
    "motion, playback). Do NOT answer such questions from memory. Call a tool only once per "
    "turn unless a result tells you to try again. After a tool runs, relay the result in one "
    "short sentence of plain speech — never read JSON aloud. If no tool matches the request, "
    "answer in plain speech without calling anything."
)

DEFAULT_MAX_TOOL_HOPS = 3

# SLM agent-loop hardening lives in ``_agent_harness``. See that module and
# ``docs/reports/slm-tool-calling-benchmark.md`` §7.4 for the research cites.
from edgevox.llm._agent_harness import (  # noqa: E402  (keep imports near the module top-level usage for clarity)
    FALLBACK_BUDGET_EXHAUSTED,
    FALLBACK_ECHOED_PAYLOAD,
    FALLBACK_LOOP_BREAK,
    LOOP_BREAK_AFTER,
    LOOP_HINT_AFTER,
    MAX_SCHEMA_RETRIES,
    build_loop_break_payload,
    build_loop_hint_payload,
    build_schema_retry_hint,
    fingerprint_call,
    is_argument_shape_error,
    looks_like_echoed_payload,
)

# Back-compat aliases (existing tests import these names).
_fingerprint_call = fingerprint_call
_is_argument_shape_error = is_argument_shape_error
_looks_like_echoed_payload = looks_like_echoed_payload


LANGUAGE_HINTS = {
    "vi": "Respond in Vietnamese (tiếng Việt). ",
    "fr": "Respond in French (français). ",
    "es": "Respond in Spanish (español). ",
    "ja": "Respond in Japanese (日本語). ",
    "zh": "Respond in Chinese (中文). ",
    "ko": "Respond in Korean (한국어). ",
    "de": "Respond in German (Deutsch). ",
    "it": "Respond in Italian (italiano). ",
    "pt": "Respond in Portuguese (português). ",
    "hi": "Respond in Hindi (हिन्दी). ",
    "th": "Respond in Thai (ภาษาไทย). ",
    "ru": "Respond in Russian (русский). ",
    "ar": "Respond in Arabic (العربية). ",
    "id": "Respond in Indonesian (Bahasa Indonesia). ",
}


def get_system_prompt(
    language: str = "en",
    has_tools: bool = False,
    persona: str | None = None,
) -> str:
    """Build the LLM system prompt.

    The final string is: ``<language hint> <persona> <voice instructions> <tool suffix?>``.
    Supplying ``persona`` overrides only the identity line so voice-mode
    guidance and language hints are preserved across every agent.
    """
    hint = LANGUAGE_HINTS.get(language, "")
    identity = persona if persona else DEFAULT_PERSONA
    base = f"{hint}{identity} {BASE_VOICE_INSTRUCTIONS}"
    if has_tools:
        base = base + TOOL_SYSTEM_SUFFIX
    return base


def _detect_gpu_layers() -> int:
    """Return number of layers to offload to GPU. 0 = CPU only."""
    from edgevox.core.gpu import get_nvidia_vram_gb, has_metal

    vram_gb = get_nvidia_vram_gb()
    if vram_gb is not None:
        if vram_gb >= 6:
            return -1  # offload all layers
        if vram_gb >= 3:
            return 20
    if has_metal():
        return -1
    return 0


def _resolve_preset(model_path: str | None):
    """Return the :class:`ModelPreset` backing ``model_path`` if any, else ``None``.

    Returns ``None`` for raw ``hf:`` shorthand or local paths that aren't
    mapped to a preset; use :func:`_resolve_model_path` to actually resolve
    to a filesystem path.
    """
    if model_path and model_path.startswith("preset:"):
        return resolve_preset(model_path[len("preset:") :])
    if model_path and model_path in PRESETS:
        return resolve_preset(model_path)
    if model_path is None:
        return resolve_preset(DEFAULT_PRESET)
    return None


def _resolve_model_path(model_path: str | None) -> str:
    """Return local path to GGUF model, downloading if needed.

    Accepts:
      - Local file path: ``/path/to/model.gguf``
      - HuggingFace shorthand: ``hf:repo/name:filename.gguf``
        e.g. ``hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf``
      - Preset slug: ``preset:qwen3-1.7b`` or bare ``qwen3-1.7b``
        (see :mod:`edgevox.llm.models` for the catalog).
      - ``None`` → downloads the default preset.
    """
    if model_path and Path(model_path).exists():
        return model_path

    from huggingface_hub import hf_hub_download

    if model_path and model_path.startswith("hf:"):
        parts = model_path[3:].split(":", 1)
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        if not filename:
            raise ValueError(f"HF model path must be 'hf:repo/name:filename', got '{model_path}'")
        log.info(f"Downloading {repo_id}/{filename} ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        log.info(f"Model cached at: {path}")
        return path

    preset = _resolve_preset(model_path)
    if preset is not None:
        log.info(f"Downloading preset '{preset.slug}' ({preset.repo}/{preset.filename}) ...")
        path = hf_hub_download(repo_id=preset.repo, filename=preset.filename)
        log.info(f"Model cached at: {path}")
        return path

    raise FileNotFoundError(
        f"Could not resolve model_path '{model_path}': not a local file, "
        f"not an 'hf:repo:file' shorthand, and not a known preset slug."
    )


ToolCallback = Callable[[ToolCallResult], None]


class LLM:
    """llama-cpp-python chat wrapper with optional tool-calling support."""

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        language: str = "en",
        tools: Iterable[Callable[..., object] | Tool] | ToolRegistry | None = None,
        max_tool_hops: int = DEFAULT_MAX_TOOL_HOPS,
        on_tool_call: ToolCallback | None = None,
        persona: str | None = None,
    ):
        from llama_cpp import Llama

        resolved = _resolve_model_path(model_path)
        preset = _resolve_preset(model_path)
        n_gpu = _detect_gpu_layers()

        llama_kwargs: dict = {
            "model_path": resolved,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu,
            "verbose": False,
            "flash_attn": True,
        }
        if preset and preset.chat_format:
            llama_kwargs["chat_format"] = preset.chat_format

        log.info(
            "Loading LLM: %s (n_gpu_layers=%s, n_ctx=%s, chat_format=%s)",
            resolved,
            n_gpu,
            n_ctx,
            preset.chat_format if preset else "(auto)",
        )
        self._llm = Llama(**llama_kwargs)
        self._language = language
        self._registry = self._build_registry(tools)
        self._max_tool_hops = max_tool_hops
        self._on_tool_call = on_tool_call
        # Tool-call parser chain — preset-specific SGLang detectors are tried
        # in order before the hand-rolled chatml / Gemma regex fallbacks.
        self._tool_call_parsers: tuple[str, ...] = preset.tool_call_parsers if preset else ()
        # Inference lock — llama_cpp.Llama is NOT thread-safe. Concurrent
        # ``create_chat_completion`` calls from parallel agents must
        # serialize here. Everything else in the agent framework
        # (tool/skill dispatch, bus publishing, workflow composition)
        # runs truly in parallel.
        self._inference_lock = threading.Lock()
        self._history: list[dict] = [
            {"role": "system", "content": get_system_prompt(language, has_tools=bool(self._registry))},
        ]
        log.info(
            "LLM loaded. Tools: %s",
            ", ".join(t.name for t in self._registry) if self._registry else "(none)",
        )

    @staticmethod
    def _build_registry(
        tools: Iterable[Callable[..., object] | Tool] | ToolRegistry | None,
    ) -> ToolRegistry:
        if tools is None:
            return ToolRegistry()
        if isinstance(tools, ToolRegistry):
            return tools
        registry = ToolRegistry()
        registry.register(*tools)
        return registry

    @property
    def tools(self) -> ToolRegistry:
        """The live tool registry. Mutate to add/remove tools at runtime."""
        return self._registry

    def register_tool(self, *funcs: Callable[..., object] | Tool) -> None:
        """Add tools after construction; updates system prompt if needed."""
        was_empty = not self._registry
        self._registry.register(*funcs)
        if was_empty and self._registry:
            self._history[0] = {
                "role": "system",
                "content": get_system_prompt(self._language, has_tools=True),
            }

    def set_language(self, language: str):
        """Update the system prompt for a new language. Keeps conversation history."""
        self._language = language
        self._history[0] = {
            "role": "system",
            "content": get_system_prompt(language, has_tools=bool(self._registry)),
        }

    def _completion_kwargs(self, stream: bool = False) -> dict:
        kwargs: dict = {
            "messages": self._history,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": stream,
        }
        if self._registry:
            kwargs["tools"] = self._registry.openai_schemas()
            kwargs["tool_choice"] = "auto"
        return kwargs

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
        """Thread-safe completion entry point for the agent framework.

        Takes explicit ``messages`` and ``tools`` (no reliance on
        ``self._history`` or ``self._registry``) so concurrent
        ``LLMAgent`` turns stay isolated. Serializes access to the
        underlying ``llama_cpp.Llama`` via :attr:`_inference_lock`
        because llama-cpp is not thread-safe.
        """
        kwargs: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        with self._inference_lock:
            return self._llm.create_chat_completion(**kwargs)

    def _run_agent(self) -> str:
        """Drive the model through tool calls until it produces a text reply.

        Always runs non-streaming. Only used when tools are registered.
        Applies SLM-specific harness hardening per the research in
        ``docs/reports/slm-tool-calling-benchmark.md``:

        * **Loop detection.** Fingerprints each ``(name, args)`` pair; a
          repeated call gets a "you already called this" hint back
          instead of a real dispatch. Three identical calls in a row
          hard-break with a fallback answer.
        * **Schema-retry hint.** When dispatch raises a ``bad arguments``
          error, the next hop receives the tool's actual JSON schema in
          the tool-result message so the model can self-correct. Retry
          budget is :data:`MAX_SCHEMA_RETRIES` per tool per turn.
        """
        call_counts: dict[str, int] = {}
        retry_budget: dict[str, int] = {}

        for hop in range(self._max_tool_hops + 1):
            result = self._llm.create_chat_completion(**self._completion_kwargs(stream=False))
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            raw_content = message.get("content") or ""
            # Strip <think>…</think> blocks (Qwen3 / R1-style reasoning)
            # before anything else so TTS doesn't read them and the fallback
            # parsers can still see embedded <tool_call> blobs.
            scrubbed = _strip_thinking(raw_content)
            content = scrubbed.strip()
            fallback_mode = False

            # Parser chain: preset-configured SGLang detectors first, then the
            # hand-rolled chatml / Gemma regex fallbacks for anything they miss.
            if not tool_calls and self._tool_call_parsers:
                from edgevox.llm.tool_parsers import parse_tool_calls

                sglang_calls = parse_tool_calls(
                    scrubbed,
                    self._registry.openai_schemas() if self._registry else None,
                    detectors=list(self._tool_call_parsers),
                )
                if sglang_calls:
                    log.debug(
                        "Recovered %d tool call(s) via SGLang detectors (%s)",
                        len(sglang_calls),
                        ", ".join(self._tool_call_parsers),
                    )
                    tool_calls = sglang_calls
                    fallback_mode = True
                    content = ""  # SGLang detectors consume the whole block

            if not tool_calls:
                # Qwen3 / chatml-style <tool_call>{...}</tool_call> inside content.
                chatml_calls = _parse_chatml_tool_calls(scrubbed)
                if chatml_calls:
                    log.debug("Recovered %d chatml tool call(s) from content", len(chatml_calls))
                    tool_calls = chatml_calls
                    fallback_mode = True
                    cut = scrubbed.find("<tool_call>")
                    content = scrubbed[:cut].strip() if cut >= 0 else ""

            if not tool_calls:
                fallback_calls = _parse_gemma_inline_tool_calls(scrubbed)
                if fallback_calls:
                    log.debug("Recovered %d inline tool call(s) from content", len(fallback_calls))
                    tool_calls = fallback_calls
                    fallback_mode = True
                    # Truncate anything from the first tool marker onward — the model
                    # often hallucinates a tool response + answer after it.
                    cut = scrubbed.find("<|tool_call>")
                    content = scrubbed[:cut].strip() if cut >= 0 else ""

            if not tool_calls:
                # Small models sometimes echo our retry/tool-result payloads
                # verbatim as their final reply. Substitute a human-friendly
                # fallback so TTS doesn't read JSON aloud.
                if looks_like_echoed_payload(content):
                    log.info("Final reply looks like echoed tool payload; substituting fallback")
                    content = FALLBACK_ECHOED_PAYLOAD
                self._history.append({"role": "assistant", "content": content})
                return content

            if hop == self._max_tool_hops:
                log.warning("Tool-call budget exhausted after %d hops", self._max_tool_hops)
                fallback = content or FALLBACK_BUDGET_EXHAUSTED
                if looks_like_echoed_payload(fallback):
                    fallback = FALLBACK_ECHOED_PAYLOAD
                self._history.append({"role": "assistant", "content": fallback})
                return fallback

            results: list[tuple[str, dict]] = []
            loop_break = False
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                arguments = fn.get("arguments", "{}")

                fp = fingerprint_call(name, arguments)
                call_counts[fp] = call_counts.get(fp, 0) + 1
                seen = call_counts[fp]

                if seen > LOOP_BREAK_AFTER:
                    log.warning("Loop detected after %d identical calls to %s; breaking", seen, name)
                    loop_break = True
                    results.append((name, build_loop_break_payload(name)))
                    continue

                if seen > LOOP_HINT_AFTER:
                    log.info("Repeated call to %s detected; injecting hint instead of dispatching", name)
                    results.append((name, build_loop_hint_payload(name)))
                    continue

                outcome = self._registry.dispatch(name, arguments)
                if self._on_tool_call is not None:
                    try:
                        self._on_tool_call(outcome)
                    except Exception:
                        log.exception("on_tool_call callback raised")

                # SCHEMA-RETRY: enrich the tool result with the real schema so
                # the next hop can self-correct. Budget is per-tool-per-turn.
                if not outcome.ok and is_argument_shape_error(outcome.error):
                    used = retry_budget.get(name, 0)
                    if used < MAX_SCHEMA_RETRIES:
                        retry_budget[name] = used + 1
                        tool_obj = self._registry.tools.get(name)
                        hint = build_schema_retry_hint(
                            name,
                            outcome.error or "",
                            tool_obj.parameters if tool_obj else None,
                        )
                        results.append((name, {"ok": False, "retry_hint": hint}))
                        continue

                payload = (
                    {"ok": True, "result": outcome.result} if outcome.ok else {"ok": False, "error": outcome.error}
                )
                results.append((name, payload))

            if loop_break:
                # Return a direct apology-style reply and skip the next LLM
                # turn entirely — the model would just re-emit the same call.
                fallback = content or FALLBACK_LOOP_BREAK
                self._history.append({"role": "assistant", "content": fallback})
                return fallback

            if fallback_mode:
                # The chat template didn't emit structured tool_calls, so a
                # ``tool`` role message won't be formatted correctly by
                # llama-cpp's Gemma handler. Feed results back as a user
                # message instead — works with any chat template.
                summary = "; ".join(f"{name} -> {json.dumps(payload, default=str)}" for name, payload in results)
                self._history.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                self._history.append(
                    {
                        "role": "user",
                        "content": f"(system: tool results — {summary}. "
                        f"Now answer the previous request in one short sentence.)",
                    }
                )
            else:
                self._history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                for call, (name, payload) in zip(tool_calls, results, strict=False):
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call_{hop}_{name}"),
                            "name": name,
                            "content": json.dumps(payload, default=str),
                        }
                    )
        return ""  # unreachable

    def chat(self, user_message: str) -> str:
        """Send a message and return the full response."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        if self._registry:
            reply = self._run_agent()
        else:
            result = self._llm.create_chat_completion(**self._completion_kwargs(stream=False))
            reply = _strip_thinking(result["choices"][0]["message"].get("content") or "").strip()
            self._history.append({"role": "assistant", "content": reply})
        elapsed = time.perf_counter() - t0

        self._truncate_history()

        log.info(f'LLM: {elapsed:.2f}s → "{reply[:80]}..."')
        return reply

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Stream the final response.

        With no tools registered, streams token-by-token. With tools
        registered, the agent loop runs non-streaming and the final
        reply is emitted as a single chunk — downstream TTS that
        sentence-splits still works naturally.
        """
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        if self._registry:
            reply = self._run_agent()
            if reply:
                yield reply
        else:
            full_reply: list[str] = []
            stream = self._llm.create_chat_completion(**self._completion_kwargs(stream=True))
            # Track <think>…</think> so we don't stream reasoning to TTS.
            accum = ""
            in_think = False
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content", "")
                if not token:
                    continue
                full_reply.append(token)
                accum += token
                if not in_think and "<think>" in accum.lower():
                    in_think = True
                    before, _, _ = accum.lower().partition("<think>")
                    # Flush anything before <think> (rare — model usually opens with it).
                    if before.strip():
                        yield accum[: len(before)]
                    accum = ""
                    continue
                if in_think:
                    if "</think>" in accum.lower():
                        _, _, after = accum.lower().partition("</think>")
                        if after.strip():
                            yield accum[-len(after) :].lstrip()
                        accum = ""
                        in_think = False
                    continue
                yield token
            reply = _strip_thinking("".join(full_reply)).strip()
            self._history.append({"role": "assistant", "content": reply})

        elapsed = time.perf_counter() - t0
        self._truncate_history()
        log.info(f'LLM stream: {elapsed:.2f}s → "{reply[:80]}..."')

    def _truncate_history(self) -> None:
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

    def reset(self):
        """Clear conversation history."""
        self._history = self._history[:1]
