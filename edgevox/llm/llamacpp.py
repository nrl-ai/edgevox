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


def parse_tool_calls_from_content(
    content: str,
    *,
    preset_parsers: tuple[str, ...] = (),
    known_tools: set[str] | None = None,
    tool_schemas: list[dict] | None = None,
) -> tuple[list[dict], str, bool]:
    """Full LLM-output parser chain used by :class:`LLMAgent._drive`.

    Runs, in order: ``<think>`` stripping → preset-specific SGLang
    detectors → chatml / bare-JSON fallback → Gemma inline / plain-call
    fallback. Returns ``(tool_calls, cleaned_content, fallback_mode)``
    where ``fallback_mode`` signals that the caller must feed the
    results back as a synthetic user message rather than a ``tool`` role
    (because the chat template didn't produce structured tool_calls).

    This is the single source of truth for "what did the model want to
    call and what's left of its text reply?". The thread-safe
    :meth:`LLM.complete` routes structured ``tool_calls`` through the
    same logic so every agent sees consistent behavior regardless of
    the GGUF in use.
    """
    scrubbed = _strip_thinking(content or "")
    stripped = scrubbed.strip()

    # 1. SGLang preset-specific detectors (Hermes, Qwen2.5, Llama3.2, …)
    if preset_parsers:
        from edgevox.llm.tool_parsers import parse_tool_calls as sglang_parse

        sglang_calls = sglang_parse(scrubbed, tool_schemas, detectors=list(preset_parsers))
        if sglang_calls:
            return sglang_calls, "", True

    # 2. Chatml / Llama-native bare JSON
    chatml_calls = _parse_chatml_tool_calls(scrubbed)
    if chatml_calls:
        cut = scrubbed.find("<tool_call>")
        cleaned = scrubbed[:cut].strip() if cut >= 0 else ""
        return chatml_calls, cleaned, True

    # 3. Gemma inline markers / plain ``name(args)`` calls
    gemma_calls = _parse_gemma_inline_tool_calls(scrubbed, known_tools=known_tools)
    if gemma_calls:
        cut = scrubbed.find("<|tool_call>")
        cleaned = scrubbed[:cut].strip() if cut >= 0 else ""
        return gemma_calls, cleaned, True

    return [], stripped, False


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

    def _ensure_agent(self) -> Any:
        """Lazily build an :class:`LLMAgent` that owns this LLM's tools.

        Used by :meth:`chat` / :meth:`chat_stream` when the legacy
        ``tools=`` constructor arg was provided. The agent is created
        with :func:`edgevox.llm.hooks_slm.default_slm_hooks` so SLM
        hardening (loop detection, echoed-payload substitution,
        schema-retry) is applied automatically — without duplicating
        the loop implementation inside :class:`LLM`.
        """
        if getattr(self, "_shim_agent", None) is None:
            from edgevox.agents import AgentContext, LLMAgent
            from edgevox.llm.hooks_slm import default_slm_hooks

            self._shim_agent = LLMAgent(
                name="llm-shim",
                description="compatibility agent for LLM.chat(tools=...)",
                instructions=DEFAULT_PERSONA,
                tools=self._registry,
                hooks=default_slm_hooks(),
                llm=self,
                max_tool_hops=self._max_tool_hops,
            )
            self._shim_ctx = AgentContext()
            if self._on_tool_call is not None:
                cb = self._on_tool_call

                def _on_event(ev: Any) -> None:
                    if getattr(ev, "kind", None) == "tool_call":
                        try:
                            cb(ev.payload)
                        except Exception:
                            log.exception("on_tool_call callback raised")

                self._shim_ctx.bus.subscribe_all(_on_event)
        return self._shim_agent

    def chat(self, user_message: str) -> str:
        """Send a message and return the full response.

        With tools registered, delegates to the hook-based
        :class:`LLMAgent` loop so SLM hardening, parser chain, and
        structured tool dispatch stay in one place.
        """
        t0 = time.perf_counter()
        if self._registry:
            agent = self._ensure_agent()
            reply = agent.run(user_message, self._shim_ctx).reply
        else:
            self._history.append({"role": "user", "content": user_message})
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
        registered, delegates to :class:`LLMAgent` (non-streaming) and
        yields the final reply as a single chunk — downstream TTS that
        sentence-splits still works naturally.
        """
        t0 = time.perf_counter()
        if self._registry:
            agent = self._ensure_agent()
            reply = agent.run(user_message, self._shim_ctx).reply
            if reply:
                yield reply
        else:
            self._history.append({"role": "user", "content": user_message})
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
