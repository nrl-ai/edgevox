"""LLM model catalog — named presets for GGUF models on HuggingFace.

A preset is a ``(repo, filename)`` pair plus metadata (approx size, whether
it's embodied/robotics-tuned, optional ``chat_format`` override for
``llama-cpp-python``). Use :func:`resolve_preset` to turn a slug into a
:class:`ModelPreset` and :func:`download_preset` to cache the GGUF file
locally.

``_resolve_model_path`` (in :mod:`edgevox.llm.llamacpp`) accepts any of:

- Local path: ``/path/to/model.gguf``
- HuggingFace shorthand: ``hf:repo/name:filename.gguf``
- Preset slug: ``preset:qwen3-1.7b`` (or just ``qwen3-1.7b``)
- ``None`` → the default preset (see :data:`DEFAULT_PRESET`)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    """A named GGUF model with enough metadata to download and load it.

    Fields:
        slug / repo / filename: locate the GGUF on HuggingFace.
        size_gb: approximate on-disk size at Q4_K_M.
        family / description / embodied: display metadata.
        chat_format: override for ``llama_cpp.Llama(chat_format=...)``. Leave
            ``None`` to let llama-cpp auto-detect from the GGUF's Jinja template.
            Set to ``"chatml-function-calling"`` for most tool-calling models
            where llama-cpp's built-in tool dispatch is useful.
        tool_call_parsers: ordered list of detector names (see
            :mod:`edgevox.llm.tool_parsers`) to try against the assistant's
            ``content`` when no structured ``tool_calls`` come back. This is
            how we recover tool calls from models whose chat template emits
            them inline (Qwen3, Llama 3.x, Mistral, etc.) — the only path
            currently usable with ``llama-cpp-python 0.3.20``.
    """

    slug: str
    repo: str
    filename: str
    size_gb: float
    family: str
    description: str
    chat_format: str | None = None
    tool_call_parsers: tuple[str, ...] = ()
    embodied: bool = False


# Registry. Ordered roughly by size within each family.
PRESETS: dict[str, ModelPreset] = {
    preset.slug: preset
    for preset in (
        # --- Gemma (current default) -------------------------------------
        ModelPreset(
            slug="gemma-4-e2b",
            repo="unsloth/gemma-4-E2B-it-GGUF",
            filename="gemma-4-E2B-it-Q4_K_M.gguf",
            size_gb=1.8,
            family="gemma",
            description="Google Gemma 4 E2B Instruct — the EdgeVox default.",
            # Gemma uses its own ``<|tool_call>call:…<tool_call|>`` syntax,
            # handled by the legacy regex fallback in llamacpp.py.
            tool_call_parsers=(),
        ),
        # --- Qwen --------------------------------------------------------
        ModelPreset(
            slug="qwen3-1.7b",
            repo="unsloth/Qwen3-1.7B-GGUF",
            filename="Qwen3-1.7B-Q4_K_M.gguf",
            size_gb=1.1,
            family="qwen",
            description="Alibaba Qwen3 1.7B — small multilingual chat model with tool calling.",
            tool_call_parsers=("qwen25", "hermes"),
        ),
        ModelPreset(
            slug="qwen2.5-1.5b",
            repo="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            size_gb=1.0,
            family="qwen",
            description="Alibaba Qwen2.5 1.5B Instruct — tiny, Apache-2.0.",
            tool_call_parsers=("qwen25", "hermes"),
        ),
        ModelPreset(
            slug="qwen2.5-3b",
            repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
            filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            size_gb=2.0,
            family="qwen",
            description="Alibaba Qwen2.5 3B Instruct — strong multilingual + tools.",
            tool_call_parsers=("qwen25", "hermes"),
        ),
        # --- Llama -------------------------------------------------------
        ModelPreset(
            slug="llama-3.2-1b",
            repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            size_gb=0.8,
            family="llama",
            description="Meta Llama 3.2 1B Instruct — smallest Llama, good on-device.",
            tool_call_parsers=("llama32", "pythonic"),
        ),
        ModelPreset(
            slug="llama-3.2-3b",
            repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size_gb=2.0,
            family="llama",
            description="Meta Llama 3.2 3B Instruct — solid dialog + tool calls.",
            tool_call_parsers=("llama32", "pythonic"),
        ),
        # --- SmolLM ------------------------------------------------------
        ModelPreset(
            slug="smollm3-3b",
            repo="unsloth/SmolLM3-3B-GGUF",
            filename="SmolLM3-3B-Q4_K_M.gguf",
            size_gb=1.9,
            family="smollm",
            description="HuggingFace SmolLM3 3B — fully open, strong at the 3B scale.",
            tool_call_parsers=("hermes", "qwen25"),
        ),
        # --- Tool-calling specialist fine-tunes ------------------------
        # Salesforce xLAM-2 — top BFCL v3 scores but CC-BY-NC-4.0 license.
        ModelPreset(
            slug="xlam-2-1b-fc",
            repo="Salesforce/xLAM-2-1b-fc-r-gguf",
            filename="xLAM-2-1B-fc-r-Q4_K_M.gguf",
            size_gb=1.0,
            family="xlam",
            description="Salesforce xLAM-2 1B fc-r — top BFCL ≤2B. License: CC-BY-NC-4.0 (non-commercial).",
            tool_call_parsers=("xlam", "hermes"),
        ),
        ModelPreset(
            slug="xlam-2-3b-fc",
            repo="Salesforce/xLAM-2-3b-fc-r-gguf",
            filename="xLAM-2-3B-fc-r-Q4_K_M.gguf",
            size_gb=2.0,
            family="xlam",
            description="Salesforce xLAM-2 3B fc-r — tool-call specialist. License: CC-BY-NC-4.0.",
            tool_call_parsers=("xlam", "hermes"),
        ),
        ModelPreset(
            slug="xlam-2-8b-fc",
            repo="Salesforce/Llama-xLAM-2-8b-fc-r-gguf",
            filename="Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf",
            size_gb=4.9,
            family="xlam",
            description="Salesforce xLAM-2 8B fc-r — BFCL top-5 at 8B. License: CC-BY-NC-4.0.",
            tool_call_parsers=("xlam", "llama32", "hermes"),
        ),
        # IBM Granite 4.0 Nano — Apache 2.0, edge-first hybrid Mamba-2.
        ModelPreset(
            slug="granite-4.0-350m",
            repo="ibm-granite/granite-4.0-h-350m-GGUF",
            filename="granite-4.0-h-350m-Q4_K_M.gguf",
            size_gb=0.3,
            family="granite",
            description="IBM Granite 4.0 H-350M — Apache-2.0, ultra-tiny tool caller (listed in BFCL).",
            tool_call_parsers=("granite", "hermes"),
        ),
        ModelPreset(
            slug="granite-4.0-1b",
            repo="ibm-granite/granite-4.0-h-1b-GGUF",
            filename="granite-4.0-h-1b-Q4_K_M.gguf",
            size_gb=0.9,
            family="granite",
            description="IBM Granite 4.0 H-1B (Nano) — Apache-2.0, BFCL v3 ~50, edge-first.",
            tool_call_parsers=("granite", "hermes"),
        ),
        # MeetKai Functionary v3.2 — MIT, BFCL v3 ~82.8.
        # chat_format="functionary-v2" requires hf_tokenizer_path=; not wired
        # through LLM() yet, so we let llama-cpp auto-detect and rely on the
        # parser chain instead. See report Section 7.3.
        ModelPreset(
            slug="functionary-v3.2",
            repo="meetkai/functionary-small-v3.2-GGUF",
            filename="functionary-small-v3.2.Q4_0.gguf",
            size_gb=4.5,
            family="functionary",
            description="MeetKai Functionary small v3.2 — MIT, native OpenAI tool_calls format, BFCL ~82.8.",
            tool_call_parsers=("hermes", "llama32", "xlam"),
        ),
        # Nous Hermes 3 on Llama-3.2-3B — designed for <tool_call> format.
        ModelPreset(
            slug="hermes-3-3b",
            repo="NousResearch/Hermes-3-Llama-3.2-3B-GGUF",
            filename="Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
            size_gb=2.0,
            family="hermes",
            description="Nous Hermes 3 on Llama-3.2-3B — designed-for-tool-calling chatml format.",
            tool_call_parsers=("hermes", "qwen25"),
        ),
        # Microsoft Phi-4-mini — MIT, 3.8B, strong tool calling.
        ModelPreset(
            slug="phi-4-mini",
            repo="unsloth/Phi-4-mini-instruct-GGUF",
            filename="Phi-4-mini-instruct-Q4_K_M.gguf",
            size_gb=2.4,
            family="phi",
            description="Microsoft Phi-4-mini Instruct — 3.8B, MIT, listed in BFCL v4 supported.",
            tool_call_parsers=("hermes", "pythonic"),
        ),
        # Team-ACE ToolACE-2 — Apache-2.0, "best 8B" per paper claim.
        ModelPreset(
            slug="toolace-2-8b",
            repo="mradermacher/ToolACE-2-Llama-3.1-8B-GGUF",
            filename="ToolACE-2-Llama-3.1-8B.Q4_K_M.gguf",
            size_gb=4.9,
            family="toolace",
            description="Team-ACE ToolACE-2-Llama-3.1-8B — Apache-2.0, SOTA 8B on BFCL v3 (paper).",
            tool_call_parsers=("pythonic", "llama32", "hermes"),
        ),
        # MadeAgents Hammer 2.1 — qwen-research license.
        ModelPreset(
            slug="hammer-2.1-0.5b",
            repo="MaayanYosef/Hammer2.1-0.5b-Q4_K_M-GGUF",
            filename="hammer2.1-0.5b-q4_k_m.gguf",
            size_gb=0.4,
            family="hammer",
            description="MadeAgents Hammer 2.1 0.5B — Qwen2.5-Coder base, best-in-class ≤1B on BFCL v3.",
            tool_call_parsers=("xlam", "hermes"),
        ),
        # --- Embodied / robotics ----------------------------------------
        ModelPreset(
            slug="robobrain-2.0-7b",
            repo="Mungert/RoboBrain2.0-7B-GGUF",
            filename="RoboBrain2.0-7B-q4_k_m.gguf",
            size_gb=4.7,
            family="robobrain",
            description=(
                "BAAI RoboBrain 2.0 7B — embodied VLM tuned for spatial reasoning, "
                "affordances, trajectory forecasting. Runs text-only under llama.cpp."
            ),
            embodied=True,
            tool_call_parsers=(),
        ),
    )
}


DEFAULT_PRESET = "gemma-4-e2b"


def list_presets() -> list[ModelPreset]:
    """Return all registered presets in insertion order."""
    return list(PRESETS.values())


def resolve_preset(slug: str) -> ModelPreset:
    """Return the preset for ``slug`` or raise ``KeyError`` with known names."""
    if slug in PRESETS:
        return PRESETS[slug]
    known = ", ".join(PRESETS)
    raise KeyError(f"Unknown LLM preset '{slug}'. Known presets: {known}")


def download_preset(slug: str) -> str:
    """Download (or return cached path to) the GGUF for a preset slug."""
    from huggingface_hub import hf_hub_download

    preset = resolve_preset(slug)
    return hf_hub_download(repo_id=preset.repo, filename=preset.filename)
