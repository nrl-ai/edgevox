"""LLM chat via llama-cpp-python, targeting Gemma 4 E2B IT.

Supports both local GGUF files and automatic download from HuggingFace.
Auto-detects GPU layers based on available VRAM.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_HF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_HF_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"

SYSTEM_PROMPT = (
    "You are Vox, an AI assistant built with EdgeVox. "
    "Keep your responses concise and conversational — aim for 1-3 sentences. "
    "You are talking to the user in real time via voice."
)

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


def get_system_prompt(language: str = "en") -> str:
    """Get system prompt with language hint."""
    hint = LANGUAGE_HINTS.get(language, "")
    return hint + SYSTEM_PROMPT


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


def _resolve_model_path(model_path: str | None) -> str:
    """Return local path to GGUF model, downloading if needed.

    Accepts:
      - Local file path: ``/path/to/model.gguf``
      - HuggingFace shorthand: ``hf:repo/name:filename.gguf``
        e.g. ``hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf``
      - None → downloads the default Gemma 4 E2B model.
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

    log.info(f"Downloading {DEFAULT_HF_REPO}/{DEFAULT_HF_FILE} ...")
    path = hf_hub_download(
        repo_id=DEFAULT_HF_REPO,
        filename=DEFAULT_HF_FILE,
    )
    log.info(f"Model cached at: {path}")
    return path


class LLM:
    """llama-cpp-python chat wrapper."""

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, language: str = "en"):
        from llama_cpp import Llama

        resolved = _resolve_model_path(model_path)
        n_gpu = _detect_gpu_layers()

        log.info(f"Loading LLM: {resolved} (n_gpu_layers={n_gpu}, n_ctx={n_ctx})")
        self._llm = Llama(
            model_path=resolved,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu,
            verbose=False,
            flash_attn=True,
        )
        self._language = language
        self._history: list[dict] = [
            {"role": "system", "content": get_system_prompt(language)},
        ]
        log.info("LLM loaded.")

    def set_language(self, language: str):
        """Update the system prompt for a new language. Keeps conversation history."""
        self._language = language
        self._history[0] = {"role": "system", "content": get_system_prompt(language)}

    def chat(self, user_message: str) -> str:
        """Send a message and return the full response."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        result = self._llm.create_chat_completion(
            messages=self._history,
            max_tokens=256,
            temperature=0.7,
        )
        reply = result["choices"][0]["message"]["content"].strip()
        elapsed = time.perf_counter() - t0

        self._history.append({"role": "assistant", "content": reply})
        # Keep history manageable
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

        log.info(f'LLM: {elapsed:.2f}s → "{reply[:80]}..."')
        return reply

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Stream response token-by-token for lower time-to-first-token."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        full_reply = []
        stream = self._llm.create_chat_completion(
            messages=self._history,
            max_tokens=256,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                full_reply.append(token)
                yield token

        reply = "".join(full_reply).strip()
        elapsed = time.perf_counter() - t0

        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

        log.info(f'LLM stream: {elapsed:.2f}s → "{reply[:80]}..."')

    def reset(self):
        """Clear conversation history."""
        self._history = self._history[:1]
