"""Shared text REPL for built-in agent examples.

Handles all the user-facing niceties (model resolution with a progress
banner, tool-call tracing, coloured prompts) so the individual agent
modules can stay focused on their tool definitions. Uses the same
``LLM.chat`` code path the voice pipeline does, so anything that works
here will work through the microphone too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from edgevox.agents import AgentContext, AgentEvent, LLMAgent
from edgevox.llm import LLM, Tool, ToolCallResult
from edgevox.llm.llamacpp import DEFAULT_HF_FILE, DEFAULT_HF_REPO


def _parse_hf_shorthand(model: str | None) -> tuple[str, str] | None:
    """Extract ``(repo, filename)`` from an ``hf:repo:file`` shorthand,
    or return ``None`` if the argument is a local path."""
    if model is None:
        return DEFAULT_HF_REPO, DEFAULT_HF_FILE
    if model.startswith("hf:"):
        rest = model[3:]
        if ":" not in rest:
            raise ValueError(f"Expected 'hf:repo/name:filename.gguf', got {model!r}")
        repo, _, filename = rest.partition(":")
        return repo, filename
    return None  # local path


def _is_cached(repo: str, filename: str) -> bool:
    """Return True if the given HF GGUF is already on disk."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=repo, filename=filename)
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:
        return False


def _load_llm_with_progress(console: Console, *, model: str | None, **kwargs: Any) -> LLM:
    """Construct the LLM with a user-facing progress banner.

    On first run for a given model, prints a panel warning about the
    one-time download so users don't assume the app has hung while
    ``hf_hub_download`` streams bytes through tqdm. On warm starts just
    shows a spinner while llama-cpp memory-maps the file.
    """
    hf_pair = _parse_hf_shorthand(model)
    if hf_pair is not None:
        repo, filename = hf_pair
        target = f"{repo}/{filename}"
        if not _is_cached(repo, filename):
            console.print(
                Panel(
                    f"[bold]First-run model download[/bold]\n"
                    f"Fetching [cyan]{target}[/] into your Hugging Face cache.\n"
                    f"This happens once per model — later runs start in a few seconds.\n"
                    f"[dim]You'll see a progress bar from huggingface_hub below.[/]",
                    title="[yellow]downloading model[/]",
                    border_style="yellow",
                )
            )
            console.print(f"[yellow]⬇ downloading {target}...[/]")
            llm = LLM(model_path=model, **kwargs)
        else:
            with console.status(f"[green]loading {target} from cache...", spinner="dots"):
                llm = LLM(model_path=model, **kwargs)
    else:
        # Local file path — no download step, just load.
        with console.status(f"[green]loading {model} ...", spinner="dots"):
            llm = LLM(model_path=model, **kwargs)
    return llm


def run_repl(
    title: str,
    tools: Iterable[Any] | None,
    *,
    language: str = "en",
    greeting: str | None = None,
    model: str | None = None,
) -> None:
    """Drive an LLM with the given tools via a stdin REPL.

    Args:
        title: banner text shown at the top.
        tools: iterable of ``@tool``-decorated functions or a registry.
        language: LLM language code, e.g. ``"en"`` or ``"vi"``.
        greeting: optional opening line from the assistant.
        model: ``None`` for the default Gemma GGUF, ``"hf:repo:file"``
            for another HF GGUF, or a local path to a ``.gguf`` file.
    """
    logging.basicConfig(level=logging.WARNING)
    console = Console()

    def on_event(ev: AgentEvent) -> None:
        if ev.kind != "tool_call":
            return
        result: ToolCallResult = ev.payload
        if result.ok:
            console.print(f"[dim cyan]↳ {result.name}({result.arguments}) → {result.result}[/]")
        else:
            console.print(f"[red]↳ {result.name} failed: {result.error}[/]")

    console.print(Panel.fit(title, border_style="green"))
    # Load the LLM without any tool registry — the agent owns tools now.
    llm = _load_llm_with_progress(console, model=model, language=language, tools=None)

    agent = LLMAgent(
        name="vox",
        description="EdgeVox REPL assistant.",
        instructions="You are Vox, a concise voice assistant.",
        tools=tools,
        llm=llm,
    )
    ctx = AgentContext()
    ctx.bus.subscribe_all(on_event)

    registered = ", ".join(t.name for t in agent.tools) or "(none)"
    console.print(f"[dim]tools: {registered}[/]")
    if greeting:
        console.print(f"[bold green]vox:[/] {greeting}")

    while True:
        try:
            user = console.input("[bold]you:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not user:
            continue
        if user.lower() in {"quit", "exit", "bye"}:
            return
        reply = agent.run(user, ctx).reply
        console.print(f"[bold green]vox:[/] {reply}")


def print_tool_summary(tools: Iterable[Tool]) -> None:
    """Dump the JSON schema the model actually sees — handy when
    debugging tool descriptions or parameter types."""
    for t in tools:
        print(json.dumps(t.openai_schema(), indent=2))
