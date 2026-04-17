"""Shared artifact store for structured agent-to-agent handoffs.

Inspired by Anthropic's harness-design guidance: context resets beat
compaction for long tasks, and *handoffs via files* give sub-agents a
clean context boundary without losing the parent's work.

An :class:`Artifact` is a named, typed blob (text, JSON, or bytes) with
metadata (author agent, created_at, tags). Agents read and write
artifacts via the :class:`ArtifactStore` protocol; a sub-agent spawn
can declare `artifacts=[...]` and the store is rendered into its
system prompt as a lightweight index so it can pull what it needs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)


ArtifactKind = Literal["text", "json", "bytes"]


@dataclass
class Artifact:
    name: str
    kind: ArtifactKind
    content: Any  # str for text, dict/list for json, bytes for bytes
    author: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    summary: str = ""  # short one-line description for index rendering


@runtime_checkable
class ArtifactStore(Protocol):
    """Thread-safe artifact store."""

    def write(self, artifact: Artifact) -> None: ...
    def read(self, name: str) -> Artifact | None: ...
    def delete(self, name: str) -> bool: ...
    def list(self, *, tag: str | None = None) -> list[Artifact]: ...
    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str: ...


# ---------------------------------------------------------------------------
# InMemoryArtifactStore (default, fast)
# ---------------------------------------------------------------------------


class InMemoryArtifactStore:
    """Dict-backed artifact store. Good for single-process pipelines."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, Artifact] = {}

    def write(self, artifact: Artifact) -> None:
        with self._lock:
            self._items[artifact.name] = artifact

    def read(self, name: str) -> Artifact | None:
        with self._lock:
            return self._items.get(name)

    def delete(self, name: str) -> bool:
        with self._lock:
            return self._items.pop(name, None) is not None

    def list(self, *, tag: str | None = None) -> list[Artifact]:
        with self._lock:
            items = list(self._items.values())
            if tag is not None:
                items = [a for a in items if tag in a.tags]
            return items

    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str:
        """Short markdown index — name, one-line summary, tags. Meant
        for injection into a sub-agent's system prompt so it knows what
        it has access to without pre-loading content."""
        items = sorted(self.list(tag=tag), key=lambda a: a.created_at, reverse=True)[:max_items]
        if not items:
            return ""
        lines = ["## Available artifacts"]
        for a in items:
            tagpart = f" [{', '.join(a.tags)}]" if a.tags else ""
            summary = a.summary or _auto_summary(a)
            lines.append(f"- `{a.name}` ({a.kind}){tagpart}: {summary}")
        return "\n".join(lines)


def _auto_summary(a: Artifact) -> str:
    if a.kind == "text" and isinstance(a.content, str):
        first = a.content.strip().splitlines()[0] if a.content.strip() else ""
        return first[:120]
    if a.kind == "json":
        try:
            keys = list(a.content.keys()) if isinstance(a.content, dict) else []
            return f"json with keys: {', '.join(keys[:5])}"
        except Exception:
            return "json blob"
    if a.kind == "bytes":
        size = len(a.content) if isinstance(a.content, (bytes, bytearray)) else 0
        return f"{size} bytes"
    return ""


# ---------------------------------------------------------------------------
# FileArtifactStore (persistent)
# ---------------------------------------------------------------------------


class FileArtifactStore:
    """File-backed store: each artifact is a file under ``base``.

    Text → ``<name>.txt``
    JSON → ``<name>.json``
    Bytes → ``<name>.bin``
    Metadata → ``<name>.meta.json``

    Names may include ``/`` for organization (subdirectories are created).
    """

    def __init__(self, base: str | Path) -> None:
        self.base = Path(base)
        self._lock = threading.RLock()

    def _paths(self, name: str, kind: ArtifactKind) -> tuple[Path, Path]:
        ext = {"text": ".txt", "json": ".json", "bytes": ".bin"}[kind]
        main = self.base / f"{name}{ext}"
        meta = self.base / f"{name}.meta.json"
        return main, meta

    def write(self, artifact: Artifact) -> None:
        with self._lock:
            main, meta = self._paths(artifact.name, artifact.kind)
            main.parent.mkdir(parents=True, exist_ok=True)
            if artifact.kind == "text":
                main.write_text(artifact.content or "", encoding="utf-8")
            elif artifact.kind == "json":
                main.write_text(json.dumps(artifact.content, indent=2, default=str), encoding="utf-8")
            else:
                main.write_bytes(artifact.content or b"")
            meta_data = {k: v for k, v in asdict(artifact).items() if k != "content"}
            meta.write_text(json.dumps(meta_data, indent=2, default=str), encoding="utf-8")

    def read(self, name: str) -> Artifact | None:
        with self._lock:
            for kind in ("text", "json", "bytes"):
                main, meta = self._paths(name, kind)  # type: ignore[arg-type]
                if main.exists() and meta.exists():
                    try:
                        meta_data = json.loads(meta.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if kind == "text":
                        content: Any = main.read_text(encoding="utf-8")
                    elif kind == "json":
                        try:
                            content = json.loads(main.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            content = None
                    else:
                        content = main.read_bytes()
                    return Artifact(content=content, **meta_data)
        return None

    def delete(self, name: str) -> bool:
        with self._lock:
            deleted = False
            for kind in ("text", "json", "bytes"):
                main, meta = self._paths(name, kind)  # type: ignore[arg-type]
                for p in (main, meta):
                    if p.exists():
                        p.unlink()
                        deleted = True
            return deleted

    def list(self, *, tag: str | None = None) -> list[Artifact]:
        if not self.base.exists():
            return []
        names: set[str] = set()
        for p in self.base.rglob("*.meta.json"):
            rel = p.relative_to(self.base).with_suffix("").with_suffix("")
            names.add(str(rel))
        items: list[Artifact] = []
        for name in names:
            a = self.read(name)
            if a is None:
                continue
            if tag is None or tag in a.tags:
                items.append(a)
        return items

    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str:
        items = sorted(self.list(tag=tag), key=lambda a: a.created_at, reverse=True)[:max_items]
        if not items:
            return ""
        lines = ["## Available artifacts"]
        for a in items:
            tagpart = f" [{', '.join(a.tags)}]" if a.tags else ""
            summary = a.summary or _auto_summary(a)
            lines.append(f"- `{a.name}` ({a.kind}){tagpart}: {summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def text_artifact(
    name: str, content: str, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="text", content=content, author=author, tags=list(tags), summary=summary)


def json_artifact(
    name: str, content: Any, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="json", content=content, author=author, tags=list(tags), summary=summary)


def bytes_artifact(
    name: str, content: bytes, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="bytes", content=content, author=author, tags=list(tags), summary=summary)


__all__ = [
    "Artifact",
    "ArtifactKind",
    "ArtifactStore",
    "FileArtifactStore",
    "InMemoryArtifactStore",
    "bytes_artifact",
    "json_artifact",
    "text_artifact",
]
