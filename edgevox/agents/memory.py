"""Persistent memory and context compaction for the agent framework.

Three layers, all optional and pluggable:

- :class:`MemoryStore` — long-term facts, preferences, and episodes that
  survive across ``run()`` calls and process restarts. Rendered into the
  system prompt at ``on_run_start`` by
  :class:`edgevox.agents.hooks_builtin.MemoryInjectionHook`.

- :class:`SessionStore` — whole-:class:`Session` persistence keyed by
  session_id. Used by :class:`~edgevox.agents.hooks_builtin.PersistSessionHook`
  to save/resume conversations.

- :class:`Compactor` — summarizes old turns when the session crosses a
  token budget. Inspired by Anthropic's context-engineering guidance:
  trigger early (50-60%% of window on SLMs), preserve system prompt +
  recent turns verbatim, compress the middle.

Both stores are Protocols — swap the default :class:`JSONMemoryStore` /
:class:`JSONSessionStore` for Redis, SQLite, or a remote service without
touching the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from edgevox.agents.base import Session

if TYPE_CHECKING:
    from edgevox.llm.llamacpp import LLM

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact, Episode, Preference
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A single durable fact an agent has learned.

    Facts are key/value pairs with optional ``scope`` (e.g. ``"user"``,
    ``"env:kitchen"``) so one store can back multiple agents cleanly.
    """

    key: str
    value: str
    scope: str = "global"
    updated_at: float = field(default_factory=time.time)
    source: str = ""  # which agent or event created it


@dataclass
class Episode:
    """A single skill/tool outcome worth remembering.

    Robotics use case: "last time you tried to grasp the red block it
    slipped". Kept lightweight so thousands fit in memory without
    summarization.
    """

    kind: str  # "tool_call" | "skill" | "user_feedback" | ...
    payload: dict[str, Any]
    outcome: str  # "ok" | "failed" | "cancelled"
    timestamp: float = field(default_factory=time.time)
    agent: str = ""


@dataclass
class Preference:
    """A user preference. Kept distinct from :class:`Fact` because
    preferences are *directional* ('user prefers X over Y') and deserve
    their own rendering bucket in the system prompt."""

    key: str
    value: str
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Long-term agent memory.

    Implementations must be thread-safe — multiple agents sharing one
    store is the common case.
    """

    # Facts ------------------------------------------------------------

    def add_fact(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        source: str = "",
    ) -> None: ...

    def get_fact(self, key: str, *, scope: str = "global") -> str | None: ...

    def facts(self, *, scope: str | None = None) -> list[Fact]: ...

    def forget_fact(self, key: str, *, scope: str = "global") -> bool: ...

    # Preferences ------------------------------------------------------

    def set_preference(self, key: str, value: str) -> None: ...

    def preferences(self) -> list[Preference]: ...

    # Episodes ---------------------------------------------------------

    def add_episode(
        self,
        kind: str,
        payload: dict[str, Any],
        outcome: str,
        *,
        agent: str = "",
    ) -> None: ...

    def recent_episodes(
        self,
        n: int = 5,
        *,
        kind: str | None = None,
    ) -> list[Episode]: ...

    # Rendering --------------------------------------------------------

    def render_for_prompt(self, *, max_facts: int = 20, max_episodes: int = 5) -> str: ...


# ---------------------------------------------------------------------------
# JSONMemoryStore (default)
# ---------------------------------------------------------------------------


def default_memory_dir() -> Path:
    """Return the default base dir for memory files (``~/.edgevox/memory``)."""
    base = os.environ.get("EDGEVOX_MEMORY_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".edgevox" / "memory"


class JSONMemoryStore:
    """File-backed :class:`MemoryStore` — one JSON file per agent.

    Writes are debounced: ``add_*`` schedules a flush; flushes happen on
    the next ``flush()`` call or after :attr:`_flush_interval` seconds.
    Callers that need durability (``on_run_end``) should call
    :meth:`flush` explicitly.

    Schema
    ------

    .. code-block:: json

        {
          "facts": [{"key": "...", "value": "...", "scope": "...", ...}],
          "preferences": [{"key": "...", "value": "...", ...}],
          "episodes": [{"kind": "...", "payload": {...}, "outcome": "...", ...}]
        }
    """

    _flush_interval = 2.0
    _max_episodes = 500  # ring buffer; oldest pruned on overflow

    def __init__(self, path: str | Path, *, autoload: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._facts: dict[tuple[str, str], Fact] = {}  # (scope, key) → Fact
        self._preferences: dict[str, Preference] = {}
        self._episodes: list[Episode] = []
        self._dirty = False
        self._last_flush = 0.0
        if autoload:
            self._load()

    # ----- loading / saving -----

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.exception("Failed to load memory %s", self.path)
            return
        with self._lock:
            for raw in data.get("facts", []):
                try:
                    f = Fact(**raw)
                except TypeError:
                    continue
                self._facts[(f.scope, f.key)] = f
            for raw in data.get("preferences", []):
                try:
                    p = Preference(**raw)
                except TypeError:
                    continue
                self._preferences[p.key] = p
            for raw in data.get("episodes", []):
                try:
                    self._episodes.append(Episode(**raw))
                except TypeError:
                    continue

    def flush(self) -> None:
        """Write pending changes to disk."""
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "facts": [asdict(f) for f in self._facts.values()],
                "preferences": [asdict(p) for p in self._preferences.values()],
                "episodes": [asdict(e) for e in self._episodes[-self._max_episodes :]],
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False
            self._last_flush = time.monotonic()

    def _mark_dirty(self) -> None:
        self._dirty = True
        # Opportunistic flush: if enough time has passed, write now.
        if time.monotonic() - self._last_flush > self._flush_interval:
            self.flush()

    # ----- facts -----

    def add_fact(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        source: str = "",
    ) -> None:
        with self._lock:
            self._facts[(scope, key)] = Fact(key=key, value=value, scope=scope, source=source)
            self._mark_dirty()

    def get_fact(self, key: str, *, scope: str = "global") -> str | None:
        with self._lock:
            f = self._facts.get((scope, key))
            return f.value if f else None

    def facts(self, *, scope: str | None = None) -> list[Fact]:
        with self._lock:
            if scope is None:
                return list(self._facts.values())
            return [f for (s, _k), f in self._facts.items() if s == scope]

    def forget_fact(self, key: str, *, scope: str = "global") -> bool:
        with self._lock:
            popped = self._facts.pop((scope, key), None)
            if popped is not None:
                self._mark_dirty()
            return popped is not None

    # ----- preferences -----

    def set_preference(self, key: str, value: str) -> None:
        with self._lock:
            self._preferences[key] = Preference(key=key, value=value)
            self._mark_dirty()

    def preferences(self) -> list[Preference]:
        with self._lock:
            return list(self._preferences.values())

    # ----- episodes -----

    def add_episode(
        self,
        kind: str,
        payload: dict[str, Any],
        outcome: str,
        *,
        agent: str = "",
    ) -> None:
        with self._lock:
            self._episodes.append(Episode(kind=kind, payload=payload, outcome=outcome, agent=agent))
            if len(self._episodes) > self._max_episodes * 2:
                # Compact in place to bounded size.
                self._episodes = self._episodes[-self._max_episodes :]
            self._mark_dirty()

    def recent_episodes(self, n: int = 5, *, kind: str | None = None) -> list[Episode]:
        with self._lock:
            source = self._episodes if kind is None else [e for e in self._episodes if e.kind == kind]
            return source[-n:]

    # ----- rendering -----

    def render_for_prompt(self, *, max_facts: int = 20, max_episodes: int = 5) -> str:
        """Render memory as a concise markdown block for the system prompt.

        Follows Anthropic context-engineering guidance: minimal high-signal
        tokens, structured sections, no verbose wrappers.
        """
        with self._lock:
            lines: list[str] = []

            if self._preferences:
                lines.append("## Known preferences")
                for p in list(self._preferences.values())[:max_facts]:
                    lines.append(f"- {p.key}: {p.value}")

            if self._facts:
                lines.append("## Known facts")
                for rendered, f in enumerate(self._facts.values()):
                    if rendered >= max_facts:
                        break
                    scope_tag = "" if f.scope == "global" else f" [{f.scope}]"
                    lines.append(f"- {f.key}{scope_tag}: {f.value}")

            if self._episodes:
                recent = self._episodes[-max_episodes:]
                if recent:
                    lines.append("## Recent outcomes")
                    for e in recent:
                        brief = ", ".join(f"{k}={v}" for k, v in list(e.payload.items())[:3])
                        lines.append(f"- [{e.kind}] {brief} → {e.outcome}")

            return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    """Whole-:class:`Session` persistence."""

    def save(self, session_id: str, session: Session) -> None: ...
    def load(self, session_id: str) -> Session | None: ...
    def delete(self, session_id: str) -> bool: ...
    def list_ids(self) -> list[str]: ...


class JSONSessionStore:
    """File-per-session session store at ``<base>/<session_id>.json``."""

    def __init__(self, base: str | Path) -> None:
        self.base = Path(base)
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        return self.base / f"{session_id}.json"

    def save(self, session_id: str, session: Session) -> None:
        with self._lock:
            self.base.mkdir(parents=True, exist_ok=True)
            data = {"messages": session.messages, "state": _jsonable(session.state)}
            p = self._path(session_id)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(p)

    def load(self, session_id: str) -> Session | None:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.exception("Failed to load session %s", p)
            return None
        return Session(messages=list(data.get("messages", [])), state=dict(data.get("state", {})))

    def delete(self, session_id: str) -> bool:
        with self._lock:
            p = self._path(session_id)
            if p.exists():
                p.unlink()
                return True
            return False

    def list_ids(self) -> list[str]:
        if not self.base.exists():
            return []
        return [p.stem for p in self.base.glob("*.json")]


def _jsonable(obj: Any) -> Any:
    """Best-effort coerce to a JSON-safe shape (private-state may hold
    threading primitives or dataclasses that json.dumps rejects even
    with default=str)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not k.startswith("__")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


def estimate_tokens(messages: Iterable[dict]) -> int:
    """Rough token estimate (chars / 4). Good enough for threshold
    checks; precise counting requires a tokenizer."""
    total = 0
    for m in messages:
        c = m.get("content") or ""
        if isinstance(c, str):
            total += len(c) // 4 + 4
        total += 4  # role + scaffolding
    return total


COMPACTION_SYSTEM_PROMPT = """You are a conversation summarizer.

Given a chat history, produce a concise bulleted summary that preserves:
- the user's intent and goal
- decisions made
- tool/skill outcomes and any failures
- unresolved questions

Omit: tool-call JSON blobs, chit-chat, repeated acknowledgments, internal formatting.

Output a single message ≤200 words, no preamble, no quotation marks."""


@dataclass
class Compactor:
    """Summarize old turns when the session crosses a token budget.

    Preservation priority (Anthropic guidance):
    1. System prompt (always kept verbatim, position 0)
    2. Last ``keep_last_turns`` user/assistant turns (verbatim)
    3. Compressed summary of everything between (single assistant msg)

    Triggered by :class:`~edgevox.agents.hooks_builtin.ContextCompactionHook`
    between turns, never mid-turn (would break tool-call chains).
    """

    trigger_tokens: int = 4000
    keep_last_turns: int = 4
    # Maximum tokens for the summary itself.
    summary_max_tokens: int = 300

    def should_compact(self, messages: list[dict]) -> bool:
        if len(messages) < self.keep_last_turns + 2:
            return False
        return estimate_tokens(messages) >= self.trigger_tokens

    def compact(self, messages: list[dict], llm: LLM | None) -> list[dict]:
        """Return a compacted copy of ``messages``.

        If ``llm`` is None (test / offline path), falls back to a
        deterministic truncation that keeps system + last N turns.
        """
        if not self.should_compact(messages):
            return list(messages)

        system = messages[0] if messages and messages[0].get("role") == "system" else None
        body = messages[1:] if system is not None else list(messages)
        if len(body) <= self.keep_last_turns:
            return list(messages)

        to_compress = body[: -self.keep_last_turns]
        keep = body[-self.keep_last_turns :]

        summary = self._summarize(to_compress, llm)
        summary_msg = {
            "role": "assistant",
            "content": f"(summary of earlier conversation)\n{summary}",
        }

        out: list[dict] = []
        if system is not None:
            out.append(system)
        out.append(summary_msg)
        out.extend(keep)
        return out

    def _summarize(self, messages: list[dict], llm: LLM | None) -> str:
        """Single-shot summary via the same LLM. Falls back to a
        bullet-list of roles + first-80-chars when LLM unavailable."""
        if llm is not None:
            try:
                prompt = [
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Summarize this conversation:\n\n" + _render_messages_for_summary(messages),
                    },
                ]
                result = llm.complete(prompt, max_tokens=self.summary_max_tokens, temperature=0.3)
                text = result["choices"][0]["message"].get("content") or ""
                return text.strip() or _fallback_summary(messages)
            except Exception:
                log.exception("Compactor LLM summarize failed; falling back")
        return _fallback_summary(messages)


def _render_messages_for_summary(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        # Trim tool-result JSON blobs to save summary tokens.
        if isinstance(content, str) and content.startswith("{") and len(content) > 200:
            content = content[:200] + "…"
        out.append(f"[{role}] {content}")
    return "\n".join(out)


def _fallback_summary(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, str) and content:
            snippet = content[:80].replace("\n", " ")
            lines.append(f"- {role}: {snippet}")
    return "\n".join(lines[:20])


# ---------------------------------------------------------------------------
# NOTES.md-style note-taking (lightweight persistent scratchpad)
# ---------------------------------------------------------------------------


class NotesFile:
    """A plain-text notes file the agent can read/write as long-term
    working memory (Anthropic NOTES.md pattern).

    Unlike :class:`MemoryStore`, this is just text. Agents can append
    structured observations ("user prefers French coffee; kettle is in
    drawer 2") and a hook injects the most recent section into the
    system prompt on ``on_run_start``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def read(self) -> str:
        with self._lock:
            if not self.path.exists():
                return ""
            return self.path.read_text(encoding="utf-8")

    def append(self, text: str, *, heading: str | None = None) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                if heading:
                    f.write(f"\n## {heading} ({time.strftime('%Y-%m-%d %H:%M')})\n")
                f.write(text.rstrip() + "\n")

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()

    def tail(self, max_chars: int = 2000) -> str:
        """Return the last ``max_chars`` of notes — cheap prompt-injection."""
        content = self.read()
        return content if len(content) <= max_chars else content[-max_chars:]


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


__all__ = [
    "COMPACTION_SYSTEM_PROMPT",
    "Compactor",
    "Episode",
    "Fact",
    "JSONMemoryStore",
    "JSONSessionStore",
    "MemoryStore",
    "NotesFile",
    "Preference",
    "SessionStore",
    "default_memory_dir",
    "estimate_tokens",
    "new_session_id",
]
