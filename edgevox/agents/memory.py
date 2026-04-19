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
import sqlite3
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

    **Bi-temporal schema** (Zep / Graphiti pattern). Every fact carries
    two time axes:

    - ``valid_from`` / ``valid_to`` — the *world-state* interval the
      fact described as true. ``valid_to=None`` means "still true".
    - ``invalidated_at`` — the *system* timestamp at which we learned
      the fact stopped being true. Always ``None`` while the fact is
      still active.

    Facts are *append-only*: when ``add_fact`` overwrites a key, the
    prior fact gets ``valid_to`` and ``invalidated_at`` set to ``now``
    and the new fact is appended with ``supersedes`` pointing at the
    old one's ``id``. ``facts_as_of(t)`` then answers "what did I
    believe at time ``t``?" — useful for robotics worldmodel queries
    ("the mug *was* on the counter at 14:02 but is in drawer 2 now").

    Old JSON files predating this schema (no ``id`` / ``valid_*``
    fields) load cleanly: missing fields fall back to ``id="legacy_…"``,
    ``valid_from=updated_at``, ``valid_to=None``.
    """

    key: str
    value: str
    scope: str = "global"
    updated_at: float = field(default_factory=time.time)
    source: str = ""  # which agent or event created it
    # Bi-temporal extensions ------------------------------------------
    id: str = ""
    valid_from: float = 0.0
    valid_to: float | None = None
    invalidated_at: float | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        # Defaults only kick in when the caller (or the JSON loader)
        # didn't provide a value — keeps the dataclass shape stable
        # without forcing every test to set every field.
        if not self.id:
            self.id = f"f_{uuid.uuid4().hex[:10]}"
        if not self.valid_from:
            self.valid_from = self.updated_at or time.time()

    @property
    def is_active(self) -> bool:
        """True iff this fact is still believed to be the truth."""
        return self.valid_to is None and self.invalidated_at is None


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


# Set of accepted ``Fact`` field names — used by :class:`JSONMemoryStore`
# to drop legacy or extension-introduced keys instead of raising.
_FACT_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(Fact))


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
        # Bi-temporal storage: every fact ever observed lives here, in
        # insertion order. ``_active_index`` keeps O(1) lookup of the
        # current believed-true value per ``(scope, key)``.
        self._facts: list[Fact] = []
        self._active_index: dict[tuple[str, str], int] = {}
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
                # Tolerate legacy schemas: drop fields the current Fact
                # dataclass doesn't recognise rather than failing the
                # whole load.
                fields = {k: v for k, v in raw.items() if k in _FACT_FIELDS}
                try:
                    f = Fact(**fields)
                except TypeError:
                    continue
                self._facts.append(f)
                if f.is_active:
                    # Last-active wins; pre-bi-temporal files only had
                    # one fact per (scope, key) so this is a no-op there.
                    self._active_index[(f.scope, f.key)] = len(self._facts) - 1
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
                "facts": [asdict(f) for f in self._facts],
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
        """Append-only fact write.

        If an active fact already exists for ``(scope, key)`` and its
        value differs, the prior fact is invalidated (``valid_to`` and
        ``invalidated_at`` set to ``now``) and a new fact is appended
        with ``supersedes`` pointing at the prior id. Identical-value
        re-writes are no-ops so callers can safely re-publish.
        """
        with self._lock:
            now = time.time()
            prior_idx = self._active_index.get((scope, key))
            prior: Fact | None = self._facts[prior_idx] if prior_idx is not None else None
            if prior is not None and prior.value == value:
                # No-op refresh: bump ``updated_at`` so callers can tell
                # we re-affirmed the fact, but keep the id and lineage.
                prior.updated_at = now
                self._mark_dirty()
                return
            if prior is not None:
                prior.valid_to = now
                prior.invalidated_at = now
            new_fact = Fact(
                key=key,
                value=value,
                scope=scope,
                source=source,
                updated_at=now,
                valid_from=now,
                supersedes=prior.id if prior is not None else None,
            )
            self._facts.append(new_fact)
            self._active_index[(scope, key)] = len(self._facts) - 1
            self._mark_dirty()

    def get_fact(self, key: str, *, scope: str = "global") -> str | None:
        """Return the *currently active* value for ``(scope, key)``."""
        with self._lock:
            idx = self._active_index.get((scope, key))
            if idx is None:
                return None
            return self._facts[idx].value

    def facts(self, *, scope: str | None = None) -> list[Fact]:
        """Return active (currently-believed) facts, optionally scoped.

        Use :meth:`fact_history` to see superseded values, or
        :meth:`facts_as_of` for time-travel queries.
        """
        with self._lock:
            active = (self._facts[i] for i in self._active_index.values())
            if scope is None:
                return list(active)
            return [f for f in active if f.scope == scope]

    def fact_history(self, key: str, *, scope: str = "global") -> list[Fact]:
        """All facts ever written for ``(scope, key)``, oldest first.
        Includes the currently-active one (last entry) and all
        superseded predecessors.
        """
        with self._lock:
            return [f for f in self._facts if f.scope == scope and f.key == key]

    def facts_as_of(self, t: float, *, scope: str | None = None) -> list[Fact]:
        """Return what the agent believed to be true at world-time ``t``.

        Bi-temporal query: a fact is "believed at ``t``" iff its
        ``valid_from <= t`` and (``valid_to is None`` or ``valid_to >
        t``). For each ``(scope, key)`` we return the single fact that
        was active at ``t``, picking the most-recent one when multiple
        rewrites happened in the same instant.
        """
        with self._lock:
            picked: dict[tuple[str, str], Fact] = {}
            for f in self._facts:
                if scope is not None and f.scope != scope:
                    continue
                if f.valid_from > t:
                    continue
                if f.valid_to is not None and f.valid_to <= t:
                    continue
                key = (f.scope, f.key)
                # Later-appended wins on ties — matches add_fact's
                # invalidate-then-append ordering.
                picked[key] = f
            return list(picked.values())

    def forget_fact(self, key: str, *, scope: str = "global") -> bool:
        """Mark the active fact for ``(scope, key)`` as no-longer-valid.

        Unlike a true delete, the fact remains in :meth:`fact_history`
        and is still returned by :meth:`facts_as_of` for any ``t`` in
        its valid interval. Returns ``True`` if a fact was active to
        forget, ``False`` otherwise.
        """
        with self._lock:
            idx = self._active_index.pop((scope, key), None)
            if idx is None:
                return False
            now = time.time()
            self._facts[idx].valid_to = now
            self._facts[idx].invalidated_at = now
            self._mark_dirty()
            return True

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

            # Render only currently-active facts. ``self._facts`` is the
            # full append-only history; the active subset lives at the
            # indices in ``_active_index``.
            active_facts = [self._facts[i] for i in self._active_index.values()]
            if active_facts:
                lines.append("## Known facts")
                for rendered, f in enumerate(active_facts):
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
# SQLite-backed MemoryStore
# ---------------------------------------------------------------------------


class SQLiteMemoryStore:
    """Durable, crash-safe :class:`MemoryStore` backed by stdlib ``sqlite3``.

    Same bi-temporal semantics as :class:`JSONMemoryStore` — facts are
    append-only, overwrites set ``valid_to`` + ``invalidated_at`` on the
    prior row and insert a new row with ``supersedes`` pointing back —
    but writes land on disk immediately via WAL-mode atomic
    transactions. Preferred when:

    * Multiple processes may touch the same store (WAL mode allows
      concurrent readers alongside a single writer).
    * A process crash / kill-9 between debounced JSON flushes would
      lose facts the agent had already committed in-memory.
    * The fact / episode set is large enough that re-reading the whole
      JSON file on open starts to sting.

    Schema (auto-created on first open)::

        facts(id PK, key, value, scope, source, updated_at,
              valid_from, valid_to, invalidated_at, supersedes)
        preferences(key PK, value, updated_at)
        episodes(rowid PK, kind, payload_json, outcome, timestamp, agent)

    Thread-safety: one connection per :class:`SQLiteMemoryStore`
    instance, guarded by an internal :class:`threading.RLock`.
    ``check_same_thread=False`` is passed on the connection so tool
    threads and the main thread can share one store.
    """

    _flush_interval = 0.0  # SQLite writes are immediate; no debounce.
    _max_episodes = 500  # trimmed on insert overflow, matches JSON store.

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS facts (
            id             TEXT PRIMARY KEY,
            key            TEXT NOT NULL,
            value          TEXT NOT NULL,
            scope          TEXT NOT NULL,
            source         TEXT NOT NULL DEFAULT '',
            updated_at     REAL NOT NULL,
            valid_from     REAL NOT NULL,
            valid_to       REAL,
            invalidated_at REAL,
            supersedes     TEXT
        );
        """,
        "CREATE INDEX IF NOT EXISTS facts_scope_key ON facts(scope, key);",
        "CREATE INDEX IF NOT EXISTS facts_active ON facts(scope, key) WHERE valid_to IS NULL;",
        """
        CREATE TABLE IF NOT EXISTS preferences (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            outcome      TEXT NOT NULL,
            timestamp    REAL NOT NULL,
            agent        TEXT NOT NULL DEFAULT ''
        );
        """,
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly.
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        for stmt in self._SCHEMA:
            self._conn.execute(stmt)

    # ----- facts -----

    def _row_to_fact(self, row: Any) -> Fact:
        return Fact(
            key=row["key"],
            value=row["value"],
            scope=row["scope"],
            source=row["source"] or "",
            updated_at=row["updated_at"],
            id=row["id"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            invalidated_at=row["invalidated_at"],
            supersedes=row["supersedes"],
        )

    def add_fact(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        source: str = "",
    ) -> None:
        """Append-only fact write; see :meth:`JSONMemoryStore.add_fact`."""
        with self._lock:
            now = time.time()
            cur = self._conn.execute(
                "SELECT id, value FROM facts WHERE scope=? AND key=? AND valid_to IS NULL LIMIT 1",
                (scope, key),
            )
            prior = cur.fetchone()
            if prior is not None and prior["value"] == value:
                self._conn.execute("UPDATE facts SET updated_at=? WHERE id=?", (now, prior["id"]))
                return
            new_id = f"f_{uuid.uuid4().hex[:10]}"
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                if prior is not None:
                    self._conn.execute(
                        "UPDATE facts SET valid_to=?, invalidated_at=? WHERE id=?",
                        (now, now, prior["id"]),
                    )
                self._conn.execute(
                    """
                    INSERT INTO facts (id, key, value, scope, source, updated_at,
                                       valid_from, valid_to, invalidated_at, supersedes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                    """,
                    (
                        new_id,
                        key,
                        value,
                        scope,
                        source,
                        now,
                        now,
                        prior["id"] if prior is not None else None,
                    ),
                )
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    def get_fact(self, key: str, *, scope: str = "global") -> str | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT value FROM facts WHERE scope=? AND key=? AND valid_to IS NULL LIMIT 1",
                (scope, key),
            )
            row = cur.fetchone()
            return row["value"] if row is not None else None

    def facts(self, *, scope: str | None = None) -> list[Fact]:
        with self._lock:
            if scope is None:
                cur = self._conn.execute("SELECT * FROM facts WHERE valid_to IS NULL ORDER BY updated_at")
            else:
                cur = self._conn.execute(
                    "SELECT * FROM facts WHERE valid_to IS NULL AND scope=? ORDER BY updated_at",
                    (scope,),
                )
            return [self._row_to_fact(r) for r in cur.fetchall()]

    def fact_history(self, key: str, *, scope: str = "global") -> list[Fact]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM facts WHERE scope=? AND key=? ORDER BY valid_from",
                (scope, key),
            )
            return [self._row_to_fact(r) for r in cur.fetchall()]

    def facts_as_of(self, t: float, *, scope: str | None = None) -> list[Fact]:
        """Return what the agent believed to be true at world-time ``t``."""
        with self._lock:
            if scope is None:
                cur = self._conn.execute(
                    """
                    SELECT * FROM facts
                    WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                    ORDER BY valid_from
                    """,
                    (t, t),
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT * FROM facts
                    WHERE scope=? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                    ORDER BY valid_from
                    """,
                    (scope, t, t),
                )
            # Dedup per (scope, key) keeping the last-inserted (matches
            # JSON store's "later-appended wins on ties" rule).
            picked: dict[tuple[str, str], Fact] = {}
            for r in cur.fetchall():
                picked[(r["scope"], r["key"])] = self._row_to_fact(r)
            return list(picked.values())

    def forget_fact(self, key: str, *, scope: str = "global") -> bool:
        with self._lock:
            now = time.time()
            cur = self._conn.execute(
                "SELECT id FROM facts WHERE scope=? AND key=? AND valid_to IS NULL LIMIT 1",
                (scope, key),
            )
            row = cur.fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE facts SET valid_to=?, invalidated_at=? WHERE id=?",
                (now, now, row["id"]),
            )
            return True

    # ----- preferences -----

    def set_preference(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, time.time()),
            )

    def preferences(self) -> list[Preference]:
        with self._lock:
            cur = self._conn.execute("SELECT key, value, updated_at FROM preferences ORDER BY key")
            return [Preference(key=r["key"], value=r["value"], updated_at=r["updated_at"]) for r in cur.fetchall()]

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
            self._conn.execute(
                "INSERT INTO episodes (kind, payload_json, outcome, timestamp, agent) VALUES (?, ?, ?, ?, ?)",
                (kind, json.dumps(payload, default=str), outcome, time.time(), agent),
            )
            # Trim to the last ``_max_episodes`` — same bounded-buffer
            # behaviour as the JSON store, at the cost of one DELETE per
            # overflow. The ``rowid > (max_rowid - N)`` form reads as a
            # no-op when there are fewer than N rows.
            self._conn.execute(
                """
                DELETE FROM episodes
                WHERE id <= (SELECT MAX(id) FROM episodes) - ?
                """,
                (self._max_episodes,),
            )

    def recent_episodes(self, n: int = 5, *, kind: str | None = None) -> list[Episode]:
        with self._lock:
            if kind is None:
                cur = self._conn.execute("SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (n,))
            else:
                cur = self._conn.execute(
                    "SELECT * FROM episodes WHERE kind=? ORDER BY id DESC LIMIT ?",
                    (kind, n),
                )
            rows = cur.fetchall()
            out: list[Episode] = []
            for r in reversed(rows):  # oldest-first matches JSON store
                try:
                    payload = json.loads(r["payload_json"])
                except json.JSONDecodeError:
                    payload = {"_malformed_json": r["payload_json"]}
                out.append(
                    Episode(
                        kind=r["kind"],
                        payload=payload,
                        outcome=r["outcome"],
                        timestamp=r["timestamp"],
                        agent=r["agent"] or "",
                    )
                )
            return out

    # ----- rendering / lifecycle -----

    def flush(self) -> None:
        """No-op — SQLite autocommit means writes are already durable."""
        return

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None  # type: ignore[assignment]

    def render_for_prompt(self, *, max_facts: int = 20, max_episodes: int = 5) -> str:
        """Render memory as a concise markdown block for the system prompt.

        Same layout as :meth:`JSONMemoryStore.render_for_prompt` so
        swapping stores doesn't change what the LLM sees.
        """
        with self._lock:
            lines: list[str] = []

            prefs = self.preferences()
            if prefs:
                lines.append("## Known preferences")
                for p in prefs[:max_facts]:
                    lines.append(f"- {p.key}: {p.value}")

            active_facts = self.facts()
            if active_facts:
                lines.append("## Known facts")
                for rendered, f in enumerate(active_facts):
                    if rendered >= max_facts:
                        break
                    scope_tag = "" if f.scope == "global" else f" [{f.scope}]"
                    lines.append(f"- {f.key}{scope_tag}: {f.value}")

            episodes = self.recent_episodes(n=max_episodes)
            if episodes:
                lines.append("## Recent outcomes")
                for e in episodes:
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


class SQLiteSessionStore:
    """SQLite-backed :class:`SessionStore` using stdlib ``sqlite3``.

    Why SQLite over JSON-per-file:

    * **Atomic writes.** ``UPDATE`` is transactional; the JSON store
      uses ``tmp.write + replace`` which is atomic at the filesystem
      layer but doesn't guard against partial reads on concurrent
      access.
    * **Crash-safe.** WAL journal mode survives mid-write SIGKILL
      without corrupting the database.
    * **Queryable.** ``list_ids`` is O(log n) instead of a directory
      scan; future ``audit_log`` tables (PR-13+) can join cleanly.
    * **Single file.** One file under the data dir for many sessions —
      easier to back up + ship across machines.

    Schema (created on first write):

    .. code-block:: sql

        CREATE TABLE sessions (
          session_id TEXT PRIMARY KEY,
          messages TEXT NOT NULL,            -- JSON array
          state TEXT NOT NULL,               -- JSON object
          updated_at REAL NOT NULL
        );

    Backwards-compatible drop-in for :class:`JSONSessionStore` — same
    Protocol surface, callers swap implementations without touching
    hooks. Pairs with :class:`PersistSessionHook`.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        messages TEXT NOT NULL,
        state TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC);
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` + a lock = thread-safe shared
        # connection. We do per-call short transactions so contention
        # is minimal even with multiple agents.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            # WAL = better concurrency + crash safety than rollback journal.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()

    def save(self, session_id: str, session: Session) -> None:
        payload_messages = json.dumps(session.messages, default=str)
        payload_state = json.dumps(_jsonable(session.state), default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, messages, state, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "messages=excluded.messages, state=excluded.state, updated_at=excluded.updated_at",
                (session_id, payload_messages, payload_state, time.time()),
            )
            self._conn.commit()

    def load(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT messages, state FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            messages = json.loads(row[0])
            state = json.loads(row[1])
        except json.JSONDecodeError:
            log.exception("Corrupt SQLite session row for %s", session_id)
            return None
        return Session(messages=list(messages), state=dict(state))

    def delete(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection. Optional — the
        connection is closed automatically on garbage collection."""
        with self._lock:
            self._conn.close()


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


def estimate_tokens(messages: Iterable[dict], llm: LLM | None = None) -> int:
    """Token estimate used for context-window decisions.

    When ``llm`` is supplied, each message's ``content`` is tokenised
    exactly via :meth:`LLM.count_tokens` (uses the loaded GGUF's
    tokenizer). Otherwise a ``chars // 4`` heuristic is used — good
    enough for rough thresholds but known-wrong for code (under-counts
    ~15-25%) and CJK/Vietnamese/Thai (under-counts heavily). Threading
    the real tokenizer through every site that calls this function is
    what keeps the Compactor + TokenBudgetHook from either triggering
    too early (heuristic over-counts ASCII) or too late (heuristic
    under-counts multilingual).
    """
    total = 0
    for m in messages:
        c = m.get("content") or ""
        if not isinstance(c, str):
            total += 4
            continue
        if llm is not None:
            try:
                total += llm.count_tokens(c) + 4
                continue
            except Exception:
                # Fall through to the heuristic on any tokenizer error.
                pass
        total += len(c) // 4 + 4
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
    # Selective compaction: when True, messages in the "compress"
    # window are split into (user/assistant turns → summarised) and
    # (tool_calls + tool results → condensed audit trail, kept as a
    # separate assistant message). Keeps the LLM aware of *what* was
    # tried in the middle of the conversation without re-inflating the
    # window with full tool payloads. Off by default (backwards
    # compat) — enable when tool chains are the dominant token cost.
    preserve_tool_calls: bool = False

    def should_compact(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Return True when compaction is warranted.

        Uses the supplied LLM's tokenizer when available so the
        threshold is counted in real tokens rather than ``chars // 4``.
        """
        if len(messages) < self.keep_last_turns + 2:
            return False
        return estimate_tokens(messages, llm) >= self.trigger_tokens

    def compact(self, messages: list[dict], llm: LLM | None) -> list[dict]:
        """Return a compacted copy of ``messages``.

        If ``llm`` is None (test / offline path), falls back to a
        deterministic truncation that keeps system + last N turns.
        """
        if not self.should_compact(messages, llm):
            return list(messages)

        system = messages[0] if messages and messages[0].get("role") == "system" else None
        body = messages[1:] if system is not None else list(messages)
        if len(body) <= self.keep_last_turns:
            return list(messages)

        to_compress = body[: -self.keep_last_turns]
        keep = body[-self.keep_last_turns :]

        out: list[dict] = []
        if system is not None:
            out.append(system)

        if self.preserve_tool_calls:
            # Split ``to_compress`` into conversational turns and tool
            # chains, summarise the former, render the latter as a
            # terse audit trail. The audit trail keeps the model aware
            # of what it's already tried (so it doesn't repeat) without
            # re-inflating the context with raw tool payloads.
            conv_only, tool_trace = _split_tool_chain(to_compress)
            if conv_only:
                conv_summary = self._summarize(conv_only, llm)
                out.append(
                    {
                        "role": "assistant",
                        "content": f"(summary of earlier conversation)\n{conv_summary}",
                    }
                )
            if tool_trace:
                out.append(
                    {
                        "role": "assistant",
                        "content": "(tool-call trace from earlier this session)\n" + tool_trace,
                    }
                )
        else:
            summary = self._summarize(to_compress, llm)
            out.append(
                {
                    "role": "assistant",
                    "content": f"(summary of earlier conversation)\n{summary}",
                }
            )

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


def _split_tool_chain(messages: list[dict]) -> tuple[list[dict], str]:
    """Separate the compress-window into conversational turns + a
    condensed tool-trace line. Used by ``Compactor`` when
    ``preserve_tool_calls=True``.

    Returns ``(conv_only, tool_trace)``:

    * ``conv_only`` — user / assistant turns **without** ``tool_calls``
      attached, suitable for LLM summarisation.
    * ``tool_trace`` — multi-line human-readable audit of every
      ``(tool_name, args_preview, result_preview)`` triple, kept as a
      single assistant message so the model can read "I tried X, got
      Y, tried Z, got W" without re-ingesting full JSON blobs.
    """
    conv: list[dict] = []
    trace_lines: list[str] = []

    for m in messages:
        role = m.get("role")
        if role == "tool":
            # Collapse the just-seen tool-call names with their result.
            name = m.get("name") or ""
            content = m.get("content") or ""
            if isinstance(content, str):
                snippet = content[:120].replace("\n", " ")
                if len(content) > 120:
                    snippet += "…"
            else:
                snippet = str(content)[:120]
            trace_lines.append(f"  → {name}: {snippet}")
            continue
        if role == "assistant" and m.get("tool_calls"):
            # Record the call intent line; skip adding the assistant
            # message to ``conv`` since its content is usually empty
            # when a turn is pure tool-dispatch.
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name", "?")
                args_raw = fn.get("arguments", "")
                if isinstance(args_raw, str) and len(args_raw) > 60:
                    args_raw = args_raw[:60] + "…"
                trace_lines.append(f"- call {name}({args_raw})")
            content = m.get("content") or ""
            if isinstance(content, str) and content.strip():
                conv.append({"role": "assistant", "content": content})
            continue
        # Plain user / assistant message.
        conv.append(m)

    trace = "\n".join(trace_lines[:40])  # hard cap — 40 entries is plenty
    if len(trace_lines) > 40:
        trace += f"\n… (+{len(trace_lines) - 40} more tool events)"
    return conv, trace


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

    The file is soft-bounded by ``max_size_chars`` (default 64 KiB).
    When :meth:`append` would take it past the bound, the file is
    rewritten keeping the newest ``max_size_chars`` characters plus a
    single ``(earlier notes truncated)`` marker line. A long-running
    voice session that logs a note per turn can otherwise grow notes
    without bound — this keeps the on-disk + in-prompt size stable.
    """

    # Default: 64 KiB holds ~12k-16k tokens of notes, well beyond the
    # typical ``NotesInjectorHook.max_chars = 1500`` prompt budget.
    DEFAULT_MAX_SIZE_CHARS = 64 * 1024

    def __init__(self, path: str | Path, *, max_size_chars: int | None = None) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.max_size_chars = max_size_chars if max_size_chars is not None else self.DEFAULT_MAX_SIZE_CHARS

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
            self._prune_if_oversized()

    def _prune_if_oversized(self) -> None:
        """Rewrite the file keeping only the newest ``max_size_chars``.

        Called under ``self._lock`` after every append. No-op when the
        file is within budget or when ``max_size_chars`` is ``0``.
        """
        if self.max_size_chars <= 0:
            return
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        # Cheap file-size test first — avoids re-reading the file until
        # we actually need to prune.
        if size <= self.max_size_chars:
            return
        content = self.path.read_text(encoding="utf-8")
        if len(content) <= self.max_size_chars:
            return
        keep = content[-self.max_size_chars :]
        # Align to a line boundary so we don't split a heading.
        nl = keep.find("\n")
        if 0 <= nl < len(keep) - 1:
            keep = keep[nl + 1 :]
        self.path.write_text(f"(earlier notes truncated)\n{keep}", encoding="utf-8")

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
    "SQLiteSessionStore",
    "SessionStore",
    "default_memory_dir",
    "estimate_tokens",
    "new_session_id",
]
