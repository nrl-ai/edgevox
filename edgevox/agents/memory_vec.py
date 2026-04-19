"""Vector-retrieval ``MemoryStore`` backed by ``sqlite-vec``.

Stores the same bi-temporal ``Fact`` / ``Preference`` / ``Episode``
objects as :class:`JSONMemoryStore` and :class:`SQLiteMemoryStore`,
plus a sidecar vec0 virtual table that holds an embedding per fact.
``search_facts(query, k=5)`` returns the top-k semantically-similar
active facts.

Design notes:

* **Embedding is user-provided.** Construct with ``embed_fn=...``,
  a callable ``(text) -> np.ndarray[float32]`` of fixed dimension.
  EdgeVox provides :func:`llama_embed` which returns a callable
  wrapping llama-cpp-python's built-in embedding mode — requires
  the ``LLM`` to have been loaded with ``embedding=True``.
* **``MemoryStore`` Protocol conformance.** All the write / read /
  render methods match :class:`SQLiteMemoryStore` so swapping is a
  drop-in upgrade — the only extension is :meth:`search_facts`.
* **Optional dep.** ``sqlite-vec`` is shipped via the
  ``[memory-vec]`` extra. Importing this module is cheap; the dep
  loads lazily in ``__init__``.
* **Model fallback.** ``llama_embed`` honours the project's
  ``nrl-ai/edgevox-models`` → upstream fallback chain via the
  existing ``LLM`` loader, so embedding models travel the same
  mirror path as STT/TTS assets.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from edgevox.agents.memory import Episode, Fact, Preference

if TYPE_CHECKING:
    from edgevox.llm.llamacpp import LLM


log = logging.getLogger(__name__)

EmbedFn = Callable[[str], "np.ndarray"]


class VectorMemoryStore:
    """Vector-retrieval ``MemoryStore`` backed by ``sqlite-vec``.

    Every fact gets an embedding row keyed by its ``id``; overwrites
    invalidate the prior fact (same bi-temporal semantics as
    :class:`SQLiteMemoryStore`) and ``forget_fact`` drops the
    embedding row so the retired fact no longer surfaces in
    similarity search. Preferences and episodes are stored
    unembedded — they're rendered whole, not retrieved.

    Example::

        from edgevox.llm.llamacpp import LLM
        from edgevox.agents.memory_vec import VectorMemoryStore, llama_embed

        llm = LLM(embedding=True)
        store = VectorMemoryStore("./vec.db", embed_fn=llama_embed(llm))
        store.add_fact("user.allergies", "peanuts, shellfish")
        store.add_fact("kitchen.fridge.contents", "milk, eggs, cheese")
        hits = store.search_facts("what's safe to cook?", k=3)
        for fact, score in hits:
            print(score, fact.key, fact.value)
    """

    _max_episodes = 500

    def __init__(
        self,
        path: str | Path,
        *,
        embed_fn: EmbedFn,
        embedding_dim: int | None = None,
    ) -> None:
        try:
            import sqlite_vec
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "VectorMemoryStore requires the ``memory-vec`` extra. Install with: pip install 'edgevox[memory-vec]'"
            ) from e

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embed_fn = embed_fn

        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        # Load the sqlite-vec extension. The sqlite_vec package
        # provides a ``load`` helper that calls ``enable_load_extension``
        # + ``load_extension`` against the bundled ``.so`` / ``.dylib``.
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._dim = embedding_dim or self._probe_dim()
        self._create_schema()

    # -- setup -----------------------------------------------------------

    def _probe_dim(self) -> int:
        """Call ``embed_fn`` once to learn the vector dimension."""
        sample = self._embed_fn("dimension probe")
        arr = np.asarray(sample, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("embed_fn returned an empty vector")
        return int(arr.size)

    def _create_schema(self) -> None:
        self._conn.execute(
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
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS facts_scope_key ON facts(scope, key);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS facts_active ON facts(scope, key) WHERE valid_to IS NULL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                kind         TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                timestamp    REAL NOT NULL,
                agent        TEXT NOT NULL DEFAULT ''
            );
            """
        )
        # vec0 virtual table for fact embeddings. Rowid is synthesised
        # from the fact id via ``_vec_rowid`` (stable hash) so the
        # embedding row can be deleted without scanning the table.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(embedding float[{self._dim}]);"
        )
        # Auxiliary map so we can delete / update embeddings by fact id.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_vec_map (
                fact_id TEXT PRIMARY KEY,
                rowid   INTEGER NOT NULL
            );
            """
        )

    # -- helpers ---------------------------------------------------------

    def _embed(self, text: str) -> bytes:
        vec = np.asarray(self._embed_fn(text), dtype=np.float32).reshape(-1)
        if vec.size != self._dim:
            raise ValueError(f"embed_fn returned dim={vec.size}, expected {self._dim} (set at __init__ time)")
        return vec.tobytes()

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

    # -- facts -----------------------------------------------------------

    def add_fact(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        source: str = "",
    ) -> None:
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
            embedding_blob = self._embed(f"{key}: {value}")
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                if prior is not None:
                    self._conn.execute(
                        "UPDATE facts SET valid_to=?, invalidated_at=? WHERE id=?",
                        (now, now, prior["id"]),
                    )
                    # Retire the old embedding.
                    self._drop_embedding(prior["id"])
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
                cur = self._conn.execute(
                    "INSERT INTO fact_embeddings(embedding) VALUES (?)",
                    (embedding_blob,),
                )
                rowid = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO fact_vec_map(fact_id, rowid) VALUES (?, ?)",
                    (new_id, rowid),
                )
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    def _drop_embedding(self, fact_id: str) -> None:
        row = self._conn.execute("SELECT rowid FROM fact_vec_map WHERE fact_id=?", (fact_id,)).fetchone()
        if row is None:
            return
        self._conn.execute("DELETE FROM fact_embeddings WHERE rowid=?", (row["rowid"],))
        self._conn.execute("DELETE FROM fact_vec_map WHERE fact_id=?", (fact_id,))

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
            fact_id = row["id"]
            self._conn.execute(
                "UPDATE facts SET valid_to=?, invalidated_at=? WHERE id=?",
                (now, now, fact_id),
            )
            # Retired facts don't surface in semantic search.
            self._drop_embedding(fact_id)
            return True

    # -- semantic search -------------------------------------------------

    def search_facts(
        self,
        query: str,
        *,
        k: int = 5,
        scope: str | None = None,
    ) -> list[tuple[Fact, float]]:
        """Return the top-``k`` active facts semantically closest to ``query``.

        Output is a list of ``(Fact, distance)`` pairs sorted by
        ascending distance (smaller = more similar, per sqlite-vec's
        L2 metric). Optionally restrict to a single ``scope``.
        """
        with self._lock:
            if k <= 0:
                return []
            qv = self._embed(query)
            # sqlite-vec's MATCH returns rowid + distance; we join
            # back to ``fact_vec_map`` → ``facts`` and filter to the
            # still-active rows (valid_to IS NULL).
            if scope is None:
                cur = self._conn.execute(
                    """
                    SELECT f.*, e.distance AS distance
                    FROM (
                        SELECT rowid, distance FROM fact_embeddings
                        WHERE embedding MATCH ? ORDER BY distance LIMIT ?
                    ) e
                    JOIN fact_vec_map m ON m.rowid = e.rowid
                    JOIN facts f ON f.id = m.fact_id
                    WHERE f.valid_to IS NULL
                    ORDER BY e.distance
                    """,
                    (qv, max(k * 4, k)),  # over-fetch to survive retired-row filtering
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT f.*, e.distance AS distance
                    FROM (
                        SELECT rowid, distance FROM fact_embeddings
                        WHERE embedding MATCH ? ORDER BY distance LIMIT ?
                    ) e
                    JOIN fact_vec_map m ON m.rowid = e.rowid
                    JOIN facts f ON f.id = m.fact_id
                    WHERE f.valid_to IS NULL AND f.scope = ?
                    ORDER BY e.distance
                    """,
                    (qv, max(k * 4, k), scope),
                )
            out: list[tuple[Fact, float]] = []
            for row in cur.fetchall():
                out.append((self._row_to_fact(row), float(row["distance"])))
                if len(out) >= k:
                    break
            return out

    # -- preferences -----------------------------------------------------

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

    # -- episodes --------------------------------------------------------

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
            self._conn.execute(
                "DELETE FROM episodes WHERE id <= (SELECT MAX(id) FROM episodes) - ?",
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
            out: list[Episode] = []
            for r in reversed(cur.fetchall()):
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

    # -- lifecycle -------------------------------------------------------

    def flush(self) -> None:
        return

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None  # type: ignore[assignment]

    def render_for_prompt(self, *, max_facts: int = 20, max_episodes: int = 5) -> str:
        with self._lock:
            lines: list[str] = []

            prefs = self.preferences()
            if prefs:
                lines.append("## Known preferences")
                for p in prefs[:max_facts]:
                    lines.append(f"- {p.key}: {p.value}")

            active = self.facts()
            if active:
                lines.append("## Known facts")
                for rendered, f in enumerate(active):
                    if rendered >= max_facts:
                        break
                    scope_tag = "" if f.scope == "global" else f" [{f.scope}]"
                    lines.append(f"- {f.key}{scope_tag}: {f.value}")

            eps = self.recent_episodes(n=max_episodes)
            if eps:
                lines.append("## Recent outcomes")
                for e in eps:
                    brief = ", ".join(f"{k}={v}" for k, v in list(e.payload.items())[:3])
                    lines.append(f"- [{e.kind}] {brief} → {e.outcome}")

            return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def llama_embed(llm: LLM) -> EmbedFn:
    """Wrap an ``LLM`` loaded with ``embedding=True`` as an :class:`EmbedFn`.

    Usage::

        llm = LLM(model_path="...", embedding=True)
        store = VectorMemoryStore("./vec.db", embed_fn=llama_embed(llm))

    The returned callable takes a string and returns a numpy float32
    array. It's thread-safe to call from multiple agents as long as the
    underlying ``LLM`` instance is (which it is by default).

    If you need a different embedding model than your main LLM, load a
    separate ``LLM`` with a dedicated embedding-optimised GGUF (e.g.
    ``nomic-embed-text-v1.5.Q4_K_M.gguf``) and pass that handle
    instead.
    """

    inner = getattr(llm, "_llm", None) or llm  # unwrap our LLM facade
    if not hasattr(inner, "embed"):
        raise TypeError(
            f"llama_embed requires a llama-cpp ``Llama`` instance loaded with "
            f"embedding=True; got {type(inner).__name__!r}"
        )

    def _embed(text: str) -> np.ndarray:
        out = inner.embed(text)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr

    return _embed


__all__ = ["EmbedFn", "VectorMemoryStore", "llama_embed"]
