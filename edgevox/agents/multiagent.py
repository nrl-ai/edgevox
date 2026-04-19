"""Multi-agent coordination primitives.

Four orthogonal pieces, all optional — use one or all:

- :class:`Blackboard` — thread-safe shared key/value store with
  watchers. Agents coordinate through it without direct coupling.
- :class:`AgentMessage` + bus helpers — direct agent-to-agent
  messaging on top of the existing :class:`EventBus`.
- :class:`BackgroundAgent` — wraps any :class:`Agent` in a background
  thread that triggers when a bus event or blackboard change matches
  a user-supplied predicate.
- :class:`AgentPool` — owns a set of agents, starts/stops them, and
  manages their shared context.

These compose with the existing ``Sequence`` / ``Fallback`` / ``Loop`` /
``Router`` / ``Parallel`` workflows in :mod:`edgevox.agents.workflow`.
The workflows are for explicit orchestration; the primitives here are
for emergent, event-driven coordination (a supervisor agent watching
sensor data, a background planner reacting to user utterances, etc.).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from edgevox.agents.base import AgentContext
from edgevox.agents.bus import EventBus

if TYPE_CHECKING:
    from edgevox.agents.base import Agent, AgentEvent, AgentResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


BlackboardHandler = Callable[[str, Any, Any], None]  # (key, old, new)


class Blackboard:
    """Thread-safe shared state across agents.

    Supports watchers that fire when a key changes. Intentionally dumb
    (dict + watchers) — if you need transactions or CAS semantics build
    on top.

    Example::

        bb = Blackboard()
        bb.set("last_sensor_reading", 37.5)
        bb.watch("task_queue", lambda k, old, new: print(f"task queue changed: {new}"))
    """

    _UNSET = object()

    def __init__(self, *, async_watchers: bool = False, max_watcher_workers: int = 4) -> None:
        """Thread-safe shared key/value store.

        When ``async_watchers=True`` watcher callbacks run on a private
        ``ThreadPoolExecutor`` (size ``max_watcher_workers``) instead of
        the writer thread. A slow observer then can't stall ``set()`` /
        ``delete()`` for other agents. The trade-off: watchers observe
        writes in scheduling order but not strict commit order under
        contention — if you need that, stay on the default synchronous
        fan-out. The executor is created lazily on first async dispatch
        and released via :meth:`close`.
        """
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._watchers: dict[str, list[BlackboardHandler]] = {}
        self._wild_watchers: list[BlackboardHandler] = []
        self._async_watchers = bool(async_watchers)
        self._max_watcher_workers = int(max_watcher_workers)
        self._watcher_pool: ThreadPoolExecutor | None = None

    # -- internal ---------------------------------------------------------

    def _dispatch(self, watchers: list[BlackboardHandler], key: str, old: Any, new: Any) -> None:
        """Fan watchers out synchronously or via the lazy pool."""
        if not watchers:
            return
        if not self._async_watchers:
            for w in watchers:
                try:
                    w(key, old, new)
                except Exception:
                    log.exception("Blackboard watcher raised on %s", key)
            return
        pool = self._ensure_pool()
        for w in watchers:
            pool.submit(self._invoke_safe, w, key, old, new)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        # Lazy: users of the synchronous Blackboard never pay for a
        # thread pool, and the test suite's ``Blackboard()`` doesn't
        # have to ``close()`` anything.
        with self._lock:
            if self._watcher_pool is None:
                self._watcher_pool = ThreadPoolExecutor(
                    max_workers=self._max_watcher_workers,
                    thread_name_prefix="bb-watcher",
                )
            return self._watcher_pool

    @staticmethod
    def _invoke_safe(handler: BlackboardHandler, key: str, old: Any, new: Any) -> None:
        try:
            handler(key, old, new)
        except Exception:
            log.exception("Blackboard async watcher raised on %s", key)

    # -- public API -------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            old = self._data.get(key, self._UNSET)
            self._data[key] = value
            watchers = list(self._watchers.get(key, ())) + list(self._wild_watchers)
        old_visible = None if old is self._UNSET else old
        self._dispatch(watchers, key, old_visible, value)

    def update(self, mapping: dict[str, Any]) -> None:
        for k, v in mapping.items():
            self.set(k, v)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            old = self._data.pop(key)
            watchers = list(self._watchers.get(key, ())) + list(self._wild_watchers)
        self._dispatch(watchers, key, old, None)
        return True

    def close(self) -> None:
        """Shut down the async-watcher pool if one was started.

        Synchronous blackboards never start a pool and ``close()`` is a
        cheap no-op; async blackboards should call this on shutdown to
        avoid leaking threads.
        """
        with self._lock:
            pool, self._watcher_pool = self._watcher_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def watch(self, key: str, handler: BlackboardHandler) -> Callable[[], None]:
        """Fire ``handler(key, old, new)`` whenever ``key`` changes.

        Pass ``key="*"`` to observe all changes.
        """
        with self._lock:
            if key == "*":
                self._wild_watchers.append(handler)
            else:
                self._watchers.setdefault(key, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if key == "*":
                    if handler in self._wild_watchers:
                        self._wild_watchers.remove(handler)
                else:
                    if handler in self._watchers.get(key, []):
                        self._watchers[key].remove(handler)

        return unsubscribe

    # -- request/reply pattern -------------------------------------------

    def post_request(
        self,
        request_key: str,
        task: Any,
        *,
        reply_key: str | None = None,
        timeout: float | None = None,
    ) -> Future[Any]:
        """Post a task on ``request_key`` and return a Future for the reply.

        Subordinate agents watching ``request_key`` self-select based on
        their capability, run the work, and call :meth:`reply_to` (or
        ``set(reply_key, result)`` directly). The returned Future
        resolves the moment a reply lands at ``reply_key``; no busy
        polling. If ``timeout`` elapses first the Future is completed
        with a ``TimeoutError``.

        This is a thin convenience wrapper on top of ``set`` + ``watch``
        — it doesn't introduce a new coordination primitive, it just
        removes the boilerplate of wiring the reply path yourself.

        Example::

            fut = bb.post_request("plan.request", {"goal": "pick cup"}, timeout=5.0)
            plan = fut.result()  # blocks up to 5 s

        ``reply_key`` defaults to ``<request_key>.reply.<uuid>`` so
        concurrent requests on the same ``request_key`` don't collide.
        """
        if reply_key is None:
            reply_key = f"{request_key}.reply.{uuid.uuid4().hex[:8]}"
        fut: Future[Any] = Future()

        def _on_reply(_k: str, _old: Any, new: Any) -> None:
            if not fut.done():
                fut.set_result(new)

        unsub = self.watch(reply_key, _on_reply)
        # Hook the future's state so callers can still do ``fut.cancel()``
        # without leaking a watcher registration.
        fut.add_done_callback(lambda _f: unsub())

        if timeout is not None:

            def _timeout() -> None:
                if not fut.done():
                    fut.set_exception(TimeoutError(f"post_request({request_key!r}) timed out after {timeout:.3f}s"))

            timer = threading.Timer(timeout, _timeout)
            timer.daemon = True
            timer.start()
            fut.add_done_callback(lambda _f: timer.cancel())

        # Set the request LAST so watchers on ``request_key`` never see
        # the task without a live reply path — avoids a race where a
        # fast volunteer replies before we've installed the watch.
        self.set(request_key, {"task": task, "reply_key": reply_key})
        return fut

    def reply_to(self, request: Any, result: Any) -> None:
        """Deliver ``result`` to the reply path embedded in ``request``.

        Works with the dict shape produced by :meth:`post_request`
        (``{"task": ..., "reply_key": ...}``). Raises ``ValueError`` if
        ``request`` isn't a valid request envelope — callers that
        constructed their own envelope should just ``set(reply_key,
        result)`` directly.
        """
        if not isinstance(request, dict) or "reply_key" not in request:
            raise ValueError("reply_to expects a request envelope produced by post_request")
        self.set(request["reply_key"], result)


# ---------------------------------------------------------------------------
# Agent messaging over the bus
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """Direct message between agents.

    Published on :class:`EventBus` with ``kind="agent_message"``.
    Use ``to="*"`` for broadcast.
    """

    kind: str = "agent_message"
    agent_name: str = ""  # the sender; required by EventBus contract
    payload: Any = None
    # Message-specific fields (wrapped in payload for bus compatibility
    # but also mirrored on the dataclass for ergonomic access).
    to: str = ""
    content: Any = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {"to": self.to, "content": self.content, "correlation_id": self.correlation_id}


def send_message(
    bus: EventBus,
    *,
    from_agent: str,
    to: str,
    content: Any,
    correlation_id: str | None = None,
) -> AgentMessage:
    """Publish a message on the bus addressed to ``to``."""
    msg = AgentMessage(
        agent_name=from_agent,
        to=to,
        content=content,
        correlation_id=correlation_id or uuid.uuid4().hex[:8],
    )
    bus.publish(msg)
    return msg


def subscribe_inbox(
    bus: EventBus,
    *,
    agent_name: str,
    handler: Callable[[AgentMessage], None],
) -> Callable[[], None]:
    """Subscribe to messages addressed to ``agent_name`` (or broadcast).

    Returns an unsubscribe callable.
    """

    def _handler(event: Any) -> None:
        if not isinstance(event, AgentMessage):
            # Events routed via bus may arrive as AgentEvent with same kind.
            payload = getattr(event, "payload", None)
            to = payload.get("to") if isinstance(payload, dict) else None
        else:
            to = event.to
        if to in (agent_name, "*"):
            try:
                if isinstance(event, AgentMessage):
                    handler(event)
                else:
                    handler(
                        AgentMessage(
                            agent_name=getattr(event, "agent_name", ""),
                            to=to or "",
                            content=(event.payload or {}).get("content") if isinstance(event.payload, dict) else None,
                            correlation_id=(event.payload or {}).get("correlation_id", "")
                            if isinstance(event.payload, dict)
                            else "",
                        )
                    )
            except Exception:
                log.exception("Inbox handler raised")

    return bus.subscribe("agent_message", _handler)


# ---------------------------------------------------------------------------
# BackgroundAgent
# ---------------------------------------------------------------------------


# Trigger returns the task string to run with, or None to ignore.
Trigger = Callable[["AgentEvent"], "str | None"]


RestartPolicy = Literal["permanent", "transient", "temporary"]


class BackgroundAgent:
    """Runs an :class:`Agent` on a dedicated thread, triggered by bus events.

    The thread sits on a :class:`queue.Queue` that receives bus events;
    each event is passed to ``trigger`` which returns either a task
    string (run the agent with it) or ``None`` (ignore).

    A single in-flight run at a time: if a trigger fires while the agent
    is running, the event is queued and picked up after the current run
    finishes. Set ``max_queue`` to bound memory — overflow behaviour is
    controlled by ``overflow``: ``"drop_oldest"`` (FIFO eviction, the
    pre-PR-4 default) or ``"drop_new"`` (preserve older events).

    ``restart`` follows Erlang OTP's supervisor contract:

    - ``"permanent"`` — any exit restarts the loop. Use for daemons that
      must always be draining their queue (sensor monitor, safety watchdog).
    - ``"transient"`` (default) — restart only after a crash, stop on
      clean shutdown. Safe default for application-level agents.
    - ``"temporary"`` — no restart; one crash and the agent stays down.

    ``max_restarts`` per ``restart_window_s`` caps how aggressively the
    supervisor loops on a crashing agent — equivalent to OTP's
    ``MaxR / MaxT``. When the budget is exhausted the loop exits and
    the agent must be manually restarted.

    Stop via :meth:`stop` — waits for the current run to finish.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        trigger: Trigger,
        subscribe_to: str = "*",
        max_queue: int = 32,
        name: str | None = None,
        overflow: Literal["drop_oldest", "drop_new"] = "drop_oldest",
        restart: RestartPolicy = "transient",
        max_restarts: int = 5,
        restart_window_s: float = 60.0,
    ) -> None:
        self.agent = agent
        self.trigger = trigger
        self.subscribe_to = subscribe_to
        self.name = name or f"bg-{agent.name}"
        # queue.Queue collapses the old (list + lock + Event) triple into
        # a single correct primitive. put_nowait/get(timeout=) give us
        # the backpressure and wake semantics without the drop-oldest
        # race that the hand-rolled version had.
        self._queue: queue.Queue[AgentEvent] = queue.Queue(maxsize=max_queue)
        self._overflow = overflow
        self._max_queue = max_queue
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._unsubscribe: Callable[[], None] | None = None
        self._ctx: AgentContext | None = None
        self._bus: EventBus | None = None
        self._results: list[AgentResult] = []
        self._restart_policy: RestartPolicy = restart
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._restart_timestamps: deque[float] = deque()
        # Public telemetry — operators can inspect these without poking
        # at private state. ``dropped_events`` is the most common
        # real-world diagnostic ("why did my agent miss that alert?").
        self.dropped_events = 0
        self.restart_count = 0

    def start(self, ctx: AgentContext, bus: EventBus) -> BackgroundAgent:
        """Attach to ``bus`` and start the worker thread."""
        if self._thread is not None:
            raise RuntimeError(f"{self.name} already started")
        self._ctx = ctx
        self._bus = bus
        self._stop.clear()
        self._unsubscribe = bus.subscribe(self.subscribe_to, self._enqueue)
        self._spawn_loop()
        return self

    def _spawn_loop(self) -> None:
        t = threading.Thread(target=self._loop, name=self.name, daemon=True)
        t.start()
        self._thread = t

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        # Kick the queue so a waiting ``get`` returns promptly.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _enqueue(self, event: Any) -> None:
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped_events += 1
            if self._overflow == "drop_oldest":
                # Best-effort FIFO eviction; a racing put from another
                # thread may fill the slot before ours lands, but the
                # queue remains bounded either way.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass
            # drop_new: discard the incoming event; older ones kept.

    def _loop(self) -> None:
        assert self._ctx is not None
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if event is None or self._stop.is_set():
                return
            try:
                task = self.trigger(event)
            except Exception:
                log.exception("BackgroundAgent trigger %s raised", self.name)
                continue
            if not task:
                continue
            try:
                result = self.agent.run(task, self._ctx)
            except Exception:
                log.exception("BackgroundAgent %s run failed", self.name)
                if self._should_restart(crashed=True):
                    self.restart_count += 1
                    log.warning(
                        "BackgroundAgent %s restarting (%d/%d) under policy=%s",
                        self.name,
                        self.restart_count,
                        self._max_restarts,
                        self._restart_policy,
                    )
                    # Fall through to the next iteration — same thread
                    # keeps running. The OTP supervisor's "restart"
                    # semantics here are cooperative: we don't respawn
                    # the thread, just resume the loop.
                    continue
                log.error("BackgroundAgent %s: restart budget exhausted; stopping", self.name)
                return
            self._results.append(result)
        # Clean exit path: the loop was asked to stop. Only the
        # ``permanent`` policy would restart on a clean exit, but since
        # stop() set the flag we respect it regardless.

    def _should_restart(self, *, crashed: bool) -> bool:
        """Apply the OTP-style restart policy + rate limit."""
        if self._restart_policy == "temporary":
            return False
        if self._restart_policy == "transient" and not crashed:
            return False
        now = time.monotonic()
        window_start = now - self._restart_window_s
        while self._restart_timestamps and self._restart_timestamps[0] < window_start:
            self._restart_timestamps.popleft()
        if len(self._restart_timestamps) >= self._max_restarts:
            return False
        self._restart_timestamps.append(now)
        return True

    @property
    def results(self) -> list[AgentResult]:
        return list(self._results)


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------


class AgentPool:
    """A bundle of agents that share one bus + one blackboard.

    The pool owns the shared state; agents are registered by name and
    their :class:`AgentContext` is constructed with the shared bus and
    blackboard automatically. Background agents are started via
    :meth:`start_background`; foreground runs go through :meth:`run`.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        blackboard: Blackboard | None = None,
    ) -> None:
        self.bus = bus or EventBus()
        self.blackboard = blackboard or Blackboard()
        self._agents: dict[str, Agent] = {}
        self._background: dict[str, BackgroundAgent] = {}

    # ----- registration -----

    def register(self, agent: Agent) -> Agent:
        if agent.name in self._agents:
            log.warning("Agent %r already registered — overwriting.", agent.name)
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return list(self._agents.keys())

    # ----- context -----

    def make_ctx(self, **overrides: Any) -> AgentContext:
        """Build an :class:`AgentContext` wired to the shared bus and
        blackboard. Additional fields passed via kwargs override
        defaults."""
        kwargs: dict[str, Any] = {"bus": self.bus, "blackboard": self.blackboard}
        kwargs.update(overrides)
        return AgentContext(**kwargs)

    # ----- foreground runs -----

    def run(
        self,
        agent_name: str,
        task: str,
        *,
        ctx: AgentContext | None = None,
    ) -> AgentResult:
        agent = self._agents.get(agent_name)
        if agent is None:
            raise KeyError(f"unknown agent: {agent_name!r}")
        return agent.run(task, ctx or self.make_ctx())

    # ----- background lifecycle -----

    def start_background(
        self,
        agent_name: str,
        *,
        trigger: Trigger,
        subscribe_to: str = "*",
        ctx: AgentContext | None = None,
    ) -> BackgroundAgent:
        agent = self._agents.get(agent_name)
        if agent is None:
            raise KeyError(f"unknown agent: {agent_name!r}")
        bg = BackgroundAgent(agent, trigger=trigger, subscribe_to=subscribe_to)
        bg.start(ctx or self.make_ctx(), self.bus)
        self._background[agent_name] = bg
        return bg

    def stop_background(self, agent_name: str) -> None:
        bg = self._background.pop(agent_name, None)
        if bg is not None:
            bg.stop()

    def stop_all(self) -> None:
        for bg in list(self._background.values()):
            bg.stop()
        self._background.clear()

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def debounce_trigger(trigger: Trigger, *, interval_s: float) -> Trigger:
    """Wrap a :data:`Trigger` so it fires at most once per ``interval_s``.

    Useful when a BackgroundAgent would otherwise fire on every
    sub-second sensor tick.
    """
    state = {"last": 0.0, "lock": threading.Lock()}

    def wrapped(event: AgentEvent) -> str | None:
        task = trigger(event)
        if task is None:
            return None
        with state["lock"]:  # type: ignore[arg-type]
            now = time.monotonic()
            if now - state["last"] < interval_s:  # type: ignore[operator]
                return None
            state["last"] = now
        return task

    return wrapped


__all__ = [
    "AgentMessage",
    "AgentPool",
    "BackgroundAgent",
    "Blackboard",
    "BlackboardHandler",
    "Trigger",
    "debounce_trigger",
    "send_message",
    "subscribe_inbox",
]
