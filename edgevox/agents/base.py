"""Agent foundations: Session, AgentContext, AgentEvent, Handoff, LLMAgent.

Design points (see ``docs/documentation/agent-loop.md`` and ``docs/documentation/hooks.md``):

- ``Agent`` is a minimal Protocol so workflows can polymorphically drive
  any mix of LLM-backed agents and composite workflows.
- ``LLMAgent`` wraps an ``edgevox.llm.LLM`` and swaps its history, tools,
  and persona per run so multiple agents can share one GGUF (memory
  budget is the hard constraint on edge devices).
- Delegation follows the **OpenAI Agents SDK "handoff-as-return-value"**
  pattern: a synthetic handoff tool returns a ``Handoff`` sentinel, and
  the agent loop short-circuits, transferring control to the target
  agent without another LLM hop on the router. That's 2 LLM calls
  (router + specialist) instead of smolagents-style 3.
- ``AgentContext.stop`` is a ``threading.Event`` the pipeline can set to
  preempt in-flight skills; the agent loop checks it between hops.
- ``AgentContext.hooks`` is a :class:`HookRegistry` that fires at six
  points in the loop so pluggable behaviors (guardrails, memory,
  compaction, plan mode, loop detection) compose without patching core.
- ``AgentContext.blackboard`` / ``artifacts`` / ``memory`` /
  ``interrupt`` are all optional, independent plug-ins used when the
  caller opts into multi-agent coordination, handoff artifacts,
  persistent memory, or barge-in respectively.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from edgevox.agents.bus import EventBus
from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_TOOL,
    ON_RUN_END,
    ON_RUN_START,
    Hook,
    HookAction,
    HookRegistry,
    HookResult,
    ToolCallRequest,
    fire_chain,
)
from edgevox.agents.skills import GoalStatus, Skill
from edgevox.llm.grammars import GrammarCache
from edgevox.llm.llamacpp import _strip_thinking, get_system_prompt, parse_tool_calls_from_content
from edgevox.llm.tools import Tool, ToolCallResult, ToolRegistry
from edgevox.llm.tools import tool as tool_decorator

if TYPE_CHECKING:
    from edgevox.agents.artifacts import ArtifactStore
    from edgevox.agents.interrupt import InterruptController
    from edgevox.agents.memory import MemoryStore
    from edgevox.agents.multiagent import Blackboard
    from edgevox.agents.skills import GoalHandle
    from edgevox.llm.llamacpp import LLM

log = logging.getLogger(__name__)


# --------- Session / events / results ---------


@dataclass
class Session:
    """Per-run conversation state — messages and a scratchpad dict.

    Replaces raw ``LLM._history`` access so tests and workflows can
    inspect and mutate state without reaching into the LLM internals.
    """

    messages: list[dict] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear the conversation history and the scratchpad."""
        self.messages.clear()
        self.state.clear()


@dataclass
class AgentEvent:
    """Unified observability. The TUI chat log subscribes and renders
    each kind with its own styling.
    """

    kind: Literal[
        "agent_start",
        "agent_end",
        "tool_call",
        "skill_goal",
        "skill_feedback",
        "skill_cancelled",
        "handoff",
        "safety_preempt",
        "agent_message",
        "hook_end_turn",
        "hook_modify",
        "render_request",
    ]
    agent_name: str
    payload: Any = None


EventCallback = Callable[[AgentEvent], None]


@dataclass
class AgentContext:
    """Dependency-injection + runtime plumbing passed through a run.

    Required fields:

    - ``session`` / ``deps`` / ``bus`` — existing.

    Optional plug-ins (each independent; pass only what you use):

    - ``hooks`` — :class:`HookRegistry` firing at 6 fire points. Empty
      by default.
    - ``blackboard`` — shared key/value store across agents in a pool.
    - ``memory`` — long-term :class:`MemoryStore` (facts, episodes).
    - ``interrupt`` — barge-in :class:`InterruptController`. When set,
      the agent loop checks ``interrupt.should_stop()`` in addition to
      ``stop`` and honors it as a safety preempt.
    - ``artifacts`` — :class:`ArtifactStore` for agent-to-agent handoff
      via structured files.
    - ``agent_name`` — populated by :class:`LLMAgent` during ``run()``
      so ctx-scoped hooks can route their behavior by agent.
    - ``state`` — per-context scratchpad separate from
      :attr:`Session.state` (not persisted by :class:`SessionStore`).
    """

    session: Session = field(default_factory=Session)
    deps: Any = None
    bus: EventBus = field(default_factory=EventBus)
    on_event: EventCallback | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    hooks: HookRegistry = field(default_factory=HookRegistry)
    blackboard: Blackboard | None = None
    memory: MemoryStore | None = None
    interrupt: InterruptController | None = None
    artifacts: ArtifactStore | None = None
    agent_name: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    # Populated by :class:`LLMAgent.run` so hooks have typed access to
    # the running agent's tool registry and LLM backend without reaching
    # into ``ctx.state`` with magic keys. ``state`` stays user-only.
    tool_registry: ToolRegistry | None = None
    llm: LLM | None = None
    # Per-hook scratchpad — hooks keyed by ``id(hook)`` get their own
    # isolated dict so fingerprint counters / retry budgets don't leak
    # across hook instances. Replaces the previous ``ctx.session.state["__xxx__"]``
    # shared keys used by :mod:`edgevox.llm.hooks_slm`.
    hook_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Deterministic-mode seed. Set once per-context (or forwarded through
    # handoffs) to make a run reproducible: the LLM gets ``seed=``, hooks
    # that need randomness draw from ``ctx.rng``, and any other stochastic
    # subsystem can read ``ctx.seed`` directly. ``None`` (default) keeps
    # today's behaviour — each call uses the LLM's default sampler seed
    # and ``ctx.rng`` is a fresh ``random.Random()``.
    seed: int | None = None
    # Populated lazily from ``seed`` by ``__post_init__``; always usable,
    # even when ``seed is None`` (then it's an unseeded generator). Using
    # ``ctx.rng`` instead of ``random.random()`` is what actually gives
    # reproducibility — global ``random`` is shared across everything.
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.on_event is not None:
            self.bus.subscribe_all(self.on_event)
        if self.seed is not None:
            self.rng.seed(self.seed)

    def emit(self, kind: str, agent_name: str, payload: Any = None) -> None:
        """Publish an :class:`AgentEvent` to the bus."""
        self.bus.publish(
            AgentEvent(kind=kind, agent_name=agent_name, payload=payload)  # type: ignore[arg-type]
        )

    def should_stop(self) -> bool:
        """True if either the safety event or the interrupt controller
        requests a stop. Checked in hot loops and between tool hops."""
        if self.stop.is_set():
            return True
        return self.interrupt is not None and self.interrupt.should_stop()


@dataclass
class AgentResult:
    """Outcome of a single ``Agent.run()`` call."""

    reply: str
    agent_name: str
    elapsed: float = 0.0
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    skill_goals: list[GoalHandle] = field(default_factory=list)
    handed_off_to: str | None = None
    preempted: bool = False
    hook_ended: str | None = None  # reason string if a hook ended the turn


# --------- Handoff ---------


@dataclass
class Handoff:
    """Sentinel return value from a handoff tool.

    When ``LLMAgent._run_agent`` sees a tool whose result is a ``Handoff``,
    it stops calling the current LLM and invokes ``target.run(task, ctx)``.
    ``task`` defaults to the original user task if omitted — the common
    case for a Router.

    ``state_update`` (LangGraph ``Command``-style): when set, the keys
    are written into ``ctx.blackboard`` before the target runs. The
    target — or any subscribed observer — sees the update as part of
    the same control transfer. Useful for routers that want to record
    *why* they handed off without polluting the target's prompt.
    """

    target: Agent
    task: str | None = None
    reason: str = ""
    state_update: dict[str, Any] | None = None


# --------- Agent protocol ---------


@runtime_checkable
class Agent(Protocol):
    """Minimal polymorphic interface. Workflows accept this type."""

    name: str
    description: str

    def run(self, task: str, ctx: AgentContext) -> AgentResult: ...
    def run_stream(self, task: str, ctx: AgentContext) -> Iterator[str]: ...


# --------- LLMAgent ---------


ToolsArg = Iterable[Callable[..., object] | Tool] | ToolRegistry | None
SkillsArg = Iterable["Skill"] | None
HooksArg = Iterable[Hook] | HookRegistry | None


class LLMAgent:
    """Concrete LLM-backed agent.

    The LLM itself may be shared across agents: pass ``llm=None`` here
    and inject a single ``LLM`` before the first ``run()`` via
    :meth:`bind_llm`. Workflows and ``AgentApp`` handle this for the
    user so examples never need to think about it.

    Pass ``hooks=[...]`` to register per-agent hooks (fire alongside
    any hooks on the :class:`AgentContext`).
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        tools: ToolsArg = None,
        skills: SkillsArg = None,
        llm: LLM | None = None,
        handoffs: list[Agent] | None = None,
        hooks: HooksArg = None,
        max_tool_hops: int = 3,
        tool_choice_policy: Literal["auto", "required_first_hop", "required_always"] = "auto",
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self._llm: LLM | None = llm
        self._max_tool_hops = max_tool_hops
        # Tool-choice lifecycle. ``auto`` (default) lets the model
        # choose; ``required_first_hop`` enforces a structured tool
        # call on hop 0 via GBNF grammar (cf. OpenAI Agents SDK's
        # canonical SLM loop-break: required → auto after first
        # dispatch); ``required_always`` enforces it on every hop
        # until the budget is exhausted.
        self._tool_choice_policy = tool_choice_policy
        self._grammar_cache = GrammarCache()

        # Skills are stored alongside tools. We synthesize a synthetic
        # @tool for each skill and handoff target so the model sees a
        # single flat tool list. Dispatch then routes each call based on
        # its source (real tool / skill / handoff).
        self._tool_registry: ToolRegistry = ToolRegistry()
        self._skill_registry: dict[str, Skill] = {}
        self._handoff_registry: dict[str, Agent] = {}

        if tools is not None:
            self._register_tools(tools)
        if skills is not None:
            self._register_skills(skills)
        if handoffs:
            self._register_handoffs(handoffs)

        self._hooks: HookRegistry = HookRegistry()
        if hooks is not None:
            if isinstance(hooks, HookRegistry):
                self._hooks.extend(hooks)
            else:
                for h in hooks:
                    self._hooks.register(h)

    # ----- registration helpers -----

    def _register_tools(self, tools: ToolsArg) -> None:
        if isinstance(tools, ToolRegistry):
            for t in tools:
                self._tool_registry.register(t)
            return
        for t in tools:  # type: ignore[union-attr]
            self._tool_registry.register(t)

    def _register_skills(self, skills: SkillsArg) -> None:
        for s in skills or []:
            if not isinstance(s, Skill):
                raise TypeError(f"{s!r} is not a Skill — did you forget @skill?")
            if s.name in self._skill_registry:
                log.warning("Skill %r already registered — overwriting.", s.name)
            self._skill_registry[s.name] = s
            # Synthesize a tool descriptor the model can call. Dispatch
            # routes to the skill registry by name, never to the tool.
            self._tool_registry.tools[s.name] = s.as_tool_descriptor()

    def _register_handoffs(self, handoffs: list[Agent]) -> None:
        for agent in handoffs:
            tool_name = f"handoff_to_{agent.name}"
            if tool_name in self._handoff_registry:
                log.warning("Handoff target %r already registered — overwriting.", agent.name)
            self._handoff_registry[tool_name] = agent
            synthetic = _make_handoff_tool(agent)
            self._tool_registry.register(synthetic)

    def register_hook(self, h: Hook) -> LLMAgent:
        """Attach a hook after construction (useful in AgentApp chains)."""
        self._hooks.register(h)
        return self

    def bind_llm(self, llm: LLM) -> None:
        """Attach a shared ``LLM`` instance.

        Workflows call this before running a sub-agent so multiple agents
        share one GGUF in memory. Can be called repeatedly.
        """
        self._llm = llm

    @property
    def llm(self) -> LLM | None:
        return self._llm

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skill_registry)

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks

    # ----- hook firing -----

    def _fire(self, point: str, ctx: AgentContext, payload: Any) -> HookResult:
        """Fire agent-level hooks then ctx-level hooks in order. The
        second layer sees any modifications from the first."""
        return fire_chain([self._hooks, ctx.hooks], point, ctx, payload)

    # ----- the run() path -----

    def _ensure_llm(self) -> LLM:
        if self._llm is None:
            raise RuntimeError(
                f"LLMAgent {self.name!r} has no LLM bound. "
                "Either construct with llm=<instance> or call bind_llm() "
                "before run(). When using AgentApp, the framework does this "
                "for you."
            )
        return self._llm

    def _tool_choice_for_hop(self, hop: int, tool_schemas: list[dict] | None) -> tuple[str | None, Any]:
        """Resolve ``(tool_choice, grammar)`` for the current hop.

        Implements the OpenAI Agents SDK SLM loop-break: under
        ``required_first_hop`` the model is forced to call a tool on
        hop 0 (grammar-constrained, so SLMs can't emit malformed JSON
        for the structural envelope) and is then released to ``auto``
        on subsequent hops so the final reply can land. Returns
        ``(None, None)`` when the policy is ``auto`` or no tools are
        registered — the historical default behaviour, unchanged.
        """
        if not tool_schemas:
            return None, None
        if self._tool_choice_policy == "auto":
            return None, None
        if self._tool_choice_policy == "required_first_hop" and hop > 0:
            return "auto", None
        # required_always, OR required_first_hop on hop 0
        grammar = self._grammar_cache.get("tool", tool_schemas)
        return "required", grammar

    def _build_system_message(self) -> dict:
        """Build the system prompt for this agent. Each run() builds a
        fresh one so multiple agents sharing an LLM don't clobber each
        other's persona."""
        llm = self._ensure_llm()
        system = get_system_prompt(
            language=llm._language,
            has_tools=bool(self._tool_registry),
            persona=self.instructions,
        )
        return {"role": "system", "content": system}

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        """Run one turn. Returns an :class:`AgentResult`.

        On handoff, returns the *target's* result but annotates
        ``handed_off_to`` so the caller can trace delegation.
        """
        ctx = ctx or AgentContext()
        ctx.agent_name = self.name
        # Per-run ctx bookkeeping: publish the typed tool registry + LLM
        # so hooks can reach them via ``ctx.tool_registry`` / ``ctx.llm``
        # (typed fields) without touching ``ctx.state``. The legacy
        # ``ctx.state["__tool_registry__"]`` / ``["__llm__"]`` keys are
        # still populated for a single release as a back-compat shim —
        # remove once no external hook depends on them.
        prev_tool_registry = ctx.tool_registry
        prev_llm = ctx.llm
        ctx.tool_registry = self._tool_registry
        if self._llm is not None:
            ctx.llm = self._llm
        # Back-compat: legacy magic keys. Safe to remove after external
        # hook packages migrate — no framework code reads these now.
        prev_state_tool_registry = ctx.state.get("__tool_registry__")
        prev_state_llm = ctx.state.get("__llm__")
        ctx.state["__tool_registry__"] = self._tool_registry
        if self._llm is not None:
            ctx.state["__llm__"] = self._llm

        ctx.emit("agent_start", self.name, {"task": task})

        # If the previous turn was cut short by a barge-in, surface the
        # interrupt as a synthetic ``role="tool"`` envelope so the model
        # can respond coherently this turn instead of hallucinating that
        # the cancelled skill succeeded. We do this *before* resetting
        # the controller — reset clears ``latest``.
        pending_interrupt_msg: dict | None = None
        if ctx.interrupt is not None:
            pending_interrupt_msg = ctx.interrupt.as_tool_result()
            ctx.interrupt.reset()

        if ctx.should_stop():
            ctx.emit("safety_preempt", self.name, "stopped before start")
            return self._finalize_ctx_state(
                ctx,
                AgentResult(reply="Stopped.", agent_name=self.name, preempted=True),
                prev_tool_registry,
                prev_llm,
                prev_state_tool_registry,
                prev_state_llm,
            )

        # Fire on_run_start.
        r = self._fire(ON_RUN_START, ctx, {"task": task})
        if r.action is HookAction.END_TURN:
            ctx.emit("hook_end_turn", self.name, {"point": ON_RUN_START, "reason": r.reason})
            result = AgentResult(
                reply=r.payload or "",
                agent_name=self.name,
                hook_ended=r.reason,
            )
            # Fire on_run_end even on early termination so loggers run.
            end_r = self._fire(ON_RUN_END, ctx, result)
            if end_r.action is HookAction.MODIFY and isinstance(end_r.payload, AgentResult):
                result = end_r.payload
            ctx.emit("agent_end", self.name, {"reply": result.reply, "hook_ended": r.reason})
            return self._finalize_ctx_state(
                ctx,
                result,
                prev_tool_registry,
                prev_llm,
                prev_state_tool_registry,
                prev_state_llm,
            )
        if r.action is HookAction.MODIFY and isinstance(r.payload, dict):
            task = r.payload.get("task", task)

        llm = self._ensure_llm()

        # History isolation — each run() owns its own messages list.
        # Seeds from the caller's Session if any, otherwise starts
        # fresh. This is what makes concurrent LLMAgent.run() calls
        # thread-safe: they never touch each other's history.
        if ctx.session.messages:
            messages = list(ctx.session.messages)
            # Ensure first message is this agent's system prompt — it
            # may have been installed by a different agent in the same
            # session previously.
            messages[0] = self._build_system_message()
        else:
            messages = [self._build_system_message()]

        # Splice the interrupt-as-tool-result *after* the system prompt
        # so the model treats it as recent context for this turn.
        if pending_interrupt_msg is not None:
            messages.append(pending_interrupt_msg)

        t0 = time.perf_counter()
        reply, handoff, captured_tools, ended_by_hook = self._drive(llm, messages, task, ctx)
        # Persist the updated messages back into the session so the next
        # run() against the same context sees them.
        ctx.session.messages = messages
        elapsed = time.perf_counter() - t0

        if handoff is not None:
            ctx.emit("handoff", self.name, {"target": handoff.target.name, "reason": handoff.reason})
            # Apply LangGraph-style state_update before the target runs
            # so its hooks / blackboard watchers see the new state.
            if handoff.state_update and ctx.blackboard is not None:
                for k, v in handoff.state_update.items():
                    ctx.blackboard.set(k, v)
            target_task = handoff.task or task
            # Sub-agent runs with a fresh Session so its tool history
            # doesn't pollute the router's. deps / stop / on_event flow
            # through unchanged.
            sub_ctx = AgentContext(
                session=Session(),
                deps=ctx.deps,
                on_event=ctx.on_event,
                stop=ctx.stop,
                hooks=ctx.hooks,  # ctx-level hooks propagate
                blackboard=ctx.blackboard,
                memory=ctx.memory,
                interrupt=ctx.interrupt,
                artifacts=ctx.artifacts,
                # Determinism rides through handoffs — same seed on the
                # sub-agent means the whole multi-agent run is
                # reproducible end-to-end, not just the first leg.
                seed=ctx.seed,
            )
            # ``tool_registry``/``llm`` on the subagent ctx are installed
            # by the target's own ``run()`` — don't pre-seed from the
            # router, or the target's hooks would see the wrong tools.
            if isinstance(handoff.target, LLMAgent) and handoff.target._llm is None:
                handoff.target.bind_llm(llm)
            sub_result = handoff.target.run(target_task, sub_ctx)
            sub_result.handed_off_to = handoff.target.name
            sub_result.elapsed = time.perf_counter() - t0
            ctx.emit("agent_end", self.name, {"reply": sub_result.reply, "via_handoff": True})
            return self._finalize_ctx_state(
                ctx,
                sub_result,
                prev_tool_registry,
                prev_llm,
                prev_state_tool_registry,
                prev_state_llm,
            )

        preempted = ctx.should_stop()
        result = AgentResult(
            reply=reply if not preempted else (reply or "Stopped."),
            agent_name=self.name,
            elapsed=elapsed,
            tool_calls=captured_tools,
            preempted=preempted,
            hook_ended=ended_by_hook,
        )

        # Fire on_run_end.
        end_r = self._fire(ON_RUN_END, ctx, result)
        if end_r.action is HookAction.MODIFY and isinstance(end_r.payload, AgentResult):
            result = end_r.payload

        ctx.emit("agent_end", self.name, {"reply": result.reply, "preempted": preempted})
        return self._finalize_ctx_state(
            ctx,
            result,
            prev_tool_registry,
            prev_llm,
            prev_state_tool_registry,
            prev_state_llm,
        )

    @staticmethod
    def _finalize_ctx_state(
        ctx: AgentContext,
        result: AgentResult,
        prev_tool_registry: ToolRegistry | None,
        prev_llm: LLM | None,
        prev_state_tool_registry: Any = None,
        prev_state_llm: Any = None,
    ) -> AgentResult:
        """Restore ctx fields + legacy scratchpad keys we set at run()
        entry so nested runs don't clobber each other's state."""
        ctx.tool_registry = prev_tool_registry
        ctx.llm = prev_llm
        # Legacy keys restored identically — back-compat only.
        if prev_state_tool_registry is None:
            ctx.state.pop("__tool_registry__", None)
        else:
            ctx.state["__tool_registry__"] = prev_state_tool_registry
        if prev_state_llm is None:
            ctx.state.pop("__llm__", None)
        else:
            ctx.state["__llm__"] = prev_state_llm
        return result

    def _drive(
        self,
        llm: LLM,
        messages: list[dict],
        task: str,
        ctx: AgentContext,
    ) -> tuple[str, Handoff | None, list[ToolCallResult], str | None]:
        """Drive the LLM loop against a caller-owned messages list.

        Returns ``(reply, handoff, captured_tool_calls, hook_end_reason)``.
        ``hook_end_reason`` is set when a hook ended the turn early.

        The ``messages`` list is mutated in place to append the user
        turn, any tool calls, and the assistant replies — but it is
        **not** shared with the LLM's internal state. Concurrent calls
        from parallel agents are isolated because each has its own
        list. The LLM itself is still thread-unsafe and is guarded by
        ``LLM.complete()``'s inference lock.
        """
        messages.append({"role": "user", "content": task})
        captured: list[ToolCallResult] = []
        tool_schemas = self._tool_registry.openai_schemas() if self._tool_registry else None

        for hop in range(self._max_tool_hops + 1):
            if ctx.should_stop():
                return ("Stopped.", None, captured, None)

            # ----- before_llm -----
            pre = self._fire(BEFORE_LLM, ctx, {"messages": messages, "hop": hop, "tools": tool_schemas})
            if pre.action is HookAction.END_TURN:
                reply = pre.payload or ""
                messages.append({"role": "assistant", "content": reply})
                ctx.emit("hook_end_turn", self.name, {"point": BEFORE_LLM, "reason": pre.reason})
                return (reply, None, captured, pre.reason or "hook ended at before_llm")
            if pre.action is HookAction.MODIFY and isinstance(pre.payload, dict):
                # Only allow messages/tools replacement — other keys ignored.
                new_msgs = pre.payload.get("messages")
                if isinstance(new_msgs, list):
                    # Rebind by mutating in place so caller sees updates.
                    messages[:] = new_msgs
                new_tools = pre.payload.get("tools", tool_schemas)
                tool_schemas = new_tools

            # Plumb the interrupt cancel-token into llama-cpp's
            # stopping_criteria. Without this, a barge-in only stops
            # downstream consumers — the LLM keeps generating until
            # ``max_tokens`` exhausts. Test doubles that don't accept
            # ``stop_event`` are tolerated below.
            cancel_token = None
            if ctx.interrupt is not None and ctx.interrupt.policy.cancel_llm:
                cancel_token = ctx.interrupt.cancel_token

            # Pick a tool-choice + grammar combination per the policy.
            # ``auto`` is the historical default (no constraint).
            tool_choice, grammar = self._tool_choice_for_hop(hop, tool_schemas)
            # Deterministic-mode plumbing: ``ctx.seed`` (when set) pins
            # the LLM sampler so a run is reproducible end-to-end. Older
            # shims that don't accept ``seed`` are handled by the
            # TypeError fallback below.
            complete_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": tool_schemas,
                "tool_choice": tool_choice,
                "stream": False,
                "stop_event": cancel_token,
                "grammar": grammar,
            }
            if ctx.seed is not None:
                complete_kwargs["seed"] = ctx.seed
            try:
                result = llm.complete(**complete_kwargs)
            except TypeError:
                # Back-compat for LLM shims (tests) that predate
                # ``stop_event`` / ``grammar`` / ``seed``. Drop the new
                # kwargs and retry with the minimal surface.
                result = llm.complete(
                    messages,
                    tools=tool_schemas,
                    tool_choice=tool_choice,
                    stream=False,
                )
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            raw_content = message.get("content") or ""

            # Run the full parser chain (think-strip → SGLang → chatml
            # → Gemma inline) only when the chat template didn't emit
            # structured tool_calls. Preset-specific parsers are read
            # off the LLM if available; non-LLM test doubles just skip
            # them.
            fallback_mode = False
            preset_parsers = getattr(llm, "_tool_call_parsers", ()) or ()
            if not tool_calls:
                recovered, cleaned, fallback_mode = parse_tool_calls_from_content(
                    raw_content,
                    preset_parsers=preset_parsers,
                    known_tools=set(self._tool_registry.tools.keys()) if self._tool_registry else None,
                    tool_schemas=tool_schemas,
                )
                if recovered:
                    tool_calls = recovered
                    content = cleaned
                else:
                    content = _strip_thinking(raw_content).strip()
            else:
                content = raw_content.strip()

            # ----- after_llm -----
            post = self._fire(AFTER_LLM, ctx, {"content": content, "tool_calls": tool_calls, "hop": hop})
            if post.action is HookAction.END_TURN:
                reply = post.payload or content
                messages.append({"role": "assistant", "content": reply})
                ctx.emit("hook_end_turn", self.name, {"point": AFTER_LLM, "reason": post.reason})
                return (reply, None, captured, post.reason or "hook ended at after_llm")
            if post.action is HookAction.MODIFY and isinstance(post.payload, dict):
                content = post.payload.get("content", content)
                tool_calls = post.payload.get("tool_calls", tool_calls)

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                return (content, None, captured, None)

            if hop == self._max_tool_hops:
                log.warning(
                    "Tool-call budget exhausted for %s after %d hops",
                    self.name,
                    self._max_tool_hops,
                )
                fallback = content or "Sorry, I couldn't finish that request."
                messages.append({"role": "assistant", "content": fallback})
                return (fallback, None, captured, None)

            # ----- dispatch this batch of tool calls (parallel) -----
            #
            # Handoffs short-circuit before any tool runs — we split the
            # batch into handoff vs non-handoff and, if any handoff is
            # present, honour the first one immediately.
            handoff_hit: Handoff | None = None
            real_calls: list[dict] = []
            for call in tool_calls:
                name = (call.get("function") or {}).get("name", "")
                if name in self._handoff_registry and handoff_hit is None:
                    handoff_hit = Handoff(target=self._handoff_registry[name])
                    captured.append(
                        ToolCallResult(
                            name=name,
                            arguments={},
                            result=f"handoff -> {handoff_hit.target.name}",
                        )
                    )
                else:
                    real_calls.append(call)

            if handoff_hit is not None:
                return ("", handoff_hit, captured, None)

            # Run each tool/skill in parallel. Tools finish in milliseconds;
            # skills run on their own worker threads but the poll loop
            # here still blocks, so we parallelize the *dispatch* so one
            # slow skill doesn't serialize the others.
            dispatched, hook_end = self._dispatch_batch(real_calls, ctx, hop=hop)
            captured.extend(out for _, _, out in dispatched)

            if hook_end is not None:
                reply = hook_end.payload or "Stopped by hook."
                messages.append({"role": "assistant", "content": reply})
                ctx.emit("hook_end_turn", self.name, {"point": BEFORE_TOOL, "reason": hook_end.reason})
                return (reply, None, captured, hook_end.reason or "hook ended during tool dispatch")

            if ctx.should_stop():
                return ("Stopped.", None, captured, None)

            if fallback_mode:
                summary = "; ".join(f"{name} -> {json.dumps(payload, default=str)}" for name, payload, _ in dispatched)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"(system: tool results — {summary}. "
                            "Now answer the previous request in one short sentence.)"
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": real_calls,
                    }
                )
                for call, (name, payload, _) in zip(real_calls, dispatched, strict=False):
                    # Preserve the model-emitted call id verbatim (Mistral
                    # expects a 9-char round-trip; Qwen/chatml carry their
                    # own ids too). Only synthesise one when the detector
                    # didn't surface one.
                    call_id = call.get("id") or f"call_{hop}_{name}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(payload, default=str),
                        }
                    )

        return ("", None, captured, None)

    def _dispatch_batch(
        self,
        calls: list[dict],
        ctx: AgentContext,
        *,
        hop: int,
    ) -> tuple[list[tuple[str, dict, ToolCallResult]], HookResult | None]:
        """Dispatch a batch of tool/skill calls, parallelized when >1.

        The LLM can emit multiple tool_calls in a single response; each
        one runs in its own worker so a slow skill doesn't serialize
        the rest. Order of the returned list matches ``calls`` order
        so the LLM sees a deterministic tool-result sequence.

        Returns ``(results, end_turn_hook_result | None)``. If any
        hook returned ``end_turn`` at before_tool or after_tool, the
        first such HookResult is propagated out so the outer loop can
        terminate the turn cleanly.
        """
        results: list[tuple[str, dict, ToolCallResult]] = [None] * len(calls)  # type: ignore[list-item]
        end_turn_lock = threading.Lock()
        end_turn_box: list[HookResult] = []

        def record_end_turn(r: HookResult) -> None:
            with end_turn_lock:
                if not end_turn_box:
                    end_turn_box.append(r)

        def run_one(idx: int, call: dict) -> None:
            if ctx.should_stop() or end_turn_box:
                return
            fn = call.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            is_skill = name in self._skill_registry

            # ----- before_tool -----
            req = ToolCallRequest(
                name=name,
                arguments=args_raw,
                hop=hop,
                is_skill=is_skill,
            )
            pre = self._fire(BEFORE_TOOL, ctx, req)
            if pre.action is HookAction.END_TURN:
                record_end_turn(pre)
                return
            if pre.action is HookAction.MODIFY and isinstance(pre.payload, ToolCallRequest):
                req = pre.payload

            if req.skip_dispatch:
                outcome = ToolCallResult(
                    name=req.name,
                    arguments=req.arguments if isinstance(req.arguments, dict) else {},
                    result=req.synthetic_result,
                )
            elif req.is_skill or req.name in self._skill_registry:
                outcome = self._dispatch_skill(req.name, req.arguments, ctx)
            else:
                outcome = self._tool_registry.dispatch(req.name, req.arguments, ctx=ctx)

            # ----- after_tool -----
            post = self._fire(AFTER_TOOL, ctx, outcome)
            if post.action is HookAction.END_TURN:
                record_end_turn(post)
                return
            if post.action is HookAction.MODIFY and isinstance(post.payload, ToolCallResult):
                outcome = post.payload

            payload = {"ok": True, "result": outcome.result} if outcome.ok else {"ok": False, "error": outcome.error}
            results[idx] = (req.name, payload, outcome)
            ctx.emit("tool_call", self.name, outcome)

        if len(calls) > 1:
            with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as ex:
                for f in [ex.submit(run_one, i, c) for i, c in enumerate(calls)]:
                    f.result()
        else:
            for i, c in enumerate(calls):
                run_one(i, c)

        dispatched = [r for r in results if r is not None]
        end = end_turn_box[0] if end_turn_box else None
        return dispatched, end

    def _dispatch_skill(self, name: str, arguments: str | dict[str, Any], ctx: AgentContext) -> ToolCallResult:
        """Dispatch a skill call through its ``GoalHandle`` lifecycle.

        Blocks until the goal completes, fails, is cancelled, or hits
        its timeout. The worker thread inside the skill is where the
        actual work happens; this method polls so ``ctx.stop`` can
        preempt immediately.
        """
        skill_obj = self._skill_registry[name]

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                return ToolCallResult(name=name, arguments={}, error=f"invalid JSON arguments: {e}")
        else:
            args = arguments or {}

        try:
            handle = skill_obj.start(ctx, **args)
        except Exception as e:
            log.exception("Skill %r failed to start", name)
            return ToolCallResult(name=name, arguments=args, error=f"start failed: {e}")

        ctx.emit("skill_goal", self.name, {"skill": name, "handle_id": handle.id, "args": args})

        # Poll the handle. Physics (for sims like IrSimEnvironment) runs
        # on a dedicated background thread in the deps, so we just wait
        # on the handle's terminal event with a short timeout and check
        # ``ctx.stop`` in between for safety preemption.
        poll_interval = 0.05
        while True:
            if ctx.should_stop():
                handle.cancel()
                ctx.emit(
                    "skill_cancelled",
                    self.name,
                    {"skill": name, "reason": "safety_preempt"},
                )
                return ToolCallResult(name=name, arguments=args, error="cancelled (safety_preempt)")
            status = handle.poll(timeout=poll_interval)
            if status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
                break

        if handle.status is GoalStatus.SUCCEEDED:
            return ToolCallResult(name=name, arguments=args, result=handle.result)
        if handle.status is GoalStatus.CANCELLED:
            return ToolCallResult(name=name, arguments=args, error="cancelled")
        return ToolCallResult(
            name=name,
            arguments=args,
            error=handle.error or "skill failed",
        )

    # ----- streaming is not supported on the multi-hop path; fall back -----

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        """Yield the final reply as one chunk.

        Multi-step / multi-agent flows can't cleanly stream tokens while
        also intercepting tool calls, so we degrade to single-chunk.
        Callers that sentence-split (e.g. the TTS pipeline) still work
        naturally. Pure chitchat paths could stream but for simplicity
        this uses the non-streaming path uniformly.
        """
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply

    # ----- subagent helper -----

    def spawn_subagent(
        self,
        task: str,
        *,
        parent_ctx: AgentContext,
        instructions: str | None = None,
        tools: ToolsArg = None,
        hooks: HooksArg = None,
        max_tool_hops: int | None = None,
    ) -> AgentResult:
        """Spawn a sub-agent with a fresh :class:`Session` and run ``task``.

        The sub-agent shares the parent's LLM, deps, bus, and any
        plug-ins (blackboard, memory, interrupt, artifacts) but starts
        with a clean context window — the recommended pattern for long
        workflows per Anthropic's harness guidance.

        Returns the sub-agent's :class:`AgentResult` so the parent can
        fold the summary into its own turn. Artifact handoffs (parent
        writes to ``ctx.artifacts`` before spawning, subagent reads)
        are the intended way to carry structured state.
        """
        sub = LLMAgent(
            name=f"{self.name}.sub",
            description=f"subagent of {self.name}",
            instructions=instructions or self.instructions,
            tools=tools if tools is not None else list(self._tool_registry),
            llm=self._llm,
            hooks=hooks,
            max_tool_hops=max_tool_hops if max_tool_hops is not None else self._max_tool_hops,
        )
        sub_ctx = AgentContext(
            session=Session(),
            deps=parent_ctx.deps,
            on_event=parent_ctx.on_event,
            stop=parent_ctx.stop,
            hooks=parent_ctx.hooks,
            blackboard=parent_ctx.blackboard,
            memory=parent_ctx.memory,
            interrupt=parent_ctx.interrupt,
            artifacts=parent_ctx.artifacts,
            seed=parent_ctx.seed,
        )
        return sub.run(task, sub_ctx)


# --------- synthetic handoff tool helper ---------


def _make_handoff_tool(target: Agent) -> Callable[..., Any]:
    """Synthesize a ``@tool``-decorated function for a handoff target.

    The returned callable is only used for its ``__edgevox_tool__``
    descriptor — the LLMAgent dispatch loop intercepts calls to this
    tool's name before executing it, so the body here is just a
    fallback so the function is callable for tests.
    """
    tool_name = f"handoff_to_{target.name}"
    description = f"Transfer this request to the {target.name!r} specialist agent. {target.description}"

    def _handoff(reason: str = "") -> Handoff:  # body only runs in tests
        return Handoff(target=target, reason=reason)

    _handoff.__name__ = tool_name
    _handoff.__doc__ = (
        f"{description}\n\n    Args:\n        reason: one short phrase explaining why you're handing off.\n"
    )
    return tool_decorator(name=tool_name, description=description)(_handoff)
