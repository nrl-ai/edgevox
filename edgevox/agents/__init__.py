"""EdgeVox agent framework.

First-class Agent abstractions layered on top of the existing tool system:

- ``Agent`` — polymorphic protocol every agent and workflow implements.
- ``LLMAgent`` — concrete agent backed by an ``edgevox.llm.LLM``.
- ``Session``, ``AgentContext``, ``AgentEvent``, ``AgentResult`` — runtime
  plumbing and observability.
- ``Handoff`` — sentinel return value that transfers control between
  agents (OpenAI-Agents-SDK-style).
- ``Skill``, ``GoalHandle``, ``@skill`` — cancellable robot actions
  distinct from pure ``@tool`` functions.
- ``SimEnvironment``, ``ToyWorld`` — simulation protocol + stdlib-only
  reference env used for examples and tests.
- Workflows: ``Sequence``, ``Fallback``, ``Loop``, ``Router``, ``Retry``,
  ``Timeout`` — Behaviour-Tree-shaped composition of agents.
- **Hooks** (`hooks.py`, `hooks_builtin.py`) — pluggable agent-loop
  behavior (guardrails, audit, plan mode, memory, compaction, SLM
  hardening) at 6 fire points. See the module docstrings for the
  full matrix.
- **Memory / compaction** (`memory.py`) — long-term `MemoryStore`,
  `SessionStore`, `Compactor`, and `NotesFile` primitives.
- **Artifacts** (`artifacts.py`) — shared file-like store for structured
  agent-to-agent handoffs (Anthropic harness-design pattern).
- **Interrupt** (`interrupt.py`) — `InterruptController` barge-in
  coordinator.
- **Multi-agent** (`multiagent.py`) — `Blackboard`, `BackgroundAgent`,
  `AgentPool`, agent-to-agent messaging.
"""

from edgevox.agents.agent_tools import agent_as_tool
from edgevox.agents.artifacts import (
    Artifact,
    ArtifactStore,
    FileArtifactStore,
    InMemoryArtifactStore,
    bytes_artifact,
    json_artifact,
    make_artifact_tools,
    text_artifact,
)
from edgevox.agents.base import (
    Agent,
    AgentContext,
    AgentEvent,
    AgentResult,
    Handoff,
    LLMAgent,
    Session,
)
from edgevox.agents.bus import EventBus, MainThreadScheduler, RenderRequest
from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_TOOL,
    FIRE_POINTS,
    ON_RUN_END,
    ON_RUN_START,
    Hook,
    HookAction,
    HookRegistry,
    HookResult,
    ToolCallRequest,
    fire_chain,
    hook,
    load_entry_point_hooks,
)
from edgevox.agents.hooks_builtin import (
    AuditLogHook,
    ContextCompactionHook,
    ContextWindowManager,
    EchoingHook,
    EpisodeLoggerHook,
    MemoryInjectionHook,
    NotesInjectorHook,
    OutputValidatorHook,
    PersistSessionHook,
    PlanModeHook,
    SafetyGuardrailHook,
    TimingHook,
    TokenBudgetHook,
    ToolErrorRetryHook,
    ToolOutputTruncatorHook,
    TracingHook,
    ValidatorError,
    console_approver,
    length_cap,
    pii_redactor,
    schema_check,
)
from edgevox.agents.interrupt import (
    EnergyBargeInWatcher,
    InterruptController,
    InterruptEvent,
    InterruptPolicy,
)
from edgevox.agents.memory import (
    Compactor,
    Episode,
    Fact,
    JSONMemoryStore,
    JSONSessionStore,
    MemoryStore,
    NotesFile,
    Preference,
    SessionStore,
    SQLiteSessionStore,
    default_memory_dir,
    estimate_tokens,
    new_session_id,
)
from edgevox.agents.memory_vec import (
    EmbedFn,
    VectorMemoryStore,
    llama_embed,
)
from edgevox.agents.multiagent import (
    AgentMessage,
    AgentPool,
    BackgroundAgent,
    Blackboard,
    Trigger,
    debounce_trigger,
    send_message,
    subscribe_inbox,
)
from edgevox.agents.sim import SimEnvironment, ToyWorld
from edgevox.agents.skills import GoalHandle, GoalStatus, Skill, skill
from edgevox.agents.vad_watchers import (
    BargeInVADWatcher,
    SileroVADWatcher,
    TENVADWatcher,
    WebRTCVADWatcher,
    create_vad_watcher,
)
from edgevox.agents.workflow import (
    Fallback,
    Loop,
    Orchestrator,
    Parallel,
    Retry,
    Router,
    Sequence,
    Supervisor,
    Timeout,
)

__all__ = [
    "AFTER_LLM",
    "AFTER_TOOL",
    "BEFORE_LLM",
    "BEFORE_TOOL",
    "FIRE_POINTS",
    "ON_RUN_END",
    "ON_RUN_START",
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentMessage",
    "AgentPool",
    "AgentResult",
    "Artifact",
    "ArtifactStore",
    "AuditLogHook",
    "BackgroundAgent",
    "BargeInVADWatcher",
    "Blackboard",
    "Compactor",
    "ContextCompactionHook",
    "ContextWindowManager",
    "EchoingHook",
    "EmbedFn",
    "EnergyBargeInWatcher",
    "Episode",
    "EpisodeLoggerHook",
    "EventBus",
    "Fact",
    "Fallback",
    "FileArtifactStore",
    "GoalHandle",
    "GoalStatus",
    "Handoff",
    "Hook",
    "HookAction",
    "HookRegistry",
    "HookResult",
    "InMemoryArtifactStore",
    "InterruptController",
    "InterruptEvent",
    "InterruptPolicy",
    "JSONMemoryStore",
    "JSONSessionStore",
    "LLMAgent",
    "Loop",
    "MainThreadScheduler",
    "MemoryInjectionHook",
    "MemoryStore",
    "NotesFile",
    "NotesInjectorHook",
    "Orchestrator",
    "OutputValidatorHook",
    "Parallel",
    "PersistSessionHook",
    "PlanModeHook",
    "Preference",
    "RenderRequest",
    "Retry",
    "Router",
    "SQLiteSessionStore",
    "SafetyGuardrailHook",
    "Sequence",
    "Session",
    "SessionStore",
    "SileroVADWatcher",
    "SimEnvironment",
    "Skill",
    "Supervisor",
    "TENVADWatcher",
    "Timeout",
    "TimingHook",
    "TokenBudgetHook",
    "ToolCallRequest",
    "ToolErrorRetryHook",
    "ToolOutputTruncatorHook",
    "ToyWorld",
    "TracingHook",
    "Trigger",
    "ValidatorError",
    "VectorMemoryStore",
    "WebRTCVADWatcher",
    "agent_as_tool",
    "bytes_artifact",
    "console_approver",
    "debounce_trigger",
    "default_memory_dir",
    "estimate_tokens",
    "fire_chain",
    "hook",
    "json_artifact",
    "length_cap",
    "load_entry_point_hooks",
    "make_artifact_tools",
    "new_session_id",
    "pii_redactor",
    "schema_check",
    "send_message",
    "skill",
    "subscribe_inbox",
    "text_artifact",
]
