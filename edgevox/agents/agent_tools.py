"""Wrap an :class:`Agent` as a :class:`Tool` the parent LLM can call.

Alternative to the :class:`Handoff` pattern: ``agent_as_tool(child)``
lets a parent agent invoke a child **inline** during its own turn —
the result is threaded back as a regular tool-call result rather than
short-circuiting the parent's loop. This mirrors the smolagents /
LangGraph "sub-agent as tool" pattern; EdgeVox already ships the
OpenAI Agents SDK "agent-as-return-value" handoff path, and this
module adds the complementary shape.

When to prefer which:

* **Handoff** — the user's question is fully re-delegated; the parent
  agent is done for this turn. One LLM hop on the parent, sub-agent
  runs its own loop, result becomes the user-facing reply.
* **agent_as_tool** — the parent needs the sub-agent's answer as one
  input among several (e.g., a research agent calls a web-lookup
  specialist + a code-runner specialist and synthesises the results).
  Parent stays in control of the turn.

Usage::

    from edgevox.agents.agent_tools import agent_as_tool

    researcher = LLMAgent(name="researcher", tools=[web_search, fetch])
    coder      = LLMAgent(name="coder",      tools=[run_python])

    lead = LLMAgent(
        name="lead",
        instructions="Use the researcher and coder as needed.",
        tools=[
            agent_as_tool(researcher, description="Factual lookup via web"),
            agent_as_tool(coder,      description="Run Python snippets"),
        ],
    )

The wrapped tool's JSON schema has one required string argument,
``task`` — the prompt that goes into ``child.run(task, child_ctx)``.
Additional forwarded fields (``deps``, ``memory``, ``interrupt``,
``blackboard``, ``artifacts``, ``seed``) come from the parent's
context so the child sees consistent shared state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgevox.llm.tools import Tool

if TYPE_CHECKING:
    from edgevox.agents.base import Agent, AgentContext


def agent_as_tool(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_description: str = "The task or question for the sub-agent.",
) -> Tool:
    """Wrap ``agent`` as a :class:`Tool` invoked by the parent LLM.

    Args:
        agent: The sub-agent to wrap. Any object implementing the
            :class:`Agent` Protocol (``LLMAgent``, workflow, custom).
        name: Tool name exposed to the model. Defaults to ``agent.name``.
        description: Tool description exposed to the model. Defaults to
            ``agent.description``, which is what routers advertise.
        arg_description: Docstring for the ``task`` argument the LLM
            sees. Override when the sub-agent expects a specialised
            input shape (e.g. "A Python snippet to execute").

    Returns:
        A :class:`Tool` whose implementation calls ``agent.run(task,
        child_ctx)`` with a fresh :class:`Session` but the parent's
        ``deps``, ``bus``, ``memory``, ``interrupt``, ``blackboard``,
        ``artifacts``, and ``seed``.
    """
    from edgevox.agents.base import AgentContext, Session

    tool_name = name or getattr(agent, "name", None) or "subagent"
    tool_desc = description or getattr(agent, "description", None) or f"Delegate to the {tool_name!r} sub-agent."

    # The Tool.func closure captures ``agent``. The parent context is
    # *not* captured here — it's supplied at call time via the
    # framework-injected ``ctx`` parameter the tool decorator honours.
    def _run_subagent(task: str, ctx: AgentContext | None = None) -> str:
        # ``ctx`` is framework-injected: see ``Tool.call``'s handling
        # of ``ctx`` / ``handle`` special parameters. If the caller
        # didn't provide one, we still run — just without shared state.
        parent = ctx or AgentContext()
        child_ctx = AgentContext(
            session=Session(),
            deps=parent.deps,
            bus=parent.bus,
            on_event=parent.on_event,
            stop=parent.stop,
            hooks=parent.hooks,
            blackboard=parent.blackboard,
            memory=parent.memory,
            interrupt=parent.interrupt,
            artifacts=parent.artifacts,
            seed=parent.seed,
        )
        result = agent.run(task, child_ctx)
        return result.reply or ""

    return Tool(
        name=tool_name,
        description=tool_desc,
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": arg_description,
                },
            },
            "required": ["task"],
        },
        func=_run_subagent,
    )


__all__ = ["agent_as_tool"]
