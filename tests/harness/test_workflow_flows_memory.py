"""Memory + flow integration tests.

Memory primitives (``JSONMemoryStore``, ``NotesFile``) are unit-tested in
``test_memory.py``. This file tests them in a *flow* context: a real
``LLMAgent`` with ``MemoryInjectionHook`` / ``NotesInjectorHook`` wired
in, exercising the full path from memory state -> system prompt
mutation -> LLM message -> tool dispatch -> world side-effect.

Probes:

- The system prompt seen by the LLM actually contains the seeded memory
  (we read it back from ``ScriptedLLM.calls[i]['messages'][0]``).
- The hook is idempotent across tool hops within one turn (no
  duplicate ``## Memory`` blocks).
- Notes appended *during* a flow are visible to subsequent turns.
"""

from __future__ import annotations

import pytest

from edgevox.agents.base import LLMAgent
from edgevox.agents.hooks_builtin import MemoryInjectionHook, NotesInjectorHook
from edgevox.agents.memory import JSONMemoryStore, NotesFile
from edgevox.agents.sim import ToyWorld
from edgevox.llm import tool

from .conftest import ScriptedLLM, calls, reply


@pytest.fixture
def world_and_tools():
    world = ToyWorld()

    @tool
    def set_light(room: str, on: bool) -> str:
        """Turn a room's light on or off.

        Args:
            room: room name.
            on: true to turn on.
        """
        with world._lock:
            world._rooms[room].light_on = on
        return f"{room} -> {'on' if on else 'off'}"

    return {"world": world, "tools": [set_light]}


class TestMemoryInjectionInFlow:
    def test_seeded_facts_reach_the_system_prompt(self, world_and_tools, tmp_path):
        bundle = world_and_tools
        memory = JSONMemoryStore(tmp_path / "memory.json")
        memory.add_fact("user_name", "Anh")
        memory.set_preference("voice", "concise")
        memory.flush()

        scripted = ScriptedLLM(
            [
                calls(("set_light", {"room": "kitchen", "on": True})),
                reply("Done."),
            ]
        )
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Help with home tasks. Read the Memory block in your prompt before replying.",
            tools=bundle["tools"],
            llm=scripted,
            hooks=[MemoryInjectionHook(memory_store=memory)],
        )
        agent.run("turn on the kitchen")

        # First LLM hop's system prompt must include the injected ## Memory block.
        first_system_prompt = scripted.calls[0]["messages"][0]
        assert first_system_prompt["role"] == "system"
        assert "## Memory" in first_system_prompt["content"]
        # The actual fact we seeded is rendered.
        assert "Anh" in first_system_prompt["content"]

    def test_memory_block_idempotent_across_tool_hops(self, world_and_tools, tmp_path):
        """A single user turn can run multiple LLM hops (one per tool batch).
        ``MemoryInjectionHook`` should add the ## Memory block once, not
        on every hop. We check by counting occurrences in the LAST hop's
        system prompt -- they're all the same prompt object, so even one
        re-injection would show up."""
        bundle = world_and_tools
        memory = JSONMemoryStore(tmp_path / "memory.json")
        memory.add_fact("note_to_self", "the kettle is in drawer 2")
        memory.flush()

        scripted = ScriptedLLM(
            [
                calls(("set_light", {"room": "kitchen", "on": True})),
                # Second hop: also calls a tool, then closes out.
                calls(("set_light", {"room": "bedroom", "on": True})),
                reply("Both lit."),
            ]
        )
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Help with home tasks.",
            tools=bundle["tools"],
            llm=scripted,
            hooks=[MemoryInjectionHook(memory_store=memory)],
        )
        agent.run("kitchen and bedroom please")

        # The same system message object is mutated across hops because
        # `messages` is the agent's internal list. Check the final
        # version.
        final_system = scripted.calls[-1]["messages"][0]["content"]
        assert final_system.count("## Memory") == 1, "memory injected more than once per turn"


class TestNotesInjectionInFlow:
    def test_notes_tail_reaches_prompt(self, world_and_tools, tmp_path):
        bundle = world_and_tools
        notes = NotesFile(tmp_path / "notes.md")
        notes.append("kitchen kettle is in drawer 2")
        notes.append("bedroom curtain is on a remote")

        scripted = ScriptedLLM([reply("noted.")])
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Help with home tasks. The Notes block in the prompt is your scratchpad.",
            tools=bundle["tools"],
            llm=scripted,
            hooks=[NotesInjectorHook(notes)],
        )
        agent.run("anything")

        sys_msg = scripted.calls[0]["messages"][0]["content"]
        # Both notes must be visible.
        assert "kitchen kettle" in sys_msg
        assert "bedroom curtain" in sys_msg


class TestMemoryAcrossTurns:
    def test_facts_added_in_one_turn_visible_in_the_next(self, world_and_tools, tmp_path):
        """The hook is constructed once and reused; facts written to the
        memory store between turns must show up in the next turn's
        prompt without any state-passing on the agent side."""
        bundle = world_and_tools
        memory = JSONMemoryStore(tmp_path / "memory.json")
        hook = MemoryInjectionHook(memory_store=memory)

        scripted = ScriptedLLM([reply("turn 1 done"), reply("turn 2 done")])
        agent = LLMAgent(
            name="HomeBot",
            description="home assistant",
            instructions="Help.",
            tools=bundle["tools"],
            llm=scripted,
            hooks=[hook],
        )

        # Turn 1: no facts seeded yet. Hook should no-op.
        agent.run("first turn")
        assert "## Memory" not in scripted.calls[0]["messages"][0]["content"]

        # Between turns: we (the user / a different hook / a
        # subscribed observer) write a fact.
        memory.add_fact("kettle_location", "drawer 2")
        memory.flush()

        # Turn 2: the new fact must show up.
        agent.run("second turn")
        # The agent's history persists across runs, so the system
        # message in turn 2 starts as the *mutated* system from turn 1.
        # Check the fact is present in the most recent hop's system.
        latest_sys = scripted.calls[-1]["messages"][0]["content"]
        assert "drawer 2" in latest_sys
