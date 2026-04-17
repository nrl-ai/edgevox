"""Tests for MemoryStore, SessionStore, Compactor, NotesFile."""

from __future__ import annotations

from edgevox.agents.base import Session
from edgevox.agents.memory import (
    Compactor,
    JSONMemoryStore,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# JSONMemoryStore
# ---------------------------------------------------------------------------


class TestJSONMemoryStore:
    def test_facts_roundtrip(self, tmp_memory_store, tmp_path):
        store = tmp_memory_store
        store.add_fact("user_name", "Anh")
        store.add_fact("kettle_location", "drawer 2", scope="kitchen")
        store.flush()

        # Reload — separate instance reads the same file.
        store2 = JSONMemoryStore(tmp_path / "memory.json")
        assert store2.get_fact("user_name") == "Anh"
        assert store2.get_fact("kettle_location", scope="kitchen") == "drawer 2"
        assert store2.get_fact("kettle_location") is None  # wrong scope

    def test_fact_overwrite(self, tmp_memory_store):
        tmp_memory_store.add_fact("x", "1")
        tmp_memory_store.add_fact("x", "2")
        assert tmp_memory_store.get_fact("x") == "2"
        assert len(tmp_memory_store.facts()) == 1

    def test_forget_fact(self, tmp_memory_store):
        tmp_memory_store.add_fact("x", "1")
        assert tmp_memory_store.forget_fact("x") is True
        assert tmp_memory_store.get_fact("x") is None
        assert tmp_memory_store.forget_fact("x") is False

    def test_preferences(self, tmp_memory_store):
        tmp_memory_store.set_preference("voice", "concise")
        assert tmp_memory_store.preferences()[0].key == "voice"
        assert tmp_memory_store.preferences()[0].value == "concise"

    def test_episodes(self, tmp_memory_store):
        for i in range(15):
            tmp_memory_store.add_episode(
                kind="tool_call",
                payload={"name": f"t{i}", "x": i},
                outcome="ok",
            )
        recent = tmp_memory_store.recent_episodes(5)
        assert len(recent) == 5
        assert recent[-1].payload["x"] == 14

    def test_recent_episodes_kind_filter(self, tmp_memory_store):
        tmp_memory_store.add_episode(kind="skill", payload={"a": 1}, outcome="ok")
        tmp_memory_store.add_episode(kind="tool_call", payload={"b": 2}, outcome="ok")
        tmp_memory_store.add_episode(kind="skill", payload={"c": 3}, outcome="failed")
        recent = tmp_memory_store.recent_episodes(10, kind="skill")
        assert len(recent) == 2
        assert {e.payload.get("a") or e.payload.get("c") for e in recent} == {1, 3}

    def test_render_for_prompt(self, tmp_memory_store):
        tmp_memory_store.add_fact("name", "Anh")
        tmp_memory_store.set_preference("voice", "concise")
        tmp_memory_store.add_episode(kind="skill", payload={"a": "grasp"}, outcome="failed")
        text = tmp_memory_store.render_for_prompt()
        assert "## Known facts" in text
        assert "name" in text and "Anh" in text
        assert "## Known preferences" in text
        assert "## Recent outcomes" in text
        assert "grasp" in text and "failed" in text

    def test_empty_render_is_empty(self, tmp_memory_store):
        assert tmp_memory_store.render_for_prompt() == ""

    def test_episode_ring_buffer_bounds(self, tmp_memory_store):
        # Push more than 2 * max_episodes; stored should compact back to max.
        max_ep = tmp_memory_store._max_episodes
        for i in range(max_ep * 3):
            tmp_memory_store.add_episode(kind="k", payload={"i": i}, outcome="ok")
        tmp_memory_store.flush()
        # Can't inspect length directly easily; check recent wraps cleanly.
        recent = tmp_memory_store.recent_episodes(n=5)
        assert len(recent) == 5
        assert recent[-1].payload["i"] == max_ep * 3 - 1


# ---------------------------------------------------------------------------
# JSONSessionStore
# ---------------------------------------------------------------------------


class TestJSONSessionStore:
    def test_save_and_load(self, tmp_session_store):
        s = Session(messages=[{"role": "user", "content": "hi"}], state={"k": 1})
        tmp_session_store.save("abc", s)
        loaded = tmp_session_store.load("abc")
        assert loaded is not None
        assert loaded.messages == [{"role": "user", "content": "hi"}]
        assert loaded.state == {"k": 1}

    def test_load_missing(self, tmp_session_store):
        assert tmp_session_store.load("nothing") is None

    def test_list_ids(self, tmp_session_store):
        tmp_session_store.save("a", Session())
        tmp_session_store.save("b", Session())
        assert set(tmp_session_store.list_ids()) == {"a", "b"}

    def test_delete(self, tmp_session_store):
        tmp_session_store.save("x", Session())
        assert tmp_session_store.delete("x") is True
        assert tmp_session_store.delete("x") is False

    def test_unserializable_state_coerced(self, tmp_session_store):
        class Weird:
            def __str__(self):
                return "weird-obj"

        s = Session(messages=[], state={"k": Weird()})
        tmp_session_store.save("x", s)  # must not raise
        # The loaded value is a string representation.
        loaded = tmp_session_store.load("x")
        assert loaded is not None
        assert loaded.state["k"] == "weird-obj"


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


class TestCompactor:
    def test_no_op_under_threshold(self):
        c = Compactor(trigger_tokens=100_000)
        msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "short"}]
        out = c.compact(msgs, llm=None)
        assert out == msgs

    def test_should_compact_detects_long(self):
        c = Compactor(trigger_tokens=100, keep_last_turns=2)
        msgs = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": "x" * 1000} for _ in range(6)]
        assert c.should_compact(msgs)

    def test_compact_preserves_system_and_last_turns(self):
        c = Compactor(trigger_tokens=100, keep_last_turns=2)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}" + ("x" * 200)})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        compacted = c.compact(msgs, llm=None)
        # First is still system.
        assert compacted[0] == msgs[0]
        # Last 2 turns preserved verbatim.
        assert compacted[-1] == msgs[-1]
        assert compacted[-2] == msgs[-2]
        # A summary assistant message was inserted.
        middle = compacted[1:-2]
        assert any("summary" in (m.get("content") or "").lower() for m in middle)

    def test_compact_uses_llm_when_provided(self):
        class StubLLM:
            def complete(self, messages, *, max_tokens=256, temperature=0.3, **kwargs):
                return {"choices": [{"message": {"content": "(fake summary)"}}]}

        c = Compactor(trigger_tokens=50, keep_last_turns=1)
        msgs = [{"role": "system", "content": "s"}]
        for _ in range(5):
            msgs.append({"role": "user", "content": "x" * 100})
            msgs.append({"role": "assistant", "content": "y"})
        out = c.compact(msgs, llm=StubLLM())
        assert any("(fake summary)" in (m.get("content") or "") for m in out)

    def test_llm_failure_falls_back(self):
        class BrokenLLM:
            def complete(self, *a, **kw):
                raise RuntimeError("boom")

        c = Compactor(trigger_tokens=50, keep_last_turns=1)
        msgs = [{"role": "system", "content": "s"}]
        for _ in range(5):
            msgs.append({"role": "user", "content": "x" * 100})
            msgs.append({"role": "assistant", "content": "y"})
        out = c.compact(msgs, llm=BrokenLLM())
        # Still produced a valid compacted list, not crashed.
        assert out[0] == msgs[0]
        assert len(out) < len(msgs)


def test_estimate_tokens_grows_with_content():
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "hi" * 1000}]
    assert estimate_tokens(small) < estimate_tokens(large)


# ---------------------------------------------------------------------------
# NotesFile
# ---------------------------------------------------------------------------


class TestNotesFile:
    def test_append_and_read(self, tmp_notes):
        tmp_notes.append("first note")
        tmp_notes.append("second note", heading="Session 1")
        content = tmp_notes.read()
        assert "first note" in content
        assert "second note" in content
        assert "## Session 1" in content

    def test_tail(self, tmp_notes):
        tmp_notes.append("a" * 500)
        tmp_notes.append("b" * 500)
        tail = tmp_notes.tail(200)
        assert len(tail) <= 200
        # End of the notes file
        assert "b" in tail

    def test_clear(self, tmp_notes):
        tmp_notes.append("x")
        tmp_notes.clear()
        assert tmp_notes.read() == ""

    def test_read_missing(self, tmp_notes):
        assert tmp_notes.read() == ""
