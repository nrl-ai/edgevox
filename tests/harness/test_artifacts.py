"""Tests for InMemoryArtifactStore and FileArtifactStore."""

from __future__ import annotations

import pytest

from edgevox.agents.artifacts import (
    FileArtifactStore,
    InMemoryArtifactStore,
    bytes_artifact,
    json_artifact,
    text_artifact,
)


@pytest.fixture(params=["mem", "file"])
def store(request, tmp_path):
    if request.param == "mem":
        return InMemoryArtifactStore()
    return FileArtifactStore(tmp_path / "art")


def test_text_roundtrip(store):
    a = text_artifact("plan.md", "# plan\nstep 1", author="planner", tags=["plan"])
    store.write(a)
    got = store.read("plan.md")
    assert got is not None
    assert got.content == "# plan\nstep 1"
    assert got.author == "planner"
    assert "plan" in got.tags


def test_json_roundtrip(store):
    a = json_artifact("state.json", {"x": 1, "y": [1, 2, 3]})
    store.write(a)
    got = store.read("state.json")
    assert got is not None
    assert got.content == {"x": 1, "y": [1, 2, 3]}


def test_bytes_roundtrip(store):
    a = bytes_artifact("blob", b"\x00\x01\x02")
    store.write(a)
    got = store.read("blob")
    assert got is not None
    assert got.content == b"\x00\x01\x02"


def test_list_and_delete(store):
    store.write(text_artifact("a", "1"))
    store.write(text_artifact("b", "2"))
    names = {a.name for a in store.list()}
    assert names == {"a", "b"}
    assert store.delete("a") is True
    assert {a.name for a in store.list()} == {"b"}


def test_list_by_tag(store):
    store.write(text_artifact("a", "1", tags=["plan"]))
    store.write(text_artifact("b", "2", tags=["result"]))
    plans = store.list(tag="plan")
    assert [a.name for a in plans] == ["a"]


def test_read_missing(store):
    assert store.read("nope") is None


def test_render_index_format(store):
    store.write(text_artifact("plan.md", "step 1\nstep 2", tags=["plan"], summary="two-step plan"))
    idx = store.render_index()
    assert "plan.md" in idx
    assert "two-step plan" in idx


def test_render_index_empty(store):
    assert store.render_index() == ""


def test_overwrite(store):
    store.write(text_artifact("a", "first"))
    store.write(text_artifact("a", "second"))
    got = store.read("a")
    assert got.content == "second"


def test_file_store_survives_reopen(tmp_path):
    store = FileArtifactStore(tmp_path / "art")
    store.write(text_artifact("persistent", "hello"))
    store2 = FileArtifactStore(tmp_path / "art")
    assert store2.read("persistent").content == "hello"
