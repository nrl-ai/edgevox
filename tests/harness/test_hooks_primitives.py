"""Unit tests for the hook primitive surface (no LLMAgent integration)."""

from __future__ import annotations

import pytest

from edgevox.agents.hooks import (
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_TOOL,
    FIRE_POINTS,
    ON_RUN_START,
    HookRegistry,
    HookResult,
    ToolCallRequest,
    fire_chain,
    hook,
)


class TestHookResult:
    def test_cont_is_continue(self):
        r = HookResult.cont()
        assert r.is_continue
        assert not r.is_modify
        assert not r.is_end

    def test_replace_is_modify(self):
        r = HookResult.replace("new")
        assert r.is_modify
        assert r.payload == "new"

    def test_end_is_end_turn(self):
        r = HookResult.end("bye", reason="done")
        assert r.is_end
        assert r.payload == "bye"
        assert r.reason == "done"


class TestHookDecorator:
    def test_three_arg_signature(self):
        called = []

        @hook(ON_RUN_START)
        def h(point, ctx, payload):
            called.append((point, payload))
            return None

        h("on_run_start", None, {"task": "hi"})
        assert called == [("on_run_start", {"task": "hi"})]

    def test_two_arg_signature(self):
        called = []

        @hook(ON_RUN_START)
        def h(ctx, payload):
            called.append(payload)
            return None

        h("on_run_start", None, {"task": "hi"})
        assert called == [{"task": "hi"}]

    def test_unknown_point_raises(self):
        with pytest.raises(ValueError, match="Unknown hook point"):

            @hook("not_a_point")
            def h(ctx, payload):
                return None

    def test_bad_arity_raises(self):
        with pytest.raises(TypeError, match="must take"):

            @hook(ON_RUN_START)
            def h(payload):
                return None

    def test_no_points_raises(self):
        with pytest.raises(ValueError, match="at least one"):

            @hook()
            def h(ctx, payload):
                return None

    def test_multiple_points(self):
        seen = []

        @hook(ON_RUN_START, BEFORE_TOOL)
        def h(point, ctx, payload):
            seen.append(point)

        h("on_run_start", None, {})
        h("before_tool", None, None)
        assert seen == ["on_run_start", "before_tool"]


class TestHookRegistry:
    def test_empty_registry_returns_continue(self):
        reg = HookRegistry()
        r = reg.fire(ON_RUN_START, None, {"task": "x"})
        assert r.is_continue
        assert r.payload == {"task": "x"}

    def test_register_requires_points(self):
        reg = HookRegistry()

        class Bad:
            pass

        with pytest.raises(TypeError):
            reg.register(Bad())

    def test_register_rejects_unknown_point(self):
        reg = HookRegistry()

        class Hook:
            points = frozenset({"nope"})

            def __call__(self, point, ctx, payload):
                return None

        with pytest.raises(ValueError):
            reg.register(Hook())

    def test_single_hook_modify(self):
        @hook(ON_RUN_START)
        def double(ctx, payload):
            return HookResult.replace({"task": payload["task"] * 2})

        reg = HookRegistry([double])
        r = reg.fire(ON_RUN_START, None, {"task": "ab"})
        assert r.is_modify
        assert r.payload == {"task": "abab"}

    def test_chain_modify(self):
        @hook(ON_RUN_START)
        def up(ctx, payload):
            return HookResult.replace({"task": payload["task"].upper()})

        @hook(ON_RUN_START)
        def bang(ctx, payload):
            return HookResult.replace({"task": payload["task"] + "!"})

        reg = HookRegistry([up, bang])
        r = reg.fire(ON_RUN_START, None, {"task": "hi"})
        assert r.is_modify
        assert r.payload == {"task": "HI!"}

    def test_end_short_circuits_subsequent(self):
        call_log = []

        @hook(ON_RUN_START)
        def first(ctx, payload):
            call_log.append("first")
            return HookResult.end("stopped", reason="first did it")

        @hook(ON_RUN_START)
        def second(ctx, payload):
            call_log.append("second")
            return None

        reg = HookRegistry([first, second])
        r = reg.fire(ON_RUN_START, None, {})
        assert r.is_end
        assert call_log == ["first"]

    def test_exception_in_hook_is_swallowed(self, caplog):
        import logging

        caplog.set_level(logging.ERROR)

        @hook(ON_RUN_START)
        def broken(ctx, payload):
            raise RuntimeError("boom")

        @hook(ON_RUN_START)
        def ok(ctx, payload):
            return HookResult.replace({"task": "safe"})

        reg = HookRegistry([broken, ok])
        r = reg.fire(ON_RUN_START, None, {"task": "orig"})
        assert r.is_modify
        assert r.payload == {"task": "safe"}

    def test_fire_unknown_point_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError):
            reg.fire("nope", None, None)

    def test_extend_with_registry(self):
        @hook(ON_RUN_START)
        def a(ctx, payload):
            return None

        @hook(BEFORE_LLM)
        def b(ctx, payload):
            return None

        r1 = HookRegistry([a])
        r2 = HookRegistry([b])
        r1.extend(r2)
        assert len(r1) == 2
        assert ON_RUN_START in r1
        assert BEFORE_LLM in r1

    def test_at_returns_registered(self):
        @hook(BEFORE_TOOL)
        def a(ctx, payload):
            return None

        reg = HookRegistry([a])
        assert len(reg.at(BEFORE_TOOL)) == 1
        assert reg.at(AFTER_TOOL) == []

    def test_copy_is_independent(self):
        @hook(ON_RUN_START)
        def a(ctx, payload):
            return None

        r1 = HookRegistry([a])
        r2 = r1.copy()

        @hook(ON_RUN_START)
        def b(ctx, payload):
            return None

        r2.register(b)
        assert len(r1) == 1
        assert len(r2) == 2


class TestToolCallRequest:
    def test_default_fields(self):
        req = ToolCallRequest(name="foo", arguments={}, hop=0)
        assert req.name == "foo"
        assert req.hop == 0
        assert req.skip_dispatch is False
        assert req.synthetic_result is None
        assert req.is_skill is False

    def test_mutable_for_hooks(self):
        req = ToolCallRequest(name="foo", arguments="{}", hop=0)
        req.skip_dispatch = True
        req.synthetic_result = {"ok": False}
        assert req.skip_dispatch is True


class TestFireChain:
    def test_two_registries_both_modify(self):
        @hook(ON_RUN_START)
        def a(ctx, payload):
            return HookResult.replace({"task": payload["task"] + "-a"})

        @hook(ON_RUN_START)
        def b(ctx, payload):
            return HookResult.replace({"task": payload["task"] + "-b"})

        r1 = HookRegistry([a])
        r2 = HookRegistry([b])
        r = fire_chain([r1, r2], ON_RUN_START, None, {"task": "x"})
        assert r.is_modify
        assert r.payload == {"task": "x-a-b"}

    def test_end_in_first_skips_second(self):
        called = []

        @hook(ON_RUN_START)
        def a(ctx, payload):
            called.append("a")
            return HookResult.end("nope")

        @hook(ON_RUN_START)
        def b(ctx, payload):
            called.append("b")
            return None

        r = fire_chain([HookRegistry([a]), HookRegistry([b])], ON_RUN_START, None, {})
        assert r.is_end
        assert called == ["a"]

    def test_none_registry_skipped(self):
        @hook(ON_RUN_START)
        def a(ctx, payload):
            return HookResult.replace("ok")

        r = fire_chain([None, HookRegistry([a]), None], ON_RUN_START, None, "init")
        assert r.is_modify
        assert r.payload == "ok"


def test_fire_points_coverage():
    """Hardcoded list sanity — must match the 6 canonical points."""
    assert {
        "on_run_start",
        "before_llm",
        "after_llm",
        "before_tool",
        "after_tool",
        "on_run_end",
    } == FIRE_POINTS
