"""Tests for InterruptController and EnergyBargeInWatcher."""

from __future__ import annotations

import threading
import time

from edgevox.agents.interrupt import (
    EnergyBargeInWatcher,
    InterruptController,
    InterruptPolicy,
)


class TestInterruptController:
    def test_initial_not_interrupted(self):
        ic = InterruptController()
        assert not ic.should_stop()
        assert ic.latest is None

    def test_trigger_sets_flag(self):
        ic = InterruptController()
        ev = ic.trigger("user_barge_in", rms=0.5)
        assert ic.should_stop()
        assert ev.reason == "user_barge_in"
        assert ev.meta["rms"] == 0.5
        assert ic.latest is ev

    def test_reset_clears_flag(self):
        ic = InterruptController()
        ic.trigger("manual")
        ic.reset()
        assert not ic.should_stop()
        # History retained.
        assert len(ic.history) == 1

    def test_cancel_token_set_on_trigger(self):
        """Barge-in must set the cancel_token so llama-cpp stopping
        criteria actually aborts. Without this the ``cancel_llm`` flag
        is only advisory."""
        ic = InterruptController()
        assert not ic.cancel_token.is_set()
        ic.trigger("user_barge_in")
        assert ic.cancel_token.is_set()

    def test_cancel_token_respects_policy(self):
        """``policy.cancel_llm = False`` must not set the cancel_token
        — some deployments want barge-in to cut TTS but let the LLM
        finish its reply silently for logging."""
        ic = InterruptController(InterruptPolicy(cancel_llm=False))
        ic.trigger("user_barge_in")
        assert ic.should_stop()
        assert not ic.cancel_token.is_set()

    def test_reset_clears_cancel_token_and_latest(self):
        """``reset()`` must clear both the ``interrupted`` flag and the
        ``cancel_token`` event; it must also drop ``latest`` so a
        stale event doesn't leak into the next turn. History is kept."""
        ic = InterruptController()
        ic.trigger("manual")
        assert ic.latest is not None
        ic.reset()
        assert not ic.cancel_token.is_set()
        assert ic.latest is None
        assert len(ic.history) == 1  # history kept

    def test_as_tool_result_returns_none_when_not_interrupted(self):
        ic = InterruptController()
        assert ic.as_tool_result() is None

    def test_as_tool_result_envelopes_latest_event(self):
        import json as _json

        ic = InterruptController()
        ic.trigger("user_barge_in", rms=0.5)
        msg = ic.as_tool_result(partial="wait, no—")
        assert msg is not None
        assert msg["role"] == "tool"
        assert msg["name"] == "__interrupt__"
        body = _json.loads(msg["content"])
        assert body["ok"] is False
        assert body["error"] == "interrupted_by_user"
        assert body["reason"] == "user_barge_in"
        assert body["partial"] == "wait, no—"
        assert body["meta"]["rms"] == 0.5

    def test_history_bounded(self):
        """Ring-buffer protects long voice sessions from unbounded
        memory growth via interrupt events."""
        ic = InterruptController(max_history=3)
        for i in range(10):
            ic.trigger("manual", i=i)
        hist = ic.history
        assert len(hist) == 3
        # Most recent kept.
        assert hist[-1].meta["i"] == 9

    def test_wait_returns_true_on_trigger(self):
        ic = InterruptController()

        def delayed():
            time.sleep(0.05)
            ic.trigger("manual")

        t = threading.Thread(target=delayed)
        t.start()
        result = ic.wait(timeout=1.0)
        t.join()
        assert result is True

    def test_wait_timeout_returns_false(self):
        ic = InterruptController()
        assert ic.wait(timeout=0.05) is False

    def test_subscribers_receive_events(self):
        ic = InterruptController()
        seen: list = []
        unsub = ic.subscribe(lambda ev: seen.append(ev.reason))
        ic.trigger("timeout")
        ic.trigger("user_cancel")
        assert seen == ["timeout", "user_cancel"]
        unsub()
        ic.trigger("manual")
        assert len(seen) == 2  # unsubscribed

    def test_subscriber_exception_doesnt_break(self, caplog):
        ic = InterruptController()

        def bad(_ev):
            raise RuntimeError("boom")

        ic.subscribe(bad)
        ic.trigger("manual")  # must not raise

    def test_history_accumulates(self):
        ic = InterruptController()
        ic.trigger("user_barge_in")
        ic.trigger("timeout")
        assert len(ic.history) == 2
        assert [e.reason for e in ic.history] == ["user_barge_in", "timeout"]


class TestInterruptPolicy:
    def test_default_policy_conservative(self):
        p = InterruptPolicy()
        assert p.cancel_llm is True
        assert p.cancel_skills is False  # don't abort mid-grasp
        assert p.cut_tts_immediately is True


class TestEnergyBargeInWatcher:
    def test_triggers_when_user_speaks_above_echo_floor(self):
        """Realistic flow: TTS prefix calibrates the echo floor (quiet
        ambient mic), then the user starts talking loudly enough to
        clear ``echo_suppression_ratio x floor``."""
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: True, frame_ms=100)

        # Prefix: 200 ms of low-RMS frames (echo only). Watcher uses
        # these to set echo_floor.
        prefix = [[0.02] * 160] * 2
        # User speech: 300 ms of clearly louder audio (well above
        # echo_floor x 2.0).
        speech = [[0.5] * 160] * 4
        watcher.run(iter(prefix + speech))
        assert ic.should_stop()

    def test_does_not_trigger_when_only_echo_present(self):
        """The whole TTS segment is at the same RMS — looks exactly
        like echo to the watcher; no real user speech, no trigger."""
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: True, frame_ms=100)
        # Constant moderate RMS — calibrated as floor, then never exceeded.
        frames = [[0.1] * 160 for _ in range(8)]
        watcher.run(iter(frames))
        assert not ic.should_stop()

    def test_does_not_trigger_when_tts_idle(self):
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: False)
        frames = [[0.5] * 160] * 10
        watcher.run(iter(frames))
        assert not ic.should_stop()

    def test_brief_noise_does_not_trigger(self):
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: True, frame_ms=100)
        # Quiet prefix to calibrate, then a 100 ms loud burst (under
        # min_duration_ms=250).
        frames = [[0.02] * 160] * 2 + [[0.5] * 160] + [[0.0] * 160] * 3
        watcher.run(iter(frames))
        assert not ic.should_stop()

    def test_tts_energy_provider_raises_threshold(self):
        """When the caller pipes in the live TTS RMS, the watcher
        scales its threshold so the user can't be drowned out — but
        also can't be fooled by loud TTS leak."""
        ic = InterruptController()
        # The TTS is "loud" (0.4 RMS) the whole time; the user is
        # equally loud (0.4) but shouldn't trigger because mic and
        # ref are matched.
        watcher = EnergyBargeInWatcher(
            ic,
            is_tts_playing=lambda: True,
            frame_ms=100,
            tts_energy_provider=lambda: 0.4,
        )
        # Even after prefix, mic (0.4) < tts_rms (0.4) x 2.0 = 0.8.
        prefix = [[0.02] * 160] * 2
        speech = [[0.4] * 160] * 4
        watcher.run(iter(prefix + speech))
        assert not ic.should_stop()

    def test_stop_signal_halts_watcher(self):
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: True, frame_ms=10)

        def gen():
            for _ in range(1000):
                yield [0.0] * 16
                time.sleep(0.001)

        t = threading.Thread(target=watcher.run, args=(gen(),), daemon=True)
        t.start()
        time.sleep(0.02)
        watcher.stop()
        t.join(timeout=1.0)
        assert not t.is_alive()


class TestCtxIntegration:
    def test_ctx_should_stop_honors_interrupt(self):
        from edgevox.agents.base import AgentContext

        ic = InterruptController()
        ctx = AgentContext(interrupt=ic)
        assert not ctx.should_stop()
        ic.trigger("manual")
        assert ctx.should_stop()

    def test_pending_interrupt_lands_in_next_run_messages(self):
        """A barge-in mid-skill on turn N must surface as a synthetic
        ``role="tool"`` message at the start of turn N+1 so the model
        has a coherent recovery cue."""
        from edgevox.agents.base import AgentContext, LLMAgent

        from .conftest import ScriptedLLM, reply

        ic = InterruptController()
        ic.trigger("user_barge_in", rms=0.7)

        llm = ScriptedLLM([reply("Got it, I stopped.")])
        agent = LLMAgent(name="t", description="", instructions="", llm=llm)
        ctx = AgentContext(interrupt=ic)
        agent.run("hi", ctx)

        # ScriptedLLM records the messages it saw on the first call.
        seen = llm.calls[0]["messages"]
        # Last message before the user turn should be the synthetic
        # ``__interrupt__`` envelope.
        roles = [m.get("role") for m in seen]
        assert "tool" in roles
        interrupt_msg = next(m for m in seen if m.get("role") == "tool")
        assert interrupt_msg["name"] == "__interrupt__"

    def test_no_pending_interrupt_does_not_inject(self):
        from edgevox.agents.base import AgentContext, LLMAgent

        from .conftest import ScriptedLLM, reply

        ic = InterruptController()
        # Never triggered.
        llm = ScriptedLLM([reply("ok")])
        agent = LLMAgent(name="t", description="", instructions="", llm=llm)
        agent.run("hi", AgentContext(interrupt=ic))

        seen = llm.calls[0]["messages"]
        assert all(m.get("role") != "tool" for m in seen)
