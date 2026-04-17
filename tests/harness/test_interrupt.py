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
    def test_triggers_when_energy_and_tts(self):
        ic = InterruptController()
        playing = {"state": True}
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: playing["state"], frame_ms=100)

        # 3 frames of high energy (300 ms at 100ms/frame) → should trigger on the 3rd frame.
        frames = [[0.5] * 160 for _ in range(5)]  # much higher than default 0.02 threshold
        watcher.run(iter(frames))
        assert ic.should_stop()

    def test_does_not_trigger_when_tts_idle(self):
        ic = InterruptController()
        watcher = EnergyBargeInWatcher(ic, is_tts_playing=lambda: False)
        frames = [[0.5] * 160] * 10
        watcher.run(iter(frames))
        assert not ic.should_stop()

    def test_brief_noise_does_not_trigger(self):
        ic = InterruptController()
        # Very short burst under min_duration_ms (300)
        watcher = EnergyBargeInWatcher(
            ic,
            is_tts_playing=lambda: True,
            frame_ms=100,
        )
        frames = [[0.5] * 160, [0.0] * 160, [0.0] * 160]  # 100ms burst only
        watcher.run(iter(frames))
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
