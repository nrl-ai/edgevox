"""ThinkTagStripHook + VoiceCleanupHook — keep TTS output clean."""

from __future__ import annotations

from edgevox.agents.base import AgentContext
from edgevox.agents.hooks import AFTER_LLM
from edgevox.examples.agents.chess_robot.sanitize import (
    BriefingLeakGuard,
    SentenceClipHook,
    ThinkTagStripHook,
    VoiceCleanupHook,
)


def _run(hook, content: str) -> str:
    ctx = AgentContext()
    payload = {"content": content, "tool_calls": []}
    result = hook(AFTER_LLM, ctx, payload)
    if result is None:
        return content
    return result.payload["content"]


class TestThinkTagStripHook:
    def test_strips_closed_think_block(self):
        hook = ThinkTagStripHook()
        out = _run(hook, "<think>reasoning</think>Hello!")
        assert out == "Hello!"

    def test_strips_multiline_think_block(self):
        hook = ThinkTagStripHook()
        content = "<think>\nline1\nline2\n</think>\n\nYour move, e4."
        out = _run(hook, content)
        assert out == "Your move, e4."

    def test_strips_multiple_think_blocks(self):
        hook = ThinkTagStripHook()
        content = "<think>a</think>Mid<think>b</think>End"
        out = _run(hook, content)
        assert out == "MidEnd"

    def test_strips_unclosed_think_prefix_to_eof(self):
        """Qwen3 sometimes runs out of tokens before closing the tag.
        The whole trailing monologue must be removed or TTS reads it."""
        hook = ThinkTagStripHook()
        content = "<think>\nLet me consider this move carefully..."
        out = _run(hook, content)
        # Any fallback from the rotation is acceptable — the hard
        # guarantee is that the raw <think> content is gone.
        assert "consider this move" not in out
        assert "<think>" not in out
        assert out  # non-empty

    def test_pure_think_fallback_is_nonempty(self):
        """If the cleanup leaves nothing, substitute a short placeholder
        so TTS isn't silent. Any of the rotation phrases is acceptable."""
        hook = ThinkTagStripHook()
        content = "<think>hmm</think>"
        out = _run(hook, content)
        assert out
        assert "<think>" not in out

    def test_fallback_rotates(self):
        """Over many calls, the fallback should pick from the rotation
        (not always emit the same string)."""
        hook = ThinkTagStripHook()
        seen = {_run(hook, "<think>x</think>") for _ in range(60)}
        # The rotation has 6 options; in 60 rolls we should see >1 with
        # overwhelming probability (P(all same) = 6 * (1/6)^60).
        assert len(seen) > 1

    def test_no_think_passthrough(self):
        hook = ThinkTagStripHook()
        content = "Just playing e4 — let's see what you've got."
        assert _run(hook, content) == content


class TestVoiceCleanupHook:
    def test_strips_bold(self):
        hook = VoiceCleanupHook()
        out = _run(hook, "Your **queen** is hanging.")
        assert "**" not in out
        assert "queen" in out

    def test_strips_italic_asterisks(self):
        """Single-asterisk italic like ``*c5*`` must be stripped —
        otherwise TTS spells them out as 'asterisk c5 asterisk'."""
        hook = VoiceCleanupHook()
        out = _run(hook, "*c5* A solid opening.")
        assert "*" not in out
        assert "c5" in out

    def test_strips_italic_underscores(self):
        hook = VoiceCleanupHook()
        out = _run(hook, "That was _clever_ of you.")
        assert "_" not in out
        assert "clever" in out

    def test_preserves_inner_asterisk_punctuation(self):
        """Don't eat asterisks inside words or single-asterisk lines
        that aren't italic markers (e.g. censored output)."""
        hook = VoiceCleanupHook()
        out = _run(hook, "a * lone one")
        # Single lone asterisk with spaces around it shouldn't match
        # ``_ITALIC_RE`` (it requires non-space surrounds).
        assert "a * lone one" in out

    def test_strips_backticks(self):
        hook = VoiceCleanupHook()
        out = _run(hook, "Try `Nf3`.")
        assert "`" not in out
        assert "Nf3" in out

    def test_strips_list_bullet(self):
        hook = VoiceCleanupHook()
        out = _run(hook, "- first line\nsecond line")
        assert not out.startswith("-")

    def test_strips_heading(self):
        hook = VoiceCleanupHook()
        out = _run(hook, "# Opening\nI play e4.")
        assert "#" not in out

    def test_plain_text_passthrough(self):
        hook = VoiceCleanupHook()
        content = "I'll play e4. Your move."
        assert _run(hook, content) == content


class TestSentenceClipHook:
    def test_clips_to_two_sentences(self):
        hook = SentenceClipHook(max_sentences=2)
        out = _run(hook, "First. Second. Third. Fourth.")
        assert out == "First. Second."

    def test_passthrough_under_limit(self):
        hook = SentenceClipHook(max_sentences=2)
        out = _run(hook, "Only one here.")
        assert out == "Only one here."

    def test_handles_exclamation_and_question(self):
        hook = SentenceClipHook(max_sentences=2)
        out = _run(hook, "Nice! Great question? Third thing. Fourth.")
        assert out == "Nice! Great question?"

    def test_empty_passthrough(self):
        hook = SentenceClipHook(max_sentences=2)
        out = _run(hook, "")
        assert out == ""

    def test_configurable_limit(self):
        hook = SentenceClipHook(max_sentences=1)
        out = _run(hook, "One. Two. Three.")
        assert out == "One."

    def test_no_trailing_punctuation_no_split(self):
        """A blob with no terminators should pass through intact."""
        hook = SentenceClipHook(max_sentences=2)
        out = _run(hook, "one blob no punctuation here")
        assert out == "one blob no punctuation here"


class TestBriefingLeakGuard:
    def test_strips_full_bracketed_block(self):
        hook = BriefingLeakGuard()
        content = (
            "Good luck!\n"
            "[CHESS BRIEFING — internal context, do not read aloud verbatim]\n"
            "Position (FEN): rnbqkbnr/...\n"
            "[END BRIEFING]"
        )
        out = _run(hook, content)
        assert "[CHESS BRIEFING" not in out
        assert "[END BRIEFING]" not in out
        assert "Position (FEN):" not in out
        assert out.startswith("Good luck!")

    def test_strips_headerless_leak(self):
        """Reproduces the RookApp opening-move leak: model drops the
        ``[CHESS BRIEFING ...]`` marker but parrots the body and the
        closing marker. Guard must still catch it."""
        hook = BriefingLeakGuard()
        content = (
            "\"Hey, let's start a new game. I'm ready to play.\"\n"
            "You (Rook) are playing the BLACK pieces. The user is playing WHITE. "
            "Any sentence starting with 'You' / 'your' refers to Rook (black); "
            "any sentence about 'the user' refers to white.\n"
            "Position (FEN): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
            "To move: white — the user's\n"
            "Phase: opening\n"
            "Material: material is even\n"
            "Evaluation: evaluation unavailable\n"
            "King safety: white uncastled (king still on e1); black uncastled (king still on e8)\n"
            "Last move: no last move (starting position)\n"
            "Engine-top top line: e4 e5 Nf3\n"
            "[END BRIEFING]"
        )
        out = _run(hook, content)
        assert "You (Rook) are playing" not in out
        assert "Position (FEN):" not in out
        assert "[END BRIEFING]" not in out
        assert "ready to play" in out

    def test_strips_dangling_end_marker_only(self):
        hook = BriefingLeakGuard()
        out = _run(hook, "Your move. [END BRIEFING]")
        assert "[END BRIEFING]" not in out
        assert out.startswith("Your move.")

    def test_empty_result_gets_filler(self):
        hook = BriefingLeakGuard()
        content = "[CHESS BRIEFING]\nPosition (FEN): x\n[END BRIEFING]"
        out = _run(hook, content)
        assert out  # non-empty filler
        assert "[CHESS BRIEFING" not in out

    def test_no_briefing_passthrough(self):
        hook = BriefingLeakGuard()
        content = "I'll play the French. Your move."
        assert _run(hook, content) == content


class TestHookPointRegistration:
    def test_think_stripper_fires_only_after_llm(self):
        assert ThinkTagStripHook().points == {AFTER_LLM}

    def test_voice_cleanup_fires_only_after_llm(self):
        assert VoiceCleanupHook().points == {AFTER_LLM}

    def test_sentence_clip_fires_only_after_llm(self):
        assert SentenceClipHook().points == {AFTER_LLM}

    def test_think_stripper_runs_before_voice_cleanup(self):
        """If both are registered, ``<think>`` removal should happen
        before markdown scrubbing so a think block with markdown inside
        doesn't produce a half-cleaned result."""
        assert ThinkTagStripHook.priority > VoiceCleanupHook.priority

    def test_sentence_clip_runs_last(self):
        """Clip runs after think + markdown so we always clip clean text."""
        assert SentenceClipHook.priority < VoiceCleanupHook.priority
        assert SentenceClipHook.priority < ThinkTagStripHook.priority
