"""Reply sanitisation for Rook.

Three cleanups run at :data:`AFTER_LLM` before the reply reaches TTS:

- :class:`ThinkTagStripHook` — removes ``<think>...</think>`` blocks
  (and truncated ``<think>`` prefixes when the model ran out of tokens
  before closing the tag). Required for Qwen3-family models which
  default to reasoning mode — without stripping, the robot literally
  reads the model's internal monologue out loud.
- :class:`VoiceCleanupHook` — strips markdown artefacts (asterisks,
  backticks, leading list dashes) that TTS engines spell out.
- :class:`SentenceClipHook` — trims the reply to the first N sentences
  so the SLM can't run on past its point. Small models tend to repeat
  or template-loop when they overshoot; capping their output defends
  against that without changing generation parameters.

All hooks are idempotent and safe to stack — if nothing matches they
return ``None`` and the payload flows through unchanged.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from edgevox.agents.hooks import AFTER_LLM, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext


# ``<think>...</think>`` — greedy-match spans, but also handle the
# truncated case where the model never emits a closing tag (ran out of
# tokens). In that case we drop everything from ``<think>`` to EOF,
# which would otherwise be spoken as stream-of-consciousness.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRUNCATED_RE = re.compile(r"<think>.*$", re.DOTALL)

# Markdown artefacts a TTS engine reads verbatim.
_MARKDOWN_RE = re.compile(
    r"("
    r"\*\*|"  # bold
    r"__|"  # bold alt
    r"`+|"  # inline code / fences
    r"^\s*[-*+]\s|"  # list bullet at line start
    r"^\s*#+\s"  # header at line start
    r")",
    re.MULTILINE,
)

# Matched pairs of single ``*foo*`` / ``_foo_`` (italic/emphasis) —
# removed after the block patterns above so we don't eat bold ``**``.
# Keep the inner text; drop the asterisks/underscores.
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^\*\n]+?)(?<!\s)\*(?!\*)")
_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!\s)([^_\n]+?)(?<!\s)_(?!_)")

# Paired outer quotes the LLM sometimes wraps its whole reply in —
# both ASCII and typographic variants. Small instruction-tuned models
# (Llama-3.2-1B notably) learn this from training data that formats
# assistant turns as ``Assistant: "..."``. TTS reads the quote mark
# as literal "quote ... unquote" which sounds broken.
_OUTER_QUOTE_PAIRS = (
    ('"', '"'),
    ("\u201c", "\u201d"),  # U+201C/U+201D curly double
    ("\u2018", "\u2019"),  # U+2018/U+2019 curly single
    ("'", "'"),
    ("\u00ab", "\u00bb"),  # U+00AB/U+00BB guillemets
)


# If the model's entire output was a <think> block, we substitute a
# short persona-agnostic filler so the speaker isn't mute. Random pick
# prevents the same phrase twice in a row when the model stalls on
# consecutive turns.
_FALLBACK_FILLERS = (
    "Hmm, let me look at that.",
    "Interesting position.",
    "Your move.",
    "Let's see how that plays.",
    "I see what you're doing.",
    "Okay, let's continue.",
)


class ThinkTagStripHook:
    """Remove ``<think>…</think>`` reasoning blocks before TTS.

    Qwen3 defaults to reasoning mode; even with ``/no_think`` in the
    system prompt, some prompts occasionally still emit them. This
    hook is a belt-and-braces cleanup at ``AFTER_LLM`` so nothing
    leaks to the speaker.
    """

    points = frozenset({AFTER_LLM})
    priority = 70  # run early in AFTER_LLM so other hooks see clean content

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        if "<think>" not in content:
            return None
        cleaned = _THINK_BLOCK_RE.sub("", content)
        cleaned = _THINK_TRUNCATED_RE.sub("", cleaned)
        cleaned = cleaned.strip()
        if cleaned == content.strip():
            return None
        if not cleaned:
            # Model produced only a think block — substitute a short
            # filler so the robot doesn't go silent. Random pick avoids
            # "Let me think about that one" appearing twice in a row.
            cleaned = random.choice(_FALLBACK_FILLERS)
        new_payload = dict(payload)
        new_payload["content"] = cleaned
        return HookResult.replace(new_payload, reason="stripped <think> block")


class VoiceCleanupHook:
    """Strip markdown artefacts that TTS engines read literally.

    Keeps commentary readable-aloud: bold, italic, backticks, list
    bullets, headers. Tool calls are unaffected — we only touch
    ``content``.
    """

    points = frozenset({AFTER_LLM})
    priority = 65

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        if not content:
            return None
        cleaned = _MARKDOWN_RE.sub("", content)
        cleaned = _ITALIC_RE.sub(r"\1", cleaned)
        cleaned = _UNDERSCORE_ITALIC_RE.sub(r"\1", cleaned)
        cleaned = _strip_outer_quotes(cleaned.strip())
        if cleaned == content.strip():
            return None
        new_payload = dict(payload)
        new_payload["content"] = cleaned
        return HookResult.replace(new_payload, reason="markdown stripped for TTS")


def _strip_outer_quotes(text: str) -> str:
    """Peel one layer of whole-reply quotes if present.

    Idempotent and conservative: only strips when the opener/closer
    form a recognised pair AND the whole payload sits between them.
    A reply that *contains* quoted dialogue inside (e.g.
    ``The book says "nice try"``) is left alone because the opener
    and closer don't bracket the entire string.
    """
    if len(text) < 2:
        return text
    for open_q, close_q in _OUTER_QUOTE_PAIRS:
        if text.startswith(open_q) and text.endswith(close_q):
            inner = text[len(open_q) : len(text) - len(close_q)].strip()
            # Bail if the inner still contains the same opener — means
            # the outer pair is actually two adjacent quoted sections,
            # not a wrapper.
            if inner and open_q not in inner:
                return inner
    return text


class BriefingLeakGuard:
    """Drop any ``[CHESS BRIEFING ...]`` preamble the model parrots back.

    Small models (1-2B) occasionally regurgitate the system-role
    briefing verbatim instead of speaking in persona. The block is
    internal context — if it reaches TTS the user hears a wall of FEN /
    eval / PV noise that makes the robot sound broken.

    Strip anything from the first ``[CHESS BRIEFING`` marker up to and
    including ``[END BRIEFING]`` (or end-of-string if the closing
    marker is missing). If the stripped result is empty we substitute a
    filler so Rook still says *something*.
    """

    points = frozenset({AFTER_LLM})
    priority = 68  # run between ThinkTagStrip (70) and VoiceCleanup (65)

    _BRIEFING_RE = re.compile(r"\[CHESS BRIEFING.*?(?:\[END BRIEFING\]|\Z)", re.DOTALL)

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        if "[CHESS BRIEFING" not in content:
            return None
        cleaned = self._BRIEFING_RE.sub("", content).strip()
        if not cleaned:
            cleaned = random.choice(_FALLBACK_FILLERS)
        if cleaned == content.strip():
            return None
        return HookResult.replace(
            {**payload, "content": cleaned},
            reason="stripped leaked chess briefing",
        )


class SentenceClipHook:
    """Clip the reply to the first N sentences.

    Small models (1-2B) often run on past their useful output,
    template-looping or repeating stock phrases ("I'll bring my bishop
    to c5 — solid" every turn). This hook bounds the damage: keep the
    first ``max_sentences`` sentences and drop the rest. The cutoff
    sits in the ``AFTER_LLM`` tier after think / markdown scrubbing
    (priority 50 so it runs last), so we always clip the clean text.

    A "sentence" ends at ``.``, ``!``, or ``?`` followed by whitespace
    or end-of-string. We keep the terminator so the clip reads
    naturally.
    """

    points = frozenset({AFTER_LLM})
    priority = 50

    # Match a sentence terminator followed by whitespace OR end of string.
    _SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, *, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = (payload.get("content") or "").strip()
        if not content:
            return None
        parts = self._SPLIT_RE.split(content)
        if len(parts) <= self.max_sentences:
            return None
        clipped = " ".join(parts[: self.max_sentences]).strip()
        if clipped == content:
            return None
        return HookResult.replace(
            {**payload, "content": clipped},
            reason=f"clipped to {self.max_sentences} sentences",
        )


__all__ = [
    "BriefingLeakGuard",
    "SentenceClipHook",
    "ThinkTagStripHook",
    "VoiceCleanupHook",
]
