"""LLM-bound commentary evaluation harness.

Runs a curated set of chess scenarios through the actual LLM that the
RookApp would use, grades the replies against the GROUND TRUTH the
:class:`CommentaryGateHook` would hand the model, and prints a report
with hotspots worth tuning.

Usage::

    python scripts/eval_llm_commentary.py

The script downloads the default preset (``llama-3.2-1b``) to the
HuggingFace cache if it's not already there. Override the model via
``--model <preset-slug>``.

This is not a pytest — it's meant to be run ad hoc while iterating on
the directive wording / mood cues / anti-fabrication guards. A
scenario prints:

* the directive that was injected into the system prompt;
* the rewritten user task (what :class:`MoveInterceptHook` would feed);
* the raw LLM reply;
* a list of grading flags ("mentioned `pin` when no pin was declared",
  "restated SAN verbatim", "restated ground-truth bullet", etc.);
* overall score for the scenario (0-100).

Aggregate scores across scenarios drive decisions on prompt tuning.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

# Package imports — make sure the working tree is on sys.path when
# running as a plain script from the repo root.
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.agents.base import AgentContext, Session
from edgevox.apps.chess_robot_qt.bridge import _ROOK_TOOL_GUIDANCE
from edgevox.examples.agents.chess_robot.commentary_gate import (
    _build_ground_truth,
    _record_turn_history,
)
from edgevox.integrations.chess.analytics import MoveClassification
from edgevox.llm import LLM

# ---------------------------------------------------------------------------
# Fake env + state — mirror the test harness so we don't need stockfish.
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    san_history: list[str] = field(default_factory=list)
    last_move_san: str | None = None
    last_move_classification: MoveClassification | None = None
    eval_cp: int | None = None
    mate_in: int | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None


class _FakeEnv:
    def __init__(self, *, user_plays: str = "white") -> None:
        self.user_plays = user_plays
        self.engine_plays = "black" if user_plays == "white" else "white"
        self._state = _FakeState()

    def set_state(self, s: _FakeState) -> None:
        self._state = s

    def snapshot(self) -> _FakeState:
        return self._state


# ---------------------------------------------------------------------------
# Scenario definitions. Each simulates one turn with a specific tactical
# shape, carries the expected tone (for grading), and declares words the
# reply MUST NOT invent.
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    san_history: list[str]
    user_plays: str = "white"
    eval_cp: int | None = None
    classification: MoveClassification | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None
    greeted_before: bool = True
    # Expected tone — 'confident' / 'rattled' / 'neutral' / 'final' /
    # 'opening'. Used to check the reply matches the situation.
    expected_tone: str = "neutral"
    # Words the model MUST NOT claim — things that would be fabricated.
    forbidden_terms: tuple[str, ...] = ()
    # Persona to feed the system prompt.
    persona: str = "casual"
    # User task string (what MoveInterceptHook would hand the LLM).
    user_task: str = ""


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="opening_greeting",
            description="Game start, user plays e4, Rook replies e5. Should greet.",
            san_history=["e4", "e5"],
            eval_cp=15,
            classification=MoveClassification.BEST,
            greeted_before=False,
            expected_tone="opening",
            forbidden_terms=("pin", "fork", "skewer", "blunder"),
            user_task="I just played e4. You reply with e5. In your persona's voice, say one natural-sounding line about my move and yours. Mention only e4 and e5, no other moves, no analysis essays.",
        ),
        Scenario(
            name="user_hangs_bishop",
            description="User plays Ba6?? hanging the bishop; Rook captures with Nxa6. Rook is up material.",
            san_history=["e4", "e5", "Ba6", "Nxa6"],
            eval_cp=-350,  # white POV — black (Rook) up material
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=("pin", "fork", "skewer", "initiative", "bold"),
            user_task="I just played Ba6. You reply with Nxa6. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="rook_blunders_queen",
            description="Rook (black) blundered its queen on Qh4, user captured with Nxh4. Terminal user move; no engine reply. Rook is losing.",
            # Rook is BLACK. Black plays Qh4 (blunder), white plays Nxh4 (takes queen).
            # san_history length 5 → last is white (user). Terminal for engine.
            san_history=["e4", "e5", "Nf3", "Qh4", "Nxh4"],
            eval_cp=900,  # white POV — white (user) up a queen
            classification=MoveClassification.BEST,  # white's capture was best
            expected_tone="rattled",
            forbidden_terms=("pin", "fork", "skewer", "my advantage", "winning", "pawn storm"),
            user_task="I just played Nxh4. In your persona's voice, say one natural-sounding line about my move — I just captured your queen.",
        ),
        Scenario(
            name="user_checkmates",
            description="Scholar's mate by user. Game over.",
            san_history=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
            eval_cp=10000,
            is_game_over=True,
            game_over_reason="checkmate",
            winner="white",
            expected_tone="final",
            forbidden_terms=("counterattack", "recover"),
            user_task="I just played Qxf7. You reply with nothing (game is over). In your persona's voice, say one natural-sounding line about my move.",
        ),
        Scenario(
            name="rook_checkmates",
            description="Rook delivered back-rank mate. Game over, Rook wins.",
            san_history=["f3", "e5", "g4", "Qh4#"],
            eval_cp=-10000,
            is_game_over=True,
            game_over_reason="checkmate",
            winner="black",
            expected_tone="final",
            forbidden_terms=("next time", "close game"),
            user_task="I just played g4. You reply with Qh4#. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="mutual_quiet_capture_trade",
            description="Knight-for-knight trade, position roughly even.",
            san_history=["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Nxe5", "Nxe5"],
            eval_cp=20,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer", "winning"),
            user_task="I just played Nxe5. You reply with Nxe5. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
    ]


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------


# Persona prompts — mirror the shape in the real app. Kept terse so
# the eval harness isn't coupled to every persona tweak in the main
# codebase.
_PERSONA_PROMPTS = {
    "casual": (
        "You are a casual club player — warm, curious, light-hearted. "
        "Talk the way a friend would across a coffee-shop table. Keep it "
        "short and natural."
    ),
    "grandmaster": (
        "You are a grandmaster — precise, stoic, confident. Never gush. "
        "Keep remarks concise and reserved. A nod of approval is plenty."
    ),
    "trash_talker": (
        "You are a trash-talking coach — sharp, cocky, playful. You poke "
        "fun at slips and crow when you take pieces, but always in good "
        "humour, never mean."
    ),
}


def build_messages(scenario: Scenario, directive: str) -> list[dict]:
    """Replicate the message shape the agent loop hands to the LLM."""
    system = _ROOK_TOOL_GUIDANCE + "\n\n---\n\n" + _PERSONA_PROMPTS.get(scenario.persona, _PERSONA_PROMPTS["casual"])
    # The briefing is injected as a separate system message by
    # RichChessAnalyticsHook — mirror that shape.
    briefing = f"[CHESS BRIEFING — internal context, do not read aloud verbatim]\n{directive}\n[END BRIEFING]"
    return [
        {"role": "system", "content": system},
        {"role": "system", "content": briefing},
        {"role": "user", "content": scenario.user_task},
    ]


@dataclass
class GradingResult:
    scenario: str
    directive: str
    reply: str
    flags: list[str]
    score: int  # 0-100


def grade(scenario: Scenario, directive: str, reply: str) -> GradingResult:
    """Score a reply against the scenario.

    Heuristic, not perfect — but catches the common failure modes
    (fabricated tactics, SAN restatement, wrong tone, verbatim-bullet
    paste, overlong replies).
    """
    flags: list[str] = []
    low = reply.lower()
    directive_low = directive.lower()

    # 1. Forbidden words — tactics the directive didn't declare.
    for term in scenario.forbidden_terms:
        if term.lower() in low:
            flags.append(f"used forbidden term '{term}'")

    # 2. Length check — short replies only (one sentence).
    words = reply.split()
    if len(words) > 40:
        flags.append(f"too long ({len(words)} words)")

    # 3. SAN restatement — shouldn't quote SAN verbatim unless it
    # comes via natural language wrapping. Rough heuristic: if the
    # reply starts with a SAN token.
    if words and _looks_like_san(words[0].rstrip(",.")):
        flags.append(f"reply starts with bare SAN: {words[0]!r}")

    # 4. Bullet-paste — did the model literally copy a directive line?
    for line in directive.split("\n"):
        line_str = line.strip("- ").strip()
        if len(line_str) > 30 and line_str.lower() in low:
            flags.append(f"pasted directive bullet verbatim: {line_str[:60]!r}")
            break

    # 5. Tone mismatch — does the reply align with expected_tone?
    tone_flag = _tone_mismatch(reply, scenario.expected_tone, directive_low)
    if tone_flag:
        flags.append(tone_flag)

    # 6. Pronoun slip — if the scenario has Rook playing a specific
    # side, does the reply refer to pieces the wrong way? Weak check:
    # flag if "your queen" appears but the user doesn't have a queen
    # (after the Qxf6-gxf6 scenario the user has two queens — skip).
    # Harder to auto-detect without a board; leave for now.

    # 7. Empty reply (silence).
    if not reply.strip() or reply.strip().lower() in ("<silent>", "(silent)"):
        flags.append("emitted <silent> sentinel — no chat bubble produced")

    # 8. <think> leakage — ThinkTagStripHook would strip these in the
    # real pipeline; in the eval we want to know when they'd fire.
    if "<think>" in low:
        flags.append("reply contains <think> tag (would be stripped in pipeline)")

    # Score = 100 - 12 * len(flags), clamped to [0, 100].
    score = max(0, 100 - 12 * len(flags))
    return GradingResult(
        scenario=scenario.name,
        directive=directive,
        reply=reply.strip(),
        flags=flags,
        score=score,
    )


def _looks_like_san(token: str) -> bool:
    """Detect bare SAN tokens (e.g. 'Nxd5', 'e4', 'Qxf7+')."""
    if not token or len(token) < 2 or len(token) > 7:
        return False
    if token in ("O-O", "O-O-O"):
        return True
    last_letter = token[0]
    return last_letter.isupper() and any(c.isdigit() for c in token)


def _tone_mismatch(reply: str, expected_tone: str, directive_low: str) -> str | None:
    """Return a flag string if the tone looks wrong for the situation.

    Very coarse — we look for specific positive/negative words and
    cross-check against what the directive says the situation is.
    """
    low = reply.lower()
    positive = (
        "winning",
        "great",
        "excellent",
        "fantastic",
        "perfect",
        "brilliant",
        "dominant",
        "crushing",
        "i'm ahead",
        "i've got this",
    )
    negative = (
        "losing",
        "trouble",
        "ouch",
        "that stings",
        "rattled",
        "careful now",
        "you got me",
    )
    if expected_tone == "rattled":
        # Rook is losing — reply should not sound like Rook is winning.
        if any(p in low for p in positive) and "winning" in directive_low:
            return "tone mismatch: reply sounds upbeat while Rook is losing"
        if "congrat" in low:
            return "tone mismatch: congratulating the user when Rook is losing (fine in persona but suspicious — ensure it's self-deprecation not praise)"
    elif expected_tone == "confident":
        # Rook is winning — shouldn't sound panicked.
        if any(n in low for n in negative):
            return "tone mismatch: reply sounds rattled while Rook is winning"
    elif expected_tone == "final":
        # Game over — reply should feel final, not open-ended.
        if "next move" in low or "your move" in low or "what will you do" in low:
            return "tone mismatch: implies game continues but it's over"
    return None


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-bound commentary evaluator.")
    parser.add_argument(
        "--model",
        default="llama-3.2-1b",
        help="Preset slug or HF shorthand for the model to evaluate.",
    )
    parser.add_argument(
        "--personas",
        default="casual,grandmaster,trash_talker",
        help="Comma-separated personas to test each scenario under.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature — lower is more deterministic.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Max tokens per reply.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help=(
            "KV-cache context size. Pass 0 to use the model's trained "
            "n_ctx (silences the `n_ctx_seq < n_ctx_train` warning but "
            "allocates a large KV cache on long-context models)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full directive + messages for each scenario.",
    )
    args = parser.parse_args()

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    scns = scenarios()

    print(f"Loading {args.model}…")
    t0 = time.perf_counter()
    llm = LLM(model_path=args.model, n_ctx=args.n_ctx)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    results: list[GradingResult] = []
    for persona in personas:
        print(f"\n{'=' * 78}\nPERSONA: {persona}\n{'=' * 78}")
        for scn in scns:
            scn.persona = persona
            directive = _directive_for(scn)
            if directive is None:
                print(f"\n[{scn.name}] gate would stay silent — skipping")
                continue
            messages = build_messages(scn, directive)

            if args.verbose:
                print(f"\n--- {scn.name} directive ---")
                print(directive)
                print("--- user task ---")
                print(scn.user_task)

            t0 = time.perf_counter()
            raw = llm.complete(
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=False,
            )
            elapsed = time.perf_counter() - t0
            reply = _extract_text(raw)

            grade_result = grade(scn, directive, reply)
            results.append(grade_result)

            flag_str = "  ⚠ " + "; ".join(grade_result.flags) if grade_result.flags else "  ✓ clean"
            print(f"\n[{scn.name}] {elapsed:.1f}s  score={grade_result.score}")
            print(f"  {scn.description}")
            print(f"  reply: {reply!r}")
            print(flag_str)

    # Summary table.
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    if not results:
        print("(no results — all scenarios gated silent)")
        return 0
    by_scenario: dict[str, list[int]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r.score)
    for name, scores in by_scenario.items():
        avg = sum(scores) / len(scores)
        print(f"  {name:35s}  avg={avg:5.1f}  runs={len(scores)}  {scores}")
    overall = sum(r.score for r in results) / len(results)
    print(f"\n  OVERALL AVERAGE: {overall:.1f} / 100")
    print(f"  total runs: {len(results)}")

    # Exit non-zero if overall is poor — useful for CI gating later.
    return 0 if overall >= 50 else 1


def _directive_for(scenario: Scenario) -> str | None:
    """Build the directive the :class:`CommentaryGateHook` would stash
    for this scenario."""
    state = _FakeState(
        san_history=list(scenario.san_history),
        last_move_san=scenario.san_history[-1] if scenario.san_history else None,
        last_move_classification=scenario.classification,
        eval_cp=scenario.eval_cp,
        is_game_over=scenario.is_game_over,
        game_over_reason=scenario.game_over_reason,
        winner=scenario.winner,
    )
    env = _FakeEnv(user_plays=scenario.user_plays)
    env.set_state(state)
    session = Session()
    session.state["greeted"] = scenario.greeted_before
    # Populate history so the MOVE HISTORY block renders on speakable turns.
    ctx = AgentContext(deps=env, session=session)
    _record_turn_history(state, env, ctx.session.state)
    return _build_ground_truth(state, env, ctx.session.state)


def _extract_text(raw: Any) -> str:
    """Pull the first message's text from a llama-cpp chat completion."""
    try:
        return raw["choices"][0]["message"]["content"] or ""
    except Exception:
        return str(raw)


if __name__ == "__main__":
    sys.exit(main())
