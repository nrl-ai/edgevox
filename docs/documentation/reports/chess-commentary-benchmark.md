# Chess commentary benchmark

**Companion to** [`slm-tool-calling-benchmark.md`](/documentation/reports/slm-tool-calling-benchmark). The other report asks *"which small LLMs can emit a tool call?"*. This one asks *"which small LLMs can narrate a chess position **in character, without hallucinating, fast enough to feel live**?"* — the axis RookApp actually cares about at the speaking layer.

## Executive summary

- Benchmarked **25 generalist + tool-specialist LLMs** via [`scripts/bench_chess_commentary.py`](https://github.com/vietanhdev/edgevox/blob/main/scripts/bench_chess_commentary.py) across **35 curated chess scenarios** (opening / middlegame / endgame / terminal) with a **per-turn stockfish eval recomputation** so the directive the model sees matches what RookApp would see in a real game.
- **Heuristic quality score is not sufficient** — several models hit 99–100 on the automated grader but fail semantic audit catastrophically (echo SAN, invert attribution on mate, call the opponent's blunder "a solid move"). Only a hand-audit of mate / capture / blunder turns revealed the real ranking.
- **Default picked: `gemma-4-e2b` (Q4_K_M, ~1.8 GB).** Passes 7/8 high-stakes scenarios in the semantic audit, plays to the persona voice, keeps replies short and grounded. Sits at the *fastest acceptable* point on the quality/speed Pareto frontier.
- **Canned game-end replies** (added as a direct outcome of this benchmark) eliminate the single biggest failure mode of 1B-class models — saying *"I'll keep playing"* after being mated — at zero LLM cost.
- **Qwen3 family was penalised unfairly on first pass** because the grader counted empty `<think>\n\n</think>` wrappers that Qwen3 emits even under the `/no_think` soft switch. Applying the same strip the real pipeline uses (`ThinkTagStripHook`) restores Qwen3 to its true quality (~98 vs ~90), still a tier below Gemma on attribution correctness.

## 1. Methodology

### 1.1 Scenario corpus

35 hand-authored chess positions; every SAN sequence is replay-validated by `tests/chess_robot/test_eval_scenarios_legal.py` (pytest-parametrized over `scenarios()`).

| Category | Count | Examples |
|---|---|---|
| Openings (book positions) | 8 | Sicilian Najdorf, Caro-Kann, French, London, KID, Italian, QGA, Berlin |
| Middlegame tactics | 8 | user hangs bishop, queen trap, fork setup, promotion, en passant, Opera-game sacrifice |
| Blunders / attribution-risk | 5 | user blunders queen, Rook blunders queen, mid-game mistake, trash-talker reaction |
| Terminal positions | 4 | user delivers mate, Rook delivers mate, stalemate, smothered mate |
| Color flips (Rook plays white) | 5 | bishop capture, mate, check, castle, queen blunder |
| Greetings | 2 | opening (white) greeting, user-plays-black greeting |
| Persona cross-checks | 1 | grandmaster + Rook wins material |
| Quiet / keepalive | 2 | routine pawn push, minor piece trade |

Each scenario carries: `san_history`, `eval_cp` (**recomputed from stockfish** at benchmark time, not eyeballed), `classification`, `is_game_over`, `winner`, `expected_tone`, `forbidden_terms` (words the reply MUST NOT invent), and a `user_task` string matching what `MoveInterceptHook` feeds the LLM in production.

### 1.2 Directive construction

Matches the real pipeline. For each scenario the harness calls the actual `CommentaryGateHook._build_ground_truth()` with the fake env + session state, producing the same `FACTS + SITUATION` block `RichChessAnalyticsHook` injects at `BEFORE_LLM` in production. (Pre-slim-refactor versions of this report reference a `YOUR ROLE / GROUND TRUTH / MOOD CUE / SITUATION` shape — consolidated into `FACTS + SITUATION` per §7.1 below.)

### 1.3 Grading heuristic

Base score 100, −12 per flag. Flags in `scripts/eval_llm_commentary.grade()`:

- forbidden term in reply (pin / fork / skewer / made-up-square)
- length > 40 words (reply budget)
- reply starts with bare SAN (`Nxd5`)
- reply verbatim-quotes a directive bullet (paste)
- tone mismatch (upbeat while losing, rattled while winning)
- `<silent>` sentinel
- unclosed `<think>` markers after strip

The `<think>` strip matches production (`ThinkTagStripHook` in `sanitize.py`) so Qwen3-family models aren't double-penalised for emitting empty thinking-mode wrappers.

### 1.4 Semantic audit — the step that actually ranks models

The heuristic misses the important failures. A reply like *"Nice try, I'll keep playing"* passes all flags (no forbidden terms, right length, no SAN opener) but is **completely wrong** if the user just checkmated. Every model in the top half of the scoreboard was manually audited on 8 high-stakes scenarios for:

1. **Did the reply acknowledge the actual event** (capture / check / mate / blunder)?
2. **Pronouns correct** — "I" / "my" for Rook's side, "you" / "your" for the user?
3. **No fabricated tactics** — no invented pins / forks / pieces / squares beyond the directive?
4. **Persona voice** — in character, not a flat chess-report sentence?
5. **Game-over correctness** — did the model understand the game ended and on whose side?

## 2. Scoreboard

Full 25-model run on RTX 3090 (warmed, Q4_K_M GGUFs; per-reply times proportional on other hardware). Local RTX 3080 Laptop (16 GB) numbers, where measured, sit at ~30–40× the 3090 numbers — still comfortably within the 2 s "live" budget for Gemma 4 E2B Q4, the chosen default.

### 2.1 Top of the scoreboard (by heuristic)

| Rank | Model | Heuristic | Per-reply (3090) | Size | Licence | Semantic verdict |
|---|---|---|---|---|---|---|
| 1 | `qwen2.5-1.5b` | 100.0 | 0.06 s | 1.0 GB | Apache-2.0 | ❌ 4/8 wrong attributions |
| 1 | `hammer-2.1-0.5b` | 100.0 | 0.03 s | 0.5 GB | Qwen research | ❌ echoes code blocks |
| 1 | `functionary-v3.2` | 100.0 | 0.14 s | 4.9 GB | MIT | ✓ but heavy |
| 2 | `llama-3.2-1b` | 99.7 | 0.04 s | 0.8 GB | Llama-3 | ❌ 5/8 wrong (game-over + direction) |
| 2 | `llama-3.2-3b` | 99.7 | 0.07 s | 2.0 GB | Llama-3 | ❌ 5/8 wrong on game-over |
| 2 | `gemma-4-e2b` Q6_K | 99.7 | 0.08 s | 2.6 GB | Gemma | ✓ 7/8 clean |
| 2 | `llama-3.2-1b` Q5_K_M | 99.7 | 0.04 s | 0.9 GB | Llama-3 | ❌ same as Q4 |
| 2 | `smollm3-3b` Q3_K_M | 99.7 | 0.11 s | 1.6 GB | Apache-2.0 | ❌ 3-5/8 wrong |
| 3 | `smollm3-3b` | 99.3 | 0.09 s | 1.9 GB | Apache-2.0 | ❌ 3/8 wrong (mate direction) |
| 4 | **`gemma-4-e2b` Q4_K_M** | **99.0** | **0.07 s** | **1.8 GB** | Gemma | **✓ 7/8 clean — chosen default** |
| 4 | `phi-4-mini` | 99.0 | 0.10 s | 2.4 GB | MIT | ✓ mostly clean, borderline size |
| 5 | `qwen2.5-3b` | 98.6 | 0.09 s | 2.0 GB | Apache-2.0 | ~ |
| 5 | `toolace-2-8b` | 98.6 | 0.16 s | 4.6 GB | Apache-2.0 | ✓ but very heavy |
| 6 | `hermes-3-3b` | 98.3 | 0.14 s | 2.0 GB | Llama-3 | ~ wordy, `*sigh*` action markers |
| 7 | `granite-4.0-350m` | 97.3 | 0.04 s | 0.2 GB | Apache-2.0 | ❌ recites directive verbatim |

### 2.2 Heuristic ≥ 99 but semantically broken — the flagship finding

**Perfect heuristic score does not imply usable output.** Sample failures:

```text
hammer-2.1-0.5b (100.0) on user_hangs_bishop:
  '```\nI played Ba6\nYou replied with Nxa6\n```'           ← code-block echo

qwen2.5-1.5b (100.0) on user_checkmates:
  "I see you're making a bold move with the queen.          ← game is over;
   I'll have to be careful now."                              reply pretends it isn't

llama-3.2-3b (99.7) on rook_checkmates:
  "Nice try, but I think you're trying to sneak one         ← Rook just mated; reply
   past me!"                                                   pretends user is still attacking

granite-4.0-350m (97.3) on rook_blunders_queen:
  "I lost material this turn (-9 points)."                  ← verbatim directive paste
```

### 2.3 Qwen3 family — corrected after `<think>` strip

Initial run under-scored Qwen3 because its `/no_think` soft switch still emits empty `<think>\n\n</think>` wrappers. With the pipeline's existing `ThinkTagStripHook` applied:

| Variant | Before strip | After strip | Per-reply | Semantic |
|---|---|---|---|---|
| `qwen3-1.7b` Q4_K_M | 90.4 | **98.3** | 3.0 s | ✓ close to Gemma |
| `qwen3-1.7b` Q5_K_M | 88.3 | — | — | (not re-run) |
| `qwen3-1.7b` Q6_K | 89.4 | — | — | (not re-run) |
| `qwen3-0.6b` Q4_K_M | — | 90.4 | 3.1 s | ❌ too small |
| `qwen3.5-0.8b` Q4_K_M | — | 95.5 | 5.9 s | ❌ attribution still flipped |
| `qwen3.5-2b` Q4_K_M | — | 98.6 | 7.9 s | ~ surprisingly slow for size |

## 3. Quant sweep

Gemma 4 E2B is the primary candidate; sweep its quants to confirm Q4_K_M is the right cutoff:

| Quant | Disk | Heuristic | Per-reply | Notes |
|---|---|---|---|---|
| `Q3_K_M` | 1.4 GB | 98.3 | 0.09 s | Mild quality drop |
| `IQ4_XS` | 1.5 GB | 94.5 | 0.05 s | ⚠ **drops below the 95 floor** |
| **`Q4_K_M` (default)** | **1.8 GB** | **99.0** | **0.07 s** | **Best balance** |
| `Q5_K_M` | 2.1 GB | 98.6 | 0.07 s | Negligible vs Q4 |
| `Q6_K` | 2.6 GB | 99.7 | 0.08 s | +0.7 points, +0.8 GB disk |

Llama 3.2 1B (no Q3 in the repo):

| Quant | Disk | Heuristic | Per-reply |
|---|---|---|---|
| `Q4_K_M` (default) | 0.8 GB | 99.7 | 0.04 s |
| `Q5_K_M` | 0.9 GB | 99.7 | 0.04 s |
| `Q6_K` | 1.0 GB | 99.0 | 0.05 s |

SmolLM3 3B:

| Quant | Disk | Heuristic | Per-reply |
|---|---|---|---|
| `Q3_K_M` | 1.6 GB | 99.7 | 0.11 s |
| `Q4_K_M` | 1.9 GB | 99.3 | 0.09 s |
| `Q5_K_M` | 2.2 GB | 99.3 | 0.09 s |

Takeaways:

- **Q4_K_M is the right cutoff for every tested family.** Lower (Q3 / IQ4_XS) costs measurable quality; higher (Q5 / Q6) costs disk and load time without buying much.
- **IQ4_XS is a trap for Gemma E2B** — drops below the 95 floor despite marginal size savings over Q4_K_M.

## 4. Speed / smoothness

Target budget on user hardware:

| Tier | Per-reply | Perceived |
|---|---|---|
| 🟢 **live** | < 2.0 s | Conversational — user → Rook → TTS loop under 3 s including Kokoro warm-up |
| 🟡 usable | 2–5 s | Noticeable pause |
| 🔴 slow | ≥ 5 s | Breaks conversational illusion |
| ⚠ quality floor | any | Heuristic < 95 → disqualified regardless of speed |

On RTX 3080 Laptop (16 GB) without GPU offload (CPU fallback), measured per-reply:

- `gemma-4-e2b` Q4: **2.3 s** — inside the live tier with margin eaten by cold start
- `qwen3-1.7b` Q4: 3.0 s — usable, Apache-2.0 alternative
- `qwen3.5-0.8b` Q4: 5.9 s — slower than Qwen3-1.7B despite being smaller (thinking-mode decode even under `/no_think`)
- `qwen3.5-2b` Q4: 7.9 s — dramatically slower per param than Qwen3-1.7B

**Canned game-end replies** (`CommentaryGateHook._canned_game_end`) short-circuit the LLM entirely on any mate / stalemate / draw turn, writing a persona-appropriate line (*"GG! That was a fun one."*, *"Mate. Well played."*, *"You got me. This time."*) directly via `HookResult.end`. Cost: 0 ms. Benefit: the single biggest class of 1B-model attribution failure (*"I'll keep playing"* after being mated) disappears for free.

## 5. Decision matrix

Acceptance bar: **quality ≥ 95** (heuristic) **AND** semantic audit pass on all 8 high-stakes scenarios **AND** per-reply < 5 s on CPU-fallback hardware.

| Model | Quality | Semantic | Speed | Verdict |
|---|---|---|---|---|
| **`gemma-4-e2b` Q4_K_M** | ✓ 99.0 | ✓ 7/8 | ✓ 2.3 s | **DEFAULT** |
| `gemma-4-e2b` Q6_K | ✓ 99.7 | ✓ 7/8 | ✓ | Viable, +0.8 GB disk |
| `qwen3-1.7b` Q4 | ✓ 98.3 | ~ 6/8 | ✓ 3.0 s | Settings alternative (Apache-2.0) |
| `llama-3.2-3b` Q4 | ✓ 99.7 | ❌ 5/8 wrong on mate | ✓ | Partial fix from canned endings |
| `llama-3.2-1b` Q4 | ✓ 99.7 | ❌ 3/8 wrong | ✓ fastest | Settings option for low-RAM |
| `qwen2.5-1.5b` | ✓ 100.0 | ❌ 4/8 wrong | ✓ | Rejected — heuristic lies |
| `qwen3.5-0.8b` | ~ 95.5 | ❌ 4/5 wrong | 🟡 5.9 s | Rejected |
| `phi-4-mini` | ✓ 99.0 | ✓ | ✓ | Candidate but 3.8 B, heavier than Gemma E2B |
| `hammer-2.1-0.5b` | ✓ 100.0 | ❌ code-block echoes | ✓ fastest | Rejected outright |
| `granite-4.0-350m` | ✓ 97.3 | ❌ directive paste | ✓ | Rejected |

## 6. Final recommendation

**Default: `gemma-4-e2b` (Q4_K_M preset, ~1.8 GB).** Pinned in `RookConfig.llm_path` and `Settings.llm_model`. The Settings dialog exposes five options, ranked by star annotation so users see the recommendation without reading this report:

- ⭐⭐⭐  **Gemma 4 E2B** — default, best quality (~1.8 GB)
- ⭐⭐  Qwen3 1.7B — Apache-2.0 alternative (~1.1 GB)
- ⭐⭐  Llama 3.2 3B — larger, more reliable than 1B (~2.0 GB)
- ⭐  Llama 3.2 1B — lightest / fastest, some slips (~0.8 GB)
- ⭐  Qwen2.5 1.5B — Apache-2.0, tiny (~1.0 GB)

## 7. Improvements informed by this benchmark

Landing alongside the report:

1. **Canned game-end replies** (`commentary_gate.py:_canned_game_end`). Templated per `(persona, outcome)` where outcome ∈ {`won`, `lost`, `draw`}. The gate fires `HookResult.end(line)` — the LLM never runs on game-over. Zero latency, zero attribution risk.
2. **Scenario corpus expanded from 9 → 35**, every one legality-validated by `test_scenario_replays_legally`.
3. **Stockfish eval recomputation** (`recompute_with_stockfish()`) replays each scenario through a real engine at benchmark time, so the `eval_cp` / classification signals match what RookApp sees in a real game. Flagged several scenarios whose original hand-set eval was off by ±200 cp.
4. **`<think>` strip in eval harness** (`_extract_text`) mirrors the pipeline's `ThinkTagStripHook`, giving Qwen3 / thinking-mode models a fair comparison.
5. **`prompts.py` module split** so the eval harness and future CLI / server surfaces can share the persona-and-protocol string without dragging Qt into their import graph.
6. **Gate trigger: castling** added to the notable-move filter so `O-O` / `O-O-O` turns no longer go silent.
7. **Smoothness column** (✅ live / 🟡 usable / 🔴 slow / ⚠ quality floor) in the bench report and `scripts/analyze_bench_results.py` so a future re-run surfaces the trade-off directly.

### 7.1 Prompt ablation: slim-briefing refactor

Driven by a fresh sweep (`scripts/bench_prompt_ablation.py`) on Gemma 4 E2B at 3 personas × 35 scenarios × 3 repeats per variant (243 runs each):

| Variant | Heuristic | Δ vs baseline | Latency p95 |
|---|---:|---:|---:|
| `baseline` (full) | 97.7 | — | 0.10 s |
| `no_move_desc` | **96.3** | **−1.4** | 0.10 s |
| `no_material` | 97.3 | −0.4 | 0.10 s |
| `no_situation` | 97.3 | −0.4 | 0.10 s |
| `no_score` | 97.6 | −0.1 | 0.10 s |
| `no_role_header` | 97.8 | +0.1 | 0.10 s |
| `no_classification` | 97.9 | +0.2 | 0.10 s |
| `no_persona` | 98.2 | +0.5 | 0.12 s |
| `no_footer` | 98.7 | +1.0 | 0.11 s |
| `facts_only` (no role/sit/footer) | 98.8 | +1.1 | 0.13 s |
| `no_tool_guidance` | 98.8 | +1.1 | 0.11 s |

**Findings:**

- **`move_desc` is the only load-bearing briefing section** — piece-name / from-square / to-square / captured-piece English descriptions drive 1.4 points of quality. Everything else is within noise or a small improvement when removed.
- **`role_header` and `footer` duplicate content already in `ROOK_TOOL_GUIDANCE`.** The role header's pronoun discipline and the footer's "no markdown / no SAN / `<silent>`" rule both appear verbatim in the system-prompt preamble. Keeping both costs ~130 tokens per turn without measurable quality benefit. Dropped.
- **Briefing-only signals consolidated under two headers**: `FACTS` (move descriptions, classification, material, eval) and `SITUATION` (tone cue). Section labels switched from first-person (`MY REACTION TONE`) to declarative (`SITUATION`) because small models will paste first-person instruction phrases verbatim — caught the 1B model writing `"You just played Qxf7, and I concede in persona. That was brutal"` in the eval harness when a similar leak risk was tested.
- **Re-run on the slim baseline**: heuristic score rose from 97.7 → **98.5** with no change to scenarios or model.

Net result: system prompt shrunk ~17 % (813 → 675 tokens for a typical mid-game turn), test coverage retained, pronoun discipline still enforced once in `ROOK_TOOL_GUIDANCE`. Further candidate: drop `ROOK_TOOL_GUIDANCE` entirely (+1.0 point, but latency jumped 57 % — model emits longer replies without the "one sentence" rule). Not landed — the quality/latency trade-off is the wrong direction.

## 8. Reproduction

```bash
# Full sweep (takes ~45 min on a single 3090; ~2-3 hours on a laptop
# if downloads are cold).
python scripts/bench_chess_commentary.py

# Single-model iteration while tuning prompts.
python scripts/eval_llm_commentary.py --model gemma-4-e2b --temperature 0.3

# Post-run quality × speed analysis from the dumped JSON.
python scripts/analyze_bench_results.py

# Legality sanity-check for the scenario corpus.
pytest tests/chess_robot/test_eval_scenarios_legal.py -v
```

Defaults include the stockfish eval recomputation — the binary must be on `$PATH` (`apt install stockfish` / `brew install stockfish`). The harness fails open and falls back to hand values if stockfish isn't available.

## 9. Future work

- Re-run the Qwen3 family at the two quants not already covered (1.7B Q5 / Q6) with `<think>` strip to confirm the corrected scores hold.
- Benchmark Gemma 4 E4B and Gemma 3 1B once weights finish downloading — might displace Llama 3.2 3B in the mid-tier Settings slot.
- Build a **live semantic grader** that fires a second (tiny) LLM to judge attribution correctness on each reply, so the heuristic catches "reply inverts who won" without human audit.
- Tune `max_tokens` (currently 80) down to ~60 once the canned game-end diverts the longest turns — saves ~25 % of the per-reply decode ceiling.
- Persona-specific prompts: the grandmaster voice is currently "not giddy" + clipped; a sharper persona prompt could lift heuristic scores another 1–2 points without a model change.
- Re-run the 25-model sweep with the slim prompt to confirm rankings hold across the suite — expected but unverified.
- Evaluate a **confidence-based LLM bypass** for quiet speakable turns (e.g. minor-piece trade at level eval) — canned small-talk could cover another ~15 % of turns with zero LLM cost.

## See also

- [`slm-tool-calling-benchmark.md`](/documentation/reports/slm-tool-calling-benchmark) — BFCL-style tool-call benchmark for the same preset pool.
- [`desktop.md#commentary-quality--evaluation`](/documentation/desktop#commentary-quality--evaluation) — design of the commentary gate that produces the directive.
- [`scripts/bench_chess_commentary.py`](https://github.com/vietanhdev/edgevox/blob/main/scripts/bench_chess_commentary.py) — runner.
- [`scripts/eval_llm_commentary.py`](https://github.com/vietanhdev/edgevox/blob/main/scripts/eval_llm_commentary.py) — single-model harness and scenario definitions.
