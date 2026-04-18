# Chess Partner

Voice-controlled chess agent with pluggable engines, live analytics, and three
selectable personas. Exercises every harness seam — tool registry, `ctx.deps`
domain state, hooks, and the upgraded WebSocket pipeline — in a single demo.

## Quick start

```bash
# Install stockfish (Linux / macOS)
sudo apt-get install -y stockfish       # or: brew install stockfish

# Play in text mode
uv run edgevox-agent chess --persona casual --engine stockfish --text-mode

# Full TUI voice — board appears in the side panel
uv run edgevox-agent chess --persona trash_talker --engine stockfish

# Browser / desktop app (upgraded pipeline — chess_state events flow to the UI)
uv run edgevox-serve \
    --agent edgevox.examples.agents.chess_partner:build_server_agent
# then open http://localhost:8765/
```

## Personas

| Slug           | Voice                 | Default engine | Style                            |
|----------------|----------------------|----------------|----------------------------------|
| `grandmaster`  | precise, terse       | Stockfish 20   | Cites openings by ECO code       |
| `casual`       | warm, chatty         | Maia 1400      | Explains ideas in plain English  |
| `trash_talker` | cocky, playful       | Maia 1800      | Teases blunders affectionately   |

Add your own in `edgevox/integrations/chess/personas.py` or at import time via
`register_persona(Persona(...))`.

## Engines

Both shipped backends run as UCI subprocesses through `python-chess`'s
`SimpleEngine` — EdgeVox stays MIT; the GPL engine binary lives out-of-process.

- **Stockfish** — strongest, `Skill Level 0-20` is the main knob.
  `apt install stockfish` / `brew install stockfish`.
- **Maia** — plays like a human at a target ELO (600-2600). Needs `lc0` +
  a `.pb.gz` weight file from https://maiachess.com/. Pass
  `--maia-weights /path/to/maia-1500.pb.gz`.

Write a new backend by implementing the `ChessEngine` protocol in
`edgevox/integrations/chess/engine.py` — the tool/hook layer doesn't care.

## Architecture

```mermaid
flowchart LR
    U[User voice / text] --> LLM
    LLM -->|@tool call| Env[ChessEnvironment]
    Env -->|UCI subprocess| E[Stockfish / LC0]
    Env -->|subscribe| TUI[Textual ChessPanel]
    Env -->|subscribe + ws.send_json| Web[React ChessBoard / EvalBar / MoveList]
    LLM --> TTS[TTS] --> Sp[Speaker]
```

The `ChessEnvironment` implements the `SimEnvironment` protocol and lives
under `ctx.deps`. Agent tools (`play_user_move`, `engine_move`,
`analyze_position`, ...) mutate board state, and the environment publishes a
typed `ChessState` snapshot to every subscribed listener on each mutation.

## Hooks

| Hook                         | Fire point     | Purpose                                   |
|------------------------------|----------------|-------------------------------------------|
| `BoardStateInjectionHook`    | `on_run_start` | Prepend compact FEN + last-move + eval    |
| `MoveCommentaryHook`         | `after_tool`   | Stash the latest state for commentary     |

Both hooks are declared once in `AgentApp(hooks=[...])`. Per
[ADR-002](/adr/002-typed-ctx-hook-state), state lives under
`ctx.hook_state[id(self)]` so two hook instances don't share buffers.

## Analytics

| Tool                 | Returns                                                   |
|----------------------|-----------------------------------------------------------|
| `get_board_state`    | FEN, side-to-move, move history, last move, eval          |
| `list_legal_moves`   | UCI strings for the side to move                          |
| `play_user_move`     | Validated user move; raises with clear error on illegal   |
| `engine_move`        | Engine's move + eval + principal variation                |
| `analyze_position`   | Centipawn eval, best line, win probability, mate-in       |
| `classify_last_move` | `best` / `good` / `inaccuracy` / `mistake` / `blunder`    |
| `undo_last_move`     | Board rollback — for voice corrections                    |
| `new_game`           | Reset with optional side + skill level                    |

Thresholds for classification (all centipawns): ±10 → best, ±50 → good, ±150
→ inaccuracy, ±300 → mistake, >300 → blunder. Edit
`edgevox/integrations/chess/analytics.py` to tune.

## Web UI wire format

The React chess panel mounts automatically the first time the server emits a
`chess_state` message. Payload fields:

```ts
{
  type: "chess_state";
  fen: string;
  ply: number;
  turn: "white" | "black";
  last_move_uci?: string | null;
  last_move_san?: string | null;
  last_move_classification?: "best" | "good" | "inaccuracy" | "mistake" | "blunder" | null;
  san_history?: string[];
  eval_cp?: number | null;
  mate_in?: number | null;
  win_prob_white?: number;
  opening?: string | null;
  is_game_over?: boolean;
  game_over_reason?: string | null;
  winner?: "white" | "black" | null;
}
```

See `webui/src/lib/ws-client.ts` for the typed message union.

## Server env vars

When running via `edgevox-serve --agent ...:build_server_agent`:

| Env var                               | Default  | Meaning                       |
|---------------------------------------|----------|-------------------------------|
| `EDGEVOX_CHESS_PERSONA`               | `casual` | Persona slug                  |
| `EDGEVOX_CHESS_ENGINE`                | persona  | Override engine backend       |
| `EDGEVOX_CHESS_USER_PLAYS`            | `white`  | `white` or `black`            |
| `EDGEVOX_CHESS_STOCKFISH_SKILL`       | persona  | 0-20                          |
| `EDGEVOX_CHESS_MAIA_WEIGHTS`          | —        | Required when engine=`maia`   |

## Next steps

- **Phase 2 (planned)** — MuJoCo Franka arm moving physical pieces on a
  tabletop chessboard scene. `skills/move_piece_arm.py` + new MJCF world.
- **Opening book** — swap the curated `_OPENING_TABLE` for a real polyglot
  book when the agent should speak like a prepared player.
- **Engine on another host** — `ChessEngine` is a protocol; wire up a
  network UCI backend if you want the Python process thin.
