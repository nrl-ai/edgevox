"""``edgevox-chess-robot`` entry point — launch the Qt app.

Runs everything in one Python process:

    QApplication  →  RookWindow  →  RookBridge  →  LLMAgent  →  llama-cpp
                                           ↓
                                      ChessEnvironment  →  stockfish subprocess

No WebSocket, no Vite, no Node, no Tauri. Models are loaded lazily on
a QThreadPool worker so the window paints immediately and the user
watches the status line update from "loading…" → "online".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edgevox-chess-robot",
        description="RookApp — voice chess against an offline robot (PySide6 desktop).",
    )
    parser.add_argument(
        "--persona",
        default=None,
        choices=["grandmaster", "casual", "trash_talker"],
        help="Which chess persona Rook plays as.",
    )
    parser.add_argument(
        "--user-plays",
        default=None,
        choices=["white", "black"],
        help="Which side the user takes. Default: white.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=["stockfish", "maia"],
        help="Chess engine backend override.",
    )
    parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=None,
        help="Stockfish skill level 0-20.",
    )
    parser.add_argument(
        "--maia-weights",
        default=None,
        help="Path to Maia weights (required when --engine maia).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def _apply_args_to_env(args: argparse.Namespace) -> None:
    """Translate CLI flags into the env vars ``RookConfig.from_env``
    reads, keeping a single config surface."""
    if args.persona is not None:
        os.environ["EDGEVOX_CHESS_PERSONA"] = args.persona
    if args.engine is not None:
        os.environ["EDGEVOX_CHESS_ENGINE"] = args.engine
    if args.user_plays is not None:
        os.environ["EDGEVOX_CHESS_USER_PLAYS"] = args.user_plays
    if args.stockfish_skill is not None:
        os.environ["EDGEVOX_CHESS_STOCKFISH_SKILL"] = str(args.stockfish_skill)
    if args.maia_weights is not None:
        os.environ["EDGEVOX_CHESS_MAIA_WEIGHTS"] = args.maia_weights


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    _apply_args_to_env(args)

    # Qt imports are deferred to this function so ``edgevox-chess-robot
    # --help`` doesn't pay Qt's startup cost. Same pattern as the older
    # Tauri CLI.
    from PySide6.QtWidgets import QApplication

    from edgevox.apps.chess_robot_qt.bridge import RookBridge, RookConfig
    from edgevox.apps.chess_robot_qt.window import RookWindow

    app = QApplication(sys.argv)
    app.setApplicationName("RookApp")
    app.setOrganizationName("EdgeVox")

    bridge = RookBridge(RookConfig.from_env())
    window = RookWindow(bridge)
    window.show()

    rc = app.exec()
    bridge.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
