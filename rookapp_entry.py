"""PyInstaller entry shim for RookApp.

Committed to the repo (rather than generated inside the workflow) so
the build is reproducible and the spec file can reference it.

Supports a ``--self-test`` flag used by the CI smoke test: it imports
every module the real app needs at startup and exits with status 0 if
nothing is missing. That catches a common PyInstaller failure mode —
``ModuleNotFoundError`` at runtime because a hidden import wasn't
picked up during analysis — without having to launch a GUI on a
headless CI runner.
"""

from __future__ import annotations

import sys


def _self_test() -> int:
    """Import-only smoke test for the frozen bundle.

    Touches every module the real launch path hits before Qt's event
    loop starts. Any missing hidden import will raise here and the
    build fails loudly instead of crashing on the user's desktop.
    """
    import importlib

    modules = [
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
        "qtawesome",
        "chess",
        "chess.engine",
        "chess.pgn",
        "edgevox.apps.chess_robot_qt.main",
        "edgevox.apps.chess_robot_qt.bridge",
        "edgevox.apps.chess_robot_qt.window",
        "edgevox.apps.chess_robot_qt.board",
        "edgevox.apps.chess_robot_qt.chat",
        "edgevox.apps.chess_robot_qt.face",
        "edgevox.apps.chess_robot_qt.lottie_face",
        "edgevox.apps.chess_robot_qt.settings",
        "edgevox.apps.chess_robot_qt.settings_dialog",
        "edgevox.agents",
        "edgevox.agents.hooks_builtin",
        "edgevox.agents.interrupt",
        "edgevox.agents.memory",
        "edgevox.examples.agents.chess_robot.face_hook",
        "edgevox.examples.agents.chess_robot.move_intercept",
        "edgevox.examples.agents.chess_robot.rich_board",
        "edgevox.examples.agents.chess_robot.sanitize",
        "edgevox.integrations.chess",
        "edgevox.llm",
        "edgevox.llm.hooks_slm",
    ]
    failed: list[tuple[str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
    if failed:
        print("Self-test FAILED — missing modules in bundle:", file=sys.stderr)
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1

    # Also verify bundled assets resolve — the SVG piece sets and the
    # Lottie JSON files must land next to the binary or the UI boots
    # blank. We check one file per asset dir via the packaged
    # ``chess_robot_qt.assets`` path.
    from pathlib import Path

    pkg = Path(importlib.import_module("edgevox.apps.chess_robot_qt").__file__).parent
    expected = [
        pkg / "assets" / "pieces-fantasy" / "wK.svg",
        pkg / "assets" / "lottie" / "robot_idle.json",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("Self-test FAILED — missing bundled assets:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    print("Self-test OK — all imports and assets resolved.")
    return 0


def main() -> None:
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    from edgevox.apps.chess_robot_qt.main import main as real_main

    real_main()


if __name__ == "__main__":
    main()
