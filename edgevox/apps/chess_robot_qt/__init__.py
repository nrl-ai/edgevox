"""RookApp — PySide6 desktop chess voice-robot.

Same agent/hooks plumbing as the ``chess_robot`` example, but the UI
lives in-process via Qt widgets instead of a React/Tauri webview. One
language, one process, one binary after packaging.

Entry point: ``edgevox-chess-robot`` → :func:`main`.
"""

from __future__ import annotations

from edgevox.apps.chess_robot_qt.main import main

__all__ = ["main"]
