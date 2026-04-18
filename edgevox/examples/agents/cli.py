"""Dispatcher for the ``edgevox-agent`` CLI entry point.

Usage::

    edgevox-agent home [--text-mode|--simple-ui] [--model hf:...]
    edgevox-agent robot ...
    edgevox-agent dev ...
    edgevox-agent list

Each subcommand forwards the remaining argv to the corresponding
example's ``AgentApp.run(...)``, so agent-level flags (``--text-mode``,
``--simple-ui``, ``--model``, ``--language``, etc.) are defined by the
framework, not duplicated here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

SUBCOMMANDS: dict[str, tuple[str, Callable[[list[str] | None], None]]] = {}


def _register(name: str, description: str, main: Callable[[list[str] | None], None]) -> None:
    SUBCOMMANDS[name] = (description, main)


def _lazy_subcommands() -> None:
    """Import example modules lazily so ``edgevox-agent --help`` stays
    fast and doesn't pull in Textual/llama-cpp just to print a usage
    line."""
    from edgevox.examples.agents.dev_toolbox import APP as DEV_APP
    from edgevox.examples.agents.dev_toolbox import main as dev_main
    from edgevox.examples.agents.home_assistant import APP as HOME_APP
    from edgevox.examples.agents.home_assistant import main as home_main
    from edgevox.examples.agents.robot_commander import APP as ROBOT_APP
    from edgevox.examples.agents.robot_commander import main as robot_main
    from edgevox.examples.agents.robot_scout import APP as SCOUT_APP
    from edgevox.examples.agents.robot_scout import main as scout_main

    _register("home", HOME_APP.description, home_main)
    _register("robot", ROBOT_APP.description, robot_main)
    _register("dev", DEV_APP.description, dev_main)
    _register("robot-scout", SCOUT_APP.description, scout_main)

    # IR-SIM demo is optional — only register if ir-sim is installed so
    # `edgevox-agent list` still works without the optional dependency.
    try:
        from edgevox.examples.agents.robot_irsim import APP as IRSIM_APP
        from edgevox.examples.agents.robot_irsim import main as irsim_main

        _register("robot-irsim", IRSIM_APP.description, irsim_main)
    except ImportError:
        pass

    # MuJoCo tabletop arm demo — optional, needs `edgevox[sim-mujoco]`.
    try:
        from edgevox.examples.agents.robot_panda import APP as PANDA_APP
        from edgevox.examples.agents.robot_panda import main as panda_main

        _register("robot-panda", PANDA_APP.description, panda_main)
    except ImportError:
        pass

    # Chess partner — python-chess is core; needs a UCI engine on $PATH.
    try:
        from edgevox.examples.agents.chess_partner import APP as CHESS_APP
        from edgevox.examples.agents.chess_partner import main as chess_main

        _register("chess", CHESS_APP.description, chess_main)
    except ImportError:
        pass

    # RookApp — voice chess against an expressive robot. Native Qt
    # desktop app; no browser, no server sidecar. Optional install:
    # ``pip install "edgevox[desktop]"``.
    try:
        from edgevox.apps.chess_robot_qt.main import main as chess_robot_main

        _register(
            "chess-robot",
            "RookApp — native Qt voice chess desktop app (PySide6).",
            chess_robot_main,
        )
    except ImportError:
        pass

    # MuJoCo humanoid demo — optional, shares the `edgevox[sim-mujoco]` dep.
    try:
        from edgevox.examples.agents.robot_humanoid import APP as HUMANOID_APP
        from edgevox.examples.agents.robot_humanoid import main as humanoid_main

        _register("robot-humanoid", HUMANOID_APP.description, humanoid_main)
    except ImportError:
        pass

    # Tier 3 external-ROS2 demo — needs a sourced ROS2 env. Registered
    # last so plain --help stays fast on non-ROS2 machines.
    try:
        from edgevox.examples.agents.robot_external import APP as EXTERNAL_APP
        from edgevox.examples.agents.robot_external import main as external_main

        _register("robot-external", EXTERNAL_APP.description, external_main)
    except ImportError:
        pass


def _print_usage() -> None:
    _lazy_subcommands()
    print("edgevox-agent — launch built-in EdgeVox voice agents\n")
    print("usage: edgevox-agent <subcommand> [--simple-ui|--text-mode] [--model ...]\n")
    print("subcommands:")
    width = max(len(name) for name in SUBCOMMANDS)
    for name, (desc, _) in SUBCOMMANDS.items():
        print(f"  {name.ljust(width)}  {desc}")
    print("  list".ljust(width + 4) + "  show available subcommands and exit")
    print("\nExamples:")
    print("  edgevox-agent home                 # full TUI voice (default)")
    print("  edgevox-agent robot --simple-ui    # lightweight CLI voice")
    print("  edgevox-agent dev --text-mode      # keyboard chat")
    print("\nPass --help to any subcommand to see its full options:")
    print("  edgevox-agent home --help")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_usage()
        return

    name = argv[0]
    rest = argv[1:]

    if name == "list":
        _print_usage()
        return

    _lazy_subcommands()
    entry = SUBCOMMANDS.get(name)
    if entry is None:
        print(f"edgevox-agent: unknown subcommand {name!r}\n", file=sys.stderr)
        _print_usage()
        sys.exit(2)

    _, run = entry
    run(rest)


if __name__ == "__main__":
    main()
