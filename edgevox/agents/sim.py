"""SimEnvironment protocol + ``ToyWorld`` stdlib-only reference.

A ``SimEnvironment`` is just whatever object the agent's tools and
skills read/write through ``ctx.deps``. The protocol formalizes the
minimum so ``ToyWorld`` (stdlib only), ``IrSimEnvironment`` (IR-SIM),
and future adapters (Gazebo, MuJoCo) stay drop-in replaceable.

``ToyWorld`` is the smallest useful reference — a grid of rooms plus a
2D robot with a pose and battery. It's what the built-in examples use
by default so ``edgevox-agent robot`` works the moment the package is
installed, with zero extra dependencies.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from edgevox.agents.skills import GoalHandle, GoalStatus

log = logging.getLogger(__name__)


@runtime_checkable
class SimEnvironment(Protocol):
    """Minimum protocol every simulation backend implements.

    The agent framework passes an instance as ``ctx.deps``. Tools and
    skills operate on it without knowing whether it's a ``ToyWorld``,
    an ``IrSimEnvironment``, or a live Gazebo bridge.
    """

    def reset(self) -> None: ...
    def step(self, dt: float) -> None: ...
    def get_world_state(self) -> dict[str, Any]: ...
    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle: ...
    def render(self) -> None: ...


# --------- ToyWorld reference implementation ---------


@dataclass
class _ToyRoom:
    name: str
    x: float
    y: float
    light_on: bool = False


@dataclass
class _ToyRobot:
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float = 92.0
    moving: bool = False


class ToyWorld:
    """A stdlib-only sim used for examples, unit tests, and the first
    run of ``edgevox-agent robot`` without any extra install.

    Offers:

    - a list of named rooms with positions and light state
    - a single mobile robot with pose, heading, battery
    - ``apply_action("navigate_to", room=...)`` which drives the robot
      toward a room over a bounded duration, respecting cancellation
    - ``apply_action("set_light", room=..., on=...)`` for light control
    - a no-op ``render()`` (override in subclasses for ASCII debug)

    Thread safety: the internal state is guarded by a lock so the
    skill worker threads and the dispatcher can safely interleave.
    """

    DEFAULT_ROOMS: ClassVar[list[tuple[str, float, float]]] = [
        ("living_room", 0.0, 0.0),
        ("kitchen", 4.0, 0.0),
        ("bedroom", 4.0, 4.0),
        ("office", 0.0, 4.0),
    ]

    def __init__(
        self,
        *,
        rooms: list[tuple[str, float, float]] | None = None,
        robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        navigate_speed: float = 1.0,
    ) -> None:
        self._lock = threading.Lock()
        room_specs = rooms if rooms is not None else self.DEFAULT_ROOMS
        self._rooms = {r[0]: _ToyRoom(r[0], r[1], r[2]) for r in room_specs}
        self._robot = _ToyRobot(x=robot_pose[0], y=robot_pose[1], heading_deg=robot_pose[2])
        self._navigate_speed = navigate_speed

    # ----- SimEnvironment protocol -----

    def reset(self) -> None:
        with self._lock:
            self._robot = _ToyRobot()
            for r in self._rooms.values():
                r.light_on = False

    def step(self, dt: float) -> None:  # pragma: no cover — ToyWorld is event-driven
        del dt

    def get_world_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "robot": {
                    "x": round(self._robot.x, 3),
                    "y": round(self._robot.y, 3),
                    "heading_deg": round(self._robot.heading_deg, 1),
                    "battery_pct": round(self._robot.battery_pct, 1),
                    "moving": self._robot.moving,
                },
                "rooms": {r.name: {"x": r.x, "y": r.y, "light_on": r.light_on} for r in self._rooms.values()},
            }

    def render(self) -> None:  # pragma: no cover — overridable
        state = self.get_world_state()
        log.info(
            "[ToyWorld] robot@(%.1f,%.1f) bat=%.0f%%",
            state["robot"]["x"],
            state["robot"]["y"],
            state["robot"]["battery_pct"],
        )

    # ----- actions -----

    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle:
        dispatcher = getattr(self, f"_action_{action}", None)
        if dispatcher is None:
            handle = GoalHandle()
            handle.fail(f"unknown action {action!r}")
            return handle
        return dispatcher(**kwargs)

    def room_names(self) -> list[str]:
        with self._lock:
            return sorted(self._rooms)

    # ----- action implementations -----

    def _action_get_pose(self) -> GoalHandle:
        handle = GoalHandle()
        handle.succeed(self.get_world_state()["robot"])
        return handle

    def _action_battery_level(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            handle.succeed({"battery_pct": round(self._robot.battery_pct, 1)})
        return handle

    def _action_set_light(self, room: str, on: bool) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if room not in self._rooms:
                handle.fail(f"unknown room {room!r}")
                return handle
            self._rooms[room].light_on = bool(on)
            handle.succeed({"room": room, "on": bool(on)})
        return handle

    def _action_list_rooms(self) -> GoalHandle:
        handle = GoalHandle()
        handle.succeed(self.room_names())
        return handle

    def _action_navigate_to(
        self, room: str | None = None, x: float | None = None, y: float | None = None
    ) -> GoalHandle:
        """Drive the robot toward a target over simulated time.

        Runs on a worker thread so the dispatcher can cancel mid-motion
        when the safety monitor fires.
        """
        handle = GoalHandle()

        with self._lock:
            if room is not None:
                if room not in self._rooms:
                    handle.fail(f"unknown room {room!r}")
                    return handle
                target_x, target_y = self._rooms[room].x, self._rooms[room].y
                target_label = room
            elif x is not None and y is not None:
                target_x, target_y = float(x), float(y)
                target_label = f"({target_x:.1f},{target_y:.1f})"
            else:
                handle.fail("navigate_to needs either room= or x=/y=")
                return handle

        def drive() -> None:
            try:
                dt = 0.1
                while True:
                    if handle.should_cancel():
                        with self._lock:
                            self._robot.moving = False
                        handle.mark_cancelled()
                        return
                    with self._lock:
                        dx = target_x - self._robot.x
                        dy = target_y - self._robot.y
                        dist = math.hypot(dx, dy)
                        if dist < 0.05:
                            self._robot.x = target_x
                            self._robot.y = target_y
                            self._robot.moving = False
                            handle.succeed(
                                {
                                    "arrived_at": target_label,
                                    "pose": (round(self._robot.x, 2), round(self._robot.y, 2)),
                                }
                            )
                            return
                        step = min(self._navigate_speed * dt, dist)
                        self._robot.x += (dx / dist) * step
                        self._robot.y += (dy / dist) * step
                        self._robot.battery_pct = max(0.0, self._robot.battery_pct - 0.02 * step)
                        self._robot.heading_deg = math.degrees(math.atan2(dy, dx)) % 360
                        self._robot.moving = True
                    handle.set_feedback(
                        {
                            "pose": (round(self._robot.x, 2), round(self._robot.y, 2)),
                            "remaining": round(dist - step, 2),
                        }
                    )
                    time.sleep(dt)
            except Exception as e:
                handle.fail(f"drive crashed: {e}")

        t = threading.Thread(target=drive, name="toyworld-navigate", daemon=True)
        handle._thread = t
        handle.status = GoalStatus.RUNNING
        t.start()
        return handle
