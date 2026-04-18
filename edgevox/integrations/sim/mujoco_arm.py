"""MuJoCo tabletop-arm adapter — Tier 2a 3D sim for EdgeVox agents.

Ships two scenes under one adapter:

- **Franka Emika Panda** (default) — a stock ``panda.xml`` from
  google-deepmind/mujoco_menagerie composed with a table and three
  coloured cubes. Downloaded on first use from
  ``nrl-ai/edgevox-models`` on HuggingFace Hub, cached under the
  standard HF cache. ~33 MB one-time network fetch.
- **XYZ gantry** (fallback) — a fully self-contained MJCF with a
  3-prismatic-joint gantry and 2-finger gripper over the same cube
  scene. Bundled with the package, no network required. Used on
  fallback when the Franka download fails or when callers explicitly
  pass ``model_source="gantry"``.

Both scenes expose the same ``SimEnvironment`` protocol — agent code
does not branch on which scene is loaded.

Threading model mirrors :class:`~edgevox.integrations.sim.irsim.IrSimEnvironment`:

- **Physics runs on a daemon thread** at wall-clock real time via
  ``mj_step``. Any thread may call :meth:`tick_physics` under the lock.
- **Rendering is main-thread only.** ``mujoco.viewer.launch_passive``
  hands back a handle; :meth:`pump_render` calls ``sync()`` from the
  main thread to push the latest ``mjData`` to the window.
- **Goal tracking and cancellation** live under an ``RLock``. A
  ``GoalHandle`` drives the active skill; the physics loop observes
  ``should_cancel()`` on every tick and freezes the arm in-place when
  safety fires.

Franka motion uses **position-only damped-least-squares IK** seeded from
the stock home keyframe, commanding joint-space targets through the
built-in panda position actuators. Orientation drift is kept small by
high damping and short step size.

Grasping uses a **kinematic attach**: once the end-effector reaches the
target object's pose during a ``grasp`` action, subsequent physics ticks
pin that body's freejoint to the gripper until ``release`` fires. This
bypasses contact-friction tuning and keeps the demo reliable across CPU
speeds.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import numpy as np

from edgevox.agents.skills import GoalHandle, GoalStatus

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mujoco is not installed. Run `pip install 'edgevox[sim-mujoco]'` or "
        "`pip install 'mujoco>=3.2'` to enable the MujocoArmEnvironment adapter."
    ) from e

log = logging.getLogger(__name__)

_BUNDLED_WORLDS_DIR = Path(__file__).parent / "worlds"
_GANTRY_SCENE = _BUNDLED_WORLDS_DIR / "tabletop_arm.xml"

_FRANKA_HF_REPO = "nrl-ai/edgevox-models"
_FRANKA_HF_SUBDIR = "mujoco_scenes/franka_tabletop"
_FRANKA_SCENE_FILE = "franka_tabletop.xml"

_POSITION_TOL = 0.04  # metres — "arrived" threshold (generous for gantry gravity sag)
_GRASP_APPROACH_HEIGHT = 0.1  # metres above object before descent

_IK_GAIN = 0.6
_IK_DAMPING = 0.05

ModelSource = Literal["franka", "gantry", "auto"]


def _download_franka_scene() -> Path:
    """Fetch the Franka tabletop bundle from HuggingFace Hub and return
    the local path to ``franka_tabletop.xml``. Raises on failure."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=_FRANKA_HF_REPO,
        repo_type="model",
        allow_patterns=f"{_FRANKA_HF_SUBDIR}/**",
    )
    scene = Path(local_dir) / _FRANKA_HF_SUBDIR / _FRANKA_SCENE_FILE
    if not scene.exists():
        raise FileNotFoundError(f"Franka scene file missing after download: {scene}")
    return scene


def _resolve_scene(
    model_source: ModelSource,
    model_path: str | Path | None,
    allow_hf_download: bool,
) -> tuple[Path, str]:
    """Pick the scene XML + robot kind (``"franka"`` | ``"gantry"``)."""
    if model_path is not None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo scene file not found: {path}")
        return path, _detect_kind(path)

    if model_source == "gantry":
        return _GANTRY_SCENE, "gantry"

    # "franka" or "auto" — try to fetch Franka, fall back to gantry.
    if allow_hf_download:
        try:
            return _download_franka_scene(), "franka"
        except Exception as e:  # network error, auth error, offline CI
            if model_source == "franka":
                log.warning(
                    "Franka scene download failed (%s); falling back to bundled gantry scene",
                    e,
                )
            return _GANTRY_SCENE, "gantry"

    return _GANTRY_SCENE, "gantry"


def _detect_kind(path: Path) -> str:
    """Infer whether a given MJCF is the gantry fallback or a Franka-style arm."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "franka"
    return "gantry" if "slide_x" in text and "slide_y" in text else "franka"


# Minimum constraint-buffer sizes that cover the tabletop scenes comfortably.
# Raw MuJoCo defaults (nconmax=-1 → small auto) occasionally trip
# ``mj_makeConstraint: nefc under-allocation`` on the downloaded Franka
# scene during viewer warmup. We pre-inject a ``<size>`` element into the
# compiled XML so the constraint pool is big enough from the start.
_MUJOCO_MIN_NCONMAX = 500
_MUJOCO_MIN_NJMAX = 1000


def _load_model_with_safe_sizes(scene_path: Path) -> mujoco.MjModel:
    """Load a scene, retrying with injected constraint-buffer sizes on failure.

    The fast path is just :func:`mujoco.MjModel.from_xml_path` — that
    handles ``<include>`` directives, relative mesh paths, and every
    other compiler quirk correctly. Only if that raises (scene is
    malformed or its default ``<size>`` under-allocates) do we rewrite
    the XML in-memory to inject a larger ``<size nconmax=... njmax=...>``
    and recompile via :func:`from_xml_string` with an asset map that
    covers both meshes and any sibling XML files referenced by
    ``<include>``.
    """
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except Exception as first_exc:
        log.info("from_xml_path failed for %s (%s); retrying with size patch", scene_path, first_exc)
        primary = first_exc

    try:
        raw = scene_path.read_text(encoding="utf-8")
        patched = _inject_size_element(raw, nconmax=_MUJOCO_MIN_NCONMAX, njmax=_MUJOCO_MIN_NJMAX)
        assets = _collect_scene_assets(scene_path.parent)
        return mujoco.MjModel.from_xml_string(patched, assets=assets)
    except Exception as recovery_exc:
        # Re-raise the ORIGINAL failure so the user sees the real cause;
        # the recovery attempt is a best-effort wrapper.
        log.debug("size-patched recompile also failed: %s", recovery_exc)
        raise primary from recovery_exc


def _inject_size_element(xml: str, *, nconmax: int, njmax: int) -> str:
    """Insert or update the ``<size nconmax=... njmax=...>`` element.

    Non-destructive: preserves an existing ``<size>`` if it already
    declares equal-or-larger values. Uses xml.etree.ElementTree so we
    don't corrupt comments or indentation more than necessary.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        log.warning("Could not parse MuJoCo XML for size injection; returning untouched")
        return xml

    size_el = root.find("size")
    if size_el is None:
        size_el = ET.Element("size")
        root.insert(0, size_el)
    # Only bump if the existing value is smaller than our min.
    _bump_attr(size_el, "nconmax", nconmax)
    _bump_attr(size_el, "njmax", njmax)
    return ET.tostring(root, encoding="unicode")


def _bump_attr(element: Any, name: str, minimum: int) -> None:
    raw = element.get(name)
    try:
        current = int(raw) if raw is not None else 0
    except ValueError:
        current = 0
    if current < minimum:
        element.set(name, str(minimum))


def _collect_scene_assets(asset_dir: Path) -> dict[str, bytes]:
    """Build an asset map for ``from_xml_string``.

    Includes sibling XMLs (for ``<include>`` directives) + meshes +
    textures. Returns empty when ``asset_dir`` has nothing loadable so
    the pure gantry scene stays zero-overhead.
    """
    assets: dict[str, bytes] = {}
    if not asset_dir.is_dir():
        return assets
    # XML first so <include> resolution works; then common asset kinds.
    for pattern in ("*.xml", "*.stl", "*.obj", "*.msh", "*.png", "*.jpg", "*.jpeg"):
        for path in asset_dir.rglob(pattern):
            try:
                rel = str(path.relative_to(asset_dir))
            except ValueError:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            assets[rel] = data
            assets.setdefault(path.name, data)
    return assets


class MujocoArmEnvironment:
    """MuJoCo-backed :class:`SimEnvironment` for tabletop manipulation.

    Args:
        model_source: which scene to load.
            - ``"franka"`` (default): download the Franka Panda scene
              from the EdgeVox HF model repo on first use; fall back to
              the bundled gantry if the download fails.
            - ``"gantry"``: always use the bundled primitive gantry
              scene (zero network).
            - ``"auto"``: like ``"franka"`` but silent on fallback.
        model_path: hard override. When set, ``model_source`` is ignored
            and the given MJCF is loaded. The kind (franka vs gantry) is
            inferred from the file contents.
        render: whether to open the passive viewer window.
        tick_interval: wall-clock seconds between physics batches.
        steps_per_tick: ``mj_step`` calls per batch.
        allow_hf_download: set ``False`` to skip the HF fetch entirely
            (useful for air-gapped CI); falls straight to the gantry.
        object_bodies: freejoint body names skills may grasp.
    """

    def __init__(
        self,
        model_source: ModelSource = "franka",
        *,
        model_path: str | Path | None = None,
        render: bool = True,
        tick_interval: float = 0.02,
        steps_per_tick: int = 10,
        allow_hf_download: bool = True,
        object_bodies: tuple[str, ...] = ("red_cube", "green_cube", "blue_cube"),
    ) -> None:
        scene_path, kind = _resolve_scene(model_source, model_path, allow_hf_download)
        log.info("MujocoArmEnvironment loading %s scene from %s", kind, scene_path)

        self._scene_path = scene_path
        self._kind = kind
        self._render = render
        self._tick_interval = tick_interval
        self._steps_per_tick = max(1, steps_per_tick)

        self._model = _load_model_with_safe_sizes(scene_path)
        self._data = mujoco.MjData(self._model)

        if kind == "franka":
            self._init_franka()
        else:
            self._init_gantry()

        self._object_bodies: dict[str, int] = {}
        for name in object_bodies:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                log.warning("object body %r missing from scene", name)
                continue
            self._object_bodies[name] = bid

        # Run a physics warmup so the arm + cubes fully settle under
        # gravity + actuator springs before we snapshot initial state.
        # 2000 steps at dt=0.002 = 4 seconds of sim — well past the
        # settling time for position actuators with moderate damping.
        for _ in range(2000):
            mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        # Capture settled pose so reset() restores a stable state.
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()
        self._initial_ctrl = self._data.ctrl.copy()
        self._ee_home = self._compute_ee_unlocked().copy()

        self._lock = threading.RLock()
        self._active_goal: GoalHandle | None = None
        self._active_kind: str | None = None
        self._active_target: np.ndarray | None = None
        self._grasped: str | None = None
        self._grasp_target: str | None = None
        self._grasp_phase: str | None = None

        # ``launch_passive`` can segfault at the C level on broken Linux
        # GL stacks (Wayland+NVIDIA, WSLg, remote X). A subprocess probe
        # tests GLFW context creation first so a crash in the probe
        # doesn't take down the main process. The physics tick still
        # takes the viewer's internal lock via ``viewer.lock()`` during
        # ``tick_physics`` so the viewer's render thread and our
        # physics thread never touch ``mjData`` concurrently.
        self._viewer: Any | None = None
        if render:
            from edgevox.integrations.sim._viewer_probe import viewer_available

            ok, reason = viewer_available()
            if not ok:
                log.warning(
                    "MuJoCo viewer unavailable (%s) — continuing headless. Pass --no-render to silence this warning.",
                    reason,
                )
                self._render = False
            else:
                try:
                    self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
                except Exception:
                    log.exception("mujoco.viewer.launch_passive failed; continuing headless")
                    self._viewer = None
                    self._render = False

        self._phys_stop = threading.Event()
        self._phys_thread = threading.Thread(target=self._physics_loop, name="mujoco-physics", daemon=True)
        self._phys_thread.start()

    # ----- robot-kind init -----

    # Stock Franka home pose (matches the panda.xml keyframe). We set
    # these qpos values directly instead of ``mj_resetDataKeyframe``
    # because the keyframe was authored for the bare panda scene
    # (nq=9); on our composed scene with freejoint cubes it would zero
    # the cube qpos and drop them through the table.
    _FRANKA_HOME_ARM = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
    _FRANKA_HOME_FINGER = 0.04

    def _init_franka(self) -> None:
        self._arm_dofs = 7  # joint1..joint7
        self._arm_actuator_start = 0  # actuator1..actuator7
        self._finger_actuator_id = self._actuator_id("actuator8")
        self._finger_open_ctrl = 255.0
        self._finger_closed_ctrl = 0.0
        self._hand_body_id = self._body_id("hand")
        # World-frame offset from hand body origin to the fingertip gap.
        self._hand_tip_offset = np.array([0.0, 0.0, 0.1])
        self._slide_joint_ids: dict[str, int] = {}

        arm_joint_ids = [self._joint_id(f"joint{i + 1}") for i in range(self._arm_dofs)]
        self._arm_joint_ids = arm_joint_ids
        self._arm_joint_range = np.array([self._model.jnt_range[j] for j in arm_joint_ids])
        self._arm_dof_idx = np.array([self._model.jnt_dofadr[j] for j in arm_joint_ids], dtype=int)

        # Seed arm + fingers to the home pose. Freejoint bodies keep
        # their default init pose so cubes stay on the table.
        for i, j in enumerate(arm_joint_ids):
            self._data.qpos[self._model.jnt_qposadr[j]] = self._FRANKA_HOME_ARM[i]
            self._data.ctrl[self._arm_actuator_start + i] = self._FRANKA_HOME_ARM[i]
        for jname in ("finger_joint1", "finger_joint2"):
            jid = self._joint_id(jname)
            self._data.qpos[self._model.jnt_qposadr[jid]] = self._FRANKA_HOME_FINGER
        self._data.ctrl[self._finger_actuator_id] = self._finger_open_ctrl
        mujoco.mj_forward(self._model, self._data)

    def _init_gantry(self) -> None:
        self._arm_dofs = 3
        self._slide_joint_ids = {axis: self._joint_id(axis) for axis in ("slide_x", "slide_y", "slide_z")}
        self._actuator_ids = {name: self._actuator_id(name) for name in ("act_x", "act_y", "act_z", "act_finger")}
        self._finger_actuator_id = self._actuator_ids["act_finger"]
        self._finger_open_ctrl = 0.0
        self._finger_closed_ctrl = 0.035
        self._ee_site_id = self._site_id("ee_site")
        self._arm_joint_range = np.array(
            [self._model.actuator_ctrlrange[self._actuator_ids[name]] for name in ("act_x", "act_y", "act_z")]
        )
        self._gantry_home_ctrl = (0.0, 0.0, 0.0)
        self._data.ctrl[self._actuator_ids["act_x"]] = 0.0
        self._data.ctrl[self._actuator_ids["act_y"]] = 0.0
        self._data.ctrl[self._actuator_ids["act_z"]] = 0.0
        self._data.ctrl[self._finger_actuator_id] = self._finger_open_ctrl
        mujoco.mj_forward(self._model, self._data)

    # ----- id lookup helpers -----

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site {name!r} not found in MuJoCo model")
        return sid

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in MuJoCo model")
        return jid

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"actuator {name!r} not found in MuJoCo model")
        return aid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body {name!r} not found in MuJoCo model")
        return bid

    # ----- physics thread -----

    def _physics_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._phys_stop.is_set():
            try:
                self.tick_physics()
            except Exception:
                log.exception("MuJoCo physics tick failed")
            next_tick += self._tick_interval
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()

    def tick_physics(self) -> None:
        """Advance physics by ``steps_per_tick`` ``mj_step`` calls and
        update any active goal. Thread-safe.

        When a viewer is attached we also take its internal lock so the
        viewer's render thread and our physics thread never touch
        ``mjData`` concurrently — MuJoCo is not thread-safe for
        concurrent data access.
        """
        viewer_lock_cm = self._viewer.lock() if self._viewer is not None else contextlib.nullcontext()
        with self._lock, viewer_lock_cm:
            for _ in range(self._steps_per_tick):
                mujoco.mj_step(self._model, self._data)
                self._apply_kinematic_attach_unlocked()
            self._update_active_goal_unlocked()

    # ----- main-thread render pump -----

    def pump_render(self) -> None:
        """Push the latest ``mjData`` to the viewer. Main-thread only."""
        if not self._render or self._viewer is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            with self._lock:
                if self._viewer.is_running():
                    self._viewer.sync()
        except Exception:
            log.exception("mujoco viewer sync failed")

    # Back-compat alias: framework text-mode looks for pump_events.
    pump_events = pump_render

    # ----- SimEnvironment protocol -----

    def reset(self) -> None:
        with self._lock:
            self._data.qpos[:] = self._initial_qpos
            self._data.qvel[:] = self._initial_qvel
            self._data.ctrl[:] = self._initial_ctrl
            mujoco.mj_forward(self._model, self._data)
            self._active_goal = None
            self._active_kind = None
            self._active_target = None
            self._grasped = None
            self._grasp_target = None
            self._grasp_phase = None

    def step(self, dt: float) -> None:  # pragma: no cover
        del dt
        self.tick_physics()

    def get_world_state(self) -> dict[str, Any]:
        with self._lock:
            ee = self._compute_ee_unlocked()
            objects: dict[str, dict[str, float]] = {}
            for name, bid in self._object_bodies.items():
                p = self._data.xpos[bid]
                objects[name] = _round3(p)
            return {
                "robot": self._kind,
                "ee": _round3(ee),
                "grasped": self._grasped,
                "objects": objects,
                "busy": self._active_goal is not None,
            }

    def render(self) -> None:
        """No-op; rendering happens via :meth:`pump_render` on main."""
        pass

    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle:
        dispatcher = getattr(self, f"_action_{action}", None)
        if dispatcher is None:
            h = GoalHandle()
            h.fail(f"unknown action {action!r}")
            return h
        return dispatcher(**kwargs)

    def object_names(self) -> list[str]:
        with self._lock:
            return sorted(self._object_bodies)

    def robot_kind(self) -> str:
        return self._kind

    def close(self) -> None:
        self._phys_stop.set()
        self._phys_thread.join(timeout=1.0)
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None
        if getattr(self, "_offscreen_renderer", None) is not None:
            with contextlib.suppress(Exception):
                self._offscreen_renderer.close()
            self._offscreen_renderer = None

    # ----- ROS2 adapter hooks — offscreen camera + pose interop -----
    # Consumed by :class:`RobotROS2Adapter` to publish
    # ``sensor_msgs/Image`` and accept ``PoseStamped`` goals.

    def get_ee_pose(self) -> tuple[float, float, float]:
        """Return end-effector ``(x, y, z)`` in the world frame."""
        with self._lock:
            ee = self._compute_ee_unlocked()
            return float(ee[0]), float(ee[1]), float(ee[2])

    def get_camera_frame(self, width: int = 320, height: int = 240, camera_id: int = -1) -> np.ndarray | None:
        """Offscreen-render the scene and return an ``HxWx3`` uint8
        array in ``rgb8`` encoding. Returns ``None`` if OpenGL context
        creation fails (typical on headless hosts without
        ``MUJOCO_GL=egl`` / ``osmesa``).

        The renderer is lazily created once per instance and resized
        on demand — creating a new Renderer per frame churns GL state.
        """
        with self._lock:
            renderer = getattr(self, "_offscreen_renderer", None)
            cur_size = getattr(self, "_offscreen_size", None)
            if renderer is None or cur_size != (width, height):
                try:
                    if renderer is not None:
                        with contextlib.suppress(Exception):
                            renderer.close()
                    renderer = mujoco.Renderer(self._model, width=width, height=height)
                    self._offscreen_renderer = renderer
                    self._offscreen_size = (width, height)
                except Exception:
                    log.debug(
                        "MuJoCo offscreen renderer init failed (set MUJOCO_GL=egl for headless GPU)",
                        exc_info=True,
                    )
                    self._offscreen_renderer = None
                    self._offscreen_size = None
                    return None
            try:
                renderer.update_scene(self._data, camera=camera_id)
                pixels = renderer.render()
                return np.ascontiguousarray(pixels)
            except Exception:
                log.debug("MuJoCo offscreen render failed", exc_info=True)
                return None

    # ----- ee computation -----

    def _compute_ee_unlocked(self) -> np.ndarray:
        if self._kind == "franka":
            hand_pos = self._data.xpos[self._hand_body_id]
            hand_mat = self._data.xmat[self._hand_body_id].reshape(3, 3)
            return np.asarray(hand_pos + hand_mat @ self._hand_tip_offset, dtype=float)
        return np.asarray(self._data.site_xpos[self._ee_site_id], dtype=float)

    # ----- goal progression -----

    def _update_active_goal_unlocked(self) -> None:
        goal = self._active_goal
        if goal is None:
            return

        if goal.should_cancel():
            self._freeze_ctrl_unlocked()
            goal.mark_cancelled()
            self._active_goal = None
            self._active_kind = None
            self._active_target = None
            self._grasp_target = None
            self._grasp_phase = None
            return

        kind = self._active_kind
        if kind == "move_to":
            self._progress_move_to_unlocked(goal)
        elif kind == "grasp":
            self._progress_grasp_unlocked(goal)
        elif kind == "release":
            self._progress_release_unlocked(goal)
        elif kind == "goto_home":
            self._progress_goto_home_unlocked(goal)

    def _progress_move_to_unlocked(self, goal: GoalHandle) -> None:
        target = self._active_target
        if target is None:
            goal.fail("move_to had no target")
            self._active_goal = None
            self._active_kind = None
            return
        ee = self._compute_ee_unlocked()
        self._drive_toward_unlocked(target)
        dist = float(np.linalg.norm(target - ee))
        goal.set_feedback({"ee": _round3(ee), "remaining": round(dist, 4)})
        if dist < _POSITION_TOL:
            goal.succeed({"ee": _round3(ee)})
            self._active_goal = None
            self._active_kind = None
            self._active_target = None

    def _progress_grasp_unlocked(self, goal: GoalHandle) -> None:
        name = self._grasp_target
        if name is None or name not in self._object_bodies:
            goal.fail(f"unknown object {name!r}")
            self._active_goal = None
            self._active_kind = None
            return
        bid = self._object_bodies[name]
        obj_pos = np.asarray(self._data.xpos[bid], dtype=float)
        ee = self._compute_ee_unlocked()

        if self._grasp_phase == "approach":
            approach = np.array([float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + _GRASP_APPROACH_HEIGHT])
            self._drive_toward_unlocked(approach)
            dist = float(np.linalg.norm(approach - ee))
            goal.set_feedback({"phase": "approach", "remaining": round(dist, 4)})
            if dist < _POSITION_TOL:
                self._grasp_phase = "descend"
            return

        if self._grasp_phase == "descend":
            target = np.array([float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])])
            self._drive_toward_unlocked(target)
            dist = float(np.linalg.norm(target - ee))
            goal.set_feedback({"phase": "descend", "remaining": round(dist, 4)})
            if dist < _POSITION_TOL:
                self._data.ctrl[self._finger_actuator_id] = self._finger_closed_ctrl
                self._grasped = name
                goal.succeed({"grasped": name, "ee": _round3(ee)})
                self._active_goal = None
                self._active_kind = None
                self._grasp_target = None
                self._grasp_phase = None
            return

    def _progress_release_unlocked(self, goal: GoalHandle) -> None:
        self._data.ctrl[self._finger_actuator_id] = self._finger_open_ctrl
        released = self._grasped
        self._grasped = None
        goal.succeed({"released": released})
        self._active_goal = None
        self._active_kind = None

    def _progress_goto_home_unlocked(self, goal: GoalHandle) -> None:
        ee = self._compute_ee_unlocked()
        target = self._ee_home
        self._drive_toward_unlocked(target)
        dist = float(np.linalg.norm(target - ee))
        goal.set_feedback({"remaining": round(dist, 4)})
        if dist < _POSITION_TOL:
            goal.succeed({"at": "home", "ee": _round3(ee)})
            self._active_goal = None
            self._active_kind = None
            self._active_target = None

    # ----- motion control -----

    def _drive_toward_unlocked(self, world_target: np.ndarray) -> None:
        if self._kind == "franka":
            self._drive_franka_ik_unlocked(world_target)
        else:
            self._drive_gantry_unlocked(world_target)

    def _drive_gantry_unlocked(self, world_target: np.ndarray) -> None:
        # Feedforward: target_ctrl = delta_from_home + home_ctrl. The
        # warmup settle ensures gravity compensation is baked into
        # ee_home, so no explicit gravity offset is needed.
        for axis, (act_name, home_ctrl) in enumerate(
            zip(("act_x", "act_y", "act_z"), self._gantry_home_ctrl, strict=True)
        ):
            aid = self._actuator_ids[act_name]
            delta = float(world_target[axis]) - float(self._ee_home[axis])
            lo, hi = self._model.actuator_ctrlrange[aid]
            self._data.ctrl[aid] = float(np.clip(home_ctrl + delta, lo, hi))

    def _drive_franka_ik_unlocked(self, world_target: np.ndarray) -> None:
        ee = self._compute_ee_unlocked()
        err = np.asarray(world_target, dtype=float) - ee
        err_norm = float(np.linalg.norm(err))
        # Cap single-step error so the IK stays well-conditioned on large targets.
        max_step = 0.05
        if err_norm > max_step:
            err = err * (max_step / err_norm)

        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jac(self._model, self._data, jacp, None, ee, self._hand_body_id)
        Ja = jacp[:, self._arm_dof_idx]

        A = Ja @ Ja.T + (_IK_DAMPING**2) * np.eye(3)
        try:
            dq = Ja.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            return

        q_cur = np.array([self._data.qpos[self._model.jnt_qposadr[j]] for j in self._arm_joint_ids])
        q_new = q_cur + _IK_GAIN * dq
        for i, (lo, hi) in enumerate(self._arm_joint_range):
            q_new[i] = float(np.clip(q_new[i], lo, hi))
        for i in range(self._arm_dofs):
            self._data.ctrl[self._arm_actuator_start + i] = float(q_new[i])

    def _freeze_ctrl_unlocked(self) -> None:
        if self._kind == "franka":
            for i, j in enumerate(self._arm_joint_ids):
                q = float(self._data.qpos[self._model.jnt_qposadr[j]])
                self._data.ctrl[self._arm_actuator_start + i] = q
            return
        qx = float(self._data.qpos[self._model.jnt_qposadr[self._slide_joint_ids["slide_x"]]])
        qy = float(self._data.qpos[self._model.jnt_qposadr[self._slide_joint_ids["slide_y"]]])
        qz = float(self._data.qpos[self._model.jnt_qposadr[self._slide_joint_ids["slide_z"]]])
        self._data.ctrl[self._actuator_ids["act_x"]] = qx
        self._data.ctrl[self._actuator_ids["act_y"]] = qy
        self._data.ctrl[self._actuator_ids["act_z"]] = qz

    def _apply_kinematic_attach_unlocked(self) -> None:
        if self._grasped is None:
            return
        bid = self._object_bodies.get(self._grasped)
        if bid is None:
            return
        jnt_adr = int(self._model.body_jntadr[bid])
        if jnt_adr < 0:
            return
        qpos_adr = int(self._model.jnt_qposadr[jnt_adr])
        qvel_adr = int(self._model.jnt_dofadr[jnt_adr])
        ee = self._compute_ee_unlocked()
        # Pin the held body just below the end-effector with identity
        # orientation, and zero its velocity so it doesn't drift.
        self._data.qpos[qpos_adr + 0] = float(ee[0])
        self._data.qpos[qpos_adr + 1] = float(ee[1])
        self._data.qpos[qpos_adr + 2] = float(ee[2]) - 0.005
        self._data.qpos[qpos_adr + 3] = 1.0
        self._data.qpos[qpos_adr + 4] = 0.0
        self._data.qpos[qpos_adr + 5] = 0.0
        self._data.qpos[qpos_adr + 6] = 0.0
        self._data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    # ----- action dispatch -----

    def _action_move_to(self, x: float, y: float, z: float) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            target = np.array([float(x), float(y), float(z)])
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "move_to"
            self._active_target = target
        return handle

    def _action_grasp(self, object: str) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if object not in self._object_bodies:
                handle.fail(f"unknown object {object!r}. Known: {sorted(self._object_bodies)}")
                return handle
            if self._grasped is not None:
                handle.fail(f"already grasping {self._grasped!r}; release first")
                return handle
            self._data.ctrl[self._finger_actuator_id] = self._finger_open_ctrl
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "grasp"
            self._grasp_target = object
            self._grasp_phase = "approach"
        return handle

    def _action_release(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if self._grasped is None:
                handle.fail("nothing to release")
                return handle
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "release"
        return handle

    def _action_get_ee_pose(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            ee = self._compute_ee_unlocked()
            handle.succeed(_round3(ee))
        return handle

    def _action_list_objects(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            out: list[dict[str, Any]] = []
            for name, bid in self._object_bodies.items():
                p = self._data.xpos[bid]
                out.append({"name": name, **_round3(p)})
        handle.succeed(out)
        return handle

    def _action_goto_home(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "goto_home"
            self._active_target = self._ee_home.copy()
        return handle


def _round3(vec: Any) -> dict[str, float]:
    return {
        "x": round(float(vec[0]), 3),
        "y": round(float(vec[1]), 3),
        "z": round(float(vec[2]), 3),
    }


__all__ = ["MujocoArmEnvironment"]
