# Robotics Examples

<table>
  <tr>
    <td align="center" width="50%">
      <a href="/robot_panda.png"><img src="/robot_panda.png" alt="MuJoCo Franka Panda pick-and-place"/></a>
      <br/><sub><b>MuJoCo · Franka arm</b> — voice-controlled pick-and-place</sub>
    </td>
    <td align="center" width="50%">
      <a href="/robot_unitree_g1.png"><img src="/robot_unitree_g1.png" alt="Unitree G1 humanoid"/></a>
      <br/><sub><b>MuJoCo · Unitree G1</b> — procedural gait + ONNX policy slot</sub>
    </td>
  </tr>
</table>

Five end-to-end robot agents ship in `edgevox.examples.agents`. Each is a single Python file wiring an `LLMAgent` to a concrete `SimEnvironment` (or ROS2 bridge) through the voice pipeline. You can launch any of them from the CLI:

```bash
edgevox-agent <name> [--text-mode | --simple-ui | default TUI]
```

Swap between text-only (fastest feedback loop), `--simple-ui` (headless voice), or the default TUI (live waveform + transcript + chat log). Every example respects the same agent harness — tools, hooks, interrupts, memory — so code you write for one simulator runs unchanged on the next tier.

---

## `robot-irsim` — 2D mobile robot navigation

```bash
pip install 'edgevox[sim]'
edgevox-setup
edgevox-agent robot-irsim --text-mode
```

A matplotlib window opens showing a 10×10 m apartment with four rooms. Type `"go to the kitchen"` and the blue robot drives visibly to the kitchen centroid. Type `"stop"` mid-flight and the skill preempts — the `SafetyMonitor` intercepts stop-words *before the LLM is consulted*, so the halt path doesn't wait on a model round-trip.

![EdgeVox TUI driving a voice agent](/screenshot.png)

**Tools exposed to the LLM:** `go_to_room`, `describe_surroundings`, `return_home`, `status`.
**Cancellable skills:** `go_to`, `patrol_rooms` — both expose `GoalHandle.feedback` so a user saying "slower" mid-skill adjusts the target speed without restarting the plan.
**Underlying:** `IrSimEnvironment` (IR-SIM, LGPL — dynamic-linked) wraps the sim clock + `cmd_vel` publisher into the `SimEnvironment` Protocol.

Good first pick for: understanding how skills + safety-monitor + `ctx.deps` fit together without heavy 3D rendering cost.

---

## `robot-panda` — MuJoCo Franka arm pick-and-place

```bash
pip install 'edgevox[sim-mujoco]'
edgevox-agent robot-panda --text-mode
```

A MuJoCo viewer opens with a Franka Panda arm above a table holding three coloured cubes. `"pick up the red cube"` moves the arm, grasps, and lifts. Voice input drives `move_to`, `grasp`, `release`, `goto_home` skills.

![MuJoCo Franka Panda pick-and-place](/robot_panda.png)

**Tools exposed:** `list_objects`, `describe_workspace`, `get_gripper_state`.
**Cancellable skills:** `move_to(target)`, `grasp`, `release`, `goto_home` — each wraps an IK + trajectory followup with poll-cancel plumbing so barge-in actually halts the arm mid-trajectory.
**Underlying:** `MujocoArmEnvironment` — MuJoCo Apache-2, models in `edgevox/examples/assets/panda/`.

Good pick for: manipulation policies, demo-ing grasp failure modes, prototyping tool-use that requires spatial reasoning.

---

## `robot-humanoid` — Unitree G1 / H1 with procedural gait

```bash
pip install 'edgevox[sim-mujoco]'
edgevox-agent robot-humanoid --simple-ui
```

A Unitree humanoid (model auto-fetched from `nrl-ai/edgevox-models` on first use — ~15 MB) appears in the MuJoCo viewer standing on its home keyframe. Say "walk forward half a meter" / "turn left ninety degrees" / "stand" — a procedural gait swings legs + arms while the root pose advances. Swap in your own ONNX walking policy via `MujocoHumanoidEnvironment.set_walking_policy(...)` for real RL locomotion.

![Unitree G1 humanoid in MuJoCo](/robot_unitree_g1.png)

**Tools exposed:** `list_joints`, `describe_pose`, `status`.
**Cancellable skills:** `walk_forward(distance_m)`, `turn(angle_deg)`, `stand`, `set_stance(...)`.
**Underlying:** `MujocoHumanoidEnvironment` — default weights under `nrl-ai/edgevox-models/humanoid/` with upstream fallback; policy-slot loaded with `onnxruntime`.

Good pick for: testing your own RL locomotion policies against a natural-language driver without writing a keyboard UI.

---

## `robot-scout` — single-agent scout, no sim dep

```bash
edgevox-agent robot-scout --text-mode
```

Minimal-dep scout agent (no IR-SIM, no MuJoCo) that operates over `ToyWorld` — a stdlib-only grid sim good for CI tests and offline demos. Treats the grid as a room layout, plans paths with BFS, and reports "I walked to the kitchen, saw a mug on the counter".

**Tools / skills:** same tool surface as `robot-irsim` (`go_to_room`, `describe_surroundings`) so scripts port with a one-line env swap.
**Underlying:** `ToyWorld` — pure stdlib, dataclass state, no native deps.

Good pick for: CI / unit tests, tutorials, offline environments. It's what the test suite uses for agent-loop smoke tests.

---

## `robot-external` — ROS2-native for real robots or external sims

```bash
source /opt/ros/jazzy/setup.bash
edgevox-agent robot-external --text-mode
```

Subscribes to `odom`, optionally `scan` / `camera/image_raw`; publishes `cmd_vel` and `goal_pose`. Drives any Gazebo Harmonic world, Isaac Sim (via the ROS2 bridge), or a physical mobile robot that speaks the standard contract — same agent code runs unchanged.

**Tools / skills:** `go_to_pose`, `stop`, `describe_surroundings`, `execute_skill` — mapped to the `edgevox_skills/<name>` action under the hood so a long-running skill reports feedback via ROS2 action topics.
**Underlying:** `edgevox.integrations.ros2_bridge` — `rclpy` is a system package (not pip-installable) provided by the sourced ROS2 workspace.

Good pick for: bridging EdgeVox into an existing Gazebo / Isaac / physical-robot stack. See also [ROS2 Integration](/documentation/ros2) for the bridge contract and QoS policy.

---

## Pick your starting point

| Want to… | Start with |
|---|---|
| Understand the harness with zero 3D setup | `robot-scout` |
| See voice → mobile-robot navigation end-to-end | `robot-irsim` |
| Prototype grasping / spatial tool-use | `robot-panda` |
| Drive a humanoid / test an RL policy | `robot-humanoid` |
| Plug into Gazebo / Isaac / a real robot | `robot-external` |

Every example is ≤500 lines of Python. Copy one, rename it, swap in your own tools — that's the recommended authoring path for new robots.
