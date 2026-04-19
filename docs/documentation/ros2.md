# ROS2 Integration

EdgeVox includes an optional ROS2 bridge for robotics applications. It exposes the full voice pipeline — streaming tokens, sentences, state changes, wake word events, and query APIs — as standard ROS2 topics with appropriate QoS profiles.

## Enable

```bash
# Source your ROS2 workspace (rclpy must be importable)
source /opt/ros/jazzy/setup.bash

# Voice pipeline (TUI) over ROS2
edgevox --ros2

# Custom namespace — typical multi-robot setup
edgevox --ros2 --ros2-namespace /robot1/voice

# Agent examples expose the same bridge plus robot_state / agent_event
edgevox-agent robot-irsim --text-mode --ros2
edgevox-agent robot-panda --text-mode --ros2 --ros2-namespace /robot1/arm
```

If `rclpy` is not available, the bridge falls back to a `NullBridge` (no-op) and EdgeVox runs normally without ROS2.

### Running the pytest suite inside a sourced ROS2 workspace

ROS2 ships a `launch_testing` pytest plugin that declares a hook incompatible with modern pytest (`pytest_launch_collect_makemodule`). When a sourced ROS2 environment is on `PYTHONPATH`, pytest loads the plugin at startup and fails collection. Disable plugin autoloading before invoking pytest:

```bash
source /opt/ros/jazzy/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest tests/
```

Outside a ROS2 env the variable has no effect.

### Using the Launch Files

```bash
# Voice pipeline only
ros2 launch edgevox edgevox.launch.py
ros2 launch edgevox edgevox.launch.py namespace:=/robot1/voice language:=vi

# IR-SIM agent demo (headless by default, voice + ROS2)
ros2 launch edgevox edgevox_irsim.launch.py
ros2 launch edgevox edgevox_irsim.launch.py namespace:=/robot1/voice

# MuJoCo Panda pick-and-place demo
ros2 launch edgevox edgevox_panda.launch.py
ros2 launch edgevox edgevox_panda.launch.py namespace:=/robot1/arm state_hz:=20.0
```

### Tier 2b — MuJoCo humanoid (`robot-humanoid`)

```bash
edgevox-agent robot-humanoid --simple-ui                        # Unitree G1, viewer + voice (default)
edgevox-agent robot-humanoid --simple-ui --model-source unitree_h1
edgevox-agent robot-humanoid --text-mode --no-render --ros2     # headless chat + ROS2 bridge
edgevox-agent robot-humanoid --mjcf /path/to/scene.xml          # custom MJCF
```

Models are auto-fetched from `nrl-ai/edgevox-models/mujoco_scenes/` on HuggingFace (~15-17 MB per robot, cached) with a git sparse-clone fallback to `google-deepmind/mujoco_menagerie`.

Skills: `walk_forward(distance)`, `walk_backward(distance)`, `turn_left(degrees)`, `turn_right(degrees)`, `stand`, `get_pose`. The adapter runs a procedural walking gait that swings `{side}_hip_pitch`, `{side}_knee`, `{side}_ankle_pitch` (and counter-swings `{side}_shoulder_pitch`, `{side}_elbow`) so legs + arms visibly step. Plug in an ONNX walking policy via `MujocoHumanoidEnvironment.set_walking_policy(...)` for real RL locomotion.

### Tier 3 — external ROS2 sim / real robot (`robot-external`)

EdgeVox drives any ROS2 process that speaks the standard mobile-robot contract — Gazebo Harmonic, Isaac Sim (via ROS2 bridge), a real Unitree Go2, etc.

```bash
# Terminal A — your sim or robot (whatever you prefer)
ros2 launch nav2_bringup tb3_simulation_launch.py use_sim_time:=True

# Terminal B — the voice agent
edgevox-agent robot-external --text-mode
edgevox-agent robot-external --text-mode --namespace /robot1
```

Contract:

| Dir | Topic               | Type                          |
|-----|---------------------|-------------------------------|
| sub | `odom`              | `nav_msgs/Odometry`           |
| sub | `scan`              | `sensor_msgs/LaserScan` (optional) |
| sub | `camera/image_raw`  | `sensor_msgs/Image` (optional)|
| pub | `cmd_vel`           | `geometry_msgs/Twist`         |
| pub | `goal_pose`         | `geometry_msgs/PoseStamped`   |

Skills: `navigate_to(location)` or `navigate_xy(x, y)`, `stop`, `get_pose`.

With the IR-SIM launch file running, drive the robot from another terminal:

```bash
ros2 topic pub --once /edgevox/text_input std_msgs/msg/String "{data: 'go to kitchen'}"
ros2 topic echo /edgevox/robot_state --once
ros2 topic echo /edgevox/agent_event
```

## Namespace and Topic Names

All topic names are **relative** to the node namespace. The default namespace is `/edgevox`, so topics appear as `/edgevox/transcription`, `/edgevox/state`, etc.

To run multiple instances (e.g., on different robots), set a custom namespace:

```bash
edgevox --ros2 --ros2-namespace /robot1/voice   # → /robot1/voice/transcription
edgevox --ros2 --ros2-namespace /robot2/voice   # → /robot2/voice/transcription
```

## QoS Profiles

Topics use QoS profiles matched to their data characteristics:

| Profile | Reliability | Durability | Depth | Used by |
|---------|------------|------------|-------|---------|
| **State** | Reliable | Transient Local | 1 | `state` — late joiners get the current state immediately |
| **Sensor** | Best Effort | Volatile | 5 | `audio_level`, `bot_token` — high-frequency, ephemeral data |
| **Reliable** | Reliable | Volatile | 10 | All other topics — commands and events must not be lost |

When subscribing from your own nodes, match the QoS profile or use a compatible one (e.g., `BEST_EFFORT` subscribers can receive from `RELIABLE` publishers).

## Node Parameters

The EdgeVox node exposes configuration as ROS2 parameters, allowing integration with launch files, YAML config, and `ros2 param`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `en` | Language ISO 639-1 code |
| `voice` | string | `""` | TTS voice name |
| `muted` | bool | `false` | Mute/unmute the microphone |

```bash
# Change language at runtime via parameter
ros2 param set /edgevox/edgevox language vi

# Mute the microphone
ros2 param set /edgevox/edgevox muted true
```

Parameters and topic-based commands (`set_language`, `command`) both work — use whichever fits your integration pattern.

## Robot interop (Nav2 / TF2 / sensors)

When `edgevox-agent --ros2` launches with a sim attached, the bridge also brings up a `RobotROS2Adapter` that wires the sim into standard ROS2 robotics topics:

| Topic | Direction | Type | Notes |
|-------|-----------|------|-------|
| `/tf` | out | `tf2_msgs/TFMessage` | `map → base_link` (IR-SIM) or `map → ee_link` (MuJoCo) at 30 Hz |
| `pose` | out | `geometry_msgs/PoseStamped` | same pose as TF for non-TF consumers |
| `scan` | out | `sensor_msgs/LaserScan` | 2D lidar (IR-SIM only) at 10 Hz |
| `image_raw` | out | `sensor_msgs/Image` | offscreen camera (MuJoCo, rgb8) at 5 Hz — set `MUJOCO_GL=egl` or `osmesa` on headless hosts |
| `cmd_vel` | in | `geometry_msgs/Twist` | Nav2-style velocity command (IR-SIM); times out after 0.5 s so a dropped stream halts the robot |
| `goal_pose` | in | `geometry_msgs/PoseStamped` | 2D goal for IR-SIM (`navigate_to`) or 3D target for MuJoCo (`move_to`) |

Each capability is enabled only if the underlying sim exposes the relevant method, so a sim without a lidar simply doesn't advertise `scan`.

## Services

Query commands are exposed as `std_srvs/srv/Trigger` services under the same namespace — `response.message` carries a JSON payload:

```bash
ros2 service call /edgevox/list_voices std_srvs/srv/Trigger
ros2 service call /edgevox/hardware_info std_srvs/srv/Trigger
```

The legacy pub/sub path (`/command` → `/info`) still works unchanged.

## Actions — `execute_skill`

Build the companion interface package once:

```bash
cd ~/ros_ws
ln -s <edgevox-repo>/edgevox_msgs src/edgevox_msgs   # or copy
colcon build --packages-select edgevox_msgs
source install/setup.bash
```

Then with `edgevox-agent --ros2` running, send a goal to the agent's skill dispatcher:

```bash
ros2 action send_goal /edgevox/execute_skill edgevox_msgs/action/ExecuteSkill \
  "{skill_name: 'navigate_to', arguments_json: '{\"room\": \"kitchen\"}'}" --feedback
```

`arguments_json` is a JSON object string that maps keyword → value, so the same action works for every skill the agent exposes (`navigate_to`, `move_to`, `grasp`, …) without per-skill IDL. Feedback arrives as JSON mirroring `/agent_event`; the result has `ok` + `result_json` + `error`.

If `edgevox_msgs` isn't built, the bridge prints a one-line skip note and the rest of the stack (topics, services, TF) continues to work.

## Published Topics

### Pipeline Output

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `transcription` | `String` | Reliable | User's speech text (STT result) |
| `response` | `String` | Reliable | Bot's full response text |
| `state` | `String` | State | Pipeline state: `listening`, `transcribing`, `thinking`, `speaking`, `interrupted` |
| `audio_level` | `Float32` | Sensor | Microphone level (0.0 - 1.0) |
| `metrics` | `String` | Reliable | JSON latency metrics (stt, llm, tts, ttfs, total) |
| `robot_state` | `String` | State | JSON snapshot of the sim/robot world (pose, battery, grasped object, …). Republished at `--ros2-state-hz` (default 10) by `edgevox-agent --ros2`. |
| `agent_event` | `String` | Reliable | JSON stream of agent events: `tool_call`, `skill_goal`, `skill_cancelled`, `handoff`, `safety_preempt`. Emitted by `edgevox-agent --ros2`. |

### Streaming Output

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `bot_token` | `String` | Sensor | Individual LLM tokens as they are generated |
| `bot_sentence` | `String` | Reliable | Complete sentences as they are sent to TTS |

### Events

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `wakeword` | `String` | Reliable | Wake word detection event (e.g. "hey jarvis") |
| `info` | `String` | Reliable | JSON responses to query commands |

## Subscribed Topics

### Actions

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `tts_request` | `String` | Reliable | Text to synthesize and play aloud |
| `text_input` | `String` | Reliable | Text sent to LLM, bypassing STT (like typing in the TUI) |
| `interrupt` | `String` | Reliable | Interrupt current bot response (any message triggers it) |
| `set_language` | `String` | Reliable | Switch language at runtime (ISO 639-1 code, e.g. `vi`, `fr`) |
| `set_voice` | `String` | Reliable | Switch TTS voice at runtime (e.g. `af_bella`, `de-thorsten`) |

### Commands

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `command` | `String` | Reliable | Commands: `reset`, `mute`, `unmute`, or query commands (see below) |

## Query Commands

Send one of these strings to `command` and the response is published on `info` as JSON:

| Command | Response fields |
|---------|-----------------|
| `list_voices` | `language`, `current_voice`, `voices[]` |
| `list_languages` | `current`, `languages{code: {name, stt_backend, tts_backend}}` |
| `hardware_info` | `cuda`, `metal`, `gpu_name`, `vram_total_gb`, `vram_used_mb`, `ram_gb` |
| `model_info` | `language`, `voice`, `stt{backend, model_size, device}`, `llm{model_path, device}`, `tts{backend, voice, sample_rate}` |

All responses include a `"query"` field with the command name.

## Docker

The `Dockerfile.ros2` provides a multi-stage build that keeps the runtime image small. The container runs as a non-root `edgevox` user for security.

```bash
# Build and run
docker compose -f docker-compose.ros2.yml up --build

# Interactive shell
docker compose -f docker-compose.ros2.yml run edgevox-ros2 bash
```

Set `ROS_DOMAIN_ID` in the compose file to isolate ROS2 traffic between robots.

## Examples

### Listen to transcriptions and responses

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VoiceListener(Node):
    def __init__(self):
        super().__init__('voice_listener')
        self.create_subscription(String, '/edgevox/transcription', self.on_user, 10)
        self.create_subscription(String, '/edgevox/response', self.on_bot, 10)

    def on_user(self, msg):
        self.get_logger().info(f'User said: {msg.data}')

    def on_bot(self, msg):
        self.get_logger().info(f'Bot replied: {msg.data}')
```

### Make the robot speak

```python
pub = node.create_publisher(String, '/edgevox/tts_request', 10)
msg = String()
msg.data = "I detected an obstacle ahead."
pub.publish(msg)
```

### Send text to the LLM (bypass microphone)

```python
pub = node.create_publisher(String, '/edgevox/text_input', 10)
msg = String()
msg.data = "What is the weather like today?"
pub.publish(msg)
```

### Interrupt the bot mid-response

```python
pub = node.create_publisher(String, '/edgevox/interrupt', 10)
pub.publish(String(data=""))
```

### Switch language at runtime

```python
# Via topic
pub = node.create_publisher(String, '/edgevox/set_language', 10)
pub.publish(String(data="vi"))  # Switch to Vietnamese

# Or via parameter
# ros2 param set /edgevox/edgevox language vi
```

### Switch voice at runtime

```python
pub = node.create_publisher(String, '/edgevox/set_voice', 10)
pub.publish(String(data="af_bella"))
```

### Query available voices

```python
import json
from std_msgs.msg import String

# Subscribe to responses
def on_info(msg):
    data = json.loads(msg.data)
    if data.get("query") == "list_voices":
        print(f"Current voice: {data['current_voice']}")
        print(f"Available: {data['voices']}")

node.create_subscription(String, '/edgevox/info', on_info, 10)

# Send query
cmd_pub = node.create_publisher(String, '/edgevox/command', 10)
cmd_pub.publish(String(data="list_voices"))
```

### Stream bot tokens for a display

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Match the BEST_EFFORT QoS used by bot_token
sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

class TokenDisplay(Node):
    def __init__(self):
        super().__init__('token_display')
        self._buffer = ""
        self.create_subscription(String, '/edgevox/bot_token', self.on_token, sensor_qos)
        self.create_subscription(String, '/edgevox/state', self.on_state, 10)

    def on_token(self, msg):
        self._buffer += msg.data
        # Update robot screen with partial response

    def on_state(self, msg):
        if msg.data == "listening":
            self._buffer = ""  # Reset for next turn
```

### React to wake word

```python
def on_wakeword(msg):
    print(f"Wake word detected: {msg.data}")
    # Robot turns to face the user, activates LEDs, etc.

node.create_subscription(String, '/edgevox/wakeword', on_wakeword, 10)
```
