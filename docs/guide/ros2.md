# ROS2 Integration

EdgeVox includes an optional ROS2 bridge for robotics applications.

## Overview

The ROS2 bridge publishes voice pipeline events as ROS2 topics and subscribes to incoming commands.

## Enable

```bash
# Install rclpy in your environment
pip install rclpy

# Launch with ROS2
python -m edgevox tui --ros2
```

If `rclpy` is not available, the bridge falls back to a `NullBridge` (no-op).

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/edgevox/transcription` | `String` | User's speech text |
| `/edgevox/response` | `String` | Bot's full reply |
| `/edgevox/state` | `String` | Pipeline state (listening, transcribing, thinking, speaking) |
| `/edgevox/audio_level` | `Float32` | Microphone level (0.0 - 1.0) |
| `/edgevox/metrics` | `String` | JSON latency metrics |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/edgevox/tts_request` | `String` | Text to synthesize and play |
| `/edgevox/command` | `String` | Slash command to execute |

## Example: Robot Integration

```python
import rclpy
from std_msgs.msg import String

# Listen for transcriptions
def on_transcription(msg):
    print(f"User said: {msg.data}")

# Send TTS request
pub.publish(String(data="I am navigating to the kitchen"))

# Send command
cmd_pub.publish(String(data="/lang vi"))
```
