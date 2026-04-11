"""ROS2 bridge for EdgeVox — publishes voice pipeline events to ROS2 topics.

Optional: requires 'rclpy' and a ROS2 environment. Install with:
    pip install edgevox[ros2]
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import ROS2 dependencies — they are entirely optional.
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


# ---------------------------------------------------------------------------
# ROS2Bridge — the real implementation (used when rclpy is present)
# ---------------------------------------------------------------------------
class ROS2Bridge:
    """Thin wrapper that exposes the voice pipeline to ROS2 topics.

    Publishers
    ----------
    /edgevox/transcription  (std_msgs/String)   user speech text
    /edgevox/response       (std_msgs/String)   bot response text
    /edgevox/state          (std_msgs/String)   pipeline state
    /edgevox/audio_level    (std_msgs/Float32)  mic audio level 0.0-1.0
    /edgevox/metrics        (std_msgs/String)   JSON-encoded latency metrics

    Subscribers
    -----------
    /edgevox/tts_request    (std_msgs/String)   external text-to-speak
    /edgevox/command        (std_msgs/String)   commands: reset / mute / unmute
    """

    def __init__(self) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not installed. Use create_bridge() for graceful fallback.")

        # Initialise rclpy if it hasn't been already.
        if not rclpy.ok():
            rclpy.init()

        self._node: Node = rclpy.create_node("edgevox")
        logger.info("ROS2 node 'edgevox' created")

        # Publishers
        self._pub_transcription = self._node.create_publisher(String, "/edgevox/transcription", 10)
        self._pub_response = self._node.create_publisher(String, "/edgevox/response", 10)
        self._pub_state = self._node.create_publisher(String, "/edgevox/state", 10)
        self._pub_audio_level = self._node.create_publisher(Float32, "/edgevox/audio_level", 10)
        self._pub_metrics = self._node.create_publisher(String, "/edgevox/metrics", 10)

        # Callbacks for subscribers (set externally via set_*_callback)
        self._tts_callback: Callable[[str], Any] | None = None
        self._command_callback: Callable[[str], Any] | None = None

        # Subscribers
        self._sub_tts_request = self._node.create_subscription(String, "/edgevox/tts_request", self._on_tts_request, 10)
        self._sub_command = self._node.create_subscription(String, "/edgevox/command", self._on_command, 10)

        self._spin_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    # -- publishers ----------------------------------------------------------

    def publish_transcription(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_transcription.publish(msg)
        logger.debug("Published transcription: %s", text[:80])

    def publish_response(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_response.publish(msg)
        logger.debug("Published response: %s", text[:80])

    def publish_state(self, state: str) -> None:
        msg = String()
        msg.data = state
        self._pub_state.publish(msg)
        logger.debug("Published state: %s", state)

    def publish_audio_level(self, level: float) -> None:
        msg = Float32()
        msg.data = max(0.0, min(1.0, float(level)))
        self._pub_audio_level.publish(msg)

    def publish_metrics(self, metrics: dict) -> None:
        msg = String()
        msg.data = json.dumps(metrics)
        self._pub_metrics.publish(msg)
        logger.debug("Published metrics")

    # -- subscriber callbacks ------------------------------------------------

    def set_tts_callback(self, callback: Callable[[str], Any]) -> None:
        self._tts_callback = callback

    def set_command_callback(self, callback: Callable[[str], Any]) -> None:
        self._command_callback = callback

    def _on_tts_request(self, msg: Any) -> None:
        text: str = msg.data
        logger.info("Received TTS request: %s", text[:80])
        if self._tts_callback is not None:
            try:
                self._tts_callback(text)
            except Exception:
                logger.exception("Error in TTS callback")

    def _on_command(self, msg: Any) -> None:
        command: str = msg.data
        logger.info("Received command: %s", command)
        if self._command_callback is not None:
            try:
                self._command_callback(command)
            except Exception:
                logger.exception("Error in command callback")

    # -- lifecycle -----------------------------------------------------------

    def spin(self) -> None:
        """Start spinning the ROS2 node in a daemon thread."""
        if self._spin_thread is not None and self._spin_thread.is_alive():
            logger.warning("Spin thread already running")
            return

        def _spin_worker() -> None:
            logger.info("ROS2 spin thread started")
            try:
                while not self._shutdown_event.is_set() and rclpy.ok():
                    rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                logger.exception("ROS2 spin thread error")
            finally:
                logger.info("ROS2 spin thread exiting")

        self._spin_thread = threading.Thread(target=_spin_worker, daemon=True, name="ros2-spin")
        self._spin_thread.start()

    def shutdown(self) -> None:
        """Gracefully shut down the node and spin thread."""
        logger.info("Shutting down ROS2 bridge")
        self._shutdown_event.set()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None

        try:
            self._node.destroy_node()
        except Exception:
            logger.debug("Node already destroyed or rclpy shut down")

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            logger.debug("rclpy already shut down")


# ---------------------------------------------------------------------------
# NullBridge — no-op stand-in when ROS2 is unavailable or disabled
# ---------------------------------------------------------------------------
class NullBridge:
    """Drop-in replacement for ROS2Bridge that silently discards everything."""

    def __init__(self) -> None:
        logger.debug("Using NullBridge (ROS2 disabled or unavailable)")

    def publish_transcription(self, text: str) -> None: ...
    def publish_response(self, text: str) -> None: ...
    def publish_state(self, state: str) -> None: ...
    def publish_audio_level(self, level: float) -> None: ...
    def publish_metrics(self, metrics: dict) -> None: ...

    def set_tts_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_command_callback(self, callback: Callable[[str], Any]) -> None: ...

    def spin(self) -> None: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_bridge(enabled: bool = False) -> ROS2Bridge | NullBridge:
    """Return a ROS2Bridge if *enabled* and rclpy is available, else NullBridge.

    Parameters
    ----------
    enabled:
        Set to ``True`` to attempt ROS2 initialisation.  When ``False``
        (the default) a lightweight :class:`NullBridge` is always returned.
    """
    if not enabled:
        return NullBridge()

    if not ROS2_AVAILABLE:
        logger.warning("ROS2 bridge requested but rclpy is not installed — falling back to NullBridge")
        return NullBridge()

    try:
        bridge = ROS2Bridge()
        bridge.spin()
        return bridge
    except Exception:
        logger.exception("Failed to create ROS2 bridge — falling back to NullBridge")
        return NullBridge()
