from __future__ import annotations

from typing import Any

from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


class ControlNode(Node):
    def __init__(self, teleop: Any, node_name: str = "mmk_control") -> None:
        super().__init__(node_name)
        self.teleop = teleop

        self.declare_parameter("observation_topic", "/mmk/observation")
        self.declare_parameter("action_topic", "/mmk/action")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("camera_topics", {})
        self.declare_parameter("connect_on_start", False)

        self.observation_topic = (
            self.get_parameter("observation_topic").get_parameter_value().string_value
        )
        self.action_topic = (
            self.get_parameter("action_topic").get_parameter_value().string_value
        )
        self.publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self.camera_topics = self._resolve_camera_topics()
        self.connect_on_start = (
            self.get_parameter("connect_on_start").get_parameter_value().bool_value
        )

        if self.connect_on_start and not self.teleop.is_connected:
            self.teleop.connect()

        self.action_pub = self.create_publisher(
            JointState, self.action_topic, qos_profile_default
        )
        self.observation_sub = self.create_subscription(
            JointState,
            self.observation_topic,
            self._on_observation,
            qos_profile_sensor_data,
        )

        self.camera_subs: dict[str, Any] = {}
        for name, topic in self.camera_topics.items():
            self.camera_subs[name] = self.create_subscription(
                Image, topic, self._on_image, qos_profile_sensor_data
            )

        self.last_observation: dict[str, float] = {}
        self.last_images: dict[str, Image] = {}

        self._publish_timer = None
        if self.publish_rate_hz > 0:
            self._publish_timer = self.create_timer(
                1.0 / self.publish_rate_hz, self._publish_action
            )

    def _resolve_camera_topics(self) -> dict[str, str]:
        declared = self.get_parameter("camera_topics").value
        if isinstance(declared, dict) and declared:
            return {str(key): str(value) for key, value in declared.items()}
        return {}

    def _on_observation(self, msg: JointState) -> None:
        names = list(msg.name)
        for name, position in zip(names, msg.position):
            self.last_observation[f"{name}.pos"] = float(position)

    def _on_image(self, msg: Image) -> None:
        self.last_images[msg.header.frame_id or "unknown"] = msg

    def _publish_action(self) -> None:
        if not self.teleop.is_connected:
            if self.connect_on_start:
                self.teleop.connect()
            else:
                return
        try:
            action = self.teleop.get_action()
        except Exception as exc:
            self.get_logger().warning("Failed to read teleop action: %s", exc)
            return

        names = []
        positions = []
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            names.append(key[:-4])
            positions.append(float(value))

        if not positions:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.action_pub.publish(msg)
