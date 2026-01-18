from __future__ import annotations

from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


def _normalize_image(image: np.ndarray) -> tuple[str, np.ndarray]:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return "mono8", array
    if array.ndim == 3 and array.shape[2] == 3:
        return "rgb8", array
    if array.ndim == 3 and array.shape[2] == 4:
        return "rgba8", array
    raise ValueError("Unsupported image shape for ROS2 Image message")


class RobotNode(Node):
    def __init__(self, robot: Any, node_name: str = "mmk_robot") -> None:
        super().__init__(node_name)
        self.robot = robot
        self.motor_names = list(robot.config.motors)

        self.declare_parameter("observation_topic", "/mmk/observation")
        self.declare_parameter("action_topic", "/mmk/action")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("camera_topics", {})
        self.declare_parameter("camera_topic_prefix", "/mmk/camera")
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

        if self.connect_on_start and not self.robot.is_connected:
            self.robot.connect()

        self.observation_pub = self.create_publisher(
            JointState, self.observation_topic, qos_profile_sensor_data
        )
        self.action_sub = self.create_subscription(
            JointState, self.action_topic, self._on_action, qos_profile_default
        )

        self.camera_pubs: dict[str, Any] = {}
        for name, topic in self.camera_topics.items():
            self.camera_pubs[name] = self.create_publisher(
                Image, topic, qos_profile_sensor_data
            )

        self._publish_timer = None
        if self.publish_rate_hz > 0:
            self._publish_timer = self.create_timer(
                1.0 / self.publish_rate_hz, self._publish_observation
            )

    def _resolve_camera_topics(self) -> dict[str, str]:
        declared = self.get_parameter("camera_topics").value
        if isinstance(declared, dict) and declared:
            return {str(key): str(value) for key, value in declared.items()}
        prefix = (
            self.get_parameter("camera_topic_prefix").get_parameter_value().string_value
        )
        return {
            name: f"{prefix}/{name}"
            for name in getattr(self.robot.config, "cameras", {})
        }

    def _on_action(self, msg: JointState) -> None:
        if not msg.position:
            return
        if not self.robot.is_connected and self.connect_on_start:
            self.robot.connect()
        names = list(msg.name) if msg.name else list(self.motor_names)
        action = {
            f"{name}.pos": float(position)
            for name, position in zip(names, msg.position)
        }
        if not action:
            return
        try:
            self.robot.send_action(action)
        except Exception as exc:
            self.get_logger().warning("Failed to apply action: %s", exc)

    def _publish_observation(self) -> None:
        if not self.robot.is_connected:
            return
        try:
            observation = self.robot.get_observation()
        except Exception as exc:
            self.get_logger().warning("Failed to read observation: %s", exc)
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(self.motor_names)
        joint_msg.position = [
            float(observation.get(f"{name}.pos", 0.0)) for name in self.motor_names
        ]
        self.observation_pub.publish(joint_msg)

        for camera_name, publisher in self.camera_pubs.items():
            image = observation.get(camera_name)
            if image is None:
                continue
            try:
                encoding, array = _normalize_image(image)
            except ValueError:
                continue
            msg = Image()
            msg.header.stamp = joint_msg.header.stamp
            msg.header.frame_id = camera_name
            msg.height = int(array.shape[0])
            msg.width = int(array.shape[1])
            msg.encoding = encoding
            channels = 1 if array.ndim == 2 else int(array.shape[2])
            msg.step = msg.width * channels
            msg.data = array.tobytes()
            publisher.publish(msg)
