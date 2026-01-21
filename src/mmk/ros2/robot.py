from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from lerobot.cameras import CameraConfig

from mmk.robot.base import CameraSetup, MotorSetup


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_motors(motors: dict[str, Any] | None) -> dict[str, MotorSetup]:
    if not motors:
        return {}
    parsed = {}
    for name, payload in motors.items():
        if isinstance(payload, MotorSetup):
            parsed[name] = payload
        elif isinstance(payload, dict):
            parsed[name] = MotorSetup(calibration=payload.get("calibration"))
        else:
            parsed[name] = MotorSetup(calibration=None)
    return parsed


def _parse_cameras(cameras: dict[str, Any] | None) -> dict[str, CameraSetup]:
    if not cameras:
        return {}
    parsed = {}
    for name, payload in cameras.items():
        if isinstance(payload, CameraSetup):
            parsed[name] = payload
            continue
        payload = payload or {}
        config_data = payload.get("config") or {}
        config = CameraConfig(
            fps=float(config_data.get("fps", 30)),
            width=int(config_data.get("width", 640)),
            height=int(config_data.get("height", 480)),
        )
        parsed[name] = CameraSetup(id=str(payload.get("id", name)), config=config)
    return parsed


def _build_robot(config_data: dict[str, Any]):
    robot_cfg = dict(config_data.get("robot", {}))
    robot_type = robot_cfg.pop("type", "mmk_mujoco_robot")
    if robot_type == "mmk_mujoco_robot":
        from mmk.robot.mujoco import Mujoco, MujocoConfig

        config_class = MujocoConfig
        robot_class = Mujoco
    elif robot_type == "mmk_isaac_robot":
        from mmk.robot.isaac import Isaac, IsaacConfig

        config_class = IsaacConfig
        robot_class = Isaac
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    robot_cfg["motors"] = _parse_motors(robot_cfg.get("motors"))
    robot_cfg["cameras"] = _parse_cameras(robot_cfg.get("cameras"))
    config = config_class(**robot_cfg)
    return robot_class(config)


class RobotNode(Node):
    def __init__(
        self,
        robot: Any,
        node_name: str = "mmk_robot",
        ros2_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_name)
        self.robot = robot
        self.motor_names = list(robot.config.motors)
        ros2_config = ros2_config or {}

        self.declare_parameter("topics", ros2_config.get("topics", []))
        self.declare_parameter(
            "publish_rate_hz", float(ros2_config.get("publish_rate_hz", 30.0))
        )
        self.declare_parameter(
            "connect_on_start", bool(ros2_config.get("connect_on_start", False))
        )

        self.topics_config = ros2_config.get("topics") or []
        self.publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self.connect_on_start = (
            self.get_parameter("connect_on_start").get_parameter_value().bool_value
        )

        if self.connect_on_start and not self.robot.is_connected:
            self.robot.connect()

        observation_entries = self._collect_topic_entries("observation")
        action_entries = self._collect_topic_entries("action")
        camera_entries = self._collect_topic_entries("camera")

        if not action_entries:
            raise ValueError("ros2.topics must include at least one action topic")
        if not observation_entries:
            raise ValueError("ros2.topics must include at least one observation topic")

        self.observation_pubs: list[tuple[list[str] | None, Any]] = []
        for entry in observation_entries:
            publisher = self.create_publisher(
                JointState, entry["topic"], qos_profile_sensor_data
            )
            self.observation_pubs.append((entry.get("motor_names"), publisher))

        self.action_subs: list[Any] = []
        for entry in action_entries:
            topic = entry["topic"]
            names = entry.get("motor_names")
            self.action_subs.append(
                self.create_subscription(
                    JointState,
                    topic,
                    lambda msg, names=names: self._on_action(msg, names),
                    qos_profile_default,
                )
            )

        self.camera_pubs: list[tuple[str, Any]] = []
        for entry in camera_entries:
            publisher = self.create_publisher(
                Image, entry["topic"], qos_profile_sensor_data
            )
            self.camera_pubs.append((entry["name"], publisher))

        self._publish_timer = None
        if self.publish_rate_hz > 0:
            self._publish_timer = self.create_timer(
                1.0 / self.publish_rate_hz, self._publish_observation
            )

    def _parse_topic_entry(self, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict):
            raise ValueError("ros2.topics entries must be dicts")
        kind = str(entry.get("kind") or entry.get("type") or "")
        topic = str(entry.get("topic") or "")
        if not kind:
            raise ValueError("ros2.topics entry missing kind")
        if not topic:
            raise ValueError("ros2.topics entry missing topic")
        parsed: dict[str, Any] = {"kind": kind, "topic": topic}
        names = entry.get("motor_names") or entry.get("names")
        if names is not None:
            parsed["motor_names"] = [str(name) for name in names]
        if kind == "camera":
            name = entry.get("name") or entry.get("camera_name")
            if not name:
                raise ValueError("camera topic entry missing name")
            parsed["name"] = str(name)
        return parsed

    def _collect_topic_entries(self, kind: str) -> list[dict[str, Any]]:
        entries = []
        for entry in self.topics_config:
            parsed = self._parse_topic_entry(entry)
            if parsed["kind"] == kind:
                entries.append(parsed)
        return entries

    def _on_action(self, msg: JointState, motor_names: list[str] | None = None) -> None:
        if not msg.position:
            return
        if not self.robot.is_connected and self.connect_on_start:
            self.robot.connect()
        if motor_names:
            names = list(motor_names)
        else:
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

        for motor_names, publisher in self.observation_pubs:
            names = motor_names or list(self.motor_names)
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = list(names)
            joint_msg.position = [
                float(observation.get(f"{name}.pos", 0.0)) for name in names
            ]
            publisher.publish(joint_msg)

        for camera_name, publisher in self.camera_pubs:
            image = observation.get(camera_name)
            if image is None:
                continue
            try:
                encoding, array = _normalize_image(image)
            except ValueError:
                continue
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = camera_name
            msg.height = int(array.shape[0])
            msg.width = int(array.shape[1])
            msg.encoding = encoding
            channels = 1 if array.ndim == 2 else int(array.shape[2])
            msg.step = msg.width * channels
            msg.data = array.tobytes()
            publisher.publish(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMK ROS2 robot bridge.")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--node_name", default="mmk_robot")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config_path:
        raise ValueError("--config_path is required")
    config_path = Path(args.config_path).expanduser()
    config = _load_yaml(config_path)
    robot = _build_robot(config)
    ros2_config = config.get("ros2", {})

    import rclpy

    rclpy.init()
    node = RobotNode(robot, node_name=args.node_name, ros2_config=ros2_config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
