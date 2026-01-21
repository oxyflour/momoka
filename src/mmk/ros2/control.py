from __future__ import annotations

import argparse
import base64
import json
from collections import deque
from pathlib import Path
from typing import Any
from urllib import request

from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _encode_image(msg: Image) -> dict[str, Any]:
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "encoding": msg.encoding,
        "step": int(msg.step),
        "data": base64.b64encode(bytes(msg.data)).decode("utf-8"),
    }


class PolicyClient:
    def __init__(
        self,
        server_url: str,
        client_id: str,
        chunk_size: int = 1,
        timeout_s: float = 1.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id
        self.chunk_size = max(int(chunk_size), 1)
        self.timeout_s = float(timeout_s)

    def request_actions(
        self,
        observation: dict[str, Any],
        action_names: list[str],
    ) -> list[dict[str, Any]]:
        payload = {
            "client_id": self.client_id,
            "observation": observation,
            "chunk_size": self.chunk_size,
            "action_names": action_names,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.server_url}/act",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        return list(data.get("actions") or [])


class ControlNode(Node):
    def __init__(
        self,
        policy_client: PolicyClient,
        node_name: str = "mmk_control",
        ros2_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_name)
        self.policy_client = policy_client
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

        observation_entries = self._collect_topic_entries("observation")
        action_entries = self._collect_topic_entries("action")
        camera_entries = self._collect_topic_entries("camera")
        if not observation_entries:
            raise ValueError("ros2.topics must include at least one observation topic")
        if not action_entries:
            raise ValueError("ros2.topics must include at least one action topic")

        self.action_pubs: list[tuple[list[str] | None, Any]] = []
        for entry in action_entries:
            publisher = self.create_publisher(
                JointState, entry["topic"], qos_profile_default
            )
            self.action_pubs.append((entry.get("motor_names"), publisher))

        self.observation_subs: list[Any] = []
        for entry in observation_entries:
            topic = entry["topic"]
            names = entry.get("motor_names")
            self.observation_subs.append(
                self.create_subscription(
                    JointState,
                    topic,
                    lambda msg, names=names: self._on_observation(msg, names),
                    qos_profile_sensor_data,
                )
            )

        self.camera_subs: dict[str, Any] = {}
        for entry in camera_entries:
            name = entry["name"]
            topic = entry["topic"]
            self.camera_subs[name] = self.create_subscription(
                Image,
                topic,
                lambda msg, name=name: self._on_image(name, msg),
                qos_profile_sensor_data,
            )

        self.last_observation: dict[str, float] = {}
        self.last_images: dict[str, Image] = {}
        self.pending_actions: deque[dict[str, Any]] = deque()

        self._publish_timer = None
        if self.publish_rate_hz > 0:
            self._publish_timer = self.create_timer(
                1.0 / self.publish_rate_hz, self._publish_action
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

    def _on_observation(
        self, msg: JointState, motor_names: list[str] | None = None
    ) -> None:
        names = motor_names or list(msg.name)
        if not names:
            return
        for name, position in zip(names, msg.position):
            self.last_observation[f"{name}.pos"] = float(position)

    def _on_image(self, name: str, msg: Image) -> None:
        self.last_images[name] = msg

    def _build_observation_payload(self) -> dict[str, Any]:
        images = {
            name: _encode_image(msg) for name, msg in self.last_images.items()
        }
        return {"state": dict(self.last_observation), "images": images}

    def _publish_action(self) -> None:
        if self.pending_actions:
            action = self.pending_actions.popleft()
            self._publish_action_message(action)
            return

        if not self.last_observation:
            return

        action_names = [
            key[:-4] for key in self.last_observation if key.endswith(".pos")
        ]
        if not action_names:
            return

        observation = self._build_observation_payload()
        try:
            actions = self.policy_client.request_actions(observation, action_names)
        except Exception as exc:
            self.get_logger().warning("Failed to fetch policy action: %s", exc)
            return

        if not actions:
            return
        self.pending_actions.extend(actions)
        action = self.pending_actions.popleft()
        self._publish_action_message(action)

    def _publish_action_message(self, action: dict[str, Any]) -> None:
        names = []
        positions = []
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            names.append(key[:-4])
            positions.append(float(value))

        if not positions:
            return

        for motor_names, publisher in self.action_pubs:
            filtered_names = []
            filtered_positions = []
            if motor_names:
                allowed = set(motor_names)
                for name, value in zip(names, positions):
                    if name in allowed:
                        filtered_names.append(name)
                        filtered_positions.append(value)
            else:
                filtered_names = list(names)
                filtered_positions = list(positions)
            if not filtered_positions:
                continue
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = filtered_names
            msg.position = filtered_positions
            publisher.publish(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMK ROS2 policy controller.")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--node_name", default="mmk_control")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config_path:
        raise ValueError("--config_path is required")
    config_path = Path(args.config_path).expanduser()
    config = _load_yaml(config_path)

    control_config = config.get("control", {})
    server_url = control_config.get("policy_server_url", "http://127.0.0.1:8000")
    chunk_size = int(control_config.get("chunk_size", 1))
    timeout_s = float(control_config.get("request_timeout_s", 1.0))
    client_id = str(control_config.get("client_id", args.node_name))
    policy_client = PolicyClient(
        server_url=server_url,
        client_id=client_id,
        chunk_size=chunk_size,
        timeout_s=timeout_s,
    )

    ros2_config = config.get("ros2", {})

    import rclpy

    rclpy.init()
    node = ControlNode(policy_client, node_name=args.node_name, ros2_config=ros2_config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
