import asyncio
import base64
import json
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String
import websockets


class WsBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_ws_bridge")
        self.publisher = self.create_publisher(String, "lerobot/inference", 10)
        self.image_topic = self.declare_parameter(
            "compressed_image_topic", "lerobot/image/compressed"
        ).value
        self.joint_state_topic = self.declare_parameter(
            "joint_state_topic", "lerobot/joint_state"
        ).value
        self.image_publishers: dict[str, Any] = {}
        self.joint_state_publishers: dict[str, Any] = {}
        self.image_publishers[self.image_topic] = self.create_publisher(
            CompressedImage, self.image_topic, 10
        )
        self.joint_state_publishers[self.joint_state_topic] = self.create_publisher(
            JointState, self.joint_state_topic, 10
        )

    def publish_json(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(msg)
        self.get_logger().info(f"ws->ros2: {msg.data}")

    def publish_payload(self, payload: dict) -> None:
        msg_type = payload.get("type")
        if msg_type == "compressed_image":
            image_msg = self._build_compressed_image(payload)
            if image_msg is None:
                return
            topic = self._resolve_topic(payload.get("topic"), self.image_topic)
            publisher = self._get_image_publisher(topic)
            publisher.publish(image_msg)
            self.get_logger().info(
                f"ws->ros2: CompressedImage {len(image_msg.data)} bytes -> {topic}"
            )
            return
        if msg_type == "joint_state":
            joint_msg = self._build_joint_state(payload)
            if joint_msg is None:
                return
            topic = self._resolve_topic(payload.get("topic"), self.joint_state_topic)
            publisher = self._get_joint_state_publisher(topic)
            publisher.publish(joint_msg)
            self.get_logger().info(f"ws->ros2: JointState -> {topic}")
            return
        self.publish_json(payload)

    def _build_compressed_image(self, payload: dict) -> Optional[CompressedImage]:
        data = payload.get("data")
        image_bytes = self._decode_bytes(data)
        if image_bytes is None:
            self.get_logger().warning("CompressedImage missing or invalid data")
            return None
        msg = CompressedImage()
        msg.data = image_bytes
        image_format = payload.get("format")
        if isinstance(image_format, str):
            msg.format = image_format
        else:
            msg.format = "jpeg"
        self._apply_header(msg, payload.get("header"))
        return msg

    def _build_joint_state(self, payload: dict) -> Optional[JointState]:
        msg = JointState()
        name = payload.get("name")
        if isinstance(name, list):
            msg.name = [str(item) for item in name]
        msg.position = self._float_list(payload.get("position"))
        msg.velocity = self._float_list(payload.get("velocity"))
        msg.effort = self._float_list(payload.get("effort"))
        self._apply_header(msg, payload.get("header"))
        return msg

    def _apply_header(self, msg: Any, header: Any) -> None:
        if not isinstance(header, dict):
            return
        frame_id = header.get("frame_id")
        if isinstance(frame_id, str):
            msg.header.frame_id = frame_id
        stamp = header.get("stamp")
        if isinstance(stamp, dict):
            sec = stamp.get("sec")
            nanosec = stamp.get("nanosec")
            if sec is not None:
                msg.header.stamp.sec = int(sec)
            if nanosec is not None:
                msg.header.stamp.nanosec = int(nanosec)
        elif isinstance(stamp, (int, float)):
            sec = int(stamp)
            nanosec = int((stamp - sec) * 1_000_000_000)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec

    def _float_list(self, value: Any) -> list[float]:
        if not isinstance(value, list):
            return []
        result: list[float] = []
        for item in value:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return result

    def _decode_bytes(self, data: Any) -> Optional[bytes]:
        if isinstance(data, str):
            try:
                return base64.b64decode(data)
            except (ValueError, TypeError):
                return None
        if isinstance(data, list):
            try:
                return bytes(data)
            except (ValueError, TypeError):
                return None
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return None

    def _resolve_topic(self, topic: Any, fallback: str) -> str:
        if isinstance(topic, str) and topic:
            return topic
        return fallback

    def _get_image_publisher(self, topic: str):
        publisher = self.image_publishers.get(topic)
        if publisher is None:
            publisher = self.create_publisher(CompressedImage, topic, 10)
            self.image_publishers[topic] = publisher
        return publisher

    def _get_joint_state_publisher(self, topic: str):
        publisher = self.joint_state_publishers.get(topic)
        if publisher is None:
            publisher = self.create_publisher(JointState, topic, 10)
            self.joint_state_publishers[topic] = publisher
        return publisher


async def ws_handler(websocket, node: WsBridgeNode) -> None:
    async for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            payload = {"raw": message}
        if isinstance(payload, dict):
            node.publish_payload(payload)
        else:
            node.publish_json({"raw": payload})


async def main_async(host: str, port: int) -> None:
    rclpy.init()
    node = WsBridgeNode()
    async with websockets.serve(lambda ws: ws_handler(ws, node), host, port):
        node.get_logger().info(f"WebSocket server listening on ws://{host}:{port}")
        try:
            await asyncio.Future()
        finally:
            node.destroy_node()
            rclpy.shutdown()


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    asyncio.run(main_async(host, port))


if __name__ == "__main__":
    main()
