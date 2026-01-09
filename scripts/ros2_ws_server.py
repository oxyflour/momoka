import asyncio
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import websockets


class WsBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_ws_bridge")
        self.publisher = self.create_publisher(String, "lerobot/inference", 10)

    def publish_json(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(msg)
        self.get_logger().info(f"ws->ros2: {msg.data}")


async def ws_handler(websocket, node: WsBridgeNode) -> None:
    async for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            payload = {"raw": message}
        node.publish_json(payload)


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
