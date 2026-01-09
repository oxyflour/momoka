import json
import threading
from typing import Optional

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_XML = """<mujoco model="ros2_action_demo">
  <option timestep="0.0166667"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 0 1" pos="0 0 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" ctrlrange="-1 1" gear="1"/>
  </actuator>
</mujoco>
"""


class MujocoRosBridge(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_ros_bridge")
        self._action_lock = threading.Lock()
        self._latest_action: Optional[np.ndarray] = None

        self.model = mujoco.MjModel.from_xml_string(DEFAULT_XML)
        self.data = mujoco.MjData(self.model)

        self.subscription = self.create_subscription(
            String, "lerobot/inference", self._on_msg, 10
        )
        self.timer = self.create_timer(1.0 / 60.0, self._on_step)

        self.get_logger().info(
            "MuJoCo sim ready. Listening on /lerobot/inference and stepping at 60 Hz."
        )

    def _on_msg(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid JSON payload")
            return

        action = payload.get("result", {}).get("action")
        print(action)
        if not isinstance(action, list) or not action:
            return

        with self._action_lock:
            self._latest_action = np.array(action, dtype=np.float32)

    def _on_step(self) -> None:
        ctrl = self.data.ctrl
        with self._action_lock:
            action = self._latest_action

        if action is not None:
            ctrl[0] = float(np.clip(action[0], -1.0, 1.0))
        else:
            ctrl[0] = 0.0

        mujoco.mj_step(self.model, self.data)


def main() -> None:
    rclpy.init()
    node = MujocoRosBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
