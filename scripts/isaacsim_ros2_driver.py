import json
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from omni.isaac.kit import SimulationApp


class Ros2ActionNode(Node):
    def __init__(self) -> None:
        super().__init__("isaacsim_lerobot_driver")
        self._latest_action: Optional[np.ndarray] = None
        self.create_subscription(String, "lerobot/inference", self._on_msg, 10)

    def _on_msg(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        action = payload.get("result", {}).get("action")
        if not isinstance(action, list) or not action:
            return

        self._latest_action = np.array(action, dtype=np.float32)

    def pop_action(self) -> Optional[np.ndarray]:
        action = self._latest_action
        self._latest_action = None
        return action


def main() -> None:
    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid

    rclpy.init()
    node = Ros2ActionNode()

    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/ActionCube",
            name="action_cube",
            position=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size=0.05,
            color=np.array([0.2, 0.6, 1.0], dtype=np.float32),
        )
    )
    world.reset()

    while simulation_app.is_running():
        rclpy.spin_once(node, timeout_sec=0.0)
        action = node.pop_action()
        if action is not None:
            pos, rot = cube.get_world_pose()
            delta = np.zeros(3, dtype=np.float32)
            delta[: min(3, action.shape[0])] = action[: min(3, action.shape[0])]
            cube.set_world_pose(pos + 0.01 * delta, rot)

        world.step(render=True)

    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
