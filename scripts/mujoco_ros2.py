import argparse
import json
import os
import sys
import threading
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.append(os.path.normpath(__file__ + "/../../"))
from utils.env_base import EnvBase
from utils.robot_loader import RobotConfig, build_ctrl_range, load_config, resolve_urdf


def _extract_action(msg: String) -> Optional[np.ndarray]:
    try:
        payload = json.loads(msg.data)
    except json.JSONDecodeError:
        return None
    action = payload.get("result", {}).get("action")
    if not isinstance(action, list) or not action:
        return None
    return np.array(action, dtype=np.float32)


class GenericMujocoEnv(EnvBase):
    def __init__(self, config: RobotConfig) -> None:
        self._config = config
        urdf_path = resolve_urdf(config)
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        self._renderer: Optional[mujoco.Renderer] = None
        self._ctrl_range = build_ctrl_range(self.model, config)
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def step(self, action: Optional[np.ndarray]) -> None:
        ctrl = self.data.ctrl
        if action is not None:
            action = np.clip(action, -1.0, 1.0)
            count = min(action.shape[0], ctrl.shape[0], self._ctrl_range.shape[0])
            if count > 0:
                lo = self._ctrl_range[:count, 0]
                hi = self._ctrl_range[:count, 1]
                ctrl[:count] = lo + (action[:count] + 1.0) * 0.5 * (hi - lo)
        else:
            ctrl[:] = 0.0
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data, camera=camera or None)
        return self._renderer.render()

    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


class MujocoRosBridge(Node):
    def __init__(self, env: GenericMujocoEnv, config: RobotConfig) -> None:
        super().__init__(f"{config.name}_mujoco_bridge")
        self._action_lock = threading.Lock()
        self._latest_action: Optional[np.ndarray] = None
        self._env = env
        self._config = config
        self.subscription = self.create_subscription(
            String, "lerobot/inference", self._on_msg, 10
        )
        self.timer = self.create_timer(1.0 / 60.0, self._on_step)
        self.get_logger().info(
            f"{config.name} ready. Listening on /lerobot/inference and stepping at 60 Hz."
        )

    def _on_msg(self, msg: String) -> None:
        action = _extract_action(msg)
        if action is None:
            return
        with self._action_lock:
            self._latest_action = action

    def _on_step(self) -> None:
        with self._action_lock:
            action = self._latest_action
        self._env.step(action)


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo ROS 2 Bridge")
    parser.add_argument(
        "--robot",
        type=str,
        default="panda",
        help="Robot name (config file without .yaml)",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.robot)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(config.urdf):
        print(f"Error: URDF file not found: {config.urdf}", file=sys.stderr)
        sys.exit(1)
    rclpy.init()
    env = GenericMujocoEnv(config)
    node = MujocoRosBridge(env, config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
