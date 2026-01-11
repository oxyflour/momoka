import json
import os
import shutil
import tempfile
import threading
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import os, sys

sys.path.append(os.path.normpath(__file__ + "/../../"))
from utils.env_base import EnvBase


def _extract_action(msg: String) -> Optional[np.ndarray]:
    try:
        payload = json.loads(msg.data)
    except json.JSONDecodeError:
        return None

    action = payload.get("result", {}).get("action")
    if not isinstance(action, list) or not action:
        return None
    return np.array(action, dtype=np.float32)


def _resolve_panda_model_path() -> str:
    env_path = os.environ.get("PANDA_URDF") or os.environ.get("MUJOCO_PANDA_XML")
    if env_path:
        return env_path
    urdf_candidate = os.path.join(
        "data", "PandaRobot.jl-master", "deps", "Panda", "panda.urdf"
    )
    if os.path.exists(urdf_candidate):
        return urdf_candidate
    return os.path.join("data", "mujoco_menagerie", "franka_emika_panda", "panda.xml")


def _prepare_urdf_path(model_path: str) -> str:
    if not model_path.lower().endswith(".urdf"):
        return model_path
    with open(model_path, "r") as f:
        content = f.read()
    if "package://" not in content:
        return model_path
    base_dir = os.path.abspath(os.path.dirname(model_path))
    temp_dir = tempfile.mkdtemp()
    collision_dir = os.path.join(base_dir, "meshes", "collision")
    if os.path.isdir(collision_dir):
        for name in os.listdir(collision_dir):
            src = os.path.join(collision_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(temp_dir, name))
    content = content.replace("package://Panda/meshes/collision/", "")
    content = content.replace("package://panda/meshes/collision/", "")
    content = content.replace("package://Panda/meshes/", "")
    content = content.replace("package://panda/meshes/", "")
    content = content.replace("package://Panda/", "")
    content = content.replace("package://panda/", "")
    temp_path = os.path.join(temp_dir, os.path.basename(model_path))
    with open(temp_path, "w") as f:
        f.write(content)
    return temp_path


def _resolve_ctrl_range(model: mujoco.MjModel) -> np.ndarray:
    ctrl_range = model.actuator_ctrlrange.copy()
    if ctrl_range.size == 0:
        return ctrl_range
    if not np.allclose(ctrl_range[:, 0], ctrl_range[:, 1]):
        return ctrl_range

    fallback = np.zeros_like(ctrl_range, dtype=np.float32)
    for idx in range(ctrl_range.shape[0]):
        joint_id = model.actuator_trnid[idx, 0]
        if 0 <= joint_id < model.jnt_range.shape[0]:
            joint_range = model.jnt_range[joint_id]
            if np.allclose(joint_range[0], joint_range[1]):
                joint_range = np.array([-1.0, 1.0], dtype=np.float32)
            fallback[idx] = joint_range
        else:
            fallback[idx] = np.array([-1.0, 1.0], dtype=np.float32)
    return fallback


class PandaMujocoEnv(EnvBase):
    def __init__(self, model_path: str) -> None:
        model_path = _prepare_urdf_path(model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._renderer: Optional[mujoco.Renderer] = None
        self._ctrl_range = _resolve_ctrl_range(self.model)
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def step(self, action: Optional[np.ndarray]) -> None:
        ctrl = self.data.ctrl
        if action is not None:
            action = np.clip(action, -1.0, 1.0)
            count = min(action.shape[0], ctrl.shape[0])
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
    def __init__(self, env: PandaMujocoEnv) -> None:
        super().__init__("mujoco_panda_bridge")
        self._action_lock = threading.Lock()
        self._latest_action: Optional[np.ndarray] = None
        self._env = env

        self.subscription = self.create_subscription(
            String, "lerobot/inference", self._on_msg, 10
        )
        self.timer = self.create_timer(1.0 / 60.0, self._on_step)

        self.get_logger().info(
            "MuJoCo Panda ready. Listening on /lerobot/inference and stepping at 60 Hz."
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
    model_path = _resolve_panda_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Panda model not found at "
            f"{model_path}. Set PANDA_URDF or MUJOCO_PANDA_XML before running."
        )

    rclpy.init()
    env = PandaMujocoEnv(model_path)
    node = MujocoRosBridge(env)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
