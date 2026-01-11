import argparse
import json
import os
import sys
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from omni.isaac.kit import SimulationApp
except Exception:  # Allows importing this module outside Isaac Sim.
    SimulationApp = None

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


def _is_urdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".urdf"


def _import_urdf(urdf_path: str, prim_path: str = "/World/Robot") -> str:
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    from isaacsim.asset.importer.urdf import _urdf
    import omni.kit.commands
    import omni.usd

    import_config = _urdf.ImportConfig()
    import_config.fix_base = True
    import_config.merge_fixed_joints = False

    result, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    stage = omni.usd.get_context().get_stage()
    if not imported_path:
        raise RuntimeError("URDF import returned an empty prim path.")
    if stage.GetPrimAtPath(imported_path).IsValid() is False:
        raise RuntimeError(f"URDF import produced invalid prim: {imported_path}")
    if prim_path and imported_path != prim_path:
        moved, _ = omni.kit.commands.execute(
            "MovePrim", path_from=imported_path, path_to=prim_path
        )
        if moved and stage.GetPrimAtPath(prim_path).IsValid():
            return prim_path
    return imported_path


def _resolve_ctrl_range_isaacsim(
    ctrl_range: Optional[np.ndarray], num_dof: int, robot
) -> np.ndarray:
    if ctrl_range is not None and ctrl_range.size:
        return ctrl_range.astype(np.float32, copy=False)
    limits = None
    try:
        limits = robot.get_joint_limits()
    except Exception:
        limits = None
    if limits is None:
        return np.tile(np.array([-1.0, 1.0], dtype=np.float32), (num_dof, 1))
    limits = np.asarray(limits, dtype=np.float32)
    if limits.ndim == 3:
        limits = limits[0]
    if limits.ndim != 2 or limits.shape[1] != 2:
        return np.tile(np.array([-1.0, 1.0], dtype=np.float32), (num_dof, 1))
    if limits.shape[0] < num_dof:
        fallback = np.tile(np.array([-1.0, 1.0], dtype=np.float32), (num_dof, 1))
        fallback[: limits.shape[0]] = limits
        return fallback
    return limits


class GenericIsaacSimEnv(EnvBase):
    def __init__(self, config: RobotConfig) -> None:
        if SimulationApp is None:
            raise ImportError("Isaac Sim Python modules are not available.")
        self._config = config
        self._kit_app = None
        self._owns_sim_app = False
        try:
            import omni.kit.app

            self._kit_app = omni.kit.app.get_app()
        except Exception:
            self._kit_app = None
        if self._kit_app is None or not self._kit_app.is_running():
            self.simulation_app = SimulationApp({"headless": False})
            self._owns_sim_app = True
            try:
                import omni.kit.app

                self._kit_app = omni.kit.app.get_app()
            except Exception:
                self._kit_app = None
        else:
            self.simulation_app = None
        try:
            from omni.isaac.core import World
            from isaacsim.core.prims import SingleArticulation

            if _is_urdf(config.urdf):
                urdf_path, _ = resolve_urdf(config)
                fallback_ctrlrange = None
            else:
                urdf_path, fallback_ctrlrange = resolve_urdf(config)
            self.world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
            if getattr(self.world, "_physics_context", None) is None:
                self.world._init_stage(
                    physics_dt=1.0 / 60.0,
                    rendering_dt=1.0 / 60.0,
                    stage_units_in_meters=1.0,
                    physics_prim_path="/physicsScene",
                    sim_params=None,
                    set_defaults=True,
                    backend="numpy",
                    device=None,
                )
            self.world.scene.add_default_ground_plane()
            prim_path = f"/World/{config.name}"
            imported_path = _import_urdf(urdf_path, prim_path)
            self.robot = SingleArticulation(prim_path=imported_path, name=config.name)
            self.world.scene.add(self.robot)
            self.world.reset()
            try:
                self.world.initialize_physics()
            except Exception:
                pass
            self._joint_indices = np.arange(self.robot.num_dof, dtype=np.int32)
            self._ctrl_range = build_ctrl_range(self.robot, config, fallback_ctrlrange)
            self._ctrl_range = _resolve_ctrl_range_isaacsim(
                self._ctrl_range, self.robot.num_dof, self.robot
            )
            self._action_dim = self._ctrl_range.shape[0]
        except Exception:
            self.request_exit()
            raise

    def step(self, action: Optional[np.ndarray]) -> None:
        if action is not None:
            action = np.clip(action, -1.0, 1.0)
            count = min(action.shape[0], self._action_dim, self._joint_indices.shape[0])
            if count > 0:
                lo = self._ctrl_range[:count, 0]
                hi = self._ctrl_range[:count, 1]
                targets = lo + (action[:count] + 1.0) * 0.5 * (hi - lo)
                self.robot.set_joint_positions(
                    targets, joint_indices=self._joint_indices[:count]
                )
        self.world.step(render=True)

    def render(self, camera: str) -> np.ndarray:
        try:
            from omni.kit.viewport.utility import (
                capture_viewport_to_buffer,
                get_active_viewport_window,
            )
        except Exception:
            return np.empty((0, 0, 3), dtype=np.uint8)
        viewport_window = get_active_viewport_window()
        if viewport_window is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        viewport_api = viewport_window.viewport_api
        if camera:
            try:
                viewport_api.set_active_camera(camera)
            except Exception:
                return np.empty((0, 0, 3), dtype=np.uint8)
        buffer = capture_viewport_to_buffer(viewport_api)
        if buffer is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        if isinstance(buffer, np.ndarray) and buffer.ndim == 3:
            return buffer[..., :3].astype(np.uint8, copy=False)
        try:
            width, height = viewport_api.get_texture_resolution()
            data = np.frombuffer(buffer, dtype=np.uint8)
            rgba = data.reshape((height, width, 4))
            return rgba[:, :, :3].copy()
        except Exception:
            return np.empty((0, 0, 3), dtype=np.uint8)

    def state(self) -> np.ndarray:
        positions = self.robot.get_joint_positions()
        velocities = self.robot.get_joint_velocities()
        if positions is None:
            positions = np.zeros(0, dtype=np.float32)
        if velocities is None:
            velocities = np.zeros(0, dtype=np.float32)
        return np.concatenate([positions, velocities]).astype(np.float32)

    def close(self) -> None:
        if self._owns_sim_app and self.simulation_app is not None:
            self.simulation_app.close()

    def is_running(self) -> bool:
        if self._kit_app is not None:
            return self._kit_app.is_running()
        if self.simulation_app is not None:
            return self.simulation_app.is_running()
        return False

    def request_exit(self) -> None:
        if self._kit_app is not None:
            for name in ("post_quit", "quit", "shutdown", "close"):
                method = getattr(self._kit_app, name, None)
                if method is None:
                    continue
                try:
                    method()
                    return
                except Exception:
                    pass
        if self.simulation_app is not None:
            self.simulation_app.close()


class IsaacSimRosBridge(Node):
    def __init__(self, env: GenericIsaacSimEnv, config: RobotConfig) -> None:
        super().__init__(f"{config.name}_isaacsim_bridge")
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
    parser = argparse.ArgumentParser(description="Isaac Sim ROS 2 Bridge")
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
    try:
        env = GenericIsaacSimEnv(config)
        node = IsaacSimRosBridge(env, config)
        while env.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        if "env" in locals():
            env.request_exit()
        raise
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()
        if "env" in locals():
            env.close()


if __name__ == "__main__":
    main()
