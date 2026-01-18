"""Isaac Sim-backed robot adapter for LeRobot-compatible APIs."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.cameras import CameraConfig

from .base import (
    BaseRobot,
    BaseRobotConfig,
    MotorSetup,
    CameraSetup,
    RobotConfig,
    RobotAction,
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)


@RobotConfig.register_subclass("mmk_isaac_robot")
@dataclass
class IsaacConfig(BaseRobotConfig):
    scene: str | None = None
    robot_prim_path: str | None = None
    joint_map: dict[str, str] = field(default_factory=dict)
    camera_map: dict[str, str] = field(default_factory=dict)


def _resolve_scene_path(scene: str | None) -> Path:
    if not scene:
        raise ValueError("IsaacConfig.scene is required for mmk_isaac_robot")
    scene_path = Path(scene)
    if not scene_path.is_absolute():
        scene_path = (Path.cwd() / scene_path).resolve()
    return scene_path


def _load_simulation_app(headless: bool):
    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError as exc:
            raise ImportError(
                "Isaac Sim Python API not available. Please run inside Isaac Sim."
            ) from exc
    return SimulationApp({"headless": headless})


def _collect_camera_setups(stage, default_width=640, default_height=480):
    try:
        from pxr import UsdGeom
    except ImportError:
        return {}
    cameras = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_path = str(prim.GetPath())
            cameras[camera_path] = CameraSetup(
                id=camera_path,
                config=CameraConfig(fps=30, width=default_width, height=default_height),
            )
    return cameras


def _find_articulation_root(stage, preferred_path: str | None = None):
    if preferred_path:
        return preferred_path
    try:
        from pxr import UsdPhysics
    except ImportError:
        return None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


# According to lerobot requirement, the module name and the classname should match
class Isaac(BaseRobot):
    """Minimal Isaac Sim robot wrapper backed by the Isaac Sim API."""

    config_class = IsaacConfig
    name = "mmk_isaac_robot"

    def __init__(self, config: IsaacConfig):
        scene_path = _resolve_scene_path(config.scene)
        self.sim_app = _load_simulation_app(config.headless)
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        import omni.usd

        usd_context = omni.usd.get_context()
        usd_context.open_stage(str(scene_path))
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to load stage: {scene_path}")

        self.world = World()
        articulation_path = _find_articulation_root(stage, config.robot_prim_path)
        if not articulation_path:
            raise ValueError(
                "robot_prim_path is required when the stage has no ArticulationRootAPI"
            )
        self.articulation = Articulation(prim_path=articulation_path)
        self.world.scene.add(self.articulation)
        self.world.reset()
        self.articulation.initialize()

        available_joints = list(self.articulation.get_joint_names())
        if not config.joint_map:
            if config.motors:
                config.joint_map = {alias: alias for alias in config.motors}
            else:
                config.joint_map = {name: name for name in available_joints}
        else:
            missing_joints = [
                target
                for target in config.joint_map.values()
                if target not in available_joints
            ]
            if missing_joints:
                raise ValueError(
                    "Unknown joint names in joint_map: " + ", ".join(missing_joints)
                )
        if not config.motors:
            config.motors = {
                alias: MotorSetup(calibration=None) for alias in config.joint_map
            }

        detected_cameras = _collect_camera_setups(stage)
        if config.cameras:
            if config.camera_map:
                for alias, target in config.camera_map.items():
                    if alias in config.cameras:
                        config.cameras[alias].id = target
                    elif target in detected_cameras:
                        config.cameras[alias] = detected_cameras[target]
        else:
            if not config.camera_map:
                config.camera_map = {name: name for name in detected_cameras}
            else:
                missing_cameras = [
                    target
                    for target in config.camera_map.values()
                    if target not in detected_cameras
                ]
                if missing_cameras:
                    raise ValueError(
                        "Unknown camera names in camera_map: "
                        + ", ".join(missing_cameras)
                    )
            config.cameras = {
                alias: detected_cameras[target]
                for alias, target in config.camera_map.items()
                if target in detected_cameras
            }

        super().__init__(config)
        self.joint_names = list(self.articulation.get_joint_names())
        self.joint_indices = {
            name: index for index, name in enumerate(self.joint_names)
        }
        self.joint_aliases = {
            alias: target for alias, target in config.joint_map.items()
        }

        self.cameras = {}
        if config.cameras:
            try:
                from omni.isaac.sensor import Camera
            except ImportError:
                Camera = None
            if Camera is not None:
                for alias, setup in config.cameras.items():
                    camera = Camera(
                        prim_path=setup.id,
                        name=alias,
                        frequency=setup.config.fps,
                        resolution=(setup.config.width, setup.config.height),
                    )
                    self.world.scene.add(camera)
                    self.cameras[alias] = camera
        self.world.reset()
        for camera in self.cameras.values():
            camera.initialize()

        self.worker: threading.Thread | None = None
        self.started = False

    def start(self):
        self.started = True
        while self.started and self.sim_app.is_running():
            self.step()

    def apply(self, action: np.ndarray):
        joint_positions = np.array(self.articulation.get_joint_positions(), copy=True)
        for alias, value in zip(self.config.motors, action, strict=False):
            target = self.joint_aliases.get(str(alias))
            if target is None:
                continue
            index = self.joint_indices.get(target)
            if index is None:
                continue
            joint_positions[index] = value
        self.articulation.set_joint_position_targets(joint_positions)

    def step(self):
        self.world.step(render=not self.config.headless)

    # override for lerobot

    def connect(self, calibrate=True):
        if self.worker:
            raise DeviceAlreadyConnectedError()
        self.worker = threading.Thread(target=self.start, daemon=True)
        self.worker.start()
        super().connect(calibrate)

    def disconnect(self):
        super().disconnect()
        if not self.worker:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.started = False
        self.worker.join()
        self.worker = None
        self.sim_app.close()

    def get_observation(self):
        joint_positions = self.articulation.get_joint_positions()
        obs = {}
        for alias in self.config.motors:
            motor_name = alias if isinstance(alias, str) else str(alias)
            target = self.joint_aliases.get(motor_name)
            if target is None:
                continue
            index = self.joint_indices.get(target)
            if index is None:
                continue
            obs[f"{motor_name}.pos"] = float(joint_positions[index])
        for name, camera in self.cameras.items():
            frame = camera.get_rgba()
            if frame is not None:
                obs[name] = frame[:, :, :3]
        return obs

    def send_action(self, action: RobotAction):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        action_data = [action[f"{motor}.pos"] for motor in self.config.motors]
        self.apply(np.array(action_data, dtype=np.float32))
        return action
