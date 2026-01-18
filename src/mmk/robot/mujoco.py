"""MuJoCo-backed robot adapter for LeRobot-compatible APIs."""

import os
import threading
from pathlib import Path

import numpy as np
import mujoco

from dataclasses import dataclass

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


@RobotConfig.register_subclass("mmk_mujoco_robot")
@dataclass
class MujocoConfig(BaseRobotConfig):
    scene: str | None = None


def get_motors(model):
    motors = {}
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name:
            continue
        motors[joint_name] = MotorSetup(calibration=None)
    return motors


def get_cameras(model, default_width=640, default_height=480):
    cameras = {}
    for camera_id in range(model.ncam):
        camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        if not camera_name:
            continue
        cameras[camera_name] = CameraSetup(
            id=camera_name,
            config=CameraConfig(fps=30, width=default_width, height=default_height),
        )
    return cameras


def get_joint_id(model, motor: str):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, motor)
    if joint_id != -1:
        return int(joint_id)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)
    if actuator_id != -1:
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id >= 0:
            return joint_id
    raise ValueError(f"Unknown joint or actuator name: {motor}")


# According to lerobot requirement, the module name and the classname should match
class Mujoco(BaseRobot):
    """Minimal MuJoCo robot wrapper backed by the MuJoCo API."""

    config_class = MujocoConfig
    name = "mmk_mujoco_robot"

    def __init__(self, config: MujocoConfig):
        if not config.scene:
            raise ValueError("MujocoConfig.scene is required for mmk_mujoco_robot")
        scene_path = Path(config.scene)
        if not scene_path.is_absolute():
            scene_path = (Path.cwd() / scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        if not config.motors:
            config.motors = get_motors(self.model)
        if not config.cameras:
            config.cameras = get_cameras(self.model)
        super().__init__(config)
        self.data = mujoco.MjData(self.model)
        self.joints = {
            motor: get_joint_id(self.model, motor) for motor in config.motors
        }
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)

        self.worker: threading.Thread | None = None
        self.started = False

    def start(self):
        self.started = True
        if self.config.headless:
            while self.started:
                self.step()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while self.started and viewer.is_running():
                    self.step()
                    viewer.sync()

    def apply(self, action: np.ndarray):
        self.data.ctrl[:] = action

    def step(self):
        mujoco.mj_step(self.model, self.data)

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

    def get_observation(self):
        obs = {}
        for motor in self.config.motors:
            motor_name = motor if isinstance(motor, str) else str(motor)
            joint_id = self.joints.get(motor_name, -1)
            obs[f"{motor_name}.pos"] = float(self.data.joint(joint_id).qpos[0])
        for name, setup in self.config.cameras.items():
            self.renderer.update_scene(self.data, camera=setup.id)
            obs[name] = self.renderer.render()
        return obs

    def send_action(self, action: RobotAction):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        action_data = [action[f"{motor}.pos"] for motor in self.config.motors]
        self.apply(np.array(action_data))
        return action
