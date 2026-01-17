"""MuJoCo-backed robot adapter for LeRobot-compatible APIs."""

import os
import threading
import numpy as np
import mujoco

from dataclasses import dataclass

from .base import (
    BaseRobot,
    BaseRobotConfig,

    RobotConfig,
    RobotAction,
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

@RobotConfig.register_subclass("mmk_mujoco_robot")
@dataclass
class MujocoConfig(BaseRobotConfig):
    pass

# According to lerobot requirement, the module name and the classname should match
class Mujoco(BaseRobot):
    """Minimal MuJoCo robot wrapper backed by the MuJoCo API."""

    config_class = MujocoConfig
    name = "mmk_mujoco_robot"

    def __init__(self, config: BaseRobotConfig):
        scene = os.path.normpath(rf"C:\Projects\mmk3\deps\lerobot-mujoco-tutorial\asset\example_scene_y.xml")
        self.model = mujoco.MjModel.from_xml_path(scene)
        if not config.motors:
            # TODO: get all joints in model and initialize that from config.motors
            pass
        if not config.cameras:
            # TODO: get all camera in model and initialize that from config.motors
            pass
        super().__init__(config)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        self.worker: threading.Thread | None = None
        self.started = False
    
    def connect(self, calibrate = True):
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
    
    def start(self):
        if self.config.headless:
            while self.started:
                self.step()
        else:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while self.started and viewer.is_running():
                    self.step()
                    viewer.sync()
    
    def apply(self, action: np.ndarray):
        self.data.ctrl[:] = action

    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    def get_observation(self):
        obs = { }
        # TODO:
        return obs
    
    def send_action(self, action: RobotAction):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        action_data = [action[f"{motor}.pos"] for motor in self.config.motors]
        self.apply(action_data)
        return action
