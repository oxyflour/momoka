"""Shared simulation robot base classes."""

from dataclasses import dataclass, field
from functools import cached_property

from lerobot.robots import Robot, RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

@dataclass
class MotorSetup:
    calibration: MotorCalibration | None

@dataclass
class CameraSetup:
    id: str
    config: CameraConfig

@RobotConfig.register_subclass("mmk_base_robot")
@dataclass
class BaseRobotConfig(RobotConfig):
    motors: dict[str, MotorSetup] | None
    cameras: dict[str, CameraSetup] | None
    calibrated = True
    headless = False

class BaseRobot(Robot):
    """Base Robot."""

    config_class = BaseRobotConfig
    name = "base_robot"

    def __init__(self, config: BaseRobotConfig):
        super().__init__(config)
        self._is_connected = False
        self._is_calibrated = bool(config.calibrated)
        self.config = config
        # undocumented but requires this field
        self.cameras = { name: setup for name, setup in config.cameras.items() }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.config.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (item.config.height, item.config.width, 3) for name, item in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._motors_ft,
            **self._cameras_ft
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        return {
            f"{motor}.pos": 0 for motor in self.config.motors
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
