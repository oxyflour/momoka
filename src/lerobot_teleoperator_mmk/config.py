"""Configuration for the MMK teleoperator."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("mmk_teleop")
@dataclass
class MmkTeleopConfig(TeleoperatorConfig):
    """Configuration for the MMK teleoperator."""

    robot_config_path: str | None = None
    speed: float = 0.1
    axis_step: float = 0.02
