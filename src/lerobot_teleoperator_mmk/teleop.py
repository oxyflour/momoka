"""Keyboard teleoperator for MMK robots."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import mujoco
import yaml

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.configs import parser as lerobot_parser

from mmk.robot.mujoco import get_motors

from .config import MmkTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as exc:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info("Could not import pynput: %s", exc)


@dataclass
class _TeleopState:
    joints: dict[str, float]
    speed: float
    axis_step: float


class MmkTeleop(Teleoperator):
    """Keyboard teleoperator for MMK robots."""

    config_class = MmkTeleopConfig
    name = "mmk_teleop"

    def __init__(self, config: MmkTeleopConfig):
        super().__init__(config)
        self.config = config
        self.event_queue: Queue[tuple[str, bool]] = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener: keyboard.Listener | None = None
        self.logs: dict[str, float] = {}
        self.motors = self._resolve_motors()
        self.state = _TeleopState(
            joints={name: 0.0 for name in self.motors},
            speed=config.speed,
            axis_step=config.axis_step,
        )

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.motors),),
            "names": {"motors": list(self.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            PYNPUT_AVAILABLE
            and isinstance(self.listener, keyboard.Listener)
            and self.listener.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("MMK teleop is already connected.")
        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _on_press(self, key) -> None:
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.event_queue.put(("shift", True))
        elif hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key) -> None:
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.event_queue.put(("shift", False))
        elif hasattr(key, "char"):
            self.event_queue.put((key.char, False))

    def _resolve_motors(self) -> list[str]:
        config_path = None
        if self.config.robot_config_path:
            config_path = Path(self.config.robot_config_path).expanduser()
        else:
            config_arg = lerobot_parser.parse_arg("config_path")
            if config_arg:
                config_path = Path(config_arg).expanduser()
        if config_path is None:
            raise ValueError("robot_config_path is required or pass --config_path")
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        robot_cfg = config_data.get("robot", {})
        if robot_cfg.get("type") == "mmk_mujoco_robot":
            scene_path = robot_cfg.get("scene")
            if not scene_path:
                raise ValueError("robot.scene is required for mmk_mujoco_robot")
            scene_path = Path(scene_path)
            if not scene_path.is_absolute():
                scene_path = (config_path.parent / scene_path).resolve()
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            return list(get_motors(model).keys())
        raise ValueError("robot_config_path must point to a config with robot.scene")

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def _apply_joint_control(self, key_state: dict[str, bool]) -> None:
        if not self.motors:
            return
        speed = self.state.speed
        step = self.state.axis_step
        direction = -1.0 if key_state.get("shift") else 1.0
        key_map = ["1", "2", "3", "4", "5", "6", "7"]
        for index, key_char in enumerate(key_map):
            if key_state.get(key_char):
                self._step_joint(index, step * speed * direction)

    def _step_joint(self, index: int, delta: float) -> None:
        if index >= len(self.motors):
            return
        motor = self.motors[index]
        self.state.joints[motor] += delta

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "MMK teleop is not connected. You need to run `connect()` before `get_action()`."
            )
        before_read_t = time.perf_counter()
        self._drain_pressed_keys()
        self._apply_joint_control(self.current_pressed)
        action = {
            f"{name}.pos": float(value) for name, value in self.state.joints.items()
        }
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "MMK teleop is not connected. You need to run `disconnect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()
