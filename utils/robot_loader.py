from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


class RobotConfig(dict):
    def __init__(
        self,
        name: str,
        urdf: str,
        mesh_base: Optional[str] = None,
        package_replacements: Optional[list[dict]] = None,
        ctrl_range: Optional[list[list[float]]] = None,
        joint_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self["name"] = name
        self["urdf"] = urdf
        self["mesh_base"] = mesh_base
        self["package_replacements"] = package_replacements or []
        self["ctrl_range"] = ctrl_range
        self["joint_names"] = joint_names

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def urdf(self) -> str:
        return self["urdf"]

    @property
    def mesh_base(self) -> Optional[str]:
        return self["mesh_base"]

    @property
    def package_replacements(self) -> list[dict]:
        return self.get("package_replacements", [])

    @property
    def ctrl_range(self) -> Optional[list[list[float]]]:
        return self.get("ctrl_range")

    @property
    def joint_names(self) -> Optional[list[str]]:
        return self.get("joint_names")


def load_config(name: str, config_dir: Optional[str] = None) -> RobotConfig:
    if config_dir is None:
        repo_root = Path(__file__).parent.parent
        config_dir = str(repo_root / "config")
    config_path = os.path.join(config_dir, "robots", f"{name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Robot config not found: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return RobotConfig(
        name=data["name"],
        urdf=data["urdf"],
        mesh_base=data.get("mesh_base"),
        package_replacements=data.get("package_replacements"),
        ctrl_range=data.get("ctrl_range"),
        joint_names=data.get("joint_names"),
    )


def resolve_urdf(config: RobotConfig) -> str:
    model_path = config.urdf
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"URDF file not found: {model_path}")
    return _prepare_urdf_path(model_path, config)


def _prepare_urdf_path(model_path: str, config: RobotConfig) -> str:
    with open(model_path, "r") as f:
        content = f.read()
    if "package://" not in content:
        return model_path
    temp_dir = tempfile.mkdtemp()
    mesh_base = config.mesh_base
    if mesh_base and os.path.isdir(mesh_base):
        for name in os.listdir(mesh_base):
            src = os.path.join(mesh_base, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(temp_dir, name))
    for repl in config.package_replacements:
        from_str = repl.get("from", "")
        to_str = repl.get("to", "")
        content = content.replace(from_str, to_str)
    temp_path = os.path.join(temp_dir, os.path.basename(model_path))
    with open(temp_path, "w") as f:
        f.write(content)
    return temp_path


def build_ctrl_range(
    model,
    config: RobotConfig,
) -> np.ndarray:
    if config.ctrl_range is not None:
        return np.array(config.ctrl_range, dtype=np.float32)
    return _infer_ctrl_range_from_model(model)


def _infer_ctrl_range_from_model(model) -> np.ndarray:
    try:
        ctrlrange = model.actuator_ctrlrange.copy()
    except AttributeError:
        return np.zeros((0, 2), dtype=np.float32)
    if ctrlrange.size == 0:
        return ctrlrange
    if not np.allclose(ctrlrange[:, 0], ctrlrange[:, 1]):
        return ctrlrange
    fallback = np.zeros_like(ctrlrange, dtype=np.float32)
    for idx in range(ctrlrange.shape[0]):
        try:
            joint_id = model.actuator_trnid[idx, 0]
        except AttributeError:
            fallback[idx] = np.array([-1.0, 1.0], dtype=np.float32)
            continue
        if 0 <= joint_id < model.jnt_range.shape[0]:
            joint_range = model.jnt_range[joint_id]
            if np.allclose(joint_range[0], joint_range[1]):
                fallback[idx] = np.array([-1.0, 1.0], dtype=np.float32)
            else:
                fallback[idx] = joint_range.astype(np.float32)
        else:
            fallback[idx] = np.array([-1.0, 1.0], dtype=np.float32)
    return fallback
