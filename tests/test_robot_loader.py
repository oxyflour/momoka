import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


class TestRobotConfig(unittest.TestCase):
    """Test RobotConfig class."""

    def test_robot_config_creation(self):
        """Test that RobotConfig can be created with all fields."""
        from utils.robot_loader import RobotConfig

        config = RobotConfig(
            name="test_robot",
            urdf="/path/to/robot.urdf",
            mesh_base="/path/to/meshes",
            package_replacements=[{"from": "pkg://", "to": ""}],
            ctrl_range=[[-1.0, 1.0], [-2.0, 2.0]],
            joint_names=["joint1", "joint2"],
        )
        self.assertEqual(config.name, "test_robot")
        self.assertEqual(config.urdf, "/path/to/robot.urdf")
        self.assertEqual(config.mesh_base, "/path/to/meshes")
        self.assertEqual(len(config.package_replacements), 1)
        self.assertEqual(len(config.ctrl_range or []), 2)
        self.assertEqual(len(config.joint_names or []), 2)

    def test_robot_config_defaults(self):
        """Test that RobotConfig has correct defaults."""
        from utils.robot_loader import RobotConfig

        config = RobotConfig(name="minimal", urdf="/path/to/robot.urdf")
        self.assertIsNone(config.mesh_base)
        self.assertEqual(config.package_replacements, [])
        self.assertIsNone(config.ctrl_range)
        self.assertIsNone(config.joint_names)

    def test_robot_config_dict_compat(self):
        """Test that RobotConfig is dict-compatible."""
        from utils.robot_loader import RobotConfig

        config = RobotConfig(name="test", urdf="/path/to/robot.urdf")
        self.assertEqual(config["name"], "test")
        self.assertEqual(config["urdf"], "/path/to/robot.urdf")


class TestLoadConfig(unittest.TestCase):
    """Test load_config function."""

    def setUp(self):
        self.temp_config_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent.parent)

    def tearDown(self):
        os.chdir(self.original_cwd)
        import shutil

        shutil.rmtree(self.temp_config_dir, ignore_errors=True)

    def test_load_existing_config(self):
        """Test loading an existing robot config."""
        from utils.robot_loader import load_config

        config = load_config("panda")
        self.assertEqual(config.name, "panda")
        self.assertIn("panda.urdf", config.urdf)

    def test_load_nonexistent_config(self):
        """Test that loading nonexistent config raises FileNotFoundError."""
        from utils.robot_loader import load_config

        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_robot", config_dir=self.temp_config_dir)

    def test_load_config_custom_dir(self):
        """Test loading config from custom directory."""
        from utils.robot_loader import load_config

        robot_dir = os.path.join(self.temp_config_dir, "robots")
        os.makedirs(robot_dir)
        config_path = os.path.join(robot_dir, "custom.yaml")
        with open(config_path, "w") as f:
            f.write("name: custom_robot\nurdf: /custom/path.urdf\n")
        config = load_config("custom", config_dir=self.temp_config_dir)
        self.assertEqual(config.name, "custom_robot")
        self.assertEqual(config.urdf, "/custom/path.urdf")


class TestResolveUrdf(unittest.TestCase):
    """Test resolve_urdf function."""

    def test_resolve_existing_urdf(self):
        """Test resolving an existing URDF file."""
        from utils.robot_loader import RobotConfig, resolve_urdf

        config = RobotConfig(
            name="test", urdf="data/PandaRobot.jl-master/deps/Panda/panda.urdf"
        )
        if not os.path.exists(config.urdf):
            self.skipTest("URDF file not found")
        urdf_path = resolve_urdf(config)
        self.assertTrue(urdf_path.endswith(".urdf"))

    def test_resolve_missing_file(self):
        """Test that resolving missing file raises FileNotFoundError."""
        from utils.robot_loader import RobotConfig, resolve_urdf

        config = RobotConfig(name="test", urdf="/nonexistent/path.urdf")
        with self.assertRaises(FileNotFoundError):
            resolve_urdf(config)


class TestBuildCtrlRange(unittest.TestCase):
    """Test build_ctrl_range function."""

    def test_build_from_config(self):
        """Test building ctrl_range from config."""
        from utils.robot_loader import RobotConfig, build_ctrl_range

        config = RobotConfig(
            name="test",
            urdf="/path/to/robot.urdf",
            ctrl_range=[[-1.5, 1.5], [-2.0, 2.0]],
        )
        mock_model = MagicMock()
        ctrl_range = build_ctrl_range(mock_model, config)
        np.testing.assert_array_equal(ctrl_range, np.array([[-1.5, 1.5], [-2.0, 2.0]]))

    def test_build_infers_from_model(self):
        """Test that ctrl_range is inferred from model when no config or fallback."""
        from utils.robot_loader import RobotConfig, build_ctrl_range

        config = RobotConfig(name="test", urdf="/path/to/robot.urdf")
        mock_model = MagicMock()
        mock_model.actuator_ctrlrange = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        ctrl_range = build_ctrl_range(mock_model, config)
        np.testing.assert_array_equal(ctrl_range, np.array([[-1.0, 1.0], [-2.0, 2.0]]))


class TestPrepareUrdfPath(unittest.TestCase):
    """Test _prepare_urdf_path function."""

    def test_no_package_replacements_needed(self):
        """Test that URDF path is returned as-is when no package:// prefixes."""
        from utils.robot_loader import RobotConfig, _prepare_urdf_path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write('<robot name="test"><link name="base"/></robot>')
            temp_path = f.name
        try:
            config = RobotConfig(name="test", urdf=temp_path)
            result = _prepare_urdf_path(temp_path, config)
            self.assertEqual(result, temp_path)
        finally:
            os.unlink(temp_path)

    def test_package_replacements_applied(self):
        """Test that package:// replacements are applied."""
        from utils.robot_loader import RobotConfig, _prepare_urdf_path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(
                '<robot name="test"><mesh filename="package://Panda/mesh.stl"/></robot>'
            )
            temp_path = f.name
        try:
            config = RobotConfig(
                name="test",
                urdf=temp_path,
                package_replacements=[{"from": "package://Panda/", "to": ""}],
            )
            result = _prepare_urdf_path(temp_path, config)
            self.assertNotEqual(result, temp_path)
            with open(result, "r") as f:
                content = f.read()
            self.assertIn('filename="mesh.stl"', content)
        finally:
            os.unlink(temp_path)


class TestInferCtrlRangeFromModel(unittest.TestCase):
    """Test _infer_ctrl_range_from_model function."""

    def test_uses_actuator_ctrlrange_when_valid(self):
        """Test that valid actuator_ctrlrange is used."""
        from utils.robot_loader import _infer_ctrl_range_from_model

        mock_model = MagicMock()
        mock_model.actuator_ctrlrange = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        result = _infer_ctrl_range_from_model(mock_model)
        np.testing.assert_array_equal(result, np.array([[-1.0, 1.0], [-2.0, 2.0]]))

    def test_falls_back_to_joint_range_when_ctrlrange_invalid(self):
        """Test fallback to joint range when ctrlrange is invalid."""
        from utils.robot_loader import _infer_ctrl_range_from_model

        mock_model = MagicMock()
        mock_model.actuator_ctrlrange = np.array([[0.0, 0.0], [0.0, 0.0]])
        mock_model.actuator_trnid = np.array([[0], [1]], dtype=np.int32)
        mock_model.jnt_range = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        result = _infer_ctrl_range_from_model(mock_model)
        np.testing.assert_array_equal(result, np.array([[-1.0, 1.0], [-2.0, 2.0]]))

    def test_returns_empty_when_no_actuators(self):
        """Test that empty array is returned when no actuators."""
        from utils.robot_loader import _infer_ctrl_range_from_model

        mock_model = MagicMock()
        mock_model.actuator_ctrlrange = np.zeros((0, 2))
        result = _infer_ctrl_range_from_model(mock_model)
        self.assertEqual(result.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
