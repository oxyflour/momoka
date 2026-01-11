import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


class TestExtractAction(unittest.TestCase):
    """Test _extract_action function from mujoco_ros2.py."""

    def test_extract_valid_action(self):
        """Test extraction of valid action from JSON payload."""
        from scripts.mujoco_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps(
            {"result": {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}}
        )
        action = _extract_action(msg)
        self.assertIsNotNone(action)
        self.assertEqual(action.shape, (7,))
        np.testing.assert_array_almost_equal(
            action, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        )

    def test_extract_invalid_json(self):
        """Test extraction with invalid JSON."""
        from scripts.mujoco_ros2 import _extract_action
        from std_msgs.msg import String

        msg = String()
        msg.data = "invalid json"
        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_no_action_field(self):
        """Test extraction when action field is missing."""
        from scripts.mujoco_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {}})
        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_empty_action(self):
        """Test extraction with empty action list."""
        from scripts.mujoco_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {"action": []}})
        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_non_list_action(self):
        """Test extraction with non-list action."""
        from scripts.mujoco_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {"action": "not a list"}})
        action = _extract_action(msg)
        self.assertIsNone(action)


class TestGenericMujocoEnv(unittest.TestCase):
    """Test GenericMujocoEnv class."""

    def setUp(self):
        self.original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent.parent)

    def tearDown(self):
        os.chdir(self.original_cwd)

    @patch("scripts.mujoco_ros2.mujoco")
    @patch("scripts.mujoco_ros2.mujoco.viewer")
    @patch("scripts.mujoco_ros2.resolve_urdf")
    def test_env_initialization(self, mock_resolve, mock_viewer, mock_mujoco):
        """Test environment initialization with mock model."""
        from utils.robot_loader import RobotConfig
        from scripts.mujoco_ros2 import GenericMujocoEnv

        mock_resolve.return_value = "/tmp/test.urdf"
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        mock_mujoco.viewer.launch_passive.return_value = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco.Renderer.return_value = mock_renderer
        mock_model.actuator_ctrlrange = np.array([[-1.0, 1.0]])

        config = RobotConfig(
            name="test", urdf="/tmp/test.urdf", ctrl_range=[[-1.0, 1.0]]
        )
        env = GenericMujocoEnv(config)
        self.assertIsNotNone(env)
        self.assertEqual(env._config.name, "test")

    @patch("scripts.mujoco_ros2.mujoco")
    @patch("scripts.mujoco_ros2.mujoco.viewer")
    @patch("scripts.mujoco_ros2.resolve_urdf")
    def test_env_step_with_action(self, mock_resolve, mock_viewer, mock_mujoco):
        """Test environment step with action."""
        from utils.robot_loader import RobotConfig
        from scripts.mujoco_ros2 import GenericMujocoEnv

        mock_resolve.return_value = "/tmp/test.urdf"
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_data.ctrl = np.zeros(7)
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        mock_mujoco.viewer.launch_passive.return_value = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco.Renderer.return_value = mock_renderer
        mock_model.actuator_ctrlrange = np.tile(np.array([-1.0, 1.0]), (7, 1))

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )
        env = GenericMujocoEnv(config)
        action = np.zeros(7)
        env.step(action)
        mock_mujoco.mj_step.assert_called()

    @patch("scripts.mujoco_ros2.mujoco")
    @patch("scripts.mujoco_ros2.mujoco.viewer")
    @patch("scripts.mujoco_ros2.resolve_urdf")
    def test_env_step_with_none(self, mock_resolve, mock_viewer, mock_mujoco):
        """Test environment step with None action (zero control)."""
        from utils.robot_loader import RobotConfig
        from scripts.mujoco_ros2 import GenericMujocoEnv

        mock_resolve.return_value = "/tmp/test.urdf"
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_data.ctrl = np.zeros(7)
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        mock_mujoco.viewer.launch_passive.return_value = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco.Renderer.return_value = mock_renderer
        mock_model.actuator_ctrlrange = np.tile(np.array([-1.0, 1.0]), (7, 1))

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )
        env = GenericMujocoEnv(config)
        env.step(None)
        np.testing.assert_array_equal(mock_data.ctrl, np.zeros(7))

    @patch("scripts.mujoco_ros2.mujoco")
    @patch("scripts.mujoco_ros2.mujoco.viewer")
    @patch("scripts.mujoco_ros2.resolve_urdf")
    def test_env_state(self, mock_resolve, mock_viewer, mock_mujoco):
        """Test environment state retrieval."""
        from utils.robot_loader import RobotConfig
        from scripts.mujoco_ros2 import GenericMujocoEnv

        mock_resolve.return_value = "/tmp/test.urdf"
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_data.qpos = np.array([0.1, 0.2, 0.3])
        mock_data.qvel = np.array([0.01, 0.02, 0.03])
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        mock_mujoco.viewer.launch_passive.return_value = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_mujoco.Renderer.return_value = mock_renderer
        mock_model.actuator_ctrlrange = np.tile(np.array([-1.0, 1.0]), (7, 1))

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )
        env = GenericMujocoEnv(config)
        state = env.state()
        self.assertEqual(len(state), 6)
        np.testing.assert_array_almost_equal(state[:3], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(state[3:], [0.01, 0.02, 0.03])


class TestMujocoRosBridge(unittest.TestCase):
    """Test MujocoRosBridge node."""

    def test_bridge_attributes(self):
        """Test bridge node attributes after construction (mock Node)."""
        from utils.robot_loader import RobotConfig
        from scripts.mujoco_ros2 import MujocoRosBridge, GenericMujocoEnv

        mock_env = MagicMock(spec=GenericMujocoEnv)
        config = RobotConfig(name="test_robot", urdf="/tmp/test.urdf")

        bridge = MujocoRosBridge.__new__(MujocoRosBridge)
        bridge._action_lock = threading.Lock()
        bridge._latest_action = None
        bridge._env = mock_env
        bridge._config = config

        self.assertEqual(bridge._config.name, "test_robot")
        self.assertEqual(bridge._env, mock_env)


if __name__ == "__main__":
    unittest.main()
