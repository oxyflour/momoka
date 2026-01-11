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


class TestIsaacSimExtractAction(unittest.TestCase):
    """Test _extract_action function from isaacsim_ros2.py."""

    def test_extract_valid_action(self):
        """Test extraction of valid action from JSON payload."""
        from scripts.isaacsim_ros2 import _extract_action
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
        from scripts.isaacsim_ros2 import _extract_action
        from std_msgs.msg import String

        msg = String()
        msg.data = "invalid json"
        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_no_action_field(self):
        """Test extraction when action field is missing."""
        from scripts.isaacsim_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {}})
        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_empty_action(self):
        """Test extraction with empty action list."""
        from scripts.isaacsim_ros2 import _extract_action
        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {"action": []}})
        action = _extract_action(msg)
        self.assertIsNone(action)


class TestIsUrdf(unittest.TestCase):
    """Test _is_urdf function."""

    def test_urdf_extension(self):
        """Test .urdf extension detection."""
        from scripts.isaacsim_ros2 import _is_urdf

        self.assertTrue(_is_urdf("/path/to/robot.urdf"))
        self.assertTrue(_is_urdf("/path/to/robot.URDF"))

    def test_non_urdf_extension(self):
        """Test non-URDF extensions."""
        from scripts.isaacsim_ros2 import _is_urdf

        self.assertFalse(_is_urdf("/path/to/robot.xml"))
        self.assertFalse(_is_urdf("/path/to/robot.mjcf"))


class TestResolveCtrlRangeIsaacSim(unittest.TestCase):
    """Test _resolve_ctrl_range_isaacsim function."""

    def test_returns_config_ctrlrange(self):
        """Test that valid config ctrlrange is returned."""
        from scripts.isaacsim_ros2 import _resolve_ctrl_range_isaacsim

        ctrl_range = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)
        mock_robot = MagicMock()
        result = _resolve_ctrl_range_isaacsim(ctrl_range, 2, mock_robot)
        np.testing.assert_array_equal(result, ctrl_range)

    def test_falls_back_to_joint_limits(self):
        """Test fallback to joint limits when ctrlrange is None."""
        from scripts.isaacsim_ros2 import _resolve_ctrl_range_isaacsim

        mock_robot = MagicMock()
        mock_robot.get_joint_limits.return_value = np.array([[-1.5, 1.5], [-2.5, 2.5]])
        result = _resolve_ctrl_range_isaacsim(None, 2, mock_robot)
        np.testing.assert_array_equal(result, np.array([[-1.5, 1.5], [-2.5, 2.5]]))

    def test_uses_fallback_when_no_limits(self):
        """Test fallback to [-1, 1] when no limits available."""
        from scripts.isaacsim_ros2 import _resolve_ctrl_range_isaacsim

        mock_robot = MagicMock()
        mock_robot.get_joint_limits.return_value = None
        result = _resolve_ctrl_range_isaacsim(None, 3, mock_robot)
        expected = np.tile(np.array([-1.0, 1.0], dtype=np.float32), (3, 1))
        np.testing.assert_array_equal(result, expected)

    def test_partial_limits_padded(self):
        """Test that partial limits are padded with defaults."""
        from scripts.isaacsim_ros2 import _resolve_ctrl_range_isaacsim

        mock_robot = MagicMock()
        mock_robot.get_joint_limits.return_value = np.array([[-1.0, 1.0]])
        result = _resolve_ctrl_range_isaacsim(None, 3, mock_robot)
        expected = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestGenericIsaacSimEnv(unittest.TestCase):
    """Test GenericIsaacSimEnv class."""

    def setUp(self):
        self.original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent.parent)

    def tearDown(self):
        os.chdir(self.original_cwd)

    @patch("scripts.isaacsim_ros2.SimulationApp")
    @patch("scripts.isaacsim_ros2.World")
    @patch("scripts.isaacsim_ros2.SingleArticulation")
    @patch("scripts.isaacsim_ros2._import_urdf")
    @patch("scripts.isaacsim_ros2.resolve_urdf")
    def test_env_initialization(
        self, mock_resolve, mock_import, mock_articulation, mock_world, mock_sim_app
    ):
        """Test environment initialization with mocks."""
        from utils.robot_loader import RobotConfig
        from scripts.isaacsim_ros2 import GenericIsaacSimEnv

        mock_sim_app.return_value = MagicMock()
        mock_sim_app.return_value.is_running.return_value = True

        mock_resolve.return_value = ("/tmp/test.urdf", None)

        mock_world_instance = MagicMock()
        mock_world.return_value = mock_world_instance

        mock_robot = MagicMock()
        mock_robot.num_dof = 7
        mock_robot.get_joint_positions.return_value = np.zeros(7)
        mock_robot.get_joint_velocities.return_value = np.zeros(7)
        mock_robot.get_joint_limits.return_value = np.tile(
            np.array([-1.0, 1.0]), (7, 1)
        )
        mock_articulation.return_value = mock_robot

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )

        with patch("scripts.isaacsim_ros2.omni.kit.app") as mock_kit_app:
            mock_kit_app.get_app.return_value = None
            env = GenericIsaacSimEnv(config)
            self.assertIsNotNone(env)
            self.assertEqual(env._config.name, "test")

    @patch("scripts.isaacsim_ros2.SimulationApp")
    @patch("scripts.isaacsim_ros2.World")
    @patch("scripts.isaacsim_ros2.SingleArticulation")
    @patch("scripts.isaacsim_ros2._import_urdf")
    @patch("scripts.isaacsim_ros2.resolve_urdf")
    def test_env_step_with_action(
        self, mock_resolve, mock_import, mock_articulation, mock_world, mock_sim_app
    ):
        """Test environment step with action."""
        from utils.robot_loader import RobotConfig
        from scripts.isaacsim_ros2 import GenericIsaacSimEnv

        mock_sim_app.return_value = MagicMock()
        mock_sim_app.return_value.is_running.return_value = True

        mock_resolve.return_value = ("/tmp/test.urdf", None)

        mock_world_instance = MagicMock()
        mock_world.return_value = mock_world_instance

        mock_robot = MagicMock()
        mock_robot.num_dof = 7
        mock_robot.set_joint_positions = MagicMock()
        mock_robot.get_joint_positions.return_value = np.zeros(7)
        mock_robot.get_joint_velocities.return_value = np.zeros(7)
        mock_robot.get_joint_limits.return_value = np.tile(
            np.array([-1.0, 1.0]), (7, 1)
        )
        mock_articulation.return_value = mock_robot

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )

        with patch("scripts.isaacsim_ros2.omni.kit.app") as mock_kit_app:
            mock_kit_app.get_app.return_value = None
            env = GenericIsaacSimEnv(config)
            action = np.zeros(7)
            env.step(action)
            mock_robot.set_joint_positions.assert_called()

    @patch("scripts.isaacsim_ros2.SimulationApp")
    @patch("scripts.isaacsim_ros2.World")
    @patch("scripts.isaacsim_ros2.SingleArticulation")
    @patch("scripts.isaacsim_ros2._import_urdf")
    @patch("scripts.isaacsim_ros2.resolve_urdf")
    def test_env_state(
        self, mock_resolve, mock_import, mock_articulation, mock_world, mock_sim_app
    ):
        """Test environment state retrieval."""
        from utils.robot_loader import RobotConfig
        from scripts.isaacsim_ros2 import GenericIsaacSimEnv

        mock_sim_app.return_value = MagicMock()
        mock_sim_app.return_value.is_running.return_value = True

        mock_resolve.return_value = ("/tmp/test.urdf", None)

        mock_world_instance = MagicMock()
        mock_world.return_value = mock_world_instance

        mock_robot = MagicMock()
        mock_robot.num_dof = 7
        mock_robot.get_joint_positions.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )
        mock_robot.get_joint_velocities.return_value = np.array(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        )
        mock_robot.get_joint_limits.return_value = np.tile(
            np.array([-1.0, 1.0]), (7, 1)
        )
        mock_articulation.return_value = mock_robot

        config = RobotConfig(
            name="test",
            urdf="/tmp/test.urdf",
            ctrl_range=np.tile(np.array([-1.0, 1.0]), (7, 1)).tolist(),
        )

        with patch("scripts.isaacsim_ros2.omni.kit.app") as mock_kit_app:
            mock_kit_app.get_app.return_value = None
            env = GenericIsaacSimEnv(config)
            state = env.state()
            self.assertEqual(len(state), 14)
            np.testing.assert_array_equal(
                state[:7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            )


class TestIsaacSimRosBridge(unittest.TestCase):
    """Test IsaacSimRosBridge node."""

    @patch("scripts.isaacsim_ros2.rclpy")
    def test_bridge_initialization(self, mock_rclpy):
        """Test bridge node initialization."""
        from utils.robot_loader import RobotConfig
        from scripts.isaacsim_ros2 import IsaacSimRosBridge, GenericIsaacSimEnv

        mock_rclpy.init.return_value = None
        mock_node = MagicMock()
        mock_rclpy.Node.return_value = mock_node

        mock_env = MagicMock(spec=GenericIsaacSimEnv)
        config = RobotConfig(name="test", urdf="/tmp/test.urdf")

        bridge = IsaacSimRosBridge(mock_env, config)
        self.assertEqual(bridge._config.name, "test")
        mock_node.create_subscription.assert_called()
        mock_node.create_timer.assert_called()


if __name__ == "__main__":
    unittest.main()
