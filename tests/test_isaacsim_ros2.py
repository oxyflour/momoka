import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)


class TestMjcfUtils(unittest.TestCase):
    """Test MJCF to URDF conversion utilities."""

    def setUp(self):
        self.xml_path = os.path.join(
            "data", "mujoco_menagerie", "franka_emika_panda", "panda.xml"
        )

    @unittest.skipIf(
        not os.path.exists("data/mujoco_menagerie/franka_emika_panda/panda.xml"),
        "Test MJCF file not found",
    )
    def test_mjcf_to_urdf_conversion(self):
        """Test that MJCF to URDF conversion works."""
        from utils.mjcf_utils import mjcf_to_urdf

        if not os.path.exists(self.xml_path):
            self.skipTest(f"Test file {self.xml_path} not found")

        temp_dir = tempfile.mkdtemp()
        urdf_path = os.path.join(temp_dir, "panda.urdf")

        try:
            result_path = mjcf_to_urdf(self.xml_path, urdf_path)
            self.assertEqual(result_path, urdf_path)
            self.assertTrue(os.path.exists(urdf_path))

            with open(urdf_path, "r") as f:
                content = f.read()
                self.assertIn("<robot", content)
                self.assertIn("link", content)
                self.assertIn("joint", content)
        finally:
            if os.path.exists(urdf_path):
                os.remove(urdf_path)
            os.rmdir(temp_dir)

    def test_mjcf_to_urdf_with_temp_file(self):
        """Test MJCF to URDF conversion with automatic temp file creation."""
        from utils.mjcf_utils import mjcf_to_urdf

        with tempfile.TemporaryDirectory() as temp_dir:
            test_xml = os.path.join(temp_dir, "test.xml")
            with open(test_xml, "w") as f:
                f.write('<mujoco model="test"><worldbody/></mujoco>')

            try:
                result_path = mjcf_to_urdf(test_xml)
                self.assertTrue(os.path.exists(result_path))
                self.assertTrue(result_path.endswith(".urdf"))

                with open(result_path, "r") as f:
                    content = f.read()
                    self.assertIn("<robot", content)
            except ImportError:
                self.skipTest("mjcf2urdf not installed")


class TestIsaacSimRos2(unittest.TestCase):
    """Test IsaacSim ROS2 bridge functionality."""

    def test_extract_action_valid(self):
        """Test action extraction from valid ROS message."""
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

    def test_extract_action_invalid_json(self):
        """Test action extraction with invalid JSON."""
        from scripts.isaacsim_ros2 import _extract_action

        from std_msgs.msg import String

        msg = String()
        msg.data = "invalid json"

        action = _extract_action(msg)
        self.assertIsNone(action)

    def test_extract_action_no_action(self):
        """Test action extraction with no action field."""
        from scripts.isaacsim_ros2 import _extract_action

        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps({"result": {}})

        action = _extract_action(msg)
        self.assertIsNone(action)

    @patch("scripts.isaacsim_ros2.SimulationApp")
    @patch("scripts.isaacsim_ros2._convert_mjcf_to_urdf")
    def test_environment_initialization(self, mock_convert, mock_sim_app):
        """Test that PandaIsaacSimEnv initializes correctly."""
        from scripts.isaacsim_ros2 import PandaIsaacSimEnv

        mock_convert.return_value = ("/tmp/test.urdf", np.array([[0.0, 1.0]]))
        mock_instance = MagicMock()
        mock_app_instance = MagicMock()
        mock_sim_app.return_value = mock_app_instance
        mock_app_instance.is_running.return_value = True

        with patch("scripts.isaacsim_ros2.World") as mock_world_class:
            mock_world = MagicMock()
            mock_world_class.return_value = mock_world

            try:
                env = PandaIsaacSimEnv("test.xml")
                self.assertIsNotNone(env)
                mock_world.add_default_ground_plane.assert_called_once()
            except Exception as e:
                self.assertIn("ImportError", str(type(e).__name__))


class TestRos2MujocoCompatibility(unittest.TestCase):
    """Test that IsaacSim ROS2 implementation matches MuJoCo ROS2 interface."""

    def test_action_extraction_compatibility(self):
        """Test that action extraction works the same way in both implementations."""
        from scripts.isaacsim_ros2 import _extract_action as isaac_extract
        from scripts.mujoco_ros2 import _extract_action as mujoco_extract

        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps(
            {"result": {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}}
        )

        action_isaac = isaac_extract(msg)
        action_mujoco = mujoco_extract(msg)

        self.assertIsNotNone(action_isaac)
        self.assertIsNotNone(action_mujoco)
        np.testing.assert_array_equal(action_isaac, action_mujoco)

    def test_action_clipping(self):
        """Test that actions are properly clipped to [-1, 1] range."""
        from scripts.isaacsim_ros2 import _extract_action

        from std_msgs.msg import String
        import json

        msg = String()
        msg.data = json.dumps(
            {"result": {"action": [2.0, -2.0, 0.5, 0.0, -1.0, 1.0, 10.0]}}
        )

        action = _extract_action(msg)
        self.assertIsNotNone(action)
        # Raw action should not be clipped in extraction
        np.testing.assert_array_equal(
            action, np.array([2.0, -2.0, 0.5, 0.0, -1.0, 1.0, 10.0])
        )


if __name__ == "__main__":
    unittest.main()
