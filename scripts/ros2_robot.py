import rclpy

from mmk.robot.mujoco import Mujoco, MujocoConfig
from mmk.ros2.robot import RobotNode

config = MujocoConfig()
robot = Mujoco(config)
node = RobotNode(robot)

rclpy.init()
import mujoco.viewer
with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    while viewer.is_running():
        rclpy.spin_once(node)
        robot.step()
        viewer.sync()
