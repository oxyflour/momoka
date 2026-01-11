call conda activate env_isaaclab
set ROS_DISTRO=humble
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;%CONDA_PREFIX%\lib\site-packages\isaacsim\exts\isaacsim.ros2.bridge\humble\lib
isaacsim --exec scripts\isaacsim_ros2.py --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
