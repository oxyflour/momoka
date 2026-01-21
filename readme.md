```bash
pixi shell -e lerobot
# must install package so that lerobot can find the lerobot_robot_xxx and lerobot_teleoperator_xxx modules
pip install -e .

lerobot-record --config_path examples/record.yaml
lerobot-replay --config_path examples/replay.yaml
lerobot-train  --config_path examples/train.yaml
lerobot-record --config_path examples/evaluate.yaml

# start a policy server and wait for connections. the server should support Real-Time Chunking (RTC)
python scripts/lerobot_serve.py
```

```bash
pixi shell -e ros2
# create a virtual mujoco robot, which would subscribe actions and publish observations
python src/mmk/ros2/robot.py --config_path examples/ros2.yaml
# create a controller which will connect to policy server, subscribe observations and publish actions
python src/mmk/ros2/control.py --config_path examples/ros2.yaml
```

```bash
# create a virtual isaacsim robot, which would subscribe actions and publish observations
conda run -n env_isaaclab isaacsim --exec src/mmk/ros2/robot.py --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```
