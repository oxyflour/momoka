# momoka

This repo uses `pixi` to set up environments for ROS 2 (Humble) and `lerobot[pi]` v0.4.2.

## Prereqs

- Install pixi: https://pixi.sh
- Make sure you have a shell with access to `pixi`

## Proxy setup

The setup uses the proxy below. Export it before running pixi commands:

PowerShell:

```powershell
$env:HTTPS_PROXY="http://proxy.yff.me:8124/"
$env:HTTP_PROXY="http://proxy.yff.me:8124/"
```

bash/zsh:

```bash
export HTTPS_PROXY="http://proxy.yff.me:8124/"
export HTTP_PROXY="http://proxy.yff.me:8124/"
```

## Create and install the environments

The environments are defined in `pixi.toml`. Install them with:

```bash
pixi install -e ros2
pixi install -e lerobot
pixi install -e mujoco-ros
```

## Activate and verify

Activate the environment:

```bash
pixi shell -e ros2
```

Verify ROS 2:

```bash
ros2 --help
```

Activate the lerobot environment:

```bash
pixi shell -e lerobot
```

Verify lerobot:

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

## MuJoCo bridge (ROS 2 -> MuJoCo)

This script subscribes to `/lerobot/inference` and drives a tiny MuJoCo
simulation using the first action value.

```bash
pixi run -e mujoco-ros python scripts/ros2_mujoco_sim.py
```

## Isaac Sim bridge (ROS 2 -> Isaac Sim 5.1)

Run this inside the Isaac Sim 5.1 Python environment with the ROS 2 Bridge extension enabled.
It subscribes to `/lerobot/inference` and moves a cube based on the first three action values.

Example (adjust to your Isaac Sim install path):

```bash
set ROS_DISTRO=humble
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;%CONDA_PREFIX%/lib/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
isaacsim --exec-script scripts\isaacsim_ros2_drive.py --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```

## WebSocket bridge (lerobot -> ROS 2)

Because ROS 2 pins `numpy<2` and lerobot requires `numpy>=2`, they run in
separate Pixi environments and communicate over WebSocket.

1) Start the ROS 2 WebSocket server (publishes to `lerobot/inference`):

```bash
pixi run -e ros2 python scripts/ros2_ws_server.py
```

2) In another shell, send a sample lerobot payload:

```bash
pixi run -e lerobot python scripts/lerobot_ws_client.py
```

3) Verify ROS 2 topic output:

```bash
pixi run -e ros2 ros2 topic echo /lerobot/inference
```

Replace the dummy payload in `scripts/lerobot_ws_client.py` with real lerobot
inference output.

## Notes

- ROS 2 packages are pulled from the `robostack-humble` channel.
- Windows is supported for both environments.
- ROS 2 and lerobot cannot share a single Pixi environment today because ROS 2 pins `numpy<2` while lerobot requires `numpy>=2` (via `rerun-sdk`).
- The lerobot environment installs from the `0.4.2` git tag to match the recommended install for `lerobot[pi]`.
- For CUDA 12.8 on Windows, install GPU Torch via pip after `pixi install -e lerobot`:
  `pixi run -e lerobot python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`
