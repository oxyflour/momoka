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
pixi install -e mujoco
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
pixi run -e mujoco python scripts/ros2_mujoco_sim.py
```

## Robot Configuration

Robots are configured via YAML files in `config/robots/`. Each config defines:

- `name`: Robot identifier
- `urdf`: Path to URDF or MJCF XML file
- `mjcf_to_urdf`: Whether to convert MJCF to URDF (default: false)
- `mesh_base`: Base path for mesh files (optional)
- `package_replacements`: List of `from`/`to` pairs for URI replacement
- `ctrl_range`: Override joint control ranges (optional)
- `joint_names`: Ordered list of joint names (optional)

Example `config/robots/panda.yaml`:

```yaml
name: panda
urdf: data/mujoco_menagerie/franka_emika_panda/panda.xml
mjcf_to_urdf: true
mesh_base: data/mujoco_menagerie/franka_emika_panda/assets
package_replacements:
  - from: "package://Panda/"
    to: ""
  - from: "package://panda/"
    to: ""
ctrl_range:
  - [-2.8973, 2.8973]
  - [-1.7628, 1.7628]
  - [-2.8973, 2.8973]
  - [-3.0718, -0.0698]
  - [-2.8973, 2.8973]
  - [-2.8973, 2.8973]
  - [-2.8973, 2.8973]
joint_names:
  - panda_joint1
  - panda_joint2
  - panda_joint3
  - panda_joint4
  - panda_joint5
  - panda_joint6
  - panda_joint7
```

## MuJoCo bridge (lerobot -> ROS 2 -> MuJoCo)

1) Ensure the robot XML and assets are present (e.g., Panda):

```bash
# Panda example
data/mujoco_menagerie/franka_emika_panda/panda.xml
data/mujoco_menagerie/franka_emika_panda/assets/  # with assets from ASSETS.txt
```

2) Run the bridge with `--robot` flag (default: panda):

```bash
pixi run -e mujoco python scripts/mujoco_ros2.py --robot panda
```

3) Run the lerobot WS client:

```bash
pixi run -e lerobot python scripts/lerobot_ws_client.py
```

## Isaac Sim bridge (lerobot -> ROS 2 -> Isaac Sim)

1) Install additional dependency:

```bash
pixi run -e mujoco pip install mjcf2urdf
```

2) Ensure robot XML and assets are present.

3) Run the bridge:

```bash
pixi run -e mujoco python scripts/isaacsim_ros2.py --robot panda
```

4) Or run inside Isaac Sim with ROS 2 Bridge extension:

```bash
set ROS_DISTRO=humble
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;%CONDA_PREFIX%/lib/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
isaacsim --exec scripts\isaacsim_ros2.py --robot panda --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```

### Action Mapping

The first N action values are applied to the robot joints, where N is the
number of actuators (controlled joints). MuJoCo uses actuator ctrlrange;
Isaac Sim applies a small delta to joint positions. Control ranges can be
overridden via `ctrl_range` in the robot config.

## WebSocket bridge (lerobot -> ROS 2)

Because ROS 2 pins `numpy<2` and lerobot requires `numpy>=2`, they run in
separate Pixi environments and communicate over WebSocket.

1) Start the ROS 2 WebSocket server (publishes to `lerobot/inference`):

```bash
pixi run -e ros2 python scripts/ros2_bridge.py
```

2) In another shell, send a sample lerobot payload:

```bash
pixi run -e lerobot python scripts/lerobot_ws_client.py
```

You can also send CompressedImage/JointState payloads directly. Include `type`
(`compressed_image` or `joint_state`) and optional `topic` to override the ROS 2
publish target:

```json
{"type":"compressed_image","topic":"/camera/image/compressed","format":"jpeg","data":"<base64>"}
```

```json
{"type":"joint_state","topic":"/joint_states","name":["joint1"],"position":[0.1]}
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
