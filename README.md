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

## Panda demo (lerobot -> ROS 2 -> MuJoCo / Isaac Sim)

Use a public Franka Panda model from the MuJoCo menagerie.

1) Ensure the Panda XML and assets are present:
   - `data/mujoco_menagerie/franka_emika_panda/panda.xml`
   - Download the assets listed in `data/mujoco_menagerie/franka_emika_panda/ASSETS.txt`
     into `data/mujoco_menagerie/franka_emika_panda/assets/`

2) Run the MuJoCo demo:

```bash
pixi run -e mujoco python scripts/ros2_mujoco_panda.py
```

3) Run the Isaac Sim demo (Franka asset expected in Isaac Sim assets):

```bash
set ROS_DISTRO=humble
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;%CONDA_PREFIX%/lib/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
isaacsim --exec scripts\isaacsim_ros2_panda.py --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
```

Isaac Sim uses the same MuJoCo `panda.xml` via the MJCF importer extension
(`omni.isaac.mjcf` must be enabled).
Action mapping: the first N action values are applied to the robot joints
(MuJoCo uses actuator ctrlrange; Isaac Sim applies a small delta to joint positions).

## Isaac Sim bridge (ROS 2 -> Isaac Sim)

Run this with the `mujoco` environment (includes ROS 2 and MuJoCo).
It subscribes to `/lerobot/inference` and drives the Panda robot from the MuJoCo menagerie.

The implementation:
- Reads the same MJCF XML file as `ros2_mujoco.py`
- Converts MJCF to URDF using `mjcf2urdf` (not Isaac Sim's MJCF importer)
- Loads the robot into Isaac Sim via URDF importer
- Maps actions to joint positions with proper range scaling

### Setup

1) Install additional dependency:

```bash
pixi run -e mujoco pip install mjcf2urdf
```

2) Ensure Panda XML and assets are present:
   - `data/mujoco_menagerie/franka_emika_panda/panda.xml`
   - Download assets listed in `data/mujoco_menagerie/franka_emika_panda/ASSETS.txt`
     into `data/mujoco_menagerie/franka_emika_panda/assets/`

3) Run the bridge:

```bash
pixi run -e mujoco python scripts/isaacsim_ros2.py
```

### Manual Testing

Run the manual test suite to verify MJCF to URDF conversion and action handling:

```bash
pixi run -e mujoco python scripts/test_isaacsim_ros2_manual.py
```

### Isaac Sim Native (Alternative)

If you want to run inside Isaac Sim 5.1 Python environment with ROS 2 Bridge extension enabled:

```bash
set ROS_DISTRO=humble
set RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set PATH=%PATH%;%CONDA_PREFIX%/lib/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib
isaacsim --exec scripts\isaacsim_ros2.py --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
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
