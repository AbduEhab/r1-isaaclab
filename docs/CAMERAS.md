# Simulated eyes: the R1 head stereo pair

The physical R1 has a camera mount on the head (`camera_joint: head_yaw_link → camera_link`, offset
`xyz (0.067, 0, −0.0216)`, pitch **0.349 rad ≈ 20° down**) and a stereo pair there. The simulation
models that pair, with RGB and metric depth.

## Where the cameras sit

Link chain in the imported USD (the URDF names them):

```
pelvis_link / waist_roll_link / waist_yaw_link / head_pitch_link / head_yaw_link
```

So each camera prim is created under `{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_roll_link/
waist_yaw_link/head_pitch_link/head_yaw_link/cam_<left|right>` and inherits the head motion (the head
yaw/pitch joints are not actuated by the walking policy; they hold their default pose).

| Property | Value |
|---|---|
| baseline | **0.065 m** (left +Y, right −Y, robot frame) |
| resolution | 320 × 240, 30 Hz update |
| intrinsics | `PinholeCameraCfg(focal_length=18 mm, horizontal_aperture=20.955 mm)` → fx ≈ 274.9 px, HFOV ≈ 60° |
| outputs | `rgb` (uint8), `distance_to_image_plane` (float, metres) |
| clipping | 0.05 – 100 m |

## Enabling them

Cameras are off for training (rendering costs time and VRAM) and enabled by an environment flag:

```powershell
set R1_CAMERAS=1
set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8
C:\isaac-sim\python.bat C:\per\thesis\r1-isaaclab\scripts\capture_stereo.py --viz none --num_envs 1
```

The capture writes, under `data/camera_samples/<run>/`:

```
cam_left_rgb.png    cam_left_depth.npy (+ .png visualisation)
cam_right_rgb.png   cam_right_depth.npy (+ .png visualisation)
meta.json           intrinsics, per-camera world pose, depth range, stereo baseline
```

`R1_CAMERAS=1` reaches the task through `r1_isaaclab/cameras.py` (`add_head_cameras`), which is why
the normal training config is untouched.

## Depth: what to use where

| Situation | Depth source | Notes |
|---|---|---|
| Training in sim | `distance_to_image_plane` (ground truth) | free, exact; a learned depth model would only add error |
| Real robot | **stereo** from the pair (rectify → SGBM / RAFT-Stereo / Isaac ROS) | metric, ~1–2 cm at 0.5–2 m, no scale ambiguity |
| Real robot, fallback | monocular metric depth (Depth Anything V2-metric, Metric3D v2, DepthPro) | 5–20 % scale error and sim→real gap; fine for coarse obstacle avoidance, risky for grasping |

A learned monocular model is therefore useful as a *fallback or teacher*, not as the primary depth
source when two calibrated cameras exist.

## Next steps this enables

- Navigation task: room + random goals, depth (or height scan) as observation, success/collision/
  time-to-goal metrics.
- Manipulation: box detection and 6-pose estimation, gripper on `wrist_roll`, pick-and-place with
  IK first, learned arm policy later.

## Gotchas found while wiring this

- `TiledCameraCfg` in this Isaac Lab version needs an explicit `spawn=PinholeCameraCfg(...)`.
- The app must be started with camera rendering on (`launch_simulation` receives
  `enable_cameras=True`; the CLI equivalent is `--enable_cameras`).
- `gym.make` returns a wrapped env; scene sensors are on `env.unwrapped.scene`.
- Isaac Sim's prebundle ships **scipy 1.17.0**; a pip scipy 1.18.1 in its Python shadows it and
  breaks `omni.replicator` with `scipy._cyutility does not export slice_memviewslice`. Pin
  `scipy==1.17.0`.
