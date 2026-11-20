# Results

All numbers were measured on **Windows 11**, **Isaac Sim 6.2.0**, **Isaac Lab v3.0.0** (repository
install), **PyTorch 2.11.0+cu130**, **Newton 1.6.1**, **RTX A5000 Laptop (16 GB)**.

## Framework sanity check (before the R1)

NVIDIA's stock humanoid task, to prove the environment:

| Task | Envs | Iterations | Wall time | Exit |
|---|---|---|---|---|
| `Isaac-Velocity-Flat-G1` | 8 | 5 | 99.4 s | 0 |

## R1 integration checkpoints

| Step | Result |
|---|---|
| R1 USD from `R1.urdf` | 26 revolute joints (12 legs, 2 waist, 10 arms, 2 head), free base, `mujoco`/`physics`/`physx` variants |
| R1 task constructs in Isaac Lab | action shape 14, observation shape 78, Newton/MuJoCo-Warp backend |
| First 3-iteration run | exit 0, 31 s (4 envs) |
| Head actuators added (all 26 joints driven) | 3-iteration run exit 0, 12 s |

## Reference run (committed)

`Isaac-Velocity-Flat-R1`, 2048 envs, 100 iterations, **461 s** (3.9â€“4.2 s/iteration):

| Iteration | Mean reward | Mean episode length | vel error xy | vel error yaw |
|---|---|---|---|---|
| 0 | âˆ’0.07 | 13 | 0.642 | 2.437 |
| 10 | âˆ’4.13 | 240 | 0.595 | 2.766 |
| 25 | âˆ’11.29 | 599 | 0.684 | 2.624 |
| 50 | âˆ’15.19 | 979 | 0.633 | 2.423 |
| 75 | âˆ’14.32 | 987 | 0.685 | 2.126 |
| 99 | âˆ’12.14 | 982 | 0.658 | 1.923 |

**Interpretation:** episode length rising from 13 to 982 of 1000 steps means the R1 learns to keep
its pelvis off the ground and survive the full 20 s episode. Tracking error is only starting to
move. This is 7 % of the stock 1500-iteration schedule; it demonstrates the pipeline, not a finished
locomotion policy.

## An earlier run with the MuJoCo-converted asset

Before the URDF rebuild was wired in, the same task trained on the older 24-joint asset
(2048 envs, 100 iterations, 515 s): episode length 767 â†’ 1000, yaw error 2.59 â†’ 1.74. The URDF
asset reaches comparable behaviour faster per iteration (3.9 s vs 5.2 s), and includes the head.

## Trained policy: 1500 iterations (committed under models/r1_flat_1500it)

Isaac-Velocity-Flat-R1, 2048 envs, 1500 iterations, **5376 s (1 h 29.6 min)**, 3.53 s/iteration:

| Iteration | Mean reward | Mean episode length | vel error xy | vel error yaw |
|---|---|---|---|---|
| 0 | -0.07 | 13 | 0.642 | 2.437 |
| 50 | -15.13 | 968 | 0.621 | 2.378 |
| 100 | -13.13 | 995 | 0.643 | 1.858 |
| 250 | -8.69 | 955 | 0.657 | 1.535 |
| 500 | -4.32 | 957 | 0.666 | 1.236 |
| 750 | +4.38 | 998 | 0.412 | 1.196 |
| 1000 | +14.47 | 1000 | 0.203 | 0.797 |
| 1250 | +19.91 | 1000 | 0.168 | 0.579 |
| 1499 | +20.73 | 994 | 0.191 | 0.529 |

Last 50 iterations: reward **20.56**, episode length **999.1/1000**, velocity error **0.193 m/s**,
yaw error **0.528 rad/s**; best iteration reward 22.07. The R1 stands, walks and tracks commanded
velocity on flat ground. Tracking only improves after ~700 iterations, once reward turns positive.

## Not measured yet

- Full 1500-iteration policy quality.
- Rough-terrain task.
- Sim-to-real, and the `deploy.yaml` export the team's POLICY_CHECKLIST requires.

