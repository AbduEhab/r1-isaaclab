# Results

All numbers were measured on **Windows 11**, **Isaac Sim 6.2.0**, **Isaac Lab v3.0.0** (repository
install), **PyTorch 2.11.0+cu130**, **Newton 1.6.1**, **RTX A5000 Laptop (16 GB)**. The 4500-iteration
run was trained inside the `isaac-env:6.2` container on the same machine.

## Summary of the three R1 policies

| | v1: 14 actions | v2: 24 actions, no base linear velocity | v3: 24 actions + base linear velocity |
|---|---|---|---|
| Checkpoint | `models/r1_flat_1500it` | — (superseded, not kept) | **`models/r1_flat_4500it`** |
| Envs / batch | 2048 / 49k | 2048 / 49k | **8192 / 393k** |
| Iterations | 1500 | 1500 | **4500** |
| Mean reward | +20.7 | −6.7 | **+33.25** |
| Velocity error xy | 0.193 m/s | 0.549 m/s | **0.115 m/s** |
| Yaw rate error | 0.528 rad/s | 1.161 rad/s | **0.219 rad/s** |
| `feet_air_time` | some (it hopped) | **0.0009 — never steps** | **0.142 — steps** |
| Behaviour | hopped in place | stood still | **walks and tracks commands** |

## Framework sanity check (before the R1)

NVIDIA's stock humanoid task, to prove the environment:

| Task | Envs | Iterations | Wall time | Exit |
|---|---|---|---|---|
| `Isaac-Velocity-Flat-G1` | 8 | 5 | 99.4 s | 0 |

## R1 integration checkpoints

| Step | Result |
|---|---|
| R1 USD from `R1.urdf` | 26 revolute joints (12 legs, 2 waist, 10 arms, 2 head), free base, `mujoco`/`physics`/`physx` variants |
| R1 task constructs in Isaac Lab | Newton/MuJoCo-Warp backend, 24 actions (head excluded) |
| First run | 3 iterations, exit 0 |

## v1 — 1500 iterations (committed under `models/r1_flat_1500it`)

Isaac-Velocity-Flat-R1, 2048 envs, 1500 iterations, **5376 s (1 h 29.6 min)**, 3.53 s/iteration:

| Iteration | Mean reward | Mean episode length | vel error xy | vel error yaw |
|---|---|---|---|---|
| 0 | -0.07 | 13 | 0.642 | 2.437 |
| 500 | -4.32 | 957 | 0.666 | 1.236 |
| 1000 | +14.47 | 1000 | 0.203 | 0.797 |
| 1250 | +19.91 | 1000 | 0.168 | 0.579 |
| 1499 | +20.73 | 994 | 0.191 | 0.529 |

This policy tracked velocity but **hopped**: with a straight-legged home pose, 14 actions and no gait
phase, a symmetric bounce satisfied the velocity reward. The inputs needed for an alternating gait
were missing.

## v2 — the Unitree contract, 1500 iterations (superseded)

Same task with the reference inputs: crouched home pose, 24 actions, gait phase, no base linear
velocity. It tracked **worse** (velocity error 0.549 m/s) and `feet_air_time` was 0.0009: the policy
never lifted a foot. Without base linear velocity it cannot estimate its own speed, and standing
still is the safe optimum.

## v3 — 4500 iterations, walking (committed under `models/r1_flat_4500it`)

v2 contract plus base linear velocity restored, 8192 envs, `num_steps_per_env = 48` (batch 393k),
4500 iterations, **19 845 s (5.5 h)**, ~4.1 s/iteration at the end:

| Iteration | Mean reward | Mean episode length | vel error xy | vel error yaw |
|---|---|---|---|---|
| 0 | -1.49 | 43 | 0.746 | 2.656 |
| 500 | +18.90 | 1000 | 0.151 | 0.678 |
| 1000 | +27.86 | 992 | 0.108 | 0.297 |
| 2000 | +30.56 | 1000 | 0.112 | 0.243 |
| 3000 | +32.33 | 1000 | 0.114 | 0.226 |
| 4000 | +33.06 | 1000 | 0.112 | 0.220 |
| 4499 | **+33.25** | **1000** | **0.115** | **0.219** |

Final reward terms: `track_lin_vel_xy_exp 0.931/1.0`, `track_ang_vel_z_exp 0.798/1.0`,
`feet_air_time 0.142`, `flat_orientation_l2 -0.0055` (upright), `action_rate_l2 -0.058`.

Tracking converges by iteration ~1000 and then improves slowly; 4500 iterations buy ~0.01 m/s over
1500. The important change is the behaviour: `feet_air_time` above zero and a steady 1000-step
episode length mean alternating steps, not a hop.

Note: v3 uses base linear velocity as an observation (86 → 89 dims). A deployment needs a state
estimator, or the term must be removed and the policy retrained — the same trade-off v2 ran into.

## Not measured yet

- Video/visual confirmation of the v3 gait (the container is headless; GUI capture on Windows is
  currently broken by an `omni.kit.widget.layers` error).
- Rough-terrain task.
- Sim-to-real, and the `deploy.yaml` export the team's POLICY_CHECKLIST requires.
