# Deploying a trained policy on the real R1

The chain is: train on the **deploy contract** → export → validate in simulation → run on the robot.
Each step has a gate; a policy that fails a gate does not go on the robot.

```
Isaac-Velocity-Flat-R1-Deploy     83 observations per frame, 5-frame history = 415 inputs
        │
        ├─ scripts/export_policy.py ──► deploy/runs/<run>/policy.onnx + deploy.yaml
        │                                   │
        │                                   ├─ arena-r1-kit/check_policy.py   (R1 rules: gains, scales, slots, sources)
        │                                   └─ scripts/validate_deploy.py     (obs parity with the sim + closed loop)
        │
        └─ python -m deploy.r1_policy_runner      (robot: LowState → obs → ONNX → LowCmd)
```

## 1. The contract

The robot can provide: IMU quaternion and angular velocity, joint positions and velocities, and the
command. It cannot provide base linear velocity, so the deploy task **removes that term** and gives
the policy **history** instead (5 frames): velocity is inferable from the motion history.

| Observation | Per frame | × 5 |
|---|---|---|
| `base_ang_vel` | 3 | 15 |
| `projected_gravity` | 3 | 15 |
| `velocity_commands` | 3 | 15 |
| `joint_pos_rel` (24 joints, minus default) | 24 | 120 |
| `joint_vel_rel` | 24 | 120 |
| `last_action` | 24 | 120 |
| `gait_phase` (0.6 s cycle) | 2 | 10 |
| **total** | **83** | **415** |

History is flattened oldest → newest for every term, matching Isaac Lab. The gait clock starts when
the policy starts. Quaternions in the policy are Isaac Lab's **(x, y, z, w)**; Unitree's IMU is
**(w, x, y, z)** and the runner converts.

Joint order is the **action term order** (pattern order: left/right per joint group), not the URDF
order; `deploy.yaml`'s `joint_ids_map` maps each entry to its Unitree motor slot and everything else
(gains, scales, default pose, limits) is in that same order.

## 2. Export

```powershell
C:\isaac-sim\python.bat scripts\export_policy.py `
  --checkpoint logs/rsl_rl/r1_flat/<run>/model_<n>.pt `
  --task Isaac-Velocity-Flat-R1-Deploy `
  --out deploy\runs\<run>_<n>
```

Writes `policy.onnx` (input `obs` [batch, 415], output `actions` [batch, 24], contract in ONNX
metadata) and `deploy.yaml` (Unitree format plus `joint_limits`, `torque_limits`, `meta`).

## 3. Gates

```powershell
# a) the team's rules as code: 13 checks, run in the kit's environment
cd C:\per\thesis\arena-r1-kit
C:\per\thesis\.venv\Scripts\python.exe check_policy.py C:\per\thesis\r1-isaaclab\deploy\runs\<run>

# b) contract in simulation: deploy.yaml observation builder vs the env, then a closed-loop rollout
C:\isaac-sim\python.bat C:\per\thesis\r1-isaaclab\scripts\validate_deploy.py `
  --policy C:\per\thesis\r1-isaaclab\deploy\runs\<run> --steps 300 --vx 0.5
```

`check_policy.py` must print **13 passed, 0 failed**. `validate_deploy.py` must print
**PARITY PASS** (max |deploy − env| < 2e-4) and show the speed/gait you trained for.

## 4. Run on the robot

```bash
# on the robot's computer, with unitree_sdk2py installed and the robot in low-level mode
python -m deploy.r1_policy_runner --policy deploy/runs/<run> --iface eth0 --vx 0.3 --ramp 3
```

What the runner does:

- subscribes `rt/lowstate`, publishes `rt/lowcmd`; policy at `step_dt` (0.02 s)
- **ramp**: for `--ramp` seconds it holds the measured pose while kp ramps 0 → 1, so nothing snaps
- **watchdog**: no fresh lowstate for 0.1 s → all gains dropped (`--strict` makes it exit)
- **clamps**: joint targets clamped to the URDF soft limits; gains/damping from the policy table
- Ctrl+C: zeroes gains and exits

Safety rules that are not in software:

- Only run in the hanger or with an assistant and a hand on the remote; low-level mode bypasses the
  built-in balance controller.
- Check the robot's joint limits and the policy's default pose on the bench first (`check_policy.py`
  passes them, but read the numbers).
- Start with `--ramp 5 --vx 0.0 --strict`, confirm the robot holds the HOME pose, then give a slow
  `--vx`, then speed up.
- Battery, foot condition, and a clear floor the robot can fall onto without damage.

## 5. Known limits of the current policy

- Trained on flat ground; no pushes, payloads or rough terrain. `r1_flat_turn2_1500it` turns by
  pivoting rather than stepping.
- No state estimator other than observation history; on a slippery floor the inferred velocity can be
  wrong, and the policy will mis-track.
- Walk speeds are limited by training: ~0.3–0.7 m/s (the 0.8 m/s command produced a running gait).

## 6. Files

| File | Role |
|---|---|
| `deploy/r1_contract.py` | deploy.yaml loader, gravity from quaternion, action → joint targets |
| `deploy/obs_builder.py` | the 415-dim observation, history buffers, gait clock |
| `deploy/r1_policy_runner.py` | robot loop (DDS, ramps, watchdog, clamps) |
| `scripts/export_policy.py` | checkpoint → ONNX + deploy.yaml (+ contract metadata) |
| `scripts/validate_deploy.py` | in-sim gate: obs parity + closed-loop rollout |

## 7. The deployable policy we ship

`deploy/runs/r1_distilled/` is a policy **distilled from the public reference walker** and trained
on the single-frame deploy contract (83 observations, no base linear velocity, no history):

| command | speed reached | both feet airborne |
|---|---|---|
| 0.3 m/s | 0.302 m/s | **0.0 %** |
| 0.5 m/s | 0.462 m/s | **0.0 %** |
| 0.7 m/s | 0.629 m/s | **0.0 %** |

It passes `check_policy.py` **13/13**, and `scripts/play_deploy.py` (the same observation builder the
robot runner uses) walks it in the simulator. The recipe is `scripts/distill_reference.py`:
behaviour cloning of 196 608 teacher-labelled steps, then two DAgger rounds where the student is
rolled and the teacher labels the student's own states (196 608 pairs each). Dataset:
`data/ref_teacher.npz`, `data/ref_dagger1.npz`, `data/ref_dagger2.npz` (gitignored, regenerate with
the script). `deploy/runs/unitree_reference/` stays as the teacher's own deploy directory and a
fallback.

Run either on the robot with `python -m deploy.r1_policy_runner --policy deploy/runs/r1_distilled`.

