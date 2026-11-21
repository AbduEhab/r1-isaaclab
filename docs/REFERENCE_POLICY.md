# The Unitree R1 walking policy (reference)

A real R1 walking policy is publicly downloadable — not from Unitree's GitHub, but from SpacemiT's
model zoo, which ships R1 (and G1, Go2, H1_2, …) policies exported to ONNX:

```
https://archive.spacemit.com/spacemit-ai/model_zoo/rl/humanoid_unitree_r1/motion/policy.onnx
```

`reference/unitree_r1_policy/` in this repository holds a copy (`policy.onnx`, `weights.npz`,
`metadata.json`) so the experiments below are reproducible offline.

## What is in the file

The ONNX graph carries its whole contract in metadata — this is the same contract the team's
`POLICY_CHECKLIST.md` describes, and it settles the open questions:

| Field | Value |
|---|---|
| input / output | 83 obs → 24 actions |
| `observation_names` | `base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, actions` |
| `joint_names` | 24, in deploy order (legs, waist, arms; head excluded) |
| `joint_stiffness` | 100 / 40 / 20 by group — matches `r1_constants.py` |
| `joint_damping` | 2.0 (1.0 for arms) |
| `default_joint_pos` | Unitree's HOME_KEYFRAME (knee 0.3, hip −0.1, ankle −0.2, shoulder pitch 0.35, elbow 0.87, roll ±0.18) |
| `action_scale` | per joint, 0.150 / 0.312 / 0.375 / 0.412 — **not one value for all** |
| network | MLP 512-256-128-24, ELU, with an observation normalizer (mean / div) |
| producer | `pytorch 2.10.0`, run `2026-03-20_14-29-13` |

The action scale and default pose match this task's action term exactly, so the 24 network outputs
are the raw actions our env expects — no remapping of the action space is needed.

## Running it in our sim

`scripts/run_unitree_policy.py` builds the task, assembles the 83-dim observation per the metadata,
runs the MLP with numpy, and steps the environment. Measured with 8 environments, 400 steps:

| Command | Forward | Lateral | Yaw rate | Heading / s | Height | Step freq | Ankle corr |
|---|---|---|---|---|---|---|---|
| `vx=0.5, wz=0` | **0.436 m/s** | 0.095 | 0.042 | — | 0.721 m (std 0.008) | 3.00 Hz | **−0.54 (alternating)** |
| `vx=0, wz=1.0` | 0.076 | 0.042 | **0.708 rad/s** | 5.69 rad / 8 s | 0.730 m (std 0.007) | 1.62 Hz | **−0.55** |

So it walks (alternating steps, stable height), tracks forward speed within 13 %, turns at 71 % of a
1 rad/s command without strafing, and needs **no base linear velocity** — the 83-dim contract is
sufficient. That refutes the earlier conclusion that dropping base linear velocity made walking
impossible; our v2 failure was a training/reward problem, not the contract.

## Why this matters

1. It is a reference walker to compare our policies against (gait frequency, alternation, height).
2. It is a teacher: behaviour cloning or teacher-student distillation in our own Isaac Lab task
   would produce a policy trained under our physics, with the deployable 83-dim contract.
3. It confirms the deployment numbers the team's checklist cites: per-joint action scale, HOME pose,
   gains, 24 joints, 83 observations, gait phase in the observation.

Note on use: this is a policy published for download by SpacemiT (a RISC-V vendor) alongside
Unitree humanoid models; it is not Unitree's official SDK artifact. Check the terms before deploying
it on hardware.

## Reproduce

```powershell
docker run --rm --gpus all -v C:\per\thesis\r1-isaaclab:/workspace isaac-env:6.2 /bin/bash -lc `
  "cd /workspace && /opt/IsaacLab/isaaclab.sh -p /workspace/scripts/run_unitree_policy.py `
   --vx 0.5 --wz 0.0 --steps 400"
```
