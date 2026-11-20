# R1 in Isaac Lab

The Unitree **R1** humanoid as a native **Isaac Lab 3.0** task: a URDF-rebuilt USD asset, the R1's
own gains and torque limits (per the team's `POLICY_CHECKLIST.md`), and a velocity-tracking
locomotion environment that trains headless with RSL-RL on Isaac Sim 6.2's Newton/MuJoCo-Warp
backend.

To our knowledge this is the first R1 integration for Isaac Lab. The stock Isaac Lab 3.0 asset
library ships A1, Go1, Go2, H1 and G1 Ã¢â‚¬â€ there is no R1, so this repository provides one.

## Status

| Piece | State |
|---|---|
| R1 USD built from the authoritative `R1.urdf` | done Ã¢â‚¬â€ 26 revolute joints, free base, head included |
| R1 asset config (gains, limits, armature) | done Ã¢â‚¬â€ the R1's table, not the G1's |
| `Isaac-Velocity-Flat-R1` task | done Ã¢â‚¬â€ trains headless |
| 100-iteration reference run (2048 envs) | done Ã¢â‚¬â€ 461 s, episode length 13 Ã¢â€ â€™ 982 of 1000 Ã¢â‚¬â€ [results](docs/RESULTS.md) |
| Full 1500-iteration policy | **done — see `models/r1_flat_1500it`** (v1: hop gait, superseded) |
| Corrected contract (reference pose, 24 actions, gait phase) | v2 retraining |
| Deployment to the real robot (`deploy.yaml`) | not started Ã¢â‚¬â€ needs the export step |

### The trained policy

`models/r1_flat_1500it/` holds the first R1 locomotion policy trained in Isaac Lab: 2048
environments, 1500 PPO iterations, 1 h 30 min on the RTX A5000.

| Metric | Iteration 0 | Iteration 1499 | Last 50 (mean) |
|---|---|---|---|
| Mean reward | Ã¢Ë†â€™0.07 | +20.73 | **+20.56** |
| Mean episode length | 13 | 994 | **999.1 / 1000** |
| Velocity error xy | 0.642 m/s | 0.191 m/s | **0.193 m/s** |
| Yaw-rate error | 2.437 rad/s | 0.529 rad/s | **0.528 rad/s** |

Watch it with `scripts/play.py` (see [docs/TRAINING.md](docs/TRAINING.md#watching-a-trained-policy));
limits and the full trend are in [models/r1_flat_1500it/README.md](models/r1_flat_1500it/README.md).

Documentation: [install](docs/INSTALL.md) Ã‚Â· [asset](docs/ASSET.md) Ã‚Â· [task](docs/TASK.md) Ã‚Â·
[training](docs/TRAINING.md) Ã‚Â· [results](docs/RESULTS.md) Ã‚Â· [troubleshooting](docs/TROUBLESHOOTING.md)

## Quick start

Prerequisites: Isaac Sim 6.2.0 (binary install) and Isaac Lab v3.0.0 installed from the repository,
with the `_isaac_sim` link present. See [docs/INSTALL.md](docs/INSTALL.md).

```powershell
# from the IsaacLab repository, with the binary install linked as _isaac_sim
cd C:\path\to\IsaacLab
isaaclab.bat -p C:\path\to\r1-isaaclab\scripts\train.py `
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none `
  --num_envs 4096 --max_iterations 1500
```

On Windows, prefix the command with `$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'` if stdout
is redirected to a file (see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)).

## Repository layout

```
r1-isaaclab/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ r1_isaaclab/              the Isaac Lab task package
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py           registers Isaac-Velocity-Flat-R1 / -Rough-R1
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ asset.py              the R1 ArticulationCfg (gains, limits, armature)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ env_cfg.py            the velocity task (actions, rewards, events, terminations)
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ agents/rsl_rl_ppo_cfg.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ scripts/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ train.py              training entry point (registers the task, then Isaac Lab's CLI)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ prepare_r1_urdf.py    stage a Unitree R1 URDF + meshes for the importer
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ import_r1_urdf.py     URDF Ã¢â€ â€™ USD with Isaac Sim's importer
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ assets/r1/                the built USD (R1.usda + payloads)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ models/                   reference checkpoints and their run parameters
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ docs/                     install, asset, task, training, results, troubleshooting
```

## What the policy sees and does

- **Actions (24):** the 12 leg joints, 2 waist joints and 10 arm joints â€” Unitree's R1 contract. The head is held by its own PD.
- **Observations (83):** base angular velocity, projected gravity, velocity command, a 2-dim gait
  phase clock, joint positions and velocities (24 joints), and the 24 actions. No base linear
  velocity â€” the real robot has no sensor for it.
- **Command:** forward 0Ã¢â‚¬â€œ1 m/s, lateral Ã‚Â±0.5 m/s, yaw Ã‚Â±1 rad/s.
- **Rewards:** Isaac Lab's stock G1 velocity set (tracking, feet air time, feet slide, joint
  deviations, torque/acceleration/action-rate penalties, termination penalty).

Details in [docs/TASK.md](docs/TASK.md).

## Provenance

- **Code in this repository** (`r1_isaaclab/`, `scripts/`, docs): written for this thesis, MIT
  licensed Ã¢â‚¬â€ see [LICENSE](LICENSE).
- **`assets/r1/`**: the Unitree R1 model (URDF and STL meshes) and the USD built from it. The
  R1 model is Unitree Robotics' material, used here under the same terms as the team's
  `arena-r1-kit`; the USD is generated, not authored, by us.
- **Task design**: Isaac Lab's velocity-locomotion templates (BSD-3-Clause, Ã‚Â© NVIDIA), adapted to
  the R1's body and joint names.


