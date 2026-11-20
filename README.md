# R1 in Isaac Lab

The Unitree **R1** humanoid as a native **Isaac Lab 3.0** task: a URDF-rebuilt USD asset, the R1's
own gains and torque limits (per the team's `POLICY_CHECKLIST.md`), and a velocity-tracking
locomotion environment that trains headless with RSL-RL on Isaac Sim 6.2's Newton/MuJoCo-Warp
backend.

To our knowledge this is the first R1 integration for Isaac Lab. The stock Isaac Lab 3.0 asset
library ships A1, Go1, Go2, H1 and G1 — there is no R1, so this repository provides one.

## Status

| Piece | State |
|---|---|
| R1 USD built from the authoritative `R1.urdf` | done — 26 revolute joints, free base, head included |
| R1 asset config (gains, limits, armature) | done — the R1's table, not the G1's |
| `Isaac-Velocity-Flat-R1` task | done — trains headless |
| 100-iteration reference run (2048 envs) | done — 461 s, episode length 13 → 982 of 1000 — [results](docs/RESULTS.md) |
| Full 1500-iteration policy | **not trained yet** — the reference run shows the pipeline works |
| Deployment to the real robot (`deploy.yaml`) | not started — needs the export step |

Documentation: [install](docs/INSTALL.md) · [asset](docs/ASSET.md) · [task](docs/TASK.md) ·
[training](docs/TRAINING.md) · [results](docs/RESULTS.md) · [troubleshooting](docs/TROUBLESHOOTING.md)

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
├── r1_isaaclab/              the Isaac Lab task package
│   ├── __init__.py           registers Isaac-Velocity-Flat-R1 / -Rough-R1
│   ├── asset.py              the R1 ArticulationCfg (gains, limits, armature)
│   ├── env_cfg.py            the velocity task (actions, rewards, events, terminations)
│   └── agents/rsl_rl_ppo_cfg.py
├── scripts/
│   ├── train.py              training entry point (registers the task, then Isaac Lab's CLI)
│   ├── prepare_r1_urdf.py    stage a Unitree R1 URDF + meshes for the importer
│   └── import_r1_urdf.py     URDF → USD with Isaac Sim's importer
├── assets/r1/                the built USD (R1.usda + payloads)
├── models/                   reference checkpoints and their run parameters
└── docs/                     install, asset, task, training, results, troubleshooting
```

## What the policy sees and does

- **Actions (14):** the 12 leg joints and 2 waist joints. Arms and head are held by their own PD,
  as on the robot while it walks.
- **Observations (78):** base linear/angular velocity, projected gravity, velocity command, joint
  positions and velocities (26 joints), the 14 actions, and a history term.
- **Command:** forward 0–1 m/s, lateral ±0.5 m/s, yaw ±1 rad/s.
- **Rewards:** Isaac Lab's stock G1 velocity set (tracking, feet air time, feet slide, joint
  deviations, torque/acceleration/action-rate penalties, termination penalty).

Details in [docs/TASK.md](docs/TASK.md).

## Provenance

- **Code in this repository** (`r1_isaaclab/`, `scripts/`, docs): written for this thesis, MIT
  licensed — see [LICENSE](LICENSE).
- **`assets/r1/`**: the Unitree R1 model (URDF and STL meshes) and the USD built from it. The
  R1 model is Unitree Robotics' material, used here under the same terms as the team's
  `arena-r1-kit`; the USD is generated, not authored, by us.
- **Task design**: Isaac Lab's velocity-locomotion templates (BSD-3-Clause, © NVIDIA), adapted to
  the R1's body and joint names.
