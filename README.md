# R1 in Isaac Lab

The Unitree **R1** humanoid as a native **Isaac Lab 3.0** task: a URDF-rebuilt USD asset, the R1's own
gains and torque limits, and a velocity-tracking locomotion environment that trains with RSL-RL on
Isaac Sim 6.2's Newton/MuJoCo-Warp backend.

Isaac Lab 3.0 ships A1, Go1, Go2, H1 and G1 in its asset library but no R1, so this repository
provides one.

## Status

| Piece | State |
|---|---|
| R1 USD from the authoritative `urdf/R1.urdf` | done — 26 revolute joints, free base, head included |
| R1 asset config (gains, limits, armature, self-collision) | done — the R1's numbers, not the G1's |
| `Isaac-Velocity-Flat-R1` task | done — trains headless, plays in the Isaac Sim window |
| Walking policy | **walks at 0.3–0.5 m/s**, hop largely removed (flight 1.5–4.4 %), [gait numbers](docs/RESULTS.md) |
| Turning | yaw-rate command tracked at ~1.05 rad/s in place (pivoting gait, not stepped turns) |
| Deployment contract (83 observations, no base linear velocity) | **not started** — needs history observations or an estimator, then `deploy.yaml` export |
| Navigation foundation | **done**: room with walls/obstacles, 91.7 % success in 12 episodes, [numbers](docs/NAVIGATION.md) |
| Shop-floor test bed | **done**: Isaac's forklift warehouse (24x39 m, 793 colliders) + conveyors, occupancy grid + A*, robot walks it, [details](docs/WAREHOUSE.md) |
| Obstacle avoidance | **done**: spawned box between robot and goal, 87.5 % success (8 trials), 0 collisions, [details](docs/NAVIGATION.md) |
| Splat prior map + novelty | **prototype**: R1 splat ? occupancy ? A*, depth-residual novelty 84.8 % recall / 100 % precision, [details](docs/SPLAT_MAP.md) |
| VGGT-SLAM 2.0 map | **done**: warehouse capture ? container run ? Sim(3)-aligned ATE 0.23 m (40 frames) / 0.92 m (70 frames), [details](docs/VGGT_SLAM.md) |
| Warehouse splat | **done**: 212-frame RGB-D walkthrough ? NVIDIA gsplat, 200k Gaussians, walkthrough video + PLY, [details](docs/SPLAT_WAREHOUSE.md) |
| Thesis gap analysis | **done**: SOTA survey (GS, VSLAM, humanoid RL, distillation) + 12 ranked gaps, [deep dive](docs/THESIS_GAPS_2026-11.md); broader landscape incl. VLA/world models/loco-manipulation in [companion](docs/RESEARCH_LANDSCAPE_BROAD_2026-11.md); handless-R1 deep dive of all 15 gaps in [deep dive](docs/THESIS_DEEP_DIVE_HANDLESS_2026-11.md) |
| Sim-to-real | not started |

The complete experimental record (every run, config, metric and finding) is in
**[docs/PAPER_NOTES.md](docs/PAPER_NOTES.md)**.

## Documentation

| Document | Purpose |
|---|---|
| [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) | research record: method, experiments, findings, reproducibility |
| [docs/RESULTS.md](docs/RESULTS.md) | measured results, gait comparisons |
| [docs/TASK.md](docs/TASK.md) | actions, observations, rewards, command contract |
| [docs/TRAINING.md](docs/TRAINING.md) | training and playback commands |
| [docs/ASSET.md](docs/ASSET.md) | how the USD is built from the URDF |
| [docs/INSTALL.md](docs/INSTALL.md) | installing Isaac Sim + Isaac Lab |
| [docs/LOCAL_GUI.md](docs/LOCAL_GUI.md) | running the window on this Windows machine |
| [docs/DISTILLATION.md](docs/DISTILLATION.md) | plan for distilling the public Unitree reference policy |
| [docs/REFERENCE_POLICY.md](docs/REFERENCE_POLICY.md) | the reference policy and its measured gait |
| [docs/CONTAINER.md](docs/CONTAINER.md), [docs/LABD.md](docs/LABD.md) | container and job-queue tooling (optional) |

## Models

| Directory | What it is |
|---|---|
| `models/r1_flat_walk2_6900it` | **current best walker** — 1.5 % flight at 0.3 m/s, 4.4 % at 0.5 |
| `models/r1_flat_gait_6000it` | first non-hopping policy (3.9 % flight) |
| `models/r1_flat_selfcoll_1000it` | walking with self-collision enabled |
| `models/r1_flat_4500it` | 4500-iteration policy: tracks well, hops (19.3 % flight) |
| `models/r1_flat_turn2_1500it` | turning experiment (turns by pivoting) |
| `models/r1_reference` | clips of the public reference policy running in this task |

## Quick start

```powershell
# headless training (Windows)
isaaclab.bat -p scripts/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --viz none --num_envs 8192 --max_iterations 1500

# continue from a checkpoint
isaaclab.bat -p scripts/finetune.py --checkpoint models/r1_flat_walk2_6900it/model_6900.pt `
  --iterations 1500 --num_envs 8192

# measure the gait (the acceptance metric)
isaaclab.bat -p scripts/gait_report.py --checkpoint models/r1_flat_walk2_6900it/model_6900.pt --vx 0.5

# watch it
isaaclab.bat -p scripts/play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --checkpoint models/r1_flat_walk2_6900it/model_6900.pt --num_envs 8 --real-time --viz kit
```

`isaaclab.bat` here means the Isaac Lab install that holds this task; with a pip-wheel Isaac Lab on a
binary Isaac Sim, use `C:\isaac-sim\python.bat` and set `PYTHONUTF8=1` (see
[docs/LOCAL_GUI.md](docs/LOCAL_GUI.md)).
