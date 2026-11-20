# Reference run: 100 iterations on the flat task

Not a deployable policy â€” a reference run that proves the pipeline and gives a checkpoint to resume
from. 100 PPO iterations, 2048 environments, 461 s of wall time on the RTX A5000 Laptop.

| | |
|---|---|
| Task | `Isaac-Velocity-Flat-R1` |
| Asset | `assets/r1/R1.usda` (URDF rebuild, 26 joints) |
| Backend | Newton / MuJoCo-Warp |
| Command | `train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none --num_envs 2048 --max_iterations 100` |
| Run directory | `logs/rsl_rl/r1_flat/2026-11-20_10-42-15` |

## Metrics

| Iteration | Mean reward | Mean episode length | vel error xy | vel error yaw |
|---|---|---|---|---|
| 0 | âˆ’0.07 | 13 | 0.642 | 2.437 |
| 1 | âˆ’0.25 | 28 | 0.808 | 2.625 |
| 10 | âˆ’4.13 | 240 | 0.595 | 2.766 |
| 25 | âˆ’11.29 | 599 | 0.684 | 2.624 |
| 50 | âˆ’15.19 | 979 | 0.633 | 2.423 |
| 75 | âˆ’14.32 | 987 | 0.685 | 2.126 |
| 99 | âˆ’12.14 | 982 | 0.658 | 1.923 |

The meaningful signal is **episode length**: 13 steps at iteration 0 â†’ **982 of 1000** by the end,
i.e. the R1 stops terminating on pelvis contact and survives the full 20 s episode. Yaw tracking
error also improves (2.44 â†’ 1.92). Reward and velocity error are not yet converged â€” 100 iterations
is 7 % of the 1500 the stock G1 config uses.

## Files

| File | What |
|---|---|
| `model_99.pt` | final checkpoint (policy + critic) |
| `model_50.pt` | midpoint checkpoint |
| `env.yaml`, `agent.yaml` | the fully resolved environment and agent configs of the run |
| `tb/events.out.tfevents.*` | TensorBoard log |

## Resuming

```powershell
isaaclab.bat -p <repo>\scripts\train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --viz none --num_envs 2048 --max_iterations 1500 --resume --load_run <run> --checkpoint model_99.pt
```

Flags match Isaac Lab's RSL-RL CLI; the checkpoint format is RSL-RL's own.

> **Superseded.** This checkpoint was trained with the pre-verification contract: straight-legged default pose, 14 actions, no gait-phase observation. It keeps its feet but moves by hopping. The corrected task (Unitree's HOME_KEYFRAME pose, 24 actions, 83-dim observation with gait phase) is what env_cfg.py now defines; this run is kept as history.

