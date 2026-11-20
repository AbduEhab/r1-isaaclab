# Training

All commands run from the Isaac Lab repository (the one with `_isaac_sim` linked), using its
bundled Python. `scripts/train.py` in this repository registers the R1 tasks first, then hands the
arguments to Isaac Lab's own CLI.

## Command

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'     # only needed when stdout is redirected
cd C:\path\to\IsaacLab
isaaclab.bat -p C:\path\to\r1-isaaclab\scripts\train.py `
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --viz none --num_envs 4096 --max_iterations 1500
```

- `--viz none` is Isaac Lab 3.0's headless flag (`--headless` no longer exists).
- `--num_envs` — robots simulated in parallel; the A5000 handles 2048 comfortably, 4096 if nothing
  else uses the GPU.
- `--max_iterations` — PPO iterations; the stock G1 flat config uses 1500.

## What a run looks like

Startup prints a summary (task, backend, action/observation shapes), then per-iteration blocks:

```
Learning iteration 33/100
Mean reward: -14.03
Mean episode length: 796.26
Metrics/base_velocity/error_vel_xy: 0.6796
Metrics/base_velocity/error_vel_yaw: 2.5957
Iteration time: 5.18s
```

Signals to watch:

| Metric | Meaning |
|---|---|
| `Mean episode length` | 1000 = full 20 s episode; below is falling or resetting |
| `Episode_Termination/base_contact` | falls (pelvis contact) — should trend down |
| `Metrics/base_velocity/error_vel_xy` | tracking error for the commanded velocity |
| `Metrics/base_velocity/error_vel_yaw` | yaw-rate tracking error |
| `Mean reward` | sum of the reward terms; goes up slowly |

## Where results go

```
IsaacLab/logs/rsl_rl/r1_flat/<timestamp>/
├── model_0.pt, model_50.pt, ...      checkpoints (every save_interval = 50 iterations)
├── events.out.tfevents.*             TensorBoard
├── params/env.yaml, params/agent.yaml  the fully resolved configs of the run
└── git/IsaacLab.diff
```

TensorBoard:

```powershell
isaaclab.bat -p -m tensorboard --logdir C:\path\to\IsaacLab\logs\rsl_rl\r1_flat
# then open http://localhost:6006
```

## Throughput measured here

| Run | Envs | Iteration time | 100 iterations |
|---|---|---|---|
| R1 flat | 2048 | ~5.2 s | ~8.6 min |

A full 1500-iteration flat run is therefore roughly **2 hours** on the A5000.

## Watching a trained policy

```powershell
isaaclab.bat -p C:\path\to\r1-isaaclab\scripts\play.py `
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --checkpoint C:\path\to\r1-isaaclab\models\r1_flat_100it\model_99.pt `
  --num_envs 16 --real-time --viz kit
```

- `--viz kit` opens the Isaac Sim viewport (omit it or use `--viz none` for headless).
- `--checkpoint` takes a path, or `latest` / `best` / `pretrained`.
- `--num_envs 16` keeps the view readable; the policy was trained with thousands.
- `--real-time` paces the simulation to wall-clock time.

## Changing the task

- Rewards, commands, terminations: `r1_isaaclab/env_cfg.py`.
- Gains, torque limits, action-scale inputs: `r1_isaaclab/asset.py`.
- PPO hyperparameters: `r1_isaaclab/agents/rsl_rl_ppo_cfg.py`.

Because the task is registered from this package at runtime, edits take effect on the next run
without reinstalling anything.

## Reproducing the R1 asset

See [ASSET.md](ASSET.md). The built USD is committed, so training needs no rebuild.
