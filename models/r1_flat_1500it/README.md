# Trained policy: 1500 iterations on the flat task

The first locomotion policy trained for the R1 in Isaac Lab. 2048 environments, 1500 PPO
iterations, **5376 s (1 h 29.6 min)** on the RTX A5000 Laptop, Newton/MuJoCo-Warp backend.

| | |
|---|---|
| Task | `Isaac-Velocity-Flat-R1` |
| Asset | `assets/r1/R1.usda` (URDF rebuild, 26 joints) |
| Command | `scripts/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none --num_envs 2048 --max_iterations 1500` |
| Run directory | `logs/rsl_rl/r1_flat/2026-11-20_11-34-15` |

## Metrics

| Iteration | Mean reward | Mean episode length | vel error xy (m/s) | vel error yaw (rad/s) |
|---|---|---|---|---|
| 0 | −0.07 | 13 | 0.642 | 2.437 |
| 1 | −0.20 | 28 | 0.810 | 2.630 |
| 10 | −4.14 | 240 | 0.596 | 2.736 |
| 50 | −15.13 | 968 | 0.621 | 2.378 |
| 100 | −13.13 | 995 | 0.643 | 1.858 |
| 250 | −8.69 | 955 | 0.657 | 1.535 |
| 500 | −4.32 | 957 | 0.666 | 1.236 |
| 750 | +4.38 | 998 | 0.412 | 1.196 |
| 1000 | +14.47 | 1000 | 0.203 | 0.797 |
| 1250 | +19.91 | 1000 | 0.168 | 0.579 |
| 1499 | +20.73 | 994 | 0.191 | 0.529 |

Last 50 iterations: **reward 20.56, episode length 999.1, vel error xy 0.193 m/s, yaw error 0.528 rad/s**.
Best single iteration: reward 22.07.

Reading it: episode length saturates at the 20 s cap almost immediately (the robot stops falling),
but tracking only takes off after ~700 iterations, once the reward turns positive — from there the
velocity error falls roughly threefold and the yaw error fourfold.

## Files

| File | What |
|---|---|
| `model_1499.pt` | final policy (actor + critic) |
| `model_1450.pt` | previous checkpoint, for comparison |
| `env.yaml`, `agent.yaml` | the fully resolved configs of the run |
| `tb/events.out.tfevents.*` | TensorBoard log (2.96 MB) |

## Using it

```powershell
isaaclab.bat -p <repo>\scripts\play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --checkpoint <repo>\models\r1_flat_1500it\model_1499.pt --num_envs 16 --real-time --viz kit
```

## Limits

- Flat ground, no height scan: not trained for steps, slopes or debris.
- Command range 0–1 m/s forward, ±0.5 m/s lateral, ±1 rad/s yaw.
- Arms and head are held by their own PD; they are not part of the policy's action.
- Not deployed to the robot yet: the `deploy.yaml` export and the team's POLICY_CHECKLIST gate
  still apply.
