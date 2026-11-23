# r1_flat_gait_6000it — the first non-hopping walker

Continues `r1_flat_selfcoll_1000it` (which was `r1_flat_4500it` + self-collision) for 966 more
iterations with the gait-shaping reward set. This is the first R1 policy here whose gait measurement
looks like walking rather than hopping.

| | |
|---|---|
| Checkpoint | `model_6000.pt` |
| Started from | `models/r1_flat_selfcoll_1000it/model_final.pt` |
| Iterations | 5499 → 6000 (stopped when the reward plateaued at ~65) |
| Envs / batch | 8192 × 48 = 393k |
| Wall time | ~1 h 55 m |

## Reward set used

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | 1.0 | follow the commanded velocity |
| `track_ang_vel_z_exp` | 2.0 | follow the yaw-rate command |
| `contact_schedule` | **1.5** | gait-clock contact pattern (left in the first half of the 0.6 s cycle, right in the second) |
| `flight_phase_penalty` | **−2.0** | no phase with both feet airborne |
| `swing_clearance` | **1.0** | `min(foot height, 8 cm)` for the swing foot |
| `feet_air_time` | 0.2 | reduced from 0.75, it used to pay for air time |
| `lin_vel_z_l2` | −1.0 | vertical bouncing is expensive |
| `undesired_contacts` | −1.0 | self-collision between the legs |
| `lateral_vel_when_turning` | −1.0 | no strafing while turning |
| `feet_slide`, `action_rate`, `dof_*` | as before | smoothness |

## Training curves (window of the fine-tune)

| Iteration | Reward | contact_schedule | flight_phase | swing_clearance |
|---|---|---|---|---|
| 5500 | 5.95 | 0.004 | −0.011 | 0.0098 |
| 5600 | 65.2 | 1.194 | −0.029 | 0.079 |
| 5800 | 65.2 | 1.302 | −0.018 | 0.078 |
| 6000 | 64.2 | 1.273 | −0.014 | 0.078 |

`contact_schedule` is near its maximum of 1.5 (≈94 % of steps phase-correct), the flight penalty is
essentially zero (0.65 % of steps have both feet up). The plateau is convergence.

## Gait measurement (`scripts/gait_report.py`, command 0.5 m/s, 400 steps)

| Metric | value | target |
|---|---|---|
| Both feet airborne | **3.9 %** | < 5 % |
| Any foot airborne | 98.3 % | — |
| Step length | **0.157 m** | > 0.15 m |
| Swing height | 0.130 m | — |
| Forward speed | +0.467 m/s | 0.5 m/s |
| Ankle correlation | alternating | < 0 |

## Evolution of the gait

| Policy | Both airborne | Step length | Notes |
|---|---|---|---|
| `r1_flat_4500it` | 19.3 % | 0.234 m | hopping |
| `r1_flat_selfcoll_1000it` | 17.7 % | 0.221 m | legs no longer intersect, still hops |
| **`r1_flat_gait_6000it`** | **3.9 %** | 0.157 m | steps alternate with single support |

## Known limits

- Single support is almost continuous (98.3 % of the time at least one foot is up, only 1.7 % double
  support), which is fast and a little stiff; a human walk has ~20 % double support.
- Only checked at 0.5 m/s; other speeds not measured yet.
- Still observes base linear velocity (89 dims), so deployment needs a state estimator.
- Turning behaviour was not re-measured after the gait-shaping fine-tune.

## Use it

```powershell
set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& C:\isaac-sim\python.bat ^
  C:\per\thesis\r1-isaaclab\scripts\play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 ^
  --checkpoint C:\per\thesis\r1-isaaclab\models\r1_flat_gait_6000it\model_6000.pt ^
  --num_envs 8 --real-time --viz kit
```
