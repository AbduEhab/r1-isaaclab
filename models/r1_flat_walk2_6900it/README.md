# r1_flat_walk2_6900it — speed-swept walker

Continues `r1_flat_gait_6000it` for 949 iterations (stopped automatically at a plateau, iteration
6948, best 40-iteration mean reward 67.42 at iteration 6697). New in this run: `double_support_bonus`
(+0.5), `swing_time_penalty` (-1.0, swing outside 0.15-0.4 s penalised), and command range
`lin_vel_x` 0.2-1.2 m/s.

## Gait at three commanded speeds (`scripts/gait_report.py`, 400 steps)

| command | both feet airborne | any foot airborne (double support = 100 - this) | step length | swing height | speed reached |
|---|---|---|---|---|---|
| 0.3 m/s | **1.5 %** | 90.7 % (9.3 %) | 0.109 m | 0.139 m | 0.303 m/s |
| 0.5 m/s | **4.4 %** | 97.8 % (2.2 %) | 0.177 m | 0.141 m | 0.489 m/s |
| 0.8 m/s | 15.9 % | 99.7 % (0.3 %) | 0.279 m | 0.147 m | 0.775 m/s |

Uniform walking at 0.3-0.5 m/s (flight under 5 %), but at 0.8 m/s the policy transitions into a
running gait with a 16 % flight phase. Double support is still low at 0.5 m/s compared with a human
walk (~20 %).

## Training

| | |
|---|---|
| started from | `models/r1_flat_gait_6000it/model_6000.pt` |
| iterations | 6000 -> 6948 (plateau stop), reward 63.4 -> 67.9 |
| envs / batch | 8192 x 48 |
| final | contact_schedule 1.37/1.5, double_support 0.018, swing_time -0.034 |

## Use it

```powershell
set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& C:\isaac-sim\python.bat ^
  C:\per\thesis\r1-isaaclab\scripts\play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 ^
  --checkpoint C:\per\thesis\r1-isaaclab\models\r1_flat_walk2_6900it\model_6900.pt ^
  --num_envs 8 --real-time --viz kit
```
