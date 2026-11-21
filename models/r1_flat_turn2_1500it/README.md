# r1_flat_turn2_1500it — turning-tuned policy

Trained 2026-11-21, 1500 PPO iterations, 8192 envs, batch 393k, 1 h 58 min. First run after the
heading-control bug was fixed.

## Training

| Metric | Value |
|---|---|
| Reward | +38.35 |
| Velocity error xy | 0.195 m/s |
| **Yaw rate error** | **0.255 rad/s** (commands up to ±1.5) |
| Episode length | 1000 (never falls) |
| `track_ang_vel_z_exp` | 1.507 / 2.0 |
| `lateral_vel_when_turning` | −0.104 (penalty paid, so it is being exercised) |
| `feet_air_time` | 0.0027 |

## Behaviour (`scripts/probe_gait.py`, fixed commands, 8 envs)

Turn in place, command `vx=0, wz=1.0`:

| Quantity | Value |
|---|---|
| Yaw rate | **+1.048 rad/s** (command 1.0 → 5% error) |
| Heading change | **+10.59 rad in 10 s** |
| Mean lateral speed | 0.028 m/s — **not strafing** |
| Forward drift | 0.018 m/s |
| Base height | 0.728 m, std 0.008 |
| Ankle correlation | **+0.64** (pivot, not alternating steps) |
| Step frequency | 5.7 Hz |

Walk and turn, command `vx=0.5, wz=1.0`:

| Quantity | Value |
|---|---|
| Forward speed | +0.429 m/s |
| Yaw rate | +1.021 rad/s |
| Mean lateral speed | 0.035 m/s |
| Ankle correlation | +0.87 (pivot while translating) |

## Comparison with the 4500-iteration policy

Same probe, `vx=0, wz=1.0`:

| | 4500it (old) | turn2 (this) |
|---|---|---|
| Yaw rate | 0.933 rad/s | **1.048 rad/s** |
| Heading change / 10 s | 9.44 rad | **10.59 rad** |
| Lateral speed | 0.027 | 0.028 |
| Ankle correlation | **−0.47 (steps)** | +0.64 (pivots) |
| Step frequency | 1.2 Hz | 5.7 Hz |
| `feet_air_time` reward | 0.142 | 0.0027 |

The new policy follows the turn command better but turns by pivoting on planted feet; the old one
steps properly and turns nearly as well. Restoring stepping is the next reward problem: the
anti-strafe penalty and the stronger foot-slide penalty are cheap to satisfy by twisting in place,
and nothing rewards lifting the feet during a turn.
