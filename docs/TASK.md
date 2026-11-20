# The task

`Isaac-Velocity-Flat-R1` is Isaac Lab 3.0's velocity-tracking locomotion MDP, with the R1 asset
and the R1's numbers. It is modelled on the stock G1 task
(`isaaclab_tasks/core/velocity/config/g1`) so the reward set is the one NVIDIA trains with.

## Backend

Newton / MuJoCo-Warp (`self.sim.physics.default = newton_mjwarp`), the same backend the stock
velocity tasks use. The USD variant selected at spawn is `mujoco` — the one carrying the Mjc joint
schemas Newton reads.

## Action space (14)

The 12 leg joints and the 2 waist joints:

```
left/right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
waist_roll, waist_yaw
```

Scale is per joint, `0.25 * torque_limit / kp` (`POLICY_CHECKLIST.md` rule 1):

| Joints | Scale (rad per unit) |
|---|---|
| hips, knees, waist | 0.15 |
| ankles | 0.3125 |

Arms and head are **not** in the action space. They hold their default pose through their own PD,
the way they behave on the robot while its legs are driven.

## Observation space (78)

| Term | Dim |
|---|---|
| base linear velocity | 3 |
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command | 3 |
| joint positions (26) | 26 |
| joint velocities (26) | 26 |
| last action (14) | 14 |

No height scan (flat task).

## Commands

Forward 0–1.0 m/s, lateral ±0.5 m/s, yaw ±1.0 rad/s.

## Rewards

Inherited from Isaac Lab's `RewardsCfg` and adjusted as the G1 task does:

| Term | Weight |
|---|---|
| `track_lin_vel_xy_exp` (yaw frame) | +1.0 |
| `track_ang_vel_z_exp` (world) | +1.0 |
| `feet_air_time` (biped positive, ankle links) | +0.75 |
| `lin_vel_z_l2` | −0.2 |
| `ang_vel_xy_l2` | −0.05 |
| `flat_orientation_l2` | −1.0 |
| `dof_torques_l2` (hips, knees) | −2.0e-6 |
| `dof_acc_l2` (hips, knees) | −1.0e-7 |
| `action_rate_l2` | −0.005 |
| `feet_slide` | −0.1 |
| `dof_pos_limits` (ankles) | −1.0 |
| `joint_deviation_hip`, `_arms`, `_torso` | −0.1 each |
| `termination_penalty` | −200 |

## Terminations and events

- Termination: `base_contact` on `pelvis_link` (in addition to time-out).
- Events: random base mass on `pelvis_link`, no COM shift, external pushes, joint-position resets
  (range 1.0 = default pose), physics material randomisation, push robot every 10–15 s.

## Physics sizes

| Task | njmax | nconmax |
|---|---|---|
| Flat | 120 | 12 |
| Rough | 300 | default |

## Files

- `r1_isaaclab/asset.py` — asset, actuators, USD path, variant.
- `r1_isaaclab/env_cfg.py` — rewards, actions, observations, events, terminations.
- `r1_isaaclab/agents/rsl_rl_ppo_cfg.py` — PPO (24 steps/env, 5 epochs, 4 mini-batches, adaptive KL).
