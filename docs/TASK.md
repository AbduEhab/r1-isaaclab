# The task

`Isaac-Velocity-Flat-R1` is Isaac Lab 3.0's velocity-tracking locomotion MDP, with the R1 asset
and the R1's numbers. It is modelled on the stock G1 task
(`isaaclab_tasks/core/velocity/config/g1`) so the reward set is the one NVIDIA trains with.

## Backend

Newton / MuJoCo-Warp (`self.sim.physics.default = newton_mjwarp`), the same backend the stock
velocity tasks use. The USD variant selected at spawn is `mujoco` â€” the one carrying the Mjc joint
schemas Newton reads.

## Action space (24)

The 12 leg joints, the 2 waist joints and the 10 arm joints — Unitree's R1 contract (deploy.yaml: 24 joints, head excluded). The head is held by its own PD:

```
left/right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
waist_roll, waist_yaw
```

Scale is per joint, `0.25 * torque_limit / kp` (`POLICY_CHECKLIST.md` rule 1):

| Joints | Scale (rad per unit) |
|---|---|
| hips, knees, waist | 0.15 |
| ankles | 0.3125 |

The head is not in the action space. It holds its default pose through its own PD.

## Observation space (83)

| Term | Dim |
|---|---|
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command | 3 |
| gait phase (period 0.6 s) | 2 |
| joint positions (24) | 24 |
| joint velocities (24) | 24 |
| last action (24) | 24 |

There is **no base linear velocity** term: the real R1 has no sensor for it, so the reference contract omits it. The gait phase is a [cos, sin] clock (1_isaaclab/mdp.py) that lets the policy represent an alternating stride instead of a symmetric bounce.

No height scan (flat task).

## Commands

Forward 0â€“1.0 m/s, lateral Â±0.5 m/s, yaw Â±1.0 rad/s.

## Rewards

Inherited from Isaac Lab's `RewardsCfg` and adjusted as the G1 task does:

| Term | Weight |
|---|---|
| `track_lin_vel_xy_exp` (yaw frame) | +1.0 |
| `track_ang_vel_z_exp` (world) | +1.0 |
| `feet_air_time` (biped positive, ankle links) | +0.75 |
| `lin_vel_z_l2` | âˆ’0.2 |
| `ang_vel_xy_l2` | âˆ’0.05 |
| `flat_orientation_l2` | âˆ’1.0 |
| `dof_torques_l2` (hips, knees) | âˆ’2.0e-6 |
| `dof_acc_l2` (hips, knees) | âˆ’1.0e-7 |
| `action_rate_l2` | âˆ’0.005 |
| `feet_slide` | âˆ’0.1 |
| `dof_pos_limits` (ankles) | âˆ’1.0 |
| `joint_deviation_hip`, `_arms`, `_torso` | âˆ’0.1 each |
| `termination_penalty` | âˆ’200 |

## Terminations and events

- Termination: `base_contact` on `pelvis_link` (in addition to time-out).
- Events: random base mass on `pelvis_link`, no COM shift, external pushes, joint-position resets
  (range 1.0 = default pose), physics material randomisation, push robot every 10â€“15 s.

## Physics sizes

| Task | njmax | nconmax |
|---|---|---|
| Flat | 120 | 12 |
| Rough | 300 | default |

## Files

- `r1_isaaclab/asset.py` â€” asset, actuators, USD path, variant.
- `r1_isaaclab/env_cfg.py` â€” rewards, actions, observations, events, terminations.
- `r1_isaaclab/agents/rsl_rl_ppo_cfg.py` â€” PPO (24 steps/env, 5 epochs, 4 mini-batches, adaptive KL).

