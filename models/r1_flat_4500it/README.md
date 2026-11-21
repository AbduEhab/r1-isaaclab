# r1_flat_4500it — walking policy

Trained on 2026-11-21, 4500 PPO iterations, 8192 environments, batch 393k
(`num_steps_per_env = 48`), 5.5 h on an RTX A5000.

| Metric (final, 50-iteration mean) | Value |
|---|---|
| Mean reward | +33.25 |
| Velocity error xy | **0.115 m/s** |
| Yaw rate error | 0.219 rad/s |
| Episode length | 1000 (never falls) |
| `track_lin_vel_xy_exp` | 0.931 / 1.0 |
| `track_ang_vel_z_exp` | 0.798 / 1.0 |
| `feet_air_time` reward | 0.142 (v2, which stood still, was 0.0009) |
| `flat_orientation_l2` penalty | −0.0055 (upright) |

This is the first R1 policy here that tracks velocity while stepping. Compare with the earlier
runs in the repository (see `docs/RESULTS.md` for the full table).

## Contract

| | |
|---|---|
| Task | `Isaac-Velocity-Flat-R1` |
| Actions | 24 (legs + waist + arms, head excluded), per-joint scale |
| Observations | 89 = base angular velocity (3) + projected gravity (3) + base linear velocity (3) + command (3) + gait phase (2) + joint positions (24) + joint velocities (24) + last actions (24) |
| Control | 0.02 s, home pose crouched, armature 0.01 |
| Agent | RSL-RL PPO, MLP 256/128/128, `num_steps_per_env = 48` |

Note the base linear velocity observation: the policy uses it (it is what made learning work after
v2 stood still), so a deployment needs a state estimator or the term removed and the policy retrained.

## Use it

```powershell
# Windows install
isaaclab.bat -p <repo>\scripts\play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 `
  --checkpoint <repo>\models\r1_flat_4500it\model_4499.pt --num_envs 16 --real-time --viz kit

# container
docker run --rm --gpus all -v <repo>:/workspace isaac-env:6.2 /bin/bash -lc `
  "cd /workspace && /opt/IsaacLab/isaaclab.sh -p /workspace/scripts/play.py --rl_library rsl_rl `
   --task Isaac-Velocity-Flat-R1 --checkpoint /workspace/models/r1_flat_4500it/model_4499.pt --viz none"
```

`params/env.yaml` is the fully resolved environment config of the training run; it is the record of
exactly what produced this checkpoint.
