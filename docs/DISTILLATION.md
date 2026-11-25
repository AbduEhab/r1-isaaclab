# Distilling the Unitree reference policy

The public R1 policy (`reference/unitree_r1_policy/`) is a proven walker: 0.4 % both-feet-airborne,
0.199 m steps, 3.0 Hz, 0.44 m/s on a 0.5 command. Our own policies now walk at low speed but still
differ from it. Distillation is the route that copies its gait into a policy trained under our
physics and our deployment contract.

## What we already have

- `policy.onnx` plus `weights.npz` and `metadata.json` — 83 observations (no base linear velocity),
  24 actions, per-joint action scale, HOME default pose, observation normaliser.
- `scripts/run_unitree_policy.py` — runs the teacher in our Isaac Lab task and measures its gait.
- `scripts/finetune.py` — continues training any RSL-RL checkpoint with the current environment.
- `scripts/gait_report.py` — the objective gate (flight phase, step length, swing height, speed).

## The contract mismatch to fix first

The teacher observes **83** dimensions; our task currently observes **89** (base linear velocity
added). The student must see what the teacher sees, and the robot must be able to provide it, so the
student policy is trained with:

```
base_ang_vel(3) + projected_gravity(3) + command(3) + gait_phase(2)
+ joint_pos(24) + joint_vel(24) + last_action(24) = 83
```

plus, optionally, **observation history** (e.g. the last 5 frames, 415 dims) so the network can infer
base velocity from the motion history — which is how the teacher's own deployment works.

## The pipeline

### Step 1 — collect teacher rollouts in our physics

Run the teacher in our task with domain randomisation on (mass, friction, pushes, terrain friction,
action latency), random commands over 0.2–0.8 m/s and ±1.0 rad/s, and record per step:

```
(observation_83, teacher_action_24, reward, done, next_observation_83)
```

One hour of simulation at 8192 envs produces millions of transitions; a few hundred thousand
randomised transitions are enough for a first clone. Store as a few `.npz` shards (~100 MB).

### Step 2 — behaviour cloning (supervised)

Train an MLP with the same shape as the deployment policy (256-128-128 is enough; the teacher uses
512-256-128) to minimise

```
L = MSE(student_obs_norm(obs), teacher_action)
```

with the observation normaliser either copied from the teacher's `weights.npz` (mean/div) or learned.
Feed the *normalised* observation into the reward-based fine-tune later, so keep the same normaliser
in both stages.

Cost: minutes on one GPU, no simulator involved. Output: a checkpoint in RSL-RL format
(`model_*.pt`) so the rest of the tooling works unchanged.

### Step 3 — DAgger (optional but recommended)

The cloned policy drifts into states the teacher never visited. Run the student for a few hundred
thousand steps, query the teacher on exactly those states, add them to the dataset, retrain. Two or
three rounds typically close the gap. With our visual gait metric, check after each round: flight
phase and step length should approach the teacher's (0.4 %, 0.20 m).

### Step 4 — RL fine-tune with a teacher regulariser

Continue with PPO from the cloned checkpoint, using the KEEP rewards (`contact_schedule`,
`flight_phase_penalty`, `swing_clearance`, velocity tracking), plus:

```
L_teacher = |student_action - teacher_action|²
```

as a reward penalty (`-k * L_teacher`) or as an auxiliary loss. The teacher action is cheap to
compute inside the reward term (numpy forward pass on the current observation), so this needs no
separate data collection. This is what prevents PPO from drifting back into a hop while it optimises
speed and turning.

### Step 5 — evaluate

`scripts/gait_report.py` at 0.3 / 0.5 / 0.8 m/s, and in the Isaac Sim window. Success = the teacher's
gait statistics (flight < 2 %, step 0.18–0.22 m, double support ≥ 10 %) at our deployment contract.
Then export `deploy.yaml` (24 joints, per-joint scales from `metadata.json`) for the robot.

## Effort

| Step | Time |
|---|---|
| 1 teacher rollouts (with randomisation) | 1–2 h |
| 2 behaviour cloning + evaluation | 1 h |
| 3 DAgger round | 1–2 h |
| 4 PPO fine-tune (plateau-stopped) | 2–3 h |
| 5 evaluation | 1 h |

## Alternative without distillation

If distillation is too much work, the reward route already used here is converging: the last run
reached 1.5 % flight and 9.3 % double support at 0.3 m/s. The remaining levers are the
single-support penalty, a longer training budget, and history observations for deployability.
Distillation is faster to a *good-looking* gait because it copies a proven one instead of searching
for it, but it adds a data pipeline that the reward route does not need.

## Implemented: teacher-clone into the deploy contract

The first deploy-contract policy trained from scratch plateaued at reward ~39.5 with a degenerate
gait (90 % both-feet-airborne, 0.1 m/s regardless of command). That is the moment the distillation
fallback was designed for, and it is now implemented for our own walker (not the Unitree ONNX, which
stays the external reference):

1. `scripts/collect_teacher.py` rolls `models/r1_flat_walk3_7300it` (the 86-observation walker that
   walks at 1.5 % flight) in the base task with randomised commands, builds the **deploy observation**
   (415 = 83 x 5 history) from the same robot state, and records the teacher action as the label.
   Sanity check: the manually built teacher observation matches the environment's own vector to 0.0.
   193 536 samples in ~3 minutes at 1024 environments.
2. `scripts/bc_train.py` behaviour-clones the deploy network (256-128-128) on those pairs: 12 epochs,
   validation action RMSE **0.065 rad**.
3. Closed-loop, the clone drifts (65 % both-feet-airborne at a 0.3 m/s command, 0.075 m/s reached) -
   classic compounding error, since a fraction of a radian per step is enough to fall.
4. `scripts/finetune.py --task Isaac-Velocity-Flat-R1-Deploy` continues with PPO from the clone:
   the RL rewards fix the drift while the clone's weights keep the gait near the teacher's.

The run logs to `logs/rsl_rl/r1_flat/deploy_bc_2026-11-25`; the plateau watcher stops it.

