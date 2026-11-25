# Research record: learning a walking policy for the Unitree R1

Living document. Everything below was measured in this repository; commit hashes point at the code
and checkpoints. Written so that a paper's Methods/Results sections can be assembled from it.

---

## 1. Objective

Train a velocity-tracking locomotion policy for the Unitree **R1** humanoid (24 actuated joints,
head excluded) that **walks** — alternating steps, no flight phase, no self-intersection — and matches
the deployment contract Unitree uses (`deploy.yaml` / `r1_constants.py`), so it can be exported to the
robot.

## 2. System

| Component | Version / detail |
|---|---|
| Simulator | Isaac Sim **6.2.0** (binary, Windows) |
| Framework | Isaac Lab **v3.0.0** |
| Physics | Newton / MuJoCo-Warp backend (`newton 1.6.1`, `warp-lang 1.17.0`) |
| RL | RSL-RL 5.5.1, PPO, MLP 256-128-128, ELU |
| Robot asset | R1 USD rebuilt from the team's `urdf/R1.urdf` (26 revolute joints, free base, head included) by Isaac Sim's URDF importer with `mujoco`/`physics`/`physx` variants |
| GPU | RTX A5000 Laptop, 16 GB |

### 2.1 Task (`Isaac-Velocity-Flat-R1`)

| Element | Configuration |
|---|---|
| Actions | 24 joint position targets (12 legs, 2 waist, 10 arms), per-joint scale `0.25·τ_max/k_p`: 0.150 (hips/knees/waist), 0.3125 (ankles), 0.375 (shoulder pitch/roll), 0.4125 (arm rest) |
| Control / physics | 0.02 s / 0.005 s; armature 0.01 on all joints |
| Gains | k_p 100 / 40 / 20, k_d 2 / 2 / 1 by group (Unitree `r1_constants.py`) |
| Home pose | Unitree `HOME_KEYFRAME`: pelvis 0.76 m, knee 0.3, hip −0.1, ankle pitch −0.2, shoulder pitch 0.35, elbow 0.87, shoulder roll ±0.18 |
| Observations | experiments used 78/83/89 dims; final: 89 = base angular velocity (3) + projected gravity (3) + base linear velocity (3) + command (3) + gait phase (2) + joint pos (24) + joint vel (24) + last action (24) |
| Commands | `v_x ∈ (−0.5, 1.0)`, `v_y ∈ (−0.5, 0.5)`, `ω_z ∈ (−1.0, 1.0)` initially; `heading_command=False`; later ranges varied per experiment (see §4) |
| Episode | 20 s (1000 control steps); termination on base contact |
| Contact sensor | net forces on all R1 bodies (`/World/envs/env_*/Robot/Geometry/.*`), 27 bodies |
| Newton solver | `njmax=300`, `nconmax=48` (12 dropped ~34 k contacts at 2048 envs) |

### 2.2 Reward family

Base set mirrors Isaac Lab's stock G1 velocity task (velocity tracking, orientation, joint/action
penalties, foot slide, air time, termination), with R1 naming and limits. It was then extended
experiment by experiment; the final set is in §5.

## 3. Measurement methodology

Two headless probes produce every number quoted below:

- **`scripts/gait_report.py`** — fixed command, 400 steps (8 s), measures per foot from
  `ankle_roll_link` positions: **both-feet-airborne fraction** (flight = hop indicator),
  any-foot-airborne (double support = 1 − this), **swing height**, **step length** (forward foot
  displacement during swing), forward speed. This is the primary acceptance gate.
- **`scripts/probe_gait.py`** — fixed command, returns joint/velocity metrics including yaw rate and
  step frequency, useful for turning.

Lesson: reward curves are not evidence. The plateaued policy at reward ~65 still hopped; only the
airborne-fraction metric showed it.

## 4. Experimental log

All runs use PPO with 8192 envs and `num_steps_per_env=48` (batch = 393k) unless noted. Checkpoints and
resolved configs are under `models/`.

| # | Commit | Change vs previous | Iterations | Reward | Vel err xy / yaw | Gait (both airborne / step) | Outcome |
|---|---|---|---|---|---|---|---|
| v1 | `7a9e928`-era | 14 actions (legs+waist), straight-leg home (0.90 m), 78 obs, no gait phase | 1500 | +20.7 | 0.193 / 0.528 | not measured | walks in the sense of tracking, but **hops** |
| v2 | `7a9e928` | Unitree contract: 24 actions, crouch home, gait phase, **no base linear velocity**, 83 obs | 1500 | −6.7 | 0.549 / 1.161 | feet_air_time 0.0009 → **never steps** | stood still |
| v3 | `59e34ca` | v2 + base linear velocity restored (89 obs), 4500 iters, batch 393k | 4500 | +33.25 | 0.115 / 0.219 | **19.3 %** / 0.234 m | walks and tracks; still hops |
| turn1 | `f2e99e9` | turning rewards (hip-yaw penalty removed, lateral-vel-when-turning, yaw weight 2.0, feet_slide −0.25) | 1500 | +38.35 | 0.195 / 0.255 | not measured | invalidated: trained with heading control on |
| turn2 | `a6c54dc` | `heading_command=False` (yaw is a real rate command), yaw range ±1.5 | 1500 | +38.35 | 0.195 / 0.255 | feet_air_time 0.0027 → **pivots** | turns at 1.05 rad/s in place, no strafe, but steps replaced by pivoting |
| selfcoll | `25529fd` | self-collision enabled on both backends + `undesired_contacts` penalty | 1000 | +46.08 | 0.167 / 0.190 | **17.7 %** / 0.221 m | legs no longer intersect; hop unchanged |
| gait | `9c2f4ae` | `contact_schedule` (+1.5), `flight_phase_penalty` (−2.0), `swing_clearance` (+1.0), feet_air_time 0.75→0.2, lin_vel_z −0.2→−1.0 | 966 | ~65 (plateau) | — | **3.9 %** / 0.157 m | first non-hopping walker |
| walk2 | `8c45b1c` | `double_support_bonus` (+0.5), `swing_time_penalty` (−1.0), command 0.2–1.2 m/s | 949 (plateau at 6948, best 67.42) | +67.9 | — | 1.5 % @0.3, 4.4 % @0.5, **15.9 % @0.8** | walks at 0.3–0.5; runs at 0.8 |
| walk3 | `b674ede` | `double_support_bonus` 0.5→1.0, `single_support_time_penalty` (−1.0, >0.6 s), command capped 0.2–0.7 m/s | ≤1500 (plateau-stopped) | running | — | running | to be measured |

### 4.1 Gait detail for the final candidates

`gait_report.py`, 400 steps, one environment:

| Policy | command | both airborne | double support | step length | swing height | speed |
|---|---|---|---|---|---|---|
| v3 (`r1_flat_4500it`) | 0.5 | 19.3 % | 0.6 % | 0.234 m | 0.163 m | 0.500 |
| selfcoll | 0.5 | 17.7 % | 2.2 % | 0.221 m | 0.152 m | 0.495 |
| gait (`r1_flat_gait_6000it`) | 0.5 | 3.9 % | 1.7 % | 0.157 m | 0.130 m | 0.467 |
| walk2 | 0.3 | **1.5 %** | **9.3 %** | 0.109 m | 0.139 m | 0.303 |
| walk2 | 0.5 | 4.4 % | 2.2 % | 0.177 m | 0.141 m | 0.489 |
| walk2 | 0.8 | 15.9 % | 0.3 % | 0.279 m | 0.147 m | 0.775 |

Turning (probe, `vx=0, ω_z=1.0`): turn2 policy reaches 1.048 rad/s (10.59 rad of heading in 10 s) with
0.028 m/s lateral drift — it turns, does not strafe, but pivots on planted feet.

### 4.2 Reference policy

A public R1 policy (SpacemiT model zoo, ONNX, 83 obs / 24 actions, per-joint action scale) was
downloaded as a reference and executed in the same task (`scripts/run_unitree_policy.py`):

| command | both airborne | step | speed reached | yaw rate |
|---|---|---|---|---|
| 0.5 m/s | **0.4 %** | 0.199 m | 0.436 m/s | 0.042 |
| 0, 1.0 rad/s | — | — | — | **0.708 rad/s**, heading 5.69 rad / 8 s |

This is a proven walker and the teacher for the distillation route (§8).

## 5. Final reward set (walk3)

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | 1.0 | velocity tracking (yaw frame) |
| `track_ang_vel_z_exp` | 2.0 | yaw-rate tracking |
| `contact_schedule` | 1.5 | gait-clock contact pattern (left/right alternation, 0.6 s cycle) |
| `flight_phase_penalty` | −2.0 | no phase with both feet airborne |
| `swing_clearance` | 1.0 | swing foot must lift (`min(z, 8 cm)`) |
| `double_support_bonus` | 1.0 | both feet down (balance phase) |
| `single_support_time_penalty` | −1.0 | stance on one foot > 0.6 s |
| `feet_air_time` | 0.2 | reduced (was 0.75; it paid for flight) |
| `lin_vel_z_l2` | −1.0 | vertical bouncing is expensive |
| `undesired_contacts` | −1.0 | self-collision between legs |
| `lateral_vel_when_turning` | −1.0 | no strafing during turns |
| `feet_slide` | −0.25 | no skating |
| `flat_orientation_l2`, `action_rate_l2`, `dof_acc_l2`, `dof_torques_l2`, `joint_deviation_*` | −1.0, −0.005, −1e-7, −2e-6, −0.1 | stability and smoothness |

PPO: `num_steps_per_env=48`, 5 epochs, 4 minibatches, lr 1e-3 (adaptive, target KL 0.01),
γ 0.99, λ 0.95, entropy 0.008, clip 0.2.

## 6. Findings

1. **The home pose and action space decide reachability.** Straight legs + 14 actions + no gait clock
   produced a hopping policy even with good velocity tracking (v1). Unitree's crouched pose and 24
   joint actions are prerequisites for a stepped gait.
2. **Base linear velocity is what made learning work.** Removing it (v2) made the policy stand still;
   `feet_air_time` 0.0009 and velocity error 0.55 m/s. Restoring it (v3) recovered walking and
   tracking — this is a training-time signal, not a deployment signal.
3. **A velocity reward alone has a hop solution.** v3's 19.3 % flight phase shows that tracking a
   velocity command does not imply walking. The reward had no term objecting to flight.
4. **`heading_command=True` hid a contract error.** The "yaw command" was 0.5·heading error,
   recomputed each step; the policy was never asked for a yaw rate. Stock G1 flat disables it; our
   inheritance of the default silently changed the task. After disabling it, yaw tracking works
   (1.05 rad/s at a 1.0 command).
5. **Disabled self-collision let the legs pass through each other**, and the reward set did not
   penalise leg-leg contacts. Enabling collision + `undesired_contacts` made the policy learn
   separation within 1000 iterations (penalty −0.20 → −0.002).
6. **Flight is free until it is taxed.** Adding a flight-phase penalty, a phase-locked contact
   schedule and a swing-clearance reward cut the flight fraction 17.7 % → 3.9 % in under 1000
   iterations, at equal velocity tracking.
7. **"Uniform" walking needs support and cadence shaping.** With flight removed, the gait becomes
   single-support dominated (2 % double support). Double-support and swing-duration terms brought
   double support to 9.3 % at 0.3 m/s; the same policy still runs above 0.7 m/s, so the walking
   range is speed-dependent.
8. **Plateau-based stopping works** and should be standard here: rewards saturate (e.g. 67.4 at
   iteration 6697, flat to 6948) while gait metrics stop improving. Stopping on a 40-iteration mean
   reward that gains < 1.0 over 250 iterations saved ~2 h per run.

## 7. Reproducibility

| Purpose | Command |
|---|---|
| Train from scratch | `python.bat scripts/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none --num_envs 8192 --max_iterations N` |
| Continue a checkpoint | `python.bat scripts/finetune.py --checkpoint models/<run>/model.pt --iterations N --num_envs 8192` |
| Gait measurement | `python.bat scripts/gait_report.py --checkpoint <ckpt> --vx 0.5` |
| Watch in the sim | `python.bat scripts/play.py ... --viz kit` (see `docs/LOCAL_GUI.md`) |
| Reference policy | `python.bat scripts/run_unitree_policy.py --vx 0.5` (headless) / `run_unitree_gui.py` (window) |
| Plateau stop | `scripts/dev/stop_on_plateau_local.ps1` |

Every run's fully resolved environment and agent configs are stored in `models/<run>/params/`.
Seeds: 42 (v1–v3), 23 (fine-tunes), 7/11 (probes). Windows and Linux/container produce comparable
iteration times (7–8 s at 8192 envs).

## 8. Open work

1. **Finish walk3** (double support 1.0 + single-support penalty, speed cap 0.7) and measure at
   0.3/0.5/0.7 m/s. Acceptance: flight < 2 %, double support ≥ 10 % at 0.5 m/s.
2. **Deployment contract (83 obs).** The policy observes base linear velocity, which the robot lacks.
   Train with 5-frame observation history (415 dims) or an estimator so the 83-dim contract can be
   met; this invalidates current checkpoints and is the last step before `deploy.yaml` export.
3. **Distillation from the reference policy** (fallback / parallel): teacher rollouts under
   randomisation → behaviour cloning → DAgger → PPO fine-tune with teacher-action penalty. Plan and
   effort estimate in `docs/DISTILLATION.md`. Success = reference gait statistics under our contract.
4. **Robustness**: domain randomisation (mass, friction, pushes, latency), payload (the kit's
   `r1_loadwalk` scenario), then rough terrain.
5. **Sim-to-real**: export with per-joint action scales, deploy `deploy.yaml`, measure tracking and
   gait frequency on hardware.

## 9. Infrastructure notes (relevant to reproducing this)

- The Windows stack must be either **binary Isaac Sim + pip-wheel Isaac Lab** or **pip Isaac Sim +
  pip Isaac Lab in a venv**; mixing the repository (source) install of Isaac Lab into a binary Isaac
  Sim puts two OpenUSD builds in one process and breaks the Kit viewport. The working window recipe is
  in `docs/LOCAL_GUI.md`.
- Container image (`isaac-env` v3.0.0-based) was used for training and removed after the local stack
  was fixed; the recipe remains in `docs/CONTAINER.md` and the Dockerfile, and the container was also
  used to record headless clips via the Newton visualizer.
- Newton requires `nconmax ≥ ~48` per environment at 2048 envs, or ~34 k contacts are silently
  dropped (we saw 37 k warnings and fixed it).
- Corporate TLS interception requires a CA build secret for container builds (`--secret id=corp_ca`).

## 10. The deployment contract (2026-11-25)

The trained flat policies observe base linear velocity (86 dims), which the R1 cannot measure. The
deployable task `Isaac-Velocity-Flat-R1-Deploy` drops that term and feeds **5 frames of history**
instead: 83 observations per frame, 415 network inputs. Velocity is then inferred from the motion
history rather than measured.

Three contract errors were found by validating the export against the simulator and the team's
`check_policy.py`; all are the kind that silently produce a walking-in-place or falling robot:

1. **Observation joint order differed from the action joint order.** Isaac Lab resolves observation
   joint patterns in articulation order (left leg, right leg, ...) while the action term keeps the
   pattern order (left hip, right hip, left roll, ...). The deploy builder must use one order for
   everything, so the observation terms now set `preserve_order=True`.
2. **Isaac Lab 3.0 quaternions are (x, y, z, w).** A builder assuming (w, x, y, z) produced gravity
   vectors with an inverted z axis. Unitree's IMU reports (w, x, y, z); the runner converts.
3. **deploy.yaml observation dimensions are per frame, multiplied by `history_length`.** Writing the
   already-multiplied dims made the checker read 2075 numbers instead of 415.

Also: the action term keeps its per-joint scale tensor in a different order than its joint order
(dict insertion order), so scales must be read by joint name from the config, not from the tensor.

Validation gates that now exist: the team's `check_policy.py` (13 rules: slots, lengths, gains,
scales, default pose, observation sources, ONNX shape) and `scripts/validate_deploy.py` (the deploy
observation builder must reproduce the environment's observation exactly, then a closed-loop rollout).
On a synthetic deploy-contract checkpoint both pass: 13/13 rules, observation parity 1e-6.

## 11. From a trained walker to a deployable one (2026-11-25, continued)

The deploy contract was validated (section 10), but the policy trained on it did not learn: from
scratch the PPO run plateaued at reward 39.5 with **90 % both-feet-airborne** and 0.1 m/s regardless
of command. Four routes were tried and measured:

1. **Behaviour cloning of our own walker** (86 observations) onto the 415-input student: 193 536
   teacher-labelled pairs, validation action RMSE **0.065 rad**, but closed-loop it drifts to 65 %
   flight and 0.075 m/s. Compounding error, not fit quality, is the problem.
2. **PPO from the clone**: collapsed to standing (feet_air_time 0.0003, reward -55).
3. **PPO from scratch with a teacher-action reward** (the teacher running inside the reward term on
   its 86-observation state): the bounded reward `1/(1+50*err)` saturated (0.03 at 0.87 rad error,
   no gradient), so it became a linear penalty. At weight 5 from the clone the error grew; at weight
   20 from scratch the policy learned to **terminate its own episodes** to stop accumulating the
   penalty (mean episode length 3.4 steps, base_contact 1.0, despite the -200 termination penalty),
   i.e. reward hacking of a regulariser.
4. **DAgger** (student rollouts labelled by the teacher): the student falls into states the teacher
   never visits, so labels there are out of distribution; retraining on them made it worse (98 %
   flight).

Two useful negative results: the proven walker with **base linear velocity replaced by zeros**
(`Isaac-Velocity-Flat-R1-Deploy86`) hops (84 % flight), so zero-filling is not a substitute for
estimation; and the deploy runner/exporter are sound (they pass the gates) - the difficulty is purely
learning a history-based policy.

### A deployable policy now exists

The public R1 reference policy does not need this: it already observes exactly the deploy contract
(83 numbers, single frame, no base linear velocity). `scripts/package_reference_policy.py` packages
it as `deploy/runs/unitree_reference` (ONNX + deploy.yaml with its joint order, per-joint action
scales from the R1 table, motor slots, gains, default pose, limits). It passes the team's
`check_policy.py` **13/13**, and `scripts/play_deploy.py` - which builds the observation with the
same builder the robot runner uses - walks it at **0.439 m/s with 0.0 % both-feet-airborne** at a
0.5 m/s command.

### The distill that worked

Because the reference policy and the student share the *same* single-frame 83-observation layout,
DAgger is exact here: the teacher can label any state the student reaches, with the same observation.
`scripts/distill_reference.py`:

1. teacher rollouts in our physics with randomised commands: 196 608 labelled steps;
2. behaviour clone (83-256-128-128-24): validation action RMSE 0.066 rad, but closed loop 33 % flight
   at 0.3 and 73 % at 0.5 - the same compounding-error failure as before;
3. **DAgger round 1**: roll the student, label its states with the teacher, retrain on the aggregate
   (val RMSE 0.143 rad - worse fit, but the states are the student's, which is the point):
   **0.0 % flight** at 0.3 and 0.5 m/s;
4. **DAgger round 2**: 0.0 % flight, and speed tracking at 0.3 / 0.5 / 0.7 m/s of
   0.302 / 0.462 / 0.629 m/s.

The result (`deploy/runs/r1_distilled`) passes the team's `check_policy.py` 13/13 and is runnable
through the robot runner. Two lessons: (a) a worse-valued RMSE can be a better policy - the
distribution matters, not the fit; (b) with matching observation contracts, DAgger works; the earlier
DAgger failure was the 86-vs-83 layout mismatch putting teacher labels out of distribution.

### Remaining

- Optional: a short PPO fine-tune of the distilled policy with the gait rewards for turning quality.
- The policy is flat-ground only; robustness (pushes, payloads, rough terrain) is future work.
- Navigation (goal-conditioned rooms) is the next phase.

