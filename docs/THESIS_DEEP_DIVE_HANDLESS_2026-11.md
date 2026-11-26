# Deep dive: every thesis gap, given a handless R1 — November 2026

**Date:** 2026-11-26.
**Constraint that reshapes the list:** the R1 in this lab has **no actuated hands/grippers**. That removes
grasping, in-hand dexterity and pick-and-place, but *not* handless interaction: 2026 work demonstrates
non-prehensile interaction on handless humanoids (pushing, pulling, two-arm carriage, locomotion-coupled
transport) — OIL (AAAI-SS 2026) on **Digit**, and VOFA (2026) for visual object-goal pushing.
**Verification:** [THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md) checks each of
the 15 gaps against the closest 2025-2026 work: most are narrowed or closed, only G1 and G6 survive as
clear claims. Read it before committing to a direction.
Companion documents: [THESIS_GAPS_2026-11.md](THESIS_GAPS_2026-11.md) (deep dive on GS/VSLAM/RL/distill),
[RESEARCH_LANDSCAPE_BROAD_2026-11.md](RESEARCH_LANDSCAPE_BROAD_2026-11.md) (broad landscape).

---

## 0. What the no-hands constraint removes / keeps / changes

| Direction | Status |
|---|---|
| Grasping, in-hand manipulation, dexterous VLA pick-and-place | **removed** |
| VLA post-training for manipulation skills | **removed** (navigation VLAs remain) |
| Non-prehensile pushing/pulling/carrying (arms + torso, no fingers) | **kept**, sim-first; hardware risk without force sensing |
| Locomotion, terrain, navigation, mapping, SLAM, safety, fall recovery, fleet | **kept** |
| Splat-based sim2real, benchmarks, distillation | **kept** |

The R1's end effectors are wrist-roll joints, so "interaction" means **body/arm contact**: pushing a cart,
nudging a box, carrying a load on the forearms, blocking a door. That is exactly the OIL/VOFA regime, which
is **under-explored on handless humanoids** compared to hand-based manipulation — a genuine gap.

Open question to verify on hardware: whether the R1's low-state exposes useful joint-torque estimates for
force-adaptive contact control (VOFA-style). This determines whether handless interaction can be validated
on the robot or stays simulation-only.

---

## 1. Additional SOTA anchors found for the handless setting

| Work | What it establishes | Why it matters here |
|---|---|---|
| [OIL: Object Interaction Layer (AAAI-SS 2026)](https://ojs.aaai.org/index.php/AAAI-SS/article/view/43503) | one learned WBC policy executes pushing, pulling, two-arm pick-and-place, locomotion-coupled transport from object-level kinematic targets; sim + sim2real on **Digit** | proves handless interaction is a real research direction, with a kinematic command interface we can copy |
| [VOFA (2026)](https://arxiv.org/html/2605.01518v2) | hierarchical visual sim-to-real for goal-conditioned **non-prehensile pushing** by a humanoid, force-adaptive | shows pushing can be learned from vision; gap: no-hands R1, no F/T sensors |
| [Unified Humanoid Fall-Safety Policy (FIRM, Jun 2026)](https://ieeexplore.ieee.org/document/11696999) | fall-avoidance + recovery policy fused from few human demos and RL, indoor/outdoor | safety layer for a deployed walker; R1-specific version is open |
| [GaussGym (arXiv 2510.15352)](https://arxiv.org/abs/2510.15352) | 3DGS as a **drop-in renderer inside vectorized physics**, >100k steps/s on one consumer GPU, locomotion from pixels | removes the "rendering is too slow for RL" objection to splat sim |
| [Learning locomotion for material handling / construction sites (2026)](https://ieeexplore.ieee.org/document/11448976) | RL locomotion policies for uneven construction terrain + transportation | template for handless "carry/transport" tasks |
| [Humanoid state estimation with IMU + kinematic odometry + VSLAM (ACRA 2025)](https://ssl.linklings.net/conferences/acra/acra2025_proceedings/views/includes/files/pap169s2.pdf) | legged state estimation with **ground-plane scale estimation** and adaptive fusion | closest precedent to metric anchoring of a feed-forward SLAM map with legged sensing |
| [LiDAR-VGGT (2026)](https://ieeexplore.ieee.org/abstract/document/11399876) | metric-scale, globally consistent dense mapping by fusing LiDAR-inertial odometry with VGGT | confirms metric-scale fusion is the open problem — but needs LiDAR, which the R1 lacks |
| [Active NBV risk-averse planning on 3DGS (2026)](https://ieeexplore.ieee.org/abstract/document/11696764) | tail-risk (AVaR) maps built on an online-updated 3DGS radiance field | closest precedent for uncertainty-aware planning on splat maps |
| [What the Elevation Map Cannot See (arXiv 2610.07396, Oct 2026)](https://arxiv.org/abs/2610.07396) | humanoid VLN benchmark with elevation-subtle hazards + command-execution deviation; G1 hardware validation | defines evaluation of hazard-aware humanoid navigation; blueprint for our warehouse benchmark |
| [MIT/Symbotic warehouse traffic RL (Mar 2026)](https://computing.mit.edu/news/ai-system-learns-to-keep-warehouse-robot-traffic-running-smoothly/) | RL for prioritising warehouse robot traffic at fleet scale | fleet coordination is a live industrial problem |
| [TwinAligner (2025/2026)](https://browse-export.arxiv.org/pdf/2512.19390) | visual + dynamic alignment for physics-aware real2sim2real, including non-prehensile manipulation | blueprint for auto-collider + dynamics alignment (gap A3) |

---

## 2. Deep dive, gap by gap

Each entry: **question → closest work / what's missing → what we build → experiments → resources →
risks/fallback → contribution → ease**.

### G1. Metric, loop-closed feed-forward SLAM for a handless legged robot (A1+A12)

- **Question:** can VGGT-SLAM 2.0 be made metric and drift-bounded for a legged robot using leg kinematics,
  IMU and contact evidence instead of LiDAR, and can we benchmark it in simulation with ground truth?
- **Closest work / missing:** VGGT-SLAM 2.0 leaves arbitrary scale and submap drift; Nov 2026 semantic
  anchoring attacks it monocularly; LiDAR-VGGT fuses LiDAR (robot has none); ACRA 2025 shows legged
  estimation can estimate ground-plane scale. Gap: **legged proprioceptive anchoring** for a
  feed-forward SLAM map.
- **Build:** (1) warehouse sequences with our VGGT-SLAM container; (2) a scale/translation anchor from
  foot-contact odometry (sim GT contact; real: joint encoders + IMU); (3) pose-graph fusion (gtsam
  Sim(3)) replacing VGGT's per-submap SL(4); (4) revisit-triggered loop closure; (5) benchmark harness
  with ATE/scale/drift metrics.
- **Experiments:** 5–8 warehouse trajectories (loop/no-loop, texture-rich/sparse), baselines stock
  VGGT-SLAM 2.0, VGGT-SLAM++ code if available, ORB-SLAM3 stereo; ablations: mono vs stereo, leg anchor
  on/off, loop closure on/off; metrics ATE, scale error, drift/m, map completeness, FPS.
- **Resources:** our container + A5000, weeks 8–10, no hardware needed; real-robot validation optional.
- **Risks:** contact detection noise; VGGT per-frame scale inconsistency. **Fallback:** sim-only benchmark
  + scale study (still publishable).
- **Contribution:** method + open benchmark. **Ease 9/10.** ✅ Recommended.

### G2. Splat novelty/change layer with safe planning (A2)

- **Question:** given a prior Gaussian-splat map, can a robot detect changed geometry (new/moved/removed
  objects) with calibrated uncertainty, and plan so it never hits the unmodelled ones?
- **Closest work / missing:** GS-LTS (long-term change), RGS-SLAM (dynamic objects during SLAM),
  uncertainty-driven GS-SLAM, Active-NBV risk maps. Missing: a *prior-map* novelty layer for navigation with
  uncertainty and a benchmark.
- **Build:** our existing render-depth vs live-depth residual (84.8 % recall, 100 % precision on injected
  object) extended with (a) opacity/scale/depth-variance uncertainty, (b) temporal filtering + removal
  (negative change), (c) moving actors, (d) novelty-aware cost inflation in our A*/potential-field planner,
  (e) benchmark in warehouse with GT (new rack, moved pallet, forklift, lighting change).
- **Experiments:** precision/recall vs object size/distance/occlusion; latency; navigation success/collisions
  with/without novelty layer; negative-change detection; localisation-noise sensitivity curves.
- **Resources:** our gsplat/PLY map + R1 walker + warehouse, weeks 10–12, hardware optional.
- **Risks:** pose error dominates residual (mitigate: use G1 localisation; report sensitivity explicitly).
  **Fallback:** detection-only benchmark.
- **Contribution:** method + benchmark; ICRA/IROS fit. **Ease 8/10.** ✅ Recommended.

### G3. Handless-humanoid warehouse benchmark (A12+B6)

- **Question:** can we define a benchmark that isolates what matters for a handless humanoid in a warehouse:
  mapping (scale/drift), navigation (hazards, narrow aisles), and locomotion (execution deviation) — with
  ground truth that no real robot can provide?
- **Closest work / missing:** HumanoidBench (15 whole-body tasks, 12 hand-free locomotion tasks — useful
  primitive suite), ObjectNav-HM3D, "What the Elevation Map Cannot See" (hazard + execution-deviation
  benchmark, G1). Missing: warehouse-scale, handless-specific, GT SLAM + navigation + locomotion, with
  scripted dynamic actors (forklifts, humans).
- **Build:** task suite on our warehouse: (i) map a route and report scale/ATE, (ii) navigate point-to-point
  with static/dynamic actors, (iii) cross cluttered aisles with execution-deviation metric, (iv) hazard
  classes (elevation-subtle objects vs. splat-map novelty), (v) auto-generated leaderboard + scripts. Metrics:
  nav success, time, collisions, hazard contacts, ATE/scale, command-execution deviation, fall rate.
- **Experiments:** baselines = our VGGT-SLAM + A*/potential-field + distilled walker; add 2–3 external
  baselines (e.g., a classical ORB-SLAM3+planner pipeline, a VLN-style baseline).
- **Resources:** all local, weeks 5–8, no hardware. **Risks:** benchmark scope creep — freeze the task
  matrix early.
- **Contribution:** dataset/benchmark + 1 small method chapter (G1). **Ease 9/10.** ✅ Recommended first.

### G4. Auto-generating physics colliders from a splat (A3)

- **Question:** can we build correct collision geometry for a splat reconstruction automatically, so a splat
  scene becomes a usable simulation of a real space for a *humanoid* (large contact surfaces, steps, doors)?
- **Closest work / missing:** TwinAligner (mesh-GS digital twin + dynamic alignment, manipulation), NuRec
  (splat USD without physics), RoboGSim/GSWorld (hand-set colliders). Missing: locomotion-scale evaluation
  of collider quality (steps, floor height, thin walls) and a humanoid contact test.
- **Build:** three pipelines — (i) voxelised opacity→convex boxes; (ii) 2DGS/depth meshes→trimesh with
  simplification; (iii) TSDF fusion from our sim depths (upper bound, since sim gives depth). Evaluate by
  replaying the R1 walker in procedural-sim, splat-visual+procedural-collider, and splat-visual+auto-collider.
- **Experiments:** collider accuracy (surface distance/IoU), step-height error, door/gap width error; then
  policy behaviour (falls, foot penetration, velocity tracking) on the three scenes.
- **Resources:** splat PLY + Isaac Sim, weeks 8–10, no hardware. **Risks:** splat surfaces too fluffy; use
  depth-based TSDF as the reliable baseline.
- **Contribution:** tool + evaluation methodology; feeds G5. **Ease 7/10.**

### G5. Photoreal splat sim for handless humanoid locomotion: does fidelity matter? (A4+A5)

- **Question:** does replacing procedural visuals with Gaussian-splat rendering (GaussGym/NuRec style) change
  locomotion policy transfer, and how much does collider fidelity matter relative to visuals?
- **Closest work / missing:** GaussGym proves 3DGS can render inside vectorized physics at >100k steps/s and
  trains locomotion from pixels; NuRec makes splat scenes interactive in Isaac Sim; GSWorld/RE3SIM/RoboGSim
  cover manipulation. Missing: **humanoid locomotion** fidelity study; **handless**; independent evaluation of
  collider vs visual contribution.
- **Build:** (i) integrate splat rendering into our Isaac Lab workflow (NuRec USD or GaussGym-style drop-in);
  (ii) 3×2 matrix: visuals {procedural, splat} × colliders {procedural, auto}; (iii) train our deployed
  walking policy family under each; (iv) evaluate zero-shot + DR; (v) optional real R1 transfer.
- **Experiments:** velocity tracking, falls, foot-slip, sim2sim transfer, PSNR/LPIPS of renders, policy
  robustness under lighting/texture DR.
- **Resources:** 1 GPU, weeks 14–20; rendering budget is the key constraint (use GaussGym-style rendering or
  1–8 envs). **Risks:** no measurable effect → still a valid negative result if designed with power analysis;
  real-robot availability.
- **Contribution:** system + empirical study; high impact. **Ease 6/10.**

### G6. Handless non-prehensile loco-manipulation on the R1 (new, C1)

- **Question:** can a handless R1 push/pull/transport a box or cart by body/arm contact using an OIL-style
  object-kinematic command layer, learned in splat-based sim, and transferred without force/torque sensors?
- **Closest work / missing:** OIL (Digit, object-level commands, precision pushing), VOFA (visual pushing,
  force-adaptive), construction material handling. Missing: R1 handless pushing, sensor-poor setting,
  splat-sim training.
- **Build:** object-level command interface (target object pose/velocity → whole-body target), RL policy on
  top of our locomotion stack, contact estimation from proprioception + vision; tasks: push box to goal,
  pull cart, carry load on forearms.
- **Experiments:** success rate vs object mass/friction/height; contact force estimation error; sim2real if
  hardware allows; ablations: command abstraction on/off, vision vs proprioception.
- **Resources:** sim weeks 10–12; hardware optional (pushing is safer than grasping but needs clearance).
  **Risks:** no wrist F/T sensing → force estimates noisy; may remain sim-only.
- **Contribution:** first R1 handless interaction study; strong novelty if hardware works. **Ease 5.5/10.**

### G7. Multi-robot warehouse traffic with learned walkers (B5)

- **Question:** when several handless humanoids share aisles with forklifts, how should goals be assigned and
  traffic resolved, and does learned locomotion (imperfect tracking) change the scheduling problem?
- **Closest work / missing:** MIT/Symbotic RL traffic prioritisation (industrial), RMFS coordination
  literature. Missing: learned-walker fleets in simulation with realistic execution noise; open benchmark.
- **Build:** N walkers (our policy) in our warehouse, scripted forklifts, a scheduler (rule-based → RL),
  metrics: throughput, makespan, deadlocks, collisions.
- **Experiments:** 4/8/16 robots, load scenarios, deadlock injection; compare FCFS, priority rules, RL
  scheduler; ablation: perfect vs noisy execution.
- **Resources:** one GPU can run multiple env instances; weeks 8–10; no hardware. **Risks:** novelty overlap
  with OR/industrial scheduling — position as *learned-execution-aware* scheduling.
- **Contribution:** benchmark + RL scheduler with execution-aware model. **Ease 7/10.**

### G8. Distillation benchmark for deployable humanoid locomotion (A10)

- **Question:** which distillation route (BC, DAgger, teacher-regularised PPO, on-policy distillation with
  privileged signals) gives the best deployable humanoid policy for a fixed compute budget?
- **Closest work / missing:** our own DAgger result (BC drifts, DAgger fixes), E2DRL, quadruped offline KD,
  Robo-MOPD (planners, not locomotion). Missing: systematic comparison for humanoid locomotion with
  privileged signals and a reproducible protocol.
- **Build:** our walker teacher (86-obs, base velocity) → students with different supervision; teacher
  signals: base velocity, contact, height scan, splat features; evaluation on gait metrics, velocity error,
  robustness pushes, and sim2real if available.
- **Experiments:** 4 methods × 3 privileged signals × 2 teacher qualities; metrics: tracking, gait (flight
  fraction, double support), robustness, training cost.
- **Resources:** local, weeks 8–10; GPU-heavy (multiple runs). **Risks:** likely a solid empirical paper, not
  a breakthrough.
- **Contribution:** protocol + empirical results. **Ease 7.5/10.**

### G9. Safety wrapper / fall-safety for the deployed walker (B4+C2)

- **Question:** can a runtime safety layer (fall prediction + recovery policy, and/or barrier-certificate
  monitor) keep our velocity-tracking walker within safe states, and can we quantify the intervention rate?
- **Closest work / missing:** FIRM fall-safety from few demos, learning-enabled barrier certificates, NVIDIA
  Halos. Missing: R1-specific, handless, integrated with a velocity-tracking policy and a warehouse map.
- **Build:** (i) fall predictor from proprioception; (ii) FIRM-style recovery policy (sim); (iii) runtime
  monitor that clamps commands/triggers recovery; (iv) barrier-certificate-style safety filter for
  velocity/height limits; evaluate interventions per hour, falls prevented, false alarms.
- **Experiments:** push disturbances, slippery floor DR, command extremes, terrain steps; baselines: no
  safety layer, simple thresholds.
- **Resources:** local, weeks 10–12; hardware optional. **Risks:** sim2real of recovery; safety claims need
  hardware caution.
- **Contribution:** safety method + evaluation. **Ease 7/10.**

### G10. Planning under splat-map uncertainty (A6)

- **Question:** how should a navigation stack consume the uncertainty of a *SLAM-built* splat map
  (scale/pose error, opacity, depth holes) rather than treating it as ground truth?
- **Closest work / missing:** Active-NBV risk-averse planning on 3DGS (AVaR), Splat-Nav (safe planning),
  uncertainty-driven GS-SLAM. Missing: online SLAM maps (not offline reconstructions), humanoid embodiment,
  end-to-end evaluation.
- **Build:** propagate G1's pose/scale covariance + per-Gaussian uncertainty into occupancy risk maps; planner
  with risk budget; replanning; benchmark on G3.
- **Experiments:** success vs risk budget, collisions, path length; sensitivity to scale error (inject ±5 %).
- **Resources:** weeks 12–14; depends on G1/G3. **Risks:** complexity; needs clean uncertainty modelling.
- **Contribution:** method. **Ease 5/10.**

### G11. Multi-session splat map maintenance (A11)

- **Question:** can a robot re-localise into a prior splat map, verify consistency, and merge a new session's
  observations into it without degrading it?
- **Closest work / missing:** GS-LTS (long-term change detection), continual-learning splats (VBGS),
  VGGT-SLAM single-session. Missing: multi-session robot loop with re-localisation + map merge + change
  bookkeeping.
- **Build:** sessions in our warehouse at intervals (move objects between captures), re-localisation via
  render-and-compare, change-aware fusion, evaluation of map quality before/after.
- **Experiments:** relocalisation success, PSNR/geometry of merged map, stale-object removal.
- **Resources:** weeks 18+, needs G2/G3. **Risks:** research-grade; scope creep.
- **Contribution:** method; strong if successful. **Ease 4/10.**

### G12. Dynamic / 4D warehouse reconstruction for navigation (A8)

- **Question:** can we reconstruct the warehouse *with* moving forklifts and people and use the dynamic
  layer for navigation (predictive avoidance) rather than deleting it?
- **Closest work / missing:** 4DGS/deformation-field methods, RGS-SLAM dynamic handling. Missing: dynamic
  navigation semantics (predict where the forklift will be), humanoid scale.
- **Resources:** weeks 18+; data capture heavy. **Ease 3/10.**

### G13. Dynamic/social navigation benchmark for humanoids (B9)

- **Question:** how do handless humanoids navigate shared warehouse space with humans and forklifts, and how
  do we measure social/hazard-aware behaviour?
- **Closest work / missing:** "What the Elevation Map Cannot See" hazard/execution benchmark (G1), social
  navigation literature. Missing: warehouse + splat map + humanoid + handless + dynamic actors with GT.
- **Build:** scripted human/ forklift actors (or simple social-force agents), hazard classes, metrics
  (success, hazard contact rate, personal-space violations, execution deviation).
- **Resources:** weeks 10–12; overlaps G3 (good merge). **Ease 6/10.**

### G14. VLA evaluation/post-training for navigation only (B1/B10)

- **Question:** with hands removed, is VLA useful for *navigation/wayfinding* commands on a humanoid, and how
  do public humanoid VLAs (GR00T N, Psi0) perform in a sim warehouse?
- **Closest work / missing:** GR00T N (open), Psi0 (humanoid VLA, RSS'26), VLN benchmarks. Missing: handless
  humanoid warehouse VLN with a walking policy underneath; compute budget is the limiting factor.
- **Resources:** GPU-heavy (16 GB) — feasible only for small models/fine-tuning evaluation; weeks 14+.
  **Ease 4/10.**

### G15. Legs-as-sensor: contact-aware metric state estimation (overlaps G1)

- **Question:** can foot-contact and leg kinematics give a metric anchor *and* a velocity estimate good
  enough to replace base-linear-velocity observations in the deployed policy?
- **Closest work / missing:** ACRA 2025 humanoid state estimation; our deploy policy needed observations
  history because base velocity is unmeasured. Missing: a contact-aware estimator integrated with a policy.
- **Build:** estimator (contact-aided EKF/UKF) + policy that consumes its output; evaluate gait/tracking vs
  history-based policy.
- **Resources:** weeks 10–12; hardware optional. **Ease 6/10.**

---

## 3. Ranked table (no hands, easiest first)

| # | Gap | Ease | Weeks | Hardware | Novelty | Verdict |
|---|---|---|---|---|---|---|
| 1 | G3 handless warehouse benchmark (A12+B6) | 9 | 5–8 | no | medium | do first |
| 2 | G1 metric legged VGGT-SLAM + benchmark | 9 | 8–10 | no | high | **top pick** |
| 3 | G2 splat novelty/change layer + planning | 8 | 10–12 | optional | high | strong |
| 4 | G8 distillation benchmark for locomotion | 7.5 | 8–10 | optional | medium | safe |
| 5 | G7 multi-robot warehouse traffic | 7 | 8–10 | no | medium | easy novel-ish |
| 6 | G9 safety wrapper / fall-safety | 7 | 10–12 | optional | medium-high | strong |
| 7 | G4 auto-colliders from splat | 7 | 8–10 | no | medium-high | enabler |
| 8 | G13 dynamic/social navigation benchmark | 6 | 10–12 | no | medium | merge into G3 |
| 9 | G5 photoreal splat sim fidelity study | 6 | 14–20 | robot ideal | high | impact/risk |
| 10 | G15 contact-aware metric state estimation | 6 | 10–12 | optional | medium | merge into G1 |
| 11 | G6 handless non-prehensile loco-manipulation | 5.5 | 10–12 | optional | high | risky |
| 12 | G10 planning under splat uncertainty | 5 | 12–14 | no | high | needs G1/G3 |
| 13 | G14 VLA navigation evaluation | 4 | 14+ | no | medium | compute-bound |
| 14 | G11 multi-session map maintenance | 4 | 18+ | no | high | long |
| 15 | G12 dynamic 4D reconstruction | 3 | 18+ | no | high | long |

---

## 4. Top three theses (handless), with the plan

### T1 — "Mapping the warehouse with a handless humanoid: metric feed-forward SLAM and a benchmark" ✅

Combine G3 (benchmark) and G1 (metric anchoring), with G15 as an optional chapter.
**Why it wins:** no hardware, all infrastructure exists, both an artifact (benchmark) and a method (legged
scale/loop fusion), publishable at workshop level in month 3 and full paper later.
**Plan:** M1 freeze task matrix + sequences; M2 VGGT-SLAM baseline numbers; M3 leg/IMU scale anchor +
pose-graph fusion; M4 loop-closure revisit tests; M5 benchmark + baselines; M6–7 ablations/writing.

### T2 — "Keeping the splat map honest: novelty detection and safe navigation" ✅

G2 + G13 merged (novelty layers, hazard classes, planning integration), on top of T1's map.
**Why:** prototype exists (84.8 % recall), benchmark matures into a paper, direct deployment relevance.
**Plan:** M1 prior map + residual pipeline; M2 uncertainty/temporal/negative-change; M3 planner integration;
M4 benchmark with dynamic actors; M5 sim2real analysis; M6–7 writing.

### T3 — "Handless interaction in photoreal simulation" (G5 + G6 + G4)

Build splat-based simulation with auto-colliders, then either (a) study visual/physics fidelity effects on
locomotion transfer, or (b) learn non-prehensile pushing/transport with OIL-style object commands.
**Why:** highest novelty (GaussGym/NuRec exist but handless-humanoid interaction is open), aligned with
NVIDIA's stack. **Risks:** rendering budget, hardware validation of contact control, scope discipline.
**Only choose this if:** the R1 is available for at least a small hardware validation and 5+ months remain.

---

## 5. What is explicitly *not* viable (no hands)

- Grasping/dexterous manipulation theses (pick-and-place with fingers, in-hand rotation, tool use).
- Manipulation VLA post-training (the action interface needs hands).
- Tactile-sensing theses at the fingertip (no fingers).
- Hand-based assembly/insertion tasks.

If the lab later adds a gripper/hand, G6 and the manipulation VLA direction immediately jump to the top of
the ranking.

---

## 6. Things to verify before committing

1. Does the R1 expose **joint torque / force estimates** in `LowState`? If not, handless contact control
   (`G6`) is sim-only or needs external sensing.
2. Is the **real R1** available, and what is the allowed experiment envelope (low-speed pushing in the
   hangar)?
3. Is **NuRec** installable/working in this Isaac Sim 6.2 setup (it is GA per GTC 2026, but we have not
   tested it here), or do we use GaussGym-style rendering?
4. Confirm the warehouse captures/dataset licensing and whether the benchmark may be released publicly.

---

## 7. References (added this round)

- OIL — https://ojs.aaai.org/index.php/AAAI-SS/article/view/43503
- VOFA — https://arxiv.org/html/2605.01518v2
- Unified Humanoid Fall-Safety Policy (FIRM) — https://ieeexplore.ieee.org/document/11696999
- GaussGym — https://arxiv.org/abs/2510.15352 (Berkeley thesis overview: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-240.html)
- Humanoid locomotion for material handling in construction — https://ieeexplore.ieee.org/document/11448976
- Real-time humanoid state estimation with IMU, kinematic odometry, VSLAM — https://ssl.linklings.net/conferences/acra/acra2025_proceedings/views/includes/files/pap169s2.pdf
- LiDAR-VGGT — https://ieeexplore.ieee.org/abstract/document/11399876
- Active NBV risk-averse planning on 3DGS — https://ieeexplore.ieee.org/abstract/document/11696764
- What the Elevation Map Cannot See — https://arxiv.org/abs/2610.07396
- MIT/Symbotic warehouse traffic RL — https://computing.mit.edu/news/ai-system-learns-to-keep-warehouse-robot-traffic-running-smoothly/
- TwinAligner — https://browse-export.arxiv.org/pdf/2512.19390
- HumanoidBench — https://www.roboticsproceedings.org/rss20/p061.pdf