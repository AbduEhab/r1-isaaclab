# Deep dive: every thesis gap, given a handless R1 — November 2026

**Date:** 2026-11-26.
**Constraint that reshapes the list:** the R1 in this lab has **no actuated hands/grippers**. That removes
grasping, in-hand dexterity and pick-and-place, but *not* handless interaction: 2026 work demonstrates
non-prehensile interaction on handless humanoids (pushing, pulling, two-arm carriage, locomotion-coupled
transport) — OIL (AAAI-SS 2026) on **Digit**, and VOFA (2026) for visual object-goal pushing.
**Verification:** [THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md) checks each of
the 15 gaps against the closest 2025-2026 work: most are narrowed or closed, only G1 and G6 survive as
clear claims. Read it before committing to a direction.
> **Direction (2026-11-26):** pure benchmark/dataset topics are excluded. Benchmarks below exist only as
> *evaluation protocols* inside a method thesis, never as the thesis contribution.

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
| Splat-based sim2real, distillation | **kept** |

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

### G1. Metric, loop-closed feed-forward SLAM for a handless legged robot (A1)

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

### G10. Planning under splat-map uncertainty (A6) - (closed in verification; not a thesis option)

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

### G12. Dynamic / 4D warehouse reconstruction for navigation (A8) - (closed in verification; not a thesis option)

- **Question:** can we reconstruct the warehouse *with* moving forklifts and people and use the dynamic
  layer for navigation (predictive avoidance) rather than deleting it?
- **Closest work / missing:** 4DGS/deformation-field methods, RGS-SLAM dynamic handling. Missing: dynamic
  navigation semantics (predict where the forklift will be), humanoid scale.
- **Resources:** weeks 18+; data capture heavy. **Ease 3/10.**

### G14. VLA evaluation/post-training for navigation only (B1/B10) - (closed in verification; not a thesis option)

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

## 3. Ranked table (method-only, handless, easiest first)

Pure benchmark topics (warehouse benchmark, fleet traffic benchmark, distillation benchmark,
social-navigation benchmark) and gaps verified as closed (G10 planning-under-uncertainty, G12 dynamic 4D
mapping, G14 VLA evaluation) are removed ? see [THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md).

| # | Gap | Ease | Weeks | Hardware | Novelty | Verdict |
|---|---|---|---|---|---|---|
| 1 | G1 contact-anchored metric feed-forward SLAM for a handless legged humanoid (G15 merged) | 9 | 8?10 | no | high | **top pick** |
| 2 | G2 prior-map novelty with calibrated uncertainty + hazard-aware planner | 8 | 10?12 | optional | high | recommended |
| 3 | G4 auto-collider generation from splats (method/tool) | 7 | 8?10 | no | medium-high | enabler for G5 |
| 4 | G9 runtime safety monitor / fall-safety for the deployed walker | 7 | 10?12 | optional | medium | optional chapter |
| 5 | G5 visual-vs-collider fidelity ablation for humanoid locomotion (empirical study) | 6 | 14?20 | robot ideal | high | risky |
| 6 | G15 contact-aware metric state estimation | 6 | 10?12 | optional | medium | merge into G1 |
| 7 | G6 handless non-prehensile interaction on the R1 | 5.5 | 10?12 | optional | high | hardware-gated |
| 8 | G11 multi-session splat map maintenance | 4 | 18+ | no | high | long bet |

## 4. Top three theses (method-only, handless)

### T1 ? "Contact-anchored metric feed-forward SLAM for a handless legged humanoid" ?

Combines G1 and G15. The claim is a **method**: leg contact + kinematics anchor the arbitrary scale of a
VGGT-SLAM 2.0 map and close submap drift; the evaluation on simulator ground truth (ATE, scale error,
drift/m) is a protocol, not a benchmark contribution.

**Plan:** M1 VGGT-SLAM baseline numbers on warehouse sequences; M2 contact/leg-odometry scale anchor;
M3 Sim(3) pose-graph fusion + loop closure; M4 ablations (mono/stereo, anchor on/off, loop on/off);
M5 evaluation protocol + comparison to FoundationVIO/ACRA-style estimators; M6?7 writing.

**Risk:** contact detection noise, VGGT per-frame scale inconsistency. **Fallback:** keep the anchor method
but evaluate only on sim sequences and report the failure cases.

### T2 ? "Keeping the splat map honest: prior-map novelty with calibrated uncertainty" ?

G2. The claim is a **method**: a camera-only novelty layer over a prior splat map that detects new *and
removed* geometry, attaches calibrated uncertainty, and feeds a hazard-cost planner. Detection alone would
be scooped by GS-LTS, so the contribution is the uncertainty calibration + planner integration + measured
safety effect.

**Plan:** M1 prior map + residual baseline; M2 uncertainty/temporal/negative change; M3 planner integration;
M4 evaluation protocol (new/moved/removed objects, forklift, lighting); M5 sim2real sensitivity; M6?7 writing.

### T3 ? "Handless interaction and photoreal contact in simulation" (G4 + G5, G6 if hardware)

G4 + G5 as a system/method thesis: auto-colliders from splat geometry, then a controlled
**visual-vs-collider fidelity ablation** for a handless humanoid; if the R1 can push safely, add G6's
object-kinematic interaction layer.

**Plan:** M1 collider pipelines + accuracy; M2 splat-visual simulation; M3 policy training under the 2x2
matrix; M4 ablations; M5 hardware validation (if available); M6?7 writing.
**Only choose this if:** rendering budget is confirmed and (for G6) the R1 can safely push.

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