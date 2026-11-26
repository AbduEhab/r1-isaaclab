# State of the Art and Master-Thesis Gaps — late November 2026

**Date:** 2026-11-26
**Scope:** Deep search across 3D Gaussian splatting (GS), visual/inertial SLAM (VSLAM), legged/humanoid RL,
policy distillation, and splat-based simulation, with a ranked list of gaps that are feasible as a master
thesis for *this* workspace (Unitree R1 + Isaac Sim warehouse + VGGT-SLAM 2.0 + NVIDIA gsplat pipeline).
**Method:** ~35 web searches over arXiv, CVF Open Access, IEEE/ACM, NVIDIA developer docs and repos,
cross-checked against the internal survey `docs/SOTA_RESEARCH_2026-11.md` (2026-11-13) and against the
artefacts actually present in this workspace (audited, see §3).

> Claims about specific numbers are attributed to the cited source. Everything about our own artefacts was
> measured in this workspace and is linked to the commit/document.

---

## TL;DR — what to do

| Rank | Thesis | Ease | Effort | Hardware needed | Risk |
|---|---|---|---|---|---|
| 1 | **Metric, loop-closed feed-forward SLAM for legged robots** (anchor VGGT-SLAM 2.0 scale with leg kinematics / IMU) | ★★★★★ | 8–10 wk | no (sim) | low |
| 2 | **Splat novelty maps: detecting and planning around new/changed obstacles in a Gaussian map** | ★★★★☆ | 10–12 wk | optional | low–med |
| 3 | **Photoreal real2sim from splats for humanoid locomotion** (splat scenes in Isaac Sim, auto-colliders, fidelity-vs-policy study) | ★★★☆☆ | 4–5 mo | yes (ideal) | medium |

Everything else is ranked in §5. The short version: the *perception/mapping* side offers two low-risk,
publication-sized theses because we already run the newest systems (VGGT-SLAM 2.0, gsplat, splats of the
warehouse); the *sim2real* side is the high-impact but riskier bet because leg locomotion on splat-rendered
physics is still open.

---

## 1. Landscape in one page

Three meta-trends set the 2026 research agenda:

1. **Feed-forward geometry displaced COLMAP as the mapping front-end.** VGGT-class models (2025) became
   complete SLAM systems in 2026 — VGGT-SLAM 2.0 (Jan 2026, real-time, dense, removes the 15-DoF/planar
   degeneracy of v1.0) and VGGT-SLAM++ (CVPRW 2026) — and the follow-up literature is now about the *residual
   problems*: arbitrary scale, submap drift, loop closure
   ([VGGT-SLAM++](https://www.openaccess.thecvf.com/content/CVPR2026W/VOCVALC/papers/Mandal_VGGT-SLAM_CVPRW_2026_paper.pdf),
   [scale anchoring for VGGT SLAM](https://export.arxiv.org/pdf/2611.06308),
   [VGGT-GS SLAM](https://arxiv.org/html/2609.19628v1)).
2. **Gaussian splatting became the map and the simulator.** GS-SLAM matured (Splat-SLAM, Gaussian-LIC,
   [uncertainty-driven GS SLAM, Sep 2026](https://ieeexplore.ieee.org/document/11677162),
   [RGS-SLAM for dynamic scenes](https://xplorestaging.ieee.org/document/11630798)), and NVIDIA productised
   the real2sim direction: **NuRec** ingests real camera/lidar data, produces a USD scene rendered with
   gsplat, and is **integrated with Isaac Sim**, with Fixer/Asset-Harvester generative clean-up
   ([NuRec](https://developer.nvidia.cn/omniverse/nurec)).
3. **Humanoid control is converging on RL + generative models.** RL remains the workhorse for velocity
   tracking and perceptive locomotion (e.g., [SOLO: omni-terrain long-horizon perceptive locomotion,
   2026](https://arxiv.org/html/2608.26583v1)), diffusion/whole-body models appear for scene interaction
   ([DreamControl](https://ieeexplore.ieee.org/document/11696795),
   [dual-view WBC](https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.70563)), and
   distillation is now the standard bridge from privileged/vision teachers to deployable students
   ([E2DRL](https://ieeexplore.ieee.org/document/11753978),
   [quadruped privileged→offline distillation](https://ieeexplore.ieee.org/document/11615876),
   [Robo-MOPD, Nov 2026](https://arxiv.org/html/2611.30997v1)).

---

## 2. SOTA snapshot by area

### 2.1 3D Gaussian splatting

| Sub-area | 2026 frontier | Representative work |
|---|---|---|
| Posed, per-scene quality | still the reference; 2DGS for surfaces; `gsplat` is the de-facto CUDA backend | Kerbl et al. 2023; 2DGS (SIGGRAPH 2024); [gsplat](https://github.com/nerfstudio-project/gsplat) |
| Feed-forward, pose-free | one pass from unposed images; iterative refinement is the current trick | [DeltaSplat (Oct 2026)](https://arxiv.org/abs/2610.09853), VGGT-GS backbones |
| GS-SLAM (splat as map) | real-time tracking + photoreal mapping; **uncertainty** and **dynamic scenes** are the open fronts | Splat-SLAM, Gaussian-LIC, [uncertainty-driven GS SLAM](https://ieeexplore.ieee.org/document/11677162), [RGS-SLAM](https://xplorestaging.ieee.org/document/11630798) |
| Semantic/language maps | CLIP features + language queryable Gaussian maps for zero-shot navigation | LangSplat lineage, [GLMap (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Multi-Scale_Gaussian-Language_Map_for_Zero-shot_Embodied_Navigation_and_Reasoning_CVPR_2026_paper.html) |
| Physics coupling | splats *inside* physics simulators for deformables/rendering | [PhysGaussian/PhysSplat (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Alfonso-Arsuaga_PHYSPLAT_a_Framework_for_Photorealistic_Hybrid_Simulation_of_Real_and_WACV_2026_paper.pdf) |
| Real2sim robots | photoreal replicas + physics, mostly manipulation | [GSWorld (2026)](https://ieeexplore.ieee.org/document/11697084), [RoboGSim](https://axi.lims.ac.uk/paper/2411.11839), [RE3SIM](https://ieeexplore.ieee.org/document/11695858), NVIDIA NuRec |
| Long-term / changing scenes | static-bias of GS maps is recognised; dynamic-update methods appear | [GS-LTS (long-term service robots)](https://export.arxiv.org/abs/2503.17733) |

### 2.2 VSLAM

- **Classical** (ORB-SLAM3, OpenVINS, VINS-Fusion, cuVSLAM v14) remains the robustness reference; cuVSLAM is
  closed-core but NVIDIA-supported.
- **Foundation-model SLAM** is the frontier: VGGT-SLAM 2.0 (BSD, ~8 FPS on a 3090-class GPU, dense submaps),
  MASt3R-SLAM, VGGT-GS SLAM, VGGT-Long. Remaining problems are exactly the ones we hit in this workspace:
  **arbitrary metric scale**, **submap/loop drift**, **dynamic scenes**, **evaluation**.
- Scale is being attacked with semantic/sequence anchors
  ([Nov 2026](https://export.arxiv.org/pdf/2611.06308)) but not with robot-specific priors (leg kinematics,
  IMU, known body geometry).
- Benchmarks: Replica/TUM for indoor, Hilti×Trimble 2026 for real industrial multi-floor, Oxford Spires for
  large-scale LiDAR-visual ([dataset](https://dl.acm.org/doi/10.1177/02783649251369905)).

### 2.3 Robot navigation with splat maps

- **Splat-Nav** established safe real-time navigation in Gaussian-splat maps (planning + collision
  checking) ([2025](https://ieeexplore.ieee.org/document/10930696)); follow-ups add language semantics
  (GLMap) and semantic SLAM (RTSG-SLAM).
- Open: maps that come from *onboard SLAM* rather than an offline reconstruction; uncertainty-aware
  safety margins; **change detection** (does the splat still match reality?); legged/humanoid embodiments.

### 2.4 Humanoid RL

- Velocity-tracking policies are commodity (our own distilled walker is one). The frontier is **perceptive
  locomotion over terrain** (SOLO: depth camera, 1.5 km route, 100+ stairs), **whole-body interaction**
  ([DreamControl](https://ieeexplore.ieee.org/document/11696795), [AMBIT on a Unitree R1](https://arxiv-org.ezproxy.obspm.fr/abs/2609.33484)),
  and **deployment** (latency/actuator modelling: [actuator digital twins, 2026](https://openurl.ebsco.com/EPDB%3Agcd%3A14%3A40630990/detailv2?sid=ebsco%3Aplink%3Aresult-item&id=ebsco%3Adoi%3A10.3390%2Fact15080451&bquery=DE%20%22ROBOTICS%22&page=1&limit=10&linkedFromId=42398435&crl=f)).
- A 2026 survey frames the field as moving from RL toward "physics-guided generative intelligence"
  ([NSF survey](https://par.nsf.gov/biblio/10711065-evolution-humanoid-locomotion-control)).

### 2.5 Distillation

- **Privileged teacher → deployable student** is the standard recipe, now formalised with DAgger and
  offline KD ([E2DRL](https://ieeexplore.ieee.org/document/11753978),
  [quadruped perception-semantic distillation](https://ieeexplore.ieee.org/document/11615876)).
- **Multi-teacher on-policy distillation** just arrived for robot planners (Robo-MOPD, Nov 2026, 9B
  generalist beats domain specialists) — mostly manipulation planning, not locomotion
  ([arXiv 2611.30997](https://arxiv.org/html/2611.30997v1)).
- Gap: distillation of **geometry foundation models** (VGGT) into onboard control/perception, and
  **distillation between simulation fidelities** (photoreal splat sim → deployable policy).

---

## 3. Our audited assets (what a thesis can stand on)

| Asset | State in this workspace | Evidence |
|---|---|---|
| Isaac Sim 6.2 + Isaac Lab 3.0 (Newton) | working locally + containerised, GUI fixed | `docs/LOCAL_GUI.md`, `docs/CONTAINER.md` |
| R1 task (24 joints, Unitree contract) | trained walkers, self-collision, turning, gait shaping | `docs/PAPER_NOTES.md`, `models/` |
| **Deployable 83-obs policy** (`deploy/runs/r1_distilled`) | 0 % flight at 0.3/0.5/0.7 m/s; `check_policy.py` 13/13; robot runner with safety ramps | `docs/DEPLOY.md` |
| Warehouse environment | 24×39 m, 793 colliders, forklifts, conveyors; A* navigation | `docs/WAREHOUSE.md` |
| **Gaussian splat pipeline** | NVIDIA gsplat on 216-flight frames: 200k Gaussians, loss 0.023; walkthrough video; PLY | `docs/SPLAT_WAREHOUSE.md` |
| **Splat novelty detection** | depth-residual change detection: 84.8 % recall / 100 % precision on injected obstacle | `docs/SPLAT_MAP.md` |
| **VGGT-SLAM 2.0** | containerised pipeline, runs on warehouse imagery: ATE 0.23 m (40 frames) / 0.92 m (70 frames); arbitrary scale 3.8–4.5 | `docs/VGGT_SLAM.md` |
| Navigation stack | potential-field + A*, 91.7 % room success; obstacle course 87.5 % | `docs/NAVIGATION.md` |
| Compute / sensors | RTX A5000 Laptop 16 GB, head stereo pair 6.5 cm baseline, warehouse assets | repo |

---

## 4. Gap catalogue

Each gap: what is missing in the literature, why *we* can do it, and the contribution type.

**A1. Metric scale for feed-forward SLAM on legged robots.**
VGGT-SLAM 2.0 output scale is arbitrary (we measured 3.8 and 4.5 across two captures); the 2026
semantic-anchoring fix targets monocular loop closure, not robots with proprioception. A legged robot has
foot-contact odometry, IMU, and a known body geometry. Gap: *scale estimation from leg odometry/contact
priors fused with VGGT submaps*, plus a benchmark of legged sequences with GT metric poses. Our sim gives
GT for free. Contribution: method + benchmark; IROS/RA-L/ICRA fit.

**A2. "Is the splat still true?" — novelty and change detection in Gaussian maps, for navigation.**
GS-LTS and RGS-SLAM handle dynamics inside mapping; no system uses a *prior splat map* as the reference and
reports geometric novelty with calibrated uncertainty for planning. We have a working protoype (84.8 %
recall) and the warehouse for evaluation. Contribution: novelty layer + uncertainty + planning integration.

**A3. Auto-generated collision geometry from splat reconstructions.**
Real2sim pipelines (GSWorld, RoboGSim, RE3SIM) hand-craft or segment colliders; NuRec gives a USD splat
scene without collision. Gap: converting Gaussian geometry (or 2DGS surfaces) to physics colliders with
calibrated fidelity. Uses our warehouse splat + Isaac Sim; contribution: tool + accuracy/robustness study.

**A4. Photoreal splat simulation for humanoid locomotion sim2real.**
RoboGSim/GSWorld/RE3SIM are manipulation-centric; humanoid *locomotion* sim2real from splat scenes is open.
We have the R1 policy stack, the warehouse splat, NuRec integration in Isaac Sim, and a real R1 (if
accessible). Contribution: system + empirical study of whether visual realism changes locomotion sim2real
(and how much physics fidelity dominates).

**A5. How much visual fidelity does a locomotion policy need?**
No controlled study of render fidelity (procedural → splat → NuRec) against control performance/sim2real
gap. Small but clean: 3–4 conditions, identical physics, measure transfer. Very easy to run in our stack;
publishable as an empirical/benchmark paper or as part of A4.

**A6. Navigation on SLAM-generated splat maps with uncertainty.**
Splat-Nav assumes a good offline map; feed-forward SLAM maps have scale/pose drift and uncertainty. Gap:
planning that consumes the SLAM map's *uncertainty* (localisation covariance, splat opacity, depth holes)
rather than treating it as ground truth. Our VGGT-SLAM + Splat-Nav lineage + R1 make this feasible.

**A7. Distilling geometry foundation models into onboard humanoid perception/control.**
VGGT features/depth are strong but cloud-scale; no work distils a VGGT teacher into an onboard policy that
uses it *for control* (not just mapping). Harder, but our deployed 83-obs policy gives a control student and
our VGGT container the teacher.

**A8. Dynamic-object-aware splat maps for warehouses.**
4DGS exists, but dynamic removal/attribution in a *navigation* map (forklift moving through) with
humanoid-scale evaluation is open. Harder data/eval.

**A9. Language-conditioned navigation over splat maps on a humanoid.**
GLMap (CVPR 2026) does zero-shot embodied navigation; no humanoid + real splat map + R1 walking stack
demonstration. Integration-heavy rather than novel.

**A10. Distillation benchmark for humanoid locomotion (privileged → deployable).**
Our DAgger experiment showed BC-only drifts and on-policy DAgger works; a systematic comparison (BC, DAgger,
teacher-regularised PPO, on-policy distillation with token/action-level supervision) across privileged
signals (base vel, depth, splat) would be a solid empirical paper. Low risk, but less novel.

**A11. Multi-session splat map maintenance.**
VGGT-SLAM and 3DGS assume one pass; re-localising a new session into a prior splat map and updating it is an
open robotics problem. Medium-hard; needs map anchoring (overlaps A1/A2).

**A12. Foundation-model SLAM evaluation for warehouses.**
We already have GT poses/depth in Isaac warehouse, and our VGGT-SLAM numbers are inconsistent across
captures (0.23 vs 0.92 m ATE). A proper benchmark (trajectories, dynamics, texture) with failure analysis
is a dataset/benchmark contribution; low novelty but high utility and easy.

---

## 5. Ranking by ease (and expected value)

Scoring: **Ease** = infrastructure readiness × data readiness × algorithmic risk × compute (10 = can start
tomorrow, 1 = new research infrastructure). **Novelty/Impact**: publication potential for a master thesis.
Weeks are for the core result, excluding writing.

| # | Gap | Ease | Weeks | Hardware | Novelty | Comment |
|---|---|---|---|---|---|---|
| 1 | **A1 metric scale anchoring for legged VGGT-SLAM** | 9 | 8–10 | sim only | high | We have VGGT-SLAM running, GT in sim, and legged contacts; the literature explicitly lists scale drift as open. Add a benchmark; strong RA-L/IROS story. |
| 2 | **A2 splat novelty/change detection** | 8 | 10–12 | sim + optional | high | Prototype exists (84.8 % recall); needs uncertainty + planning + benchmark. Directly useful for robot deployment. |
| 3 | **A12 foundation-SLAM evaluation for warehouse** | 9 | 4–6 | sim only | medium | Easiest publishable artifact; best done *inside* A1 as the benchmark section. |
| 4 | **A3 auto-colliders from splats** | 7 | 8–10 | sim only | medium-high | Tool + study; feeds A4. Risk: geometry quality from squishy splats; use 2DGS/depth to help. |
| 5 | **A5 visual fidelity vs control** | 7 | 8–10 | sim only | medium | Cheap ablation, but needs GPU-rendered RL (rendering cost is the main constraint). |
| 6 | **A10 distillation benchmark for locomotion** | 7 | 8–10 | sim; hardware optional | medium | We have the full pipeline; contribution is empirical. Good fallback if A1 stalls. |
| 7 | **A4 splat real2sim for humanoid locomotion** | 5 | 16–20 | real R1 ideally | very high | Highest impact, highest risk: physics/visual coupling, real-robot availability, sim2real eval. |
| 8 | **A6 planning under splat-map uncertainty** | 5 | 12–14 | sim | high | Needs A1/A2 results first; Splat-Nav lineage is strong. |
| 9 | **A9 language-conditioned humanoid navigation** | 5 | 12–14 | sim + real | medium | Integration work; GLMap exists, novelty is the humanoid loop. |
| 10 | **A7 VGGT distillation into onboard control** | 4 | 14–16 | sim | high | Requires designing the teacher-student interface; compute-heavy. |
| 11 | **A11 multi-session map maintenance** | 3 | 18+ | sim | high | Research-grade; overlaps A1/A2; scope risk. |
| 12 | **A8 dynamic 4D warehouse maps** | 3 | 18+ | sim | high | 4DGS data + evaluation; heavy. |

**Ranking logic:** the easiest theses are those where the hard system already exists in this workspace and
the open problem is a *measurement or fusion* question (A1, A2, A12, A3). The hardest are those that require
new infrastructure or real hardware (A4, A7, A11, A8).

---

## 6. Recommended theses (concrete)

### Recommendation 1 — "Metric, drift-bounded feed-forward SLAM for legged robots" (best ease/value)

**Thesis question:** can a VGGT-family SLAM system produce metric, drift-bounded maps for a legged robot by
fusing leg-odometry scale evidence and contact-consistent loop closure, and can we quantify it on a
warehouse benchmark?

**Why now:** VGGT-SLAM 2.0 is open (BSD), real-time, and explicitly leaves scale/loop drift to follow-ups;
the 2026 semantic-anchoring paper shows the community agrees this is the open problem — but nobody uses leg
kinematics.

**Method (what we build):**
1. Take our working `vggt-slam:2.0` container and warehouse captures (we already have ATE 0.23/0.92 m and
   scale 3.8/4.5 measurements).
2. Add a metric anchor: per-submap scale from wheel/foot-contact odometry (sim: ground-truth foot contact;
   real: R1 joint encoders + IMU), fused with VGGT's submap transforms.
3. Add a loop-closure test keyed on robot trajectory revisits (we control the walkthrough route).
4. Build the evaluation: Isaac warehouse sequences + GT poses/depth, plus our existing flyover splat data;
   report ATE, scale error, map completeness, runtime.
5. Ablate: monocular vs stereo, with/without leg anchor, with/without loop closure.

**Experiments/metrics:** ATE RMSE, scale error %, drift per metre, submap alignment error, runtime
(FPS), map point coverage. Baseline = stock VGGT-SLAM 2.0 and ORB-SLAM3 stereo.

**Deliverables:** open-source patch + benchmark + paper (IROS/RA-L). Reuses 80 % of existing code.

**Timeline (8 months):** M1 integration + sequences; M2 scale anchor; M3 loop closure; M4 benchmark;
M5–6 ablations + real/sim comparison; M7 writing; M8 defence prep.

**Risks:** leg odometry noise on the real robot; VGGT scale may not be per-frame consistent (mitigate with
submap-level fusion). **Fallback:** pure-sim benchmark + uncertainty analysis (A12) is still a thesis.

### Recommendation 2 — "Splat novelty maps: keeping a Gaussian map honest for safe navigation"

**Thesis question:** given a prior Gaussian-splat map of an environment, can a robot detect geometry that
changed (new/moved/removed objects) with calibrated uncertainty, fuse it into the map, and plan around it
safely?

**Why now:** Splat maps are becoming the robot map representation (Splat-Nav, GS-SLAM, NuRec), but every
system assumes the map is current. GS-LTS shows long-term changes matter; nobody gives a robot a
change-detection layer with planning integration and a benchmark.

**Method:**
1. Export prior map from our warehouse splat (we have 200k Gaussians, PLY + poses).
2. Render-depth vs live-depth residual (we already measured 84.8 % recall / 100 % precision on an injected
   obstacle) → upgrade with: (a) uncertainty from opacity/depth variance, (b) temporal filtering, (c) object
   removal (negative change), (d) dynamic objects (moving forklift).
3. Novelty-aware planning: inflate cost/clearance using novelty uncertainty; integrate with our A*/potential-
   field navigation and the deployed R1 walker.
4. Benchmark: new objects, moved racks, removed pallets, moving forklift, lighting change; report
   detection precision/recall, latency, false-positive cost, navigation success/collisions.

**Deliverables:** novelty layer + planner integration + benchmark; ICRA/IROS/CoRL workshop or full paper.
**Timeline:** M1 prior map + residual baseline (mostly done); M2 uncertainty/temporal; M3 planning;
M4 benchmark; M5 real/sim transfer; M6–7 writing.

**Risks:** render-vs-live residual sensitivity to localisation error (we saw this: pose error dominates) — so
localisation from A1 is the dependency. **Fallback:** pure detection benchmark (still publishable).

### Recommendation 3 — "Photoreal real2sim from splats for humanoid locomotion" (high impact, riskier)

**Thesis question:** does replacing procedural warehouse visuals with a Gaussian-splat reconstruction (and
how much of it) change a humanoid walking policy's sim2real transfer, and can the physics side (colliders)
be generated automatically from the splat?

**Why now:** NuRec makes splat scenes interactive in Isaac Sim, GSWorld/RoboGSim/RE3SIM demonstrate the
idea for manipulation — humanoid locomotion is untouched, and our policy stack is ready.

**Method:**
1. Auto-collider generation from the splat (A3) — evaluate three strategies (voxelised opacity, 2DGS
   surfaces, VGGT depth meshes) against ground-truth colliders in our warehouse sim.
2. Build 3 simulation conditions: procedural warehouse (current), splat-visual + procedural colliders, splat
   visual + auto colliders.
3. Retrain/fine-tune the deployed walking policy in each; evaluate zero-shot and with domain randomisation.
4. Transfer to the real R1 in the hangar if access allows; otherwise evaluate on a second visual domain and
   report the perception gap; use our distilled walker as the baseline.

**Metrics:** distance walked, velocity tracking error, foot-slip, falls, sim2real success; plus collider
accuracy (IoU/distance to GT surfaces) and rendering metrics (PSNR/LPIPS).

**Deliverables:** real2sim toolkit + empirical study. **Timeline:** M1 colliders; M2 splat visual sim;
M3 policy training; M4 ablations; M5 real/sim eval; M6–7 writing.

**Risks:** (i) real R1 availability and safety; (ii) splat renderer throughput for RL (rendering at 8192 envs
is impossible — plan for small-env visual RL or image-conditioned eval, which itself is a finding);
(iii) physics/visual coupling may show no effect, which is still a valid negative result if designed
properly.

---

## 7. What to avoid (saturated or out of scope)

- **Another basic GS-SLAM on Replica/TUM**: crowded, and we would compete with dedicated groups with more
  compute.
- **Training a general VLA from scratch**: out of scope on one A5000; even fine-tuning is a project of its
  own.
- **Pure VGGT fine-tuning for reconstruction quality**: unclear licence/compute and dominated by NVIDIA/
  Meta-scale labs.
- **New 3DGS rasteriser**: gsplat/3DGUT/NuRec cover it.

---

## 8. Cross-cutting advice

- **Pick the thesis where this workspace is already at the frontier**: we have runnable VGGT-SLAM 2.0,
  gsplat, a splat of the warehouse, a deployed R1 policy, and a simulated warehouse with GT. That
  combination is exactly A1/A2/A3/A5.
- **Hardware is the main unknown.** If the R1 is available, A4 jumps to the top for impact; if not, do A1 +
  A12 (sim-only) and treat A4 as future work.
- **Publish early:** A1's benchmark section alone (A12) is a workshop paper; A2's detection benchmark is
  another. Sequence: workshop → full paper.
- **Instrument everything:** ATE/scale, novelty precision/recall, navigation success, and rendering metrics;
  our repo already has the measurement scripts (`gait_report.py`, `eval_vggt_warehouse.py`,
  `nav_obstacles.py`, `splat_novelty.py`).

---

## 9. References

External (accessed 2026-11-26):

- VGGT-SLAM 2.0 — https://arxiv.org/abs/2601.19887 ; MIT-SPARK repo https://github.com/MIT-SPARK/VGGT-SLAM
- VGGT-SLAM++ (CVPRW 2026) — https://www.openaccess.thecvf.com/content/CVPR2026W/VOCVALC/papers/Mandal_VGGT-SLAM_CVPRW_2026_paper.pdf
- Scale anchoring for VGGT SLAM (Nov 2026) — https://export.arxiv.org/pdf/2611.06308
- VGGT-GS SLAM (2026) — https://arxiv.org/html/2609.19628v1
- Uncertainty-driven 3DGS RGB-D SLAM (Sep 2026) — https://ieeexplore.ieee.org/document/11677162
- RGS-SLAM (dynamic scenes) — https://xplorestaging.ieee.org/document/11630798
- RTSG-SLAM (semantic GS SLAM for VLN) — https://dl.acm.org/doi/10.1145/3833142.3833144
- Splat-Nav — https://ieeexplore.ieee.org/document/10930696
- GLMap (CVPR 2026) — https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Multi-Scale_Gaussian-Language_Map_for_Zero-shot_Embodied_Navigation_and_Reasoning_CVPR_2026_paper.html
- GS-LTS (long-term service robots) — https://export.arxiv.org/abs/2503.17733
- DeltaSplat (Oct 2026) — https://arxiv.org/abs/2610.09853
- gsplat — https://github.com/nerfstudio-project/gsplat
- NVIDIA NuRec — https://developer.nvidia.cn/omniverse/nurec
- PhysSplat (WACV 2026) — https://openaccess.thecvf.com/content/WACV2026/papers/Alfonso-Arsuaga_PHYSPLAT_a_Framework_for_Photorealistic_Hybrid_Simulation_of_Real_and_WACV_2026_paper.pdf
- GSWorld (2026) — https://ieeexplore.ieee.org/document/11697084
- RoboGSim — https://axi.lims.ac.uk/paper/2411.11839
- RE3SIM — https://ieeexplore.ieee.org/document/11695858
- SOLO (perceptive humanoid locomotion, 2026) — https://arxiv.org/html/2608.26583v1
- DreamControl (whole-body humanoid, 2026) — https://ieeexplore.ieee.org/document/11696795
- Dual-view WBC — https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.70563
- AMBIT on Unitree R1 (Sep 2026) — https://arxiv-org.ezproxy.obspm.fr/abs/2609.33484
- E2DRL (quadruped teacher/DAgger) — https://ieeexplore.ieee.org/document/11753978
- Privileged→offline distillation for quadruped (2026) — https://ieeexplore.ieee.org/document/11615876
- Robo-MOPD (multi-teacher on-policy distillation, Nov 2026) — https://arxiv.org/html/2611.30997v1
- Actuator digital twins for sim2real (2026) — https://openurl.ebsco.com/EPDB%3Agcd%3A14%3A40630990/detailv2?sid=ebsco%3Aplink%3Aresult-item&id=ebsco%3Adoi%3A10.3390%2Fact15080451
- Humanoid locomotion survey (2026) — https://par.nsf.gov/biblio/10711065-evolution-humanoid-locomotion-control
- Oxford Spires benchmark — https://dl.acm.org/doi/10.1177/02783649251369905

Internal:

- `docs/SOTA_RESEARCH_2026-11.md` — earlier survey (VSLAM + 3DGS), 2026-11-13
- `r1-isaaclab/docs/PAPER_NOTES.md`, `DEPLOY.md`, `VGGT_SLAM.md`, `SPLAT_WAREHOUSE.md`, `SPLAT_MAP.md`,
  `NAVIGATION.md`, `WAREHOUSE.md` — measured results of our artefacts
- `docs/REPORT.md` — thesis report so far
