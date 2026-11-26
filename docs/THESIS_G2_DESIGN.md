# G2 design: calibrated novelty layer for a splat-mapped warehouse

**Status:** design document for thesis Claim B (see [THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md)).
**Date:** 2026-11-26. **Constraint:** handless R1, no force sensing assumed; sim-first evaluation.

---

## 1. Thesis statement (one paragraph)

A robot that carries a Gaussian-splat map must treat it as *evidence, not truth*. We build a camera-only
**novelty layer** on top of a prior splat map that detects changed geometry — new, moved and *removed*
objects — and, crucially, **calibrates** its novelty scores with split conformal prediction so the planner
receives a *false-safe rate* guarantee instead of a hand-tuned threshold. The layer feeds a risk-aware
planner (chance-constrained / AVaR cost) driving a handless humanoid walker, and we measure the effect on
safety outcomes (collisions, hazard contacts, aborted commands), not on detection IoU alone.

The claim is **not** "we do change detection". As verified in
[THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md), change detection in 3DGS maps is
now published: [GS-LTS](http://export.arxiv.org/pdf/2503.17733) (object-level change + Gaussian editing) and
[Argos / Argos-SLAM (Oct 2026)](https://export.arxiv.org/pdf/2610.10181) (learned online scene-change
detection, integrated with VGGT-SLAM, +42 % change IoU, +27.9 % F1 over baselines). The thesis claim is the
**statistically calibrated safety layer and its planner integration**, which neither of those provides.

---

## 2. Positioning against the closest work

| Work | What it does | What it does NOT do | Consequence for us |
|---|---|---|---|
| [GS-LTS (2025/26)](http://export.arxiv.org/pdf/2503.17733) | object-level change detection and adaptive editing in a 3DGS map for long-term indoor service | no uncertainty calibration, no planner, no negative-change safety claim | detection baseline / demo of the problem |
| [Argos / Argos-SLAM (Oct 2026)](https://export.arxiv.org/pdf/2610.10181) | learned online scene-change detection with rich geometric priors, VGGT-SLAM integration | no statistical guarantee, no navigation/planning evaluation | strongest baseline; our claim must be orthogonal |
| [DUAL-GS (Nov 2026)](https://ieeexplore.ieee.org/document/11805938) | static 3DGS + dynamic 4DGS mapping (LiDAR-inertial-camera) | needs LiDAR; mapping, not prior-map novelty | shows dynamics are handled *inside* SLAM, not at planning time |
| [SemSafe-3DGS (Sep 2026)](https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1) | semantic risk-aware active navigation in uncertain splat maps | assumes the map is static/current; no change detection | closest planning precedent; our novelty layer plugs in front |
| [Active-NBV AVaR (2026)](https://ieeexplore.ieee.org/abstract/document/11696764) | tail-risk maps on an online-updated 3DGS field | reconstruction task, no map staleness | risk metric we can borrow |
| [Perceive with confidence (IJRR 2026)](https://acm-stag.literatumonline.com/doi/10.1177/02783649251378151), [learnable CP for robotics (2026)](https://ieeexplore.ieee.org/document/11697249) | conformal guarantees for learning-based perception/control | not for 3DGS change maps or map staleness | calibration toolkit we can adapt |
| [U-OBCA (Mar 2026)](https://browse-export.arxiv.org/abs/2603.04914), chance-constrained planning (2026) | distributionally robust collision avoidance | assumes an obstacle/uncertainty model, not a *change-detected splat map* | planner formulation |
| [Splat-Nav (2025)](https://ieeexplore.ieee.org/document/10930696) | safe real-time navigation in splat maps | assumes the map is correct | shows the community's assumption we attack |
| [What the Elevation Map Cannot See (Oct 2026)](https://arxiv.org/abs/2610.07396) | hazard taxonomy + execution deviation for humanoid navigation | benchmark only, no mapping-quality/novelty link | hazard classes and metrics we reuse |

**Residual claim (defensible):** *calibrated* novelty scores (with a false-safe guarantee under stated
assumptions) driving an explicit risk-aware planner for a *handless humanoid* in a warehouse whose prior map
is a splat. Neither detection (Argos), nor splat planning (SemSafe), nor calibration (Perceive-with-confidence
lineage) covers that combination.

---

## 3. Problem definition

Given:
- a prior Gaussian-splat map `M0` (our flyover splat: 200k Gaussians + poses) with rendered depth `ẑ(x; p)`
  and opacity `α(x; p)` from pose `p`,
- live RGB-D frames `D_t`, `I_t` and a pose estimate `p_t ~ N(p̂_t, Σ_t)` (from GT with injected noise in
  the standalone version; from Claim A / G1 in the fused version),
- a hazard definition (what counts as unsafe proximity/contact),

produce per-episode:
1. a **novelty map** `N(x)` over free space with scores `s(x) ∈ [0,1]` and classification
   {new, moved, removed, transient},
2. **calibrated** thresholds/quantiles such that the false-safe rate (declaring space safe when occupied) is
   controlled,
3. a **plan** `π` that respects `P(collision) ≤ δ` under the calibrated uncertainty and the walker's
   execution limits,
4. and measured safety outcomes.

Formally we calibrate the map-to-reality discrepancy: for each grid cell `c`,
`P(z_c = occupied | s(c), α(c), Σ_t) ≥ 1 − α_fpr` with a conformal guarantee under exchangeability of
calibration and test episodes.

---

## 4. System architecture

```
                 prior splat M0 (gsplat npz)                live RGB-D + pose (sim GT / G1)
                          │                                            │
                 render prior depth/opacity at p_t             depth/color preprocessing
                          └──────────────┬─────────────────────────────┘
                                         ▼
                         residual module: robust depth/color residual
                         + validity mask (opacity, out-of-range, dynamic)
                                         ▼
                     change voxeliser: log-odds volume L(v) across frames
                     (positive = new occupancy, negative = removed, transient)
                                         ▼
                 clustering/attribution → per-object change hypotheses + class
                                         ▼
                    calibration module: split conformal quantile τ(α_fpr)
                    using sim-GT change labels from calibration episodes
                                         ▼
              risk map: P_occ(c) from calibrated score + opacity + Σ_t
                                         ▼
         planner: A*/D*-lite with AVaR/chance constraint + safety shield
                                         ▼
                 controller: our distilled walker (deploy/runs/r1_distilled)
                                         ▼
                 measured outcomes: collisions, hazard contacts, aborts, path length
```

### 4.1 Residual module

For pixel `u` with live depth `z` and prior render `ẑ`, and prior opacity `α`:

```
r(u) = |z(u) − ẑ(u)| / (σ_d + k·ẑ(u))          # robust, depth-dependent
valid(u) = α(u) > α_min  ∧  σ_d ok  ∧  z in [z_min, z_max]
```

- depth noise model `σ_d` fit from flat-wall regions in the prior data (per-distance).
- colour residual `δ_I` (SSIM/L1) as a secondary cue for *texture-only* changes (e.g., spill, decal).
- explicit dynamic mask: reprojection inconsistency across two frames +, if available, the walker's commanded
  motion; dynamic pixels are *tracked separately* (transient class), not turned into static obstacles.

### 4.2 Change voxeliser (log-odds)

For each valid pixel backproject to world point `w`; update voxels along the ray:
- occupancy evidence if residual > θ_hi (new geometry),
- free-space evidence if the ray passes through voxels where the prior claimed occupancy but the live depth
  continues past (removed geometry — *negative change*),
- `L(v) ← L(v) + log-odds`, with decay for transient evidence (time constant tuned).

The negative-change handling is the part we specifically claim: removing a pallet makes previously blocked
free space usable, and *stale* obstacles cause needless detours — neither GS-LTS nor Argos evaluates this
from a planning perspective.

### 4.3 Attribution and hazard classes

Cluster changed voxels (connected components at 10 cm resolution) into objects; assign class and hazard
level:
- **occupancy hazards** (new/moved object in the path);
- **clearance hazards** (object closer than the walker's swept width to the planned path);
- **elevation-subtle hazards** (cable, spill) — detected only by colour/texture or a VLM/segmentation
  channel; kept as an explicit *residual-risk* category so the thesis does not overclaim geometry-only
  detection;
- **transient** (forklift, human) — excluded from the static map but tracked for prediction.

### 4.4 Calibration (the core contribution)

Use **split conformal prediction** on the novelty score, per hazard class:

```
calibration set: B episodes with sim-GT change labels
score: s_i = novelty confidence for hypothesis i
τ = Quantile_{1−α}({s_i : hypothesis i is actually safe/free})
guarantee: P(declaring a free cell "safe" while occupied) ≤ α   (exchangeable episodes)
```

Also report and calibrate the *false-safe* rate per metre of travel (the operationally meaningful quantity),
and check coverage under:
- localisation noise (inject Σ_t), 
- prior-map quality (different splat training budgets), 
- dynamics (moving actors).

`P_occ(c)` for planning is derived from the calibrated score, the render opacity (holes → unknown, treated
conservatively) and pose covariance (inflate residual by projected Σ_t).

### 4.5 Planner and safety shield

- Grid (5 cm) cost = occupancy probability + hazard cost + uncertainty margin.
- Chance-constrained variant: require `P(collision) ≤ δ` using the calibrated distribution; implement as
  AVaR/CVaR inflation of obstacle footprint (borrow Active-NBV AVaR formulation).
- Safety shield on the walker: if the next control step's predicted risk exceeds δ, stop/replan; log
  interventions. This makes the comparison "with/without calibrated layer" meaningful at the controller
  level, not just at the planner level.

---

## 5. Experimental design

### 5.1 Scenarios (sim; 30 episodes each, randomized placement)

| Scenario | Change type | Tests |
|---|---|---|
| S1 new obstacle in aisle | positive | baseline detection + avoidance |
| S2 rack moved | moved | attribution, path re-planning |
| S3 pallet removed | negative | stale-obstacle avoidance cost |
| S4 forklift crossing | transient | dynamic mask, no false static map |
| S5 spill / cable | elevation-subtle | residual-risk reporting (geometry fails by design) |
| S6 lighting change | appearance-only | false-positive rate (calibration stress) |
| S7 localisation noise | pose error | calibration under Σ_t (sensitivity) |
| S8 prior-map quality degradation | map error | calibration validity vs splat training budget |

### 5.2 Baselines

1. **No map**: pure online depth obstacle avoidance (our existing `nav_obstacles.py` course).
2. **Fixed-threshold novelty** (our current prototype: 84.8 % recall / 100 % precision on one injected box).
3. **GS-LTS-style change detection** (if code available) without planning.
4. **Argos-style learned change detector** (if code available) without calibration/planning; otherwise
   cite as reference numbers.
5. **SemSafe-3DGS-style risk-aware plan** on the *static* prior map (no change detection).
6. **Oracle**: GT occupancy from the simulator (upper bound).

### 5.3 Metrics

- **Detection:** per-object precision/recall/F1 at a fixed false-alarm budget; change IoU (for comparability
  with Argos); negative-change recall.
- **Calibration:** ECE and reliability diagrams; empirical false-safe rate vs α (conformal coverage);
  false alarms per 10 m travelled.
- **Planning/safety:** navigation success, collisions (with *true* geometry), hazard contact rate,
  unnecessary detours (stale-map cost), aborted commands, intervention rate, time-to-goal.
- **System:** latency, FPS, memory.
- **Ablations:** calibration off; temporal decay off; negative change off; localisation noise on; prior
  quality low/high; shield on/off.

### 5.4 Statistics

≥30 randomized episodes/scenario, 3 seeds for map/training randomness, bootstrap 95 % CIs, per-scenario
breakdown; pre-register the primary metric (collision rate / hazard contacts) to avoid cherry-picking.

---

## 6. Implementation plan (what to build, where)

| Module | New/extended | Reuses |
|---|---|---|
| prior renderer | `scripts/splat/render_prior.py` (gsplat depth + opacity at arbitrary pose) | `train_gsplat_warehouse.py` (activated params, `prim_to_cv`) |
| residual + voxeliser | `r1_isaaclab/novelty.py` | extends `scripts/splat/splat_novelty.py` (depth-residual prototype) |
| calibration | `r1_isaaclab/calibration.py` (split conformal + coverage tests) | new |
| risk planner + shield | `r1_isaaclab/risk_nav.py` | `warehouse.Occupancy`, `nav_warehouse.py`, `nav_obstacles.py` |
| scenario generator | `scripts/nav_changes.py` (scripted changes, GT labels, logging) | `warehouse.spawn`, A* |
| policy integration | controller wrapper | `deploy/runs/r1_distilled` walker |
| pose input | GT + injected noise (standalone), G1 pose+cov (fused) | `scripts/capture_warehouse_flycam.py` data, gsplat model |

Deliverable APIs:
```python
novelty = NoveltyLayer(prior_npz, K, depth_noise_model, alpha_min=0.5)
novelty.update(rgb, depth, pose, cov)
N = novelty.map()                     # scores, classes, P_occ
plan = risk_plan(N, goal, delta=0.05) # chance-constrained
shield = SafetyShield(plan, novelty)  # runtime intervention
```

---

## 7. Effort, compute, risks

**Effort:** 12 weeks (W1–2 prior renderer + residual; W3–4 voxeliser + negative change; W5–6 calibration;
W7–8 risk planner + shield; W9–10 scenarios + baselines; W11 ablations; W12 writing/demo).
**Compute:** one A5000; renders at 640×480 are ~10–30 ms; the workload is trajectory-level, not RL-scale.

**Risks and mitigations**

| Risk | Mitigation |
|---|---|
| Localisation error dominates the residual (we saw this in the prototype) | GT poses for the calibration study; simulate Σ_t; integrate G1 covariance; report failure thresholds |
| Argos or GS-LTS ships planner integration | our claim is calibration + guarantee + humanoid safety evaluation, not detection; cite them as detectors and report their scores as baselines |
| Conformal exchangeability breaks (dynamics, lighting shift) | per-scenario calibration and explicit violation analysis; report where the guarantee fails — that *is* a result |
| Sim2real splat render vs real sensor mismatch | optional real capture; otherwise report sim-fidelity limitation and use depth-derived priors |
| Splat holes cause false positives | opacity-based unknown handling + conservative inflation |

**Fallback if calibration is not tight:** deliver the *risk analysis* — which novelty signals are calibrated
and which are not, with hazard-class breakdown — still an ICRA-style paper contribution and a chapter.

---

## 8. What this would look like in the thesis

Chapter 2 ("Keeping the map honest"):
1. Motivation: splat maps are stale; navigation stacks assume truth (Splat-Nav etc.).
2. Method: residual + negative change + calibration + risk plan.
3. Evaluation: 8 scenarios, baselines, calibration curves, safety outcomes.
4. Threat model and guarantees: exchangeability assumptions, breakdown points.
5. Limitations: geometry-only hazards, no real-robot guarantee yet.

Target venue: ICRA/IROS (full), workshop version early with scenario S1/S3.

**Do not claim:** "first change detection in splat maps" (Argos/GS-LTS), "first splat navigation"
(Splat-Nav/SemSafe), "benchmark contribution" (excluded by direction).

---

## 9. Open issues to resolve before starting

1. Is Argos code available? If yes, run it as a detection baseline; if not, cite numbers and use GS-LTS.
2. Decide the guarantee semantics: conformal false-safe rate per cell vs per metre — pick one primary.
3. Decide whether S5 (spills/cables) is in scope or explicitly out (geometry-only claim). Recommendation:
   out of the core claim, reported as a failure mode.
4. Confirm the walker's swept width / stopping distance to define hazard classes (measure in sim).
5. Decide the sim2real story: real R1 available? If not, the thesis stays sim-only with a fidelity
   limitation section.
