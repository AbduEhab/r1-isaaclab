# Gap verification — do these gaps actually exist? (late Nov 2026)

**Date:** 2026-11-26.
**Purpose:** the previous documents listed 15 candidate gaps and ranked them by ease. This document checks
each gap against the closest work published *up to now*, because a gap that has been closed makes a bad
thesis. Verdicts: **CLEAR** (no system covers the claim; a thesis can be built), **NARROWED** (the core idea
exists; only a sharply-scoped residual survives), **CLOSED** (the claim is already covered; do not use).

Method: for each gap, the closest 2025–2026 systems were searched (arXiv/CVF/IEEE/ACM + repos), their scope
compared against the gap claim, and the *residual* claim written down. The residual is what a thesis may
actually claim as novel.

---

## 1. Verdict matrix

| # | Gap (as previously stated) | Closest work (2025–2026) | Coverage | Verdict | Defensible residual (if any) |
|---|---|---|---|---|---|
| G1 | Metric, loop-closed VGGT SLAM for legged robot | [FoundationVIO (Oct 2026)](https://export.arxiv.org/pdf/2610.14951) enforces metric scale in feed-forward VIO; [scale-anchoring VGGT-SLAM (Nov 2026)](https://export.arxiv.org/pdf/2611.06308); [Contact-Anchored Proprioceptive Odometry (Feb 2026)](https://export.arxiv.org/abs/2602.17393); [ACRA humanoid state estimation](https://ssl.linklings.net/conferences/acra/acra2025_proceedings/views/includes/files/pap169s2.pdf); [LiDAR-VGGT](https://ieeexplore.ieee.org/abstract/document/11399876) | metric scaling exists generically and for VIO; contact odometry exists for quadruped/wheel-legged; LiDAR-VGGT needs LiDAR | **NARROWED** | **Contact-anchored metric VGGT SLAM for a *legged humanoid* with no LiDAR/wheels**, plus a legged warehouse benchmark (no such system found) |
| G2 | Prior-splat novelty/change layer + safe planning | [GS-LTS (2025/26)](http://export.arxiv.org/pdf/2503.17733) does object-level change detection + Gaussian editing for indoor robots; [DUAL-GS (Nov 2026)](https://ieeexplore.ieee.org/document/11805938) static+dynamic 4D fields (LiDAR-inertial-camera); [SemSafe-3DGS (Sep 2026)](https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1) semantic risk-aware active navigation in uncertain splat maps | change detection exists; uncertainty-aware splat navigation exists; but not together, and dynamic GS SLAM needs LiDAR | **NARROWED** | **Camera-only prior-map novelty layer (new *and* removed objects) with calibrated uncertainty that is consumed by a planner**, benchmarked with hazards; explicitly not "change detection" as the claim |
| G3 | Handless-humanoid warehouse benchmark (mapping+navigation) | [HumanoidVLN (Aug 2026)](https://export.arxiv.org/abs/2608.12860) physics-grounded VLN benchmark across embodiments; [DPed-VLN (Sep 2026)](https://zenodo.org/records/22816620) social VLN benchmark; [What the Elevation Map Cannot See (Oct 2026)](https://arxiv.org/abs/2610.07396) hazard + execution-deviation benchmark; [COMPASS intralogistics benchmark (Oct 2026)](https://export.arxiv.org/pdf/2610.28432) industrial sim-to-real navigation incl. a humanoid mobile platform | navigation benchmarks exist; none evaluates **SLAM/mapping quality (ATE/scale)** + **handless legged locomotion** + **industrial dynamic actors** in one suite | **NARROWED** | **Mapping+localisation benchmark** (not another VLN benchmark): SLAM scale/ATE across trajectories, with navigation/hazard tasks layered on it |
| G4 | Auto-collider generation from splats | [UCL 3DGS digital twin (2026)](https://discovery.ucl.ac.uk/id/eprint/10232226/) converts splats to **collision-ready** geometry (Unity/ROS2/MoveIt); [TwinAligner (2026)](https://browse-export.arxiv.org/pdf/2512.19390) mesh-GS twin + dynamics alignment; Mesh-GS variants | the core idea is published (for manipulation) | **NARROWED/weak** | Only the **evaluation methodology for locomotion-scale colliders** (steps, gaps, floor height) survives; best as a chapter, not a thesis |
| G5 | Photoreal splat sim fidelity vs control | [GaussGym (2025/26)](https://arxiv.org/abs/2510.15352) 3DGS as renderer inside vectorized physics, locomotion from pixels at >100k steps/s; [COMPASS+Cosmos intralogistics sim2real (Oct 2026)](https://export.arxiv.org/pdf/2610.28432); [RE3SIM](https://ieeexplore.ieee.org/document/11695858) | splat-sim locomotion exists; industrial sim2real benchmark exists | **NARROWED** | **Controlled ablation of *visual* vs *collider* fidelity for a humanoid locomotion policy** (GaussGym shows rendering works; nobody isolates the two factors with a real-robot check) |
| G6 | Handless non-prehensile interaction on the R1 | [OIL (Nov 2026)](https://ojs.aaai.org/index.php/AAAI-SS/article/view/43503) pushing/pulling/transport on Digit; [VOFA (2026)](https://arxiv.org/html/2605.01518v2) visual pushing; [iGPC (Oct 2026)](https://arxiv-org.ezproxy.obspm.fr/html/2610.08120v1) object-aware humanoid interaction with onboard sensing | handless interaction demonstrated on other humanoids | **CLEAR (narrow)** | **R1-class handless interaction without force/torque sensing**, splat-sim trained; hard to claim broadly, strong if hardware validated |
| G7 | Fleet traffic with learned walkers | [MIT/Symbotic warehouse traffic RL (Mar 2026)](https://computing.mit.edu/news/ai-system-learns-to-keep-warehouse-robot-traffic-running-smoothly/); [humanoid fleet orchestration (2026)](https://www.theengineer.co.uk/content/news/ai-framework-aims-to-orchestrate-humanoid-robot-fleets) | industrial traffic learning exists | **NARROWED/weak** | *Execution-error-aware* scheduling (learned walkers deviate from commanded velocity) as a benchmark+method; modest novelty |
| G8 | Distillation benchmark for locomotion | [E2DRL](https://ieeexplore.ieee.org/document/11753978); [offline KD for quadruped](https://ieeexplore.ieee.org/document/11615876); small IsaacLab comparisons already reported publicly (e.g., a student study comparing PPO/teacher-student/P2PDRL/CPD) | comparison-style work exists in bits | **WEAK** | Only a *methodological* protocol paper; not recommended as a main thesis |
| G9 | Safety wrapper / fall-safety for deployed walker | [FIRM (Jun 2026)](https://ieeexplore.ieee.org/document/11696999); [Humanoid-WAM (2026)](https://zenodo.org/records/20313417) includes balance-aware risk estimation + unsafe action filtering; barrier-certificate literature | safety filtering exists at system level | **NARROWED** | R1-specific, policy-integrated safety monitor with measured intervention statistics; chapter-level |
| G10 | Planning under splat-map uncertainty | [SemSafe-3DGS (Sep 2026)](https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1) risk-aware active nav on uncertain splat maps; [Active NBV AVaR (2026)](https://ieeexplore.ieee.org/abstract/document/11696764); [MAGICIAN (CVPR'26)](https://blog.csdn.net/Yong_Qi2015/article/details/160749185); LLM-guided active mapping with Fisher information (2026) | this gap is essentially closed | **CLOSED** | — |
| G11 | Multi-session splat map maintenance | multi-session GS via LIVO alignment (CCDC 2026); LiLoc lifelong localisation; GS-LTS long-term updates | offline multi-session exists; LiLoc is lifelong | **NARROWED** | Robot-loop online maintenance on a *legged* platform; hard, 18+ weeks, risky |
| G12 | Dynamic/4D warehouse reconstruction for navigation | [DUAL-GS (Nov 2026)](https://ieeexplore.ieee.org/document/11805938); ProDyG (NeurIPS 2025); 4DGS-SLAM family; RGS-SLAM | crowded | **CLOSED** (for mapping) | Predictive use of the dynamic layer in navigation is unexplored, but that is a hard, long bet |
| G13 | Dynamic/social navigation benchmark | DPed-VLN; "Elevation Map Cannot See"; social navigation literature | covered at benchmark level | **CLOSED/weak** | Only warehouse-specific (forklift) social navigation; merge into G3 |
| G14 | VLA navigation evaluation | HumanoidVLN (Aug 2026); TANGO (Sep 2026); GR00T/Psi0 ecosystem | broadly covered | **CLOSED** | — |
| G15 | Contact-aware metric state estimation | [Contact-Anchored Proprioceptive Odometry (Feb 2026)](https://export.arxiv.org/abs/2602.17393) already does IMU+motor pose/velocity estimation for legged/wheel-legged robots | covered for the estimator | **NARROWED** | Integration with a policy that *uses* the estimate instead of history observations; a chapter, not a thesis |

**Net result: 0 gaps are "clear" in the strong sense; 1 is clear in the narrow sense (G6 if hardware
exists); 8 are narrowed; 6 are closed or too weak.** The honest conclusion is that the 2026 field moves fast
enough that broad "we will do X" claims are usually already done by someone — a master's thesis has to claim
a *combination* or a *specific robot/benchmark* that the literature does not cover.

---

## 2. The surviving claims (written as they would appear in a paper)

### Claim A — "Metric feed-forward SLAM for handless legged robots" (from G1; strongest)

> Foundation-model SLAM (VGGT-SLAM 2.0) produces maps with arbitrary scale and submap drift; existing
> metric fixes use VIO (FoundationVIO), semantics (scale anchoring) or LiDAR (LiDAR-VGGT). No system anchors
> a feed-forward SLAM map using **leg contact and kinematics** on a **handless legged humanoid**, and no
> benchmark measures scale/ATE for that setting. We provide the contact-anchored fusion and the benchmark.

Why the residual is defensible: contact anchoring is genuinely different from VIO (no wheel, intermittent
contacts, foot-slip), and the eval target (legged humanoid, industrial warehouse, no LiDAR) is unoccupied.
Risk of scooping: medium — the components exist and a group with a legged platform could publish it.

### Claim B — "Prior-map novelty with uncertainty consumed by a planner" (from G2; needs sharp framing)

> Change detection for GS maps exists (GS-LTS, object-level editing) and uncertainty-aware splat navigation
> exists (SemSafe-3DGS), but no system treats a **prior splat map of a warehouse** as a safety reference:
> detect new *and* removed geometry, attach calibrated uncertainty, and feed it to a planner that carries a
> hazard cost. We provide the novelty layer, uncertainty calibration and a hazard benchmark for a handless
> humanoid.

Defensible only as an integration+benchmark claim. Presenting it as "we do change detection" would be
scooped by GS-LTS.

### Claim C — "Handless interaction on an R1-class robot without force sensing" (from G6, hardware-gated)

> OIL and VOFA show handless pushing/transport on Digit and other humanoids; no system demonstrates it on an
> R1-class platform **without wrist force/torque sensing**, and none trains it in a splat-based simulator.
> We provide the object-kinematic interaction layer, contact estimation from proprioception+vision, and
> sim2real evidence.

Only viable if the R1 can push objects safely and if torque/sensing is sufficient. Check first.

---

## 3. Revised recommendation

**Do not build the thesis around G10/G12/G13/G14 (closed) or G8/G4/G7/G11/G15/G9 as main claims
(narrowed/weak).** Instead structure one thesis with one clear claim + one integration claim + benchmark:

```
Thesis: "Feed-forward mapping and hazard-aware navigation for a handless humanoid in industrial warehouses"

Chapter 1 (claim A): contact-anchored metric VGGT-SLAM 2.0 for the R1       [CLEAR residual]
Chapter 2 (benchmark): mapping+localisation+hazard benchmark, GT in sim     [clear artifact]
Chapter 3 (claim B):    prior-map novelty layer with uncertainty + planner   [NARROWED residual]
Chapter 4 (optional):   R1 handless pushing demo / sim2real (claim C)        [hardware-gated]
```

This gives one strong method claim, one benchmark artifact, one integration claim, and an optional
hardware flourish — and every chapter is defensible against the 2026 literature. It also survives if any
single chapter is scooped (the benchmark remains).

**Fallback if the R1 hardware is unavailable:** drop Chapter 4; the thesis is still complete (sim-only), and
Claim A + benchmark is the strongest sim-only combination.

---

## 4. What would change the verdicts

- If the R1 has no usable torque estimate → Claim C drops to sim-only (still a workshop contribution).
- If a legged group publishes contact-anchored VGGT-SLAM before us → Claim A weakens to benchmark + ablation;
  pivot to Claim B as the main claim.
- If GS-LTS releases a planning integration → Claim B closes; pivot to the benchmark + Claim A only.
- If NuRec/GaussGym-style splat rendering is impractical here → Chapter 3's evaluation stays depth-based
  (already works) and the fidelity study is dropped.

---

## 5. References (verification round)

- FoundationVIO, metric scale in feed-forward VIO — https://export.arxiv.org/pdf/2610.14951
- Contact-Anchored Proprioceptive Odometry — https://export.arxiv.org/abs/2602.17393
- ACRA humanoid state estimation — https://ssl.linklings.net/conferences/acra/acra2025_proceedings/views/includes/files/pap169s2.pdf
- LiDAR-VGGT — https://ieeexplore.ieee.org/abstract/document/11399876
- Scale anchoring VGGT-SLAM — https://export.arxiv.org/pdf/2611.06308
- Adaptive World Memory 3D foundation model — https://export.arxiv.org/pdf/2609.21502
- GS-LTS — http://export.arxiv.org/pdf/2503.17733
- DUAL-GS — https://ieeexplore.ieee.org/document/11805938
- ProDyG (NeurIPS 2025) — https://dl.acm.org/doi/10.52202/085713-2943
- SemSafe-3DGS — https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1
- Active NBV risk-averse planning — https://ieeexplore.ieee.org/abstract/document/11696764
- MAGICIAN (CVPR'26 oral, secondary source) — https://blog.csdn.net/Yong_Qi2015/article/details/160749185
- ATLAS Navigator — https://ongdexter.github.io/atlasnav/
- HumanoidVLN — https://export.arxiv.org/abs/2608.12860
- DPed-VLN — https://zenodo.org/records/22816620
- What the Elevation Map Cannot See — https://arxiv.org/abs/2610.07396
- TANGO — https://browse-export.arxiv.org/pdf/2609.09158
- COMPASS intralogistics benchmark + Cosmos — https://export.arxiv.org/pdf/2610.28432
- VL-N3RD-Bench — https://inventee-ai-center.github.io/vl-n3rd-bench/ (URL as listed: https://inventec-ai-center.github.io/vl-n3rd-bench/)
- UCL 3DGS digital twin (collision conversion) — https://discovery.ucl.ac.uk/id/eprint/10232226/
- TwinAligner — https://browse-export.arxiv.org/pdf/2512.19390
- OIL — https://ojs.aaai.org/index.php/AAAI-SS/article/view/43503
- VOFA — https://arxiv.org/html/2605.01518v2
- iGPC — https://arxiv-org.ezproxy.obspm.fr/html/2610.08120v1
- Humanoid-WAM — https://zenodo.org/records/20313417
- FIRM (fall-safety) — https://ieeexplore.ieee.org/document/11696999
- GaussGym — https://arxiv.org/abs/2510.15352
- LoGoPlanner (metric-aware VGGT navigation) — https://browse-export.arxiv.org/pdf/2512.19629
- MIT/Symbotic warehouse traffic — https://computing.mit.edu/news/ai-system-learns-to-keep-warehouse-robot-traffic-running-smoothly/
- Humanoid fleet orchestration — https://www.theengineer.co.uk/content/news/ai-framework-aims-to-orchestrate-humanoid-robot-fleets
- LiLoc lifelong localisation / multi-session GS (CCDC 2026) — https://ieeexplore.ieee.org/abstract/document/11127560/citations
- LLM-guided active mapping with Fisher information — https://xplorestaging.ieee.org/document/11445668
- HumanoidBench — https://www.roboticsproceedings.org/rss20/p061.pdf