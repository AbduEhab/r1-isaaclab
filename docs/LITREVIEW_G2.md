# Literature review — calibrated novelty for splat-map navigation (G2)

**Date:** 2026-11-26. **Scope:** the literature a thesis on "calibrated novelty/change layer over a prior
Gaussian-splat map, feeding a risk-aware planner for a handless humanoid" must review.
**Companion:** [THESIS_G2_DESIGN.md](THESIS_G2_DESIGN.md) (method), [THESIS_GAP_VERIFICATION_2026-11.md](THESIS_GAP_VERIFICATION_2026-11.md) (gap check).

**Search method:** ~45 queries across arXiv, CVF Open Access, IEEE Xplore, ACM DL, Wiley, ScienceDirect,
Google Scholar and NVIDIA developer docs; date window 2005–2026 (anchored on 2025–2026 frontier); keywords
combined from: *Gaussian splatting, 3DGS SLAM, neural rendering map, change detection, map maintenance,
long-term mapping, lifelong mapping, occupancy change, novelty detection, uncertainty quantification,
calibration, conformal prediction, chance-constrained planning, CVaR/AVaR, risk-aware navigation, safety
shield, humanoid navigation, hazard evaluation, VLN benchmark*. Inclusion: peer-reviewed or arXiv with
system or benchmark; exclusion: pure rendering-quality papers with no robot/map contribution.

---

## 1. Conceptual framing (what the review has to establish)

The thesis sits at the intersection of four literatures that are usually reviewed separately:

```
long-term mapping / maintenance   ─┐
change detection in 3D maps       ─┼─►  map staleness as a safety problem
splat maps + GS-SLAM              ─┤
risk-aware planning               ─┘
              ▲
      uncertainty calibration (conformal)  ── makes the arrow from detection to planning *guaranteed*
```

The review therefore has to show: (i) maps go stale and robots currently assume they don't, (ii) change
detection exists but is evaluated as a perception task, (iii) splat maps are becoming the robot map and have
their own uncertainty, (iv) calibration and risk-aware planning exist but were not connected to 3DGS map
novelty for a legged robot.

A useful taxonomy the review should introduce (used in the design):

| Change type | Meaning for navigation | Detected by |
|---|---|---|
| appearance | texture/lighting changes only | colour residual, calibration stress |
| structural addition | new object in previously free space | depth residual (positive) |
| removal | prior object gone (stale obstacle) | free-space carving (negative) |
| movement | object moved (addition + removal pair) | clustering + attribution |
| transient | forklift/human passing | temporal filtering, dynamic mask |
| semantic/elevation-subtle | spill, cable, decal | colour/VLM only — explicit failure mode |

---

## 2. Theme A — Long-term mapping and map maintenance

**Why it is needed:** establishes that staleness is a first-class robotics problem, and provides the
maintenance vocabulary (update vs re-map vs detect-and-ignore).

- **Systematic review of long-term localization and mapping** for mobile robots (PRISMA-based), *Robotica*
  2026 — the umbrella survey; establishes the field's open problems and the terminology
  ([Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002%2frob.22170)).
- **Classics:** Biber & Duckett, *Dynamic Maps for Long-Term Operation of Mobile Service Robots*, RSS 2005
  (occupancy with time scales); Konolige & Bowman, *Towards Lifelong Visual Maps*, IROS 2009; Krajník et
  al., *Fremen: Frequency Map Enhancement*, IEEE T-RO 2017 (frequency-domain long-term map representation).
  These are the standard "maps must change over time" anchors.
- **Map maintenance with change detection:** dual-layer octree change detection for long-term SLAM with
  online maintenance, IEEE 2025/26 ([link](https://ieeexplore.ieee.org/abstract/document/11153719));
  [Chamelion (2026)](https://ieeexplore.ieee.org/abstract/document/11397150), dual-head network for
  online change detection + long-term map maintenance in transient LiDAR environments.
- **Multi-session / lifelong localisation:** [LiLoc](https://ieeexplore.ieee.org/abstract/document/11127560/citations)
  (lifelong localisation with adaptive submap joining); multi-session Gaussian-splat reconstruction via LIVO
  trajectory alignment (CCDC 2026).
- **Stale-map autonomy:** *Beyond Waypoints: Semantic-Centric Autonomy with Unreliable Maps* (2026) argues
  metric maps are fragile to change and proposes learned semantic abstractions as the fix
  ([IEEE](https://ieeexplore.ieee.org/abstract/document/11696272)) — the main conceptual competitor to our
  "calibrate the geometric map" approach; review it as a contrast, not a baseline.

**Takeaway for G2:** maintenance is studied mostly for occupancy/LiDAR maps or via offline multi-session
alignment; splat-map *safety* under staleness is not covered.

---

## 3. Theme B — Change detection in 2D/3D/neural maps

**Why needed:** this is the closest technical neighbour and the source of baselines; the review must show
that detection is solved as perception, not as a safety input.

- **Classical probabilistic change detection in feature maps:** Nobre et al., *Online probabilistic change
  detection in feature-based maps* — the canonical formulation of map-change likelihood.
- **3D change detection / LiDAR:** Chamelion (2026, above); long-term LiDAR mapping pipelines
  (e.g., Berrio et al., *Long-term map maintenance pipeline for autonomous vehicles*, IEEE T-ITS).
- **Gaussian-splat change:** [GS-LTS (2025/26)](http://export.arxiv.org/pdf/2503.17733) — object-level change
  detection + Gaussian editing for long-term indoor service robots; the first splat-map change system.
- **Learned online change with geometric priors:** [Argos / Argos-SLAM (Oct 2026)](https://export.arxiv.org/pdf/2610.10181)
  — rich geometric priors, integrated with VGGT-SLAM, +42 % change IoU / +27.9 % F1 over baselines. **This is
  the current SOTA detector and the reason G2 cannot claim "change detection".**
- **Dynamic SLAM (changes during mapping, not after):** DUAL-GS (static 3DGS + dynamic 4DGS, LiDAR-inertial-camera,
  Nov 2026 [link](https://ieeexplore.ieee.org/document/11805938)); RGS-SLAM; ProDyG (NeurIPS 2025);
  4DGS-SLAM family — reviewed to argue that *mapping-time* dynamics ≠ *planning-time* map staleness.
- **Negative change:** removal detection appears rarely and is usually treated as map repair, not as a
  navigation-relevant free-space change. This is a genuine sub-gap (stale obstacles cause detours).

**Takeaway for G2:** detection is crowded; negative change and the *use* of detection for planning safety are
not. Detection work is cited as baselines, not as the claim.

---

## 4. Theme C — Gaussian splatting as the robot map

**Why needed:** the map representation and its own uncertainty sources (opacity, floaters, holes, scale).

- **Representation:** Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH
  2023); 2DGS (SIGGRAPH 2024) for surfaces; [gsplat](https://github.com/nerfstudio-project/gsplat) as the
  open CUDA rasteriser we use.
- **GS-SLAM:** MonoGS / *Gaussian Splatting SLAM* (Matsuki et al., CVPR 2024); Splat-SLAM; Gaussian-LIC;
  [uncertainty-driven GS RGB-D SLAM (Sep 2026)](https://ieeexplore.ieee.org/document/11677162);
  [BayesianGS-SLAM (2026)](https://ieeexplore.ieee.org/document/11727033) — predictive colour/depth
  uncertainty reused across tracking and mapping.
- **Feed-forward geometry:** VGGT and VGGT-SLAM 2.0 (Maggio & Carlone, 2026); [VGGT-GS SLAM
  (2026)](https://arxiv.org/html/2609.19628v1); scale anchoring ([Nov 2026](https://export.arxiv.org/pdf/2611.06308)).
- **Industrial real2sim:** [NVIDIA NuRec](https://developer.nvidia.cn/omniverse/nurec) (splat USD scenes
  rendered in Isaac Sim), GaussGym (3DGS renderer inside vectorized physics,
  [arXiv 2510.15352](https://arxiv.org/abs/2510.15352)).
- **Uncertainty of splats as maps:** opacity/scale are not calibrated uncertainty; BayesianGS-SLAM is the
  closest to treating splat uncertainty as a first-class quantity. This supports our choice of a *separate*
  calibrated novelty layer rather than trusting splat opacity.

**Takeaway for G2:** the representation is mature; what is missing is an operational uncertainty contract
between the splat map and the planner.

---

## 5. Theme D — Splat maps for navigation and risk-aware planning

**Why needed:** the planning half of G2, and the "safe in a splat map" lineage we extend.

- **Splat navigation:** [Splat-Nav (RA-L 2025)](https://ieeexplore.ieee.org/document/10930696) — planning +
  collision checking in Gaussian-splat maps, assuming the map is correct.
- **Risk-aware splat navigation:** [SemSafe-3DGS (Sep 2026)](https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1)
  — semantic risk-aware active navigation in *uncertain* splat maps (static-map assumption);
  [Active-NBV AVaR (2026)](https://ieeexplore.ieee.org/abstract/document/11696764) — tail-risk (AVaR) maps
  on an online-updated 3DGS field; MAGICIAN (CVPR'26) active mapping with imagined Gaussians.
- **Language/semantic splat maps:** [GLMap (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Multi-Scale_Gaussian-Language_Map_for_Zero-shot_Embodied_Navigation_and_Reasoning_CVPR_2026_paper.html),
  [ATLAS Navigator](https://ongdexter.github.io/atlasnav/), AOV-GS.
- **Risk measures and chance constraints:** Majumdar & Pavone, *How Should a Robot Assess Risk?* (ISRR 2017)
  for coherent risk measures; CVaR/chance-constrained navigation formulations
  ([2026 swarm risk-measure paper](https://export.arxiv.org/pdf/2609.12959)); [U-OBCA (Mar 2026)](https://browse-export.arxiv.org/abs/2603.04914)
  Wasserstein-robust chance constraints; learning-enabled barrier certificates (2026) for runtime safety.
- **Safety shields / runtime monitors:** [Humanoid-WAM (2026)](https://zenodo.org/records/20313417) includes
  balance-aware risk estimation and unsafe-action filtering; our shield is a lightweight variant.

**Takeaway for G2:** risk-aware planning in splat maps exists but consumes *map uncertainty*, not
*map-vs-reality change*; the planner integration of a calibrated novelty layer is the open piece.

---

## 6. Theme E — Uncertainty quantification and calibration

**Why needed:** the core claim is calibration, so this theme must be reviewed carefully and honestly.

- **UQ review:** *Uncertainty Quantification for Robotic Visual Perception: A Focused Review* (IEEE 2026) —
  covers MC dropout, deep ensembles, evidential learning and conformal prediction; the standard taxonomy
  ([link](https://ieeexplore.ieee.org/document/11608052)).
- **Calibration:** ECE/reliability diagrams for perception; "when a robot misidentifies an obstacle with high
  confidence, the consequence is a safety failure" framing from the review above.
- **Conformal prediction in robotics:** Lindemann et al., *Safe Planning in Dynamic Environments Using
  Conformal Prediction* (RA-L 2023); *Perceive with Confidence: Statistical Safety Assurances for Navigation
  with Learning-Based Perception* (IJRR 2026,
  [link](https://acm-stag.literatumonline.com/doi/10.1177/02783649251378151)); *Learnable Conformal Prediction
  for Safe and Efficient Robotics* (2026, [link](https://ieeexplore.ieee.org/document/11697249)); STL control
  synthesis with CP (Automatica 2026).
- **Depth/perception uncertainty for navigation:** uncertainty-aware depth attention and depth-uncertainty
  papers in monocular UAV navigation (2026) — useful for the depth-noise model `σ_d(z)`.

**Takeaway for G2:** the calibration machinery exists; applying split conformal to *novelty scores over a
splat map* and reporting a false-safe rate per metre is the specific, defensible contribution. The review
must also state the assumptions (exchangeability) and where they break (lighting shift, dynamics).

---

## 7. Theme F — Humanoid navigation and hazard evaluation

**Why needed:** the embodiment and the evaluation protocol; benchmarks are used as evaluation instruments,
not as thesis contributions.

- [What the Elevation Map Cannot See (Oct 2026)](https://arxiv.org/abs/2610.07396) — hazard taxonomy
  (elevation-subtle hazards) + command-execution deviation; G1 hardware validation.
- [HumanoidVLN (Aug 2026)](https://export.arxiv.org/abs/2608.12860) — physics-grounded VLN benchmark across
  humanoid embodiments; [TANGO (Sep 2026)](https://browse-export.arxiv.org/pdf/2609.09158) — whole-body
  VLA navigation in clutter; [DPed-VLN (Sep 2026)](https://zenodo.org/records/22816620) — socially compliant
  VLN in dynamic pedestrian environments.
- [HumanoidBench (RSS 2024)](https://www.roboticsproceedings.org/rss20/p061.pdf) — whole-body control task
  suite incl. 12 hand-free locomotion tasks; useful for reporting walking-side metrics.
- Humanoid locomotion for material handling in construction (2026,
  [link](https://ieeexplore.ieee.org/document/11448976)) — handless transport tasks.

**Takeaway for G2:** hazard evaluation for humanoids is emerging, but no benchmark links map staleness to
hazard contact for a handless legged robot; we use these metrics inside our own evaluation protocol.

---

## 8. Synthesis: what each theme covers vs what G2 claims

| Theme | Representative work | Detection | Calibration | Planner integration | Handless humanoid | Stale-splat map |
|---|---|---|---|---|---|---|
| Long-term mapping | surveys, Chamelion, octree maintenance | ● | ○ | ○ | ○ | ● |
| Change detection | GS-LTS, Argos | ● | ○ | ○ | ○ | ● |
| Splat maps / GS-SLAM | 3DGS, MonoGS, VGGT-SLAM, BayesianGS-SLAM | ○ | ◐ (SPLAT uncertainty) | ○ | ○ | ◐ |
| Splat navigation | Splat-Nav, SemSafe-3DGS, Active-NBV | ○ | ◐ | ● | ○ | ◐ |
| Calibration/UQ | CP lineage, UQ review | ○ | ● | ◐ | ○ | ○ |
| Risk planning | chance constraints, CVaR, U-OBCA, shields | ○ | ◐ | ● | ◐ | ○ |
| Humanoid navigation | Elevation-Map, HumanoidVLN, TANGO | ○ | ○ | ● | ● | ○ |
| **G2** | — | ● (baseline) | **●** | **●** | **●** | **●** |

● covered ◐ partial ○ not covered.

**Gap sentence for the thesis:** *No system calibrates novelty/change evidence from a prior Gaussian-splat
map into a false-safe guarantee and consumes it in a risk-aware planner for a handless legged humanoid; the
literature covers detection (GS-LTS, Argos), static-map risk planning (SemSafe-3DGS, Active-NBV) and
calibration (conformal line) separately.*

---

## 9. Annotated reading list (what each paper is *for* in the thesis)

| Paper | Role | Use |
|---|---|---|
| Agis et al., long-term localization/mapping review (Robotica 2026) | umbrella | intro + definitions |
| Biber & Duckett 2005; Konolige & Bowman 2009; Krajník 2017 | classics | staleness motivation |
| Chamelion 2026; octree maintenance 2025/26 | maintenance systems | related work + baseline concepts |
| GS-LTS | closest splat-change system | baseline / contrast |
| Argos / Argos-SLAM 2026 | SOTA detector | **main detection baseline** |
| DUAL-GS, RGS-SLAM, ProDyG | dynamic mapping | contrast (mapping-time vs planning-time) |
| 3DGS (Kerbl 2023), 2DGS, gsplat | representation/tools | method section |
| MonoGS/Splat-SLAM/Gaussian-LIC | GS-SLAM | related work |
| VGGT-SLAM 2.0, VGGT-GS | feed-forward SLAM | map source / integration with G1 |
| Uncertainty-driven GS-SLAM; BayesianGS-SLAM | splat uncertainty | motivates separate calibration |
| Splat-Nav, SemSafe-3DGS, Active-NBV, MAGICIAN | splat planning | planner baselines/contrast |
| Perceive with Confidence (IJRR 2026); learnable CP (2026); Lindemann RA-L 2023 | conformal | calibration method |
| UQ review (IEEE 2026); ECE literature | calibration metrics | evaluation |
| Majumdar & Pavone 2017; CVaR/chance-constrained papers; U-OBCA | risk measures | planner formulation |
| Humanoid-WAM; FIRM | safety filtering | shield design |
| Elevation-Map 2026; HumanoidVLN; TANGO; DPed-VLN; HumanoidBench | humanoid evaluation | metrics/scenarios |
| Beyond Waypoints 2026 | competing philosophy | discussion: calibrate geometry vs learn abstraction |

---

## 10. Threats to the claim (what the review must pre-empt)

1. **"Argos already does this."** No: Argos is a detector, evaluated with change IoU/F1; it provides no
   calibrated safety guarantee and no planner. Cite its numbers as the detection ceiling and differentiate on
   the guarantee + safety outcomes.
2. **"SemSafe-3DGS already plans in uncertain splats."** No: it assumes the map is otherwise current; we add
   the map-reality discrepancy as the uncertainty source.
3. **"Conformal prediction is standard."** Standard *machinery*, unstandardised *application* to splat
   novelty + per-metre false-safe metric + breakdown analysis under lighting/dynamics; the negative-change
   case and the humanoid safety evaluation are new.
4. **"Staleness is solved by re-mapping / lifelong SLAM."** Lifelong SLAM updates the map online but does not
   give a planning-time guarantee about what is *currently* wrong; the two are complementary (we can consume
   an updated map).
5. **"Learned semantic abstractions (Beyond Waypoints) are the better answer."** Discuss as an alternative
   research direction; our contribution is complementary (geometry-level guarantee) and cheaper to validate.

---

## 11. Open questions the review leaves for the thesis

- What is the right guarantee unit: per-cell false-safe probability, per-metre false-alarm rate, or
  per-episode collision risk? (Recommendation: per-metre primary, per-cell reported.)
- How do guarantees degrade when prior maps come from different sessions/qualities? (S8 in the design.)
- Does calibration transfer from simulation to a real splat map? (Only if real capture is possible.)
- Can splat opacity/uncertainty from BayesianGS-SLAM replace part of the calibration set? (Future work.)

---

## 12. References

- Systematic review, long-term localization/mapping — https://onlinelibrary.wiley.com/doi/full/10.1002%2frob.22170
- Chamelion (change detection, LiDAR map maintenance) — https://ieeexplore.ieee.org/abstract/document/11397150
- Octree online map maintenance — https://ieeexplore.ieee.org/abstract/document/11153719
- LiLoc — https://ieeexplore.ieee.org/abstract/document/11127560/citations
- Beyond Waypoints (unreliable maps, learned abstractions) — https://ieeexplore.ieee.org/abstract/document/11696272
- GS-LTS — http://export.arxiv.org/pdf/2503.17733
- Argos / Argos-SLAM — https://export.arxiv.org/pdf/2610.10181
- DUAL-GS — https://ieeexplore.ieee.org/document/11805938
- ProDyG (NeurIPS 2025) — https://dl.acm.org/doi/10.52202/085713-2943
- Uncertainty-driven 3DGS RGB-D SLAM — https://ieeexplore.ieee.org/document/11677162
- BayesianGS-SLAM — https://ieeexplore.ieee.org/document/11727033
- VGGT-GS SLAM — https://arxiv.org/html/2609.19628v1
- VGGT scale anchoring — https://export.arxiv.org/pdf/2611.06308
- gsplat — https://github.com/nerfstudio-project/gsplat
- NVIDIA NuRec — https://developer.nvidia.cn/omniverse/nurec
- GaussGym — https://arxiv.org/abs/2510.15352
- Splat-Nav — https://ieeexplore.ieee.org/document/10930696
- SemSafe-3DGS — https://arxiv-org.ezproxy.obspm.fr/html/2609.19330v1
- Active-NBV AVaR — https://ieeexplore.ieee.org/abstract/document/11696764
- GLMap (CVPR 2026) — https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Multi-Scale_Gaussian-Language_Map_for_Zero-shot_Embodied_Navigation_and_Reasoning_CVPR_2026_paper.html
- ATLAS Navigator — https://ongdexter.github.io/atlasnav/
- UQ for robotic visual perception review — https://ieeexplore.ieee.org/document/11608052
- Perceive with Confidence (IJRR 2026) — https://acm-stag.literatumonline.com/doi/10.1177/02783649251378151
- Learnable Conformal Prediction (2026) — https://ieeexplore.ieee.org/document/11697249
- CVaR / chance-constrained risk measures (2026) — https://export.arxiv.org/pdf/2609.12959
- U-OBCA (2026) — https://browse-export.arxiv.org/abs/2603.04914
- Humanoid-WAM — https://zenodo.org/records/20313417
- FIRM fall-safety — https://ieeexplore.ieee.org/document/11696999
- What the Elevation Map Cannot See — https://arxiv.org/abs/2610.07396
- HumanoidVLN — https://export.arxiv.org/abs/2608.12860
- TANGO — https://browse-export.arxiv.org/pdf/2609.09158
- DPed-VLN — https://zenodo.org/records/22816620
- HumanoidBench — https://www.roboticsproceedings.org/rss20/p061.pdf
- Humanoid material handling (2026) — https://ieeexplore.ieee.org/document/11448976

Bibliographic entries without links (classics to cite by hand):
Biber & Duckett, RSS 2005; Konolige & Bowman, IROS 2009; Krajník et al., IEEE T-RO 2017; Kerbl et al.,
SIGGRAPH 2023; Huang et al., 2DGS, SIGGRAPH 2024; Matsuki et al., Gaussian Splatting SLAM, CVPR 2024;
Majumdar & Pavone, ISRR 2017; Lindemann et al., RA-L 2023; Nobre et al., online probabilistic change
detection in feature-based maps.
