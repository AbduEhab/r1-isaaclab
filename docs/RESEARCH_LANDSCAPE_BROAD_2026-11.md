# Broader research landscape — late November 2026

Companion to [THESIS_GAPS_2026-11.md](THESIS_GAPS_2026-11.md). That document went deep on Gaussian
splatting, VSLAM, humanoid RL and distillation. This one zooms out: what else is moving in robotics and
embodied AI at the end of 2026, and which of those directions are realistic master-thesis gaps for *this*
workspace.

**Date:** 2026-11-26. **Method:** ~20 additional web searches + page fetches, filtered against the earlier
internal survey (`SOTA_RESEARCH_2026-11.md`, 2026-11-13) and our audited artefacts.

---

## 1. What was searched (transparency)

Round 1 (the deep dive in `THESIS_GAPS_2026-11.md`, ~24 queries) covered:
3DGS surveys/feed-forward/GS-SLAM, VGGT-SLAM 2.0/++ and scale anchoring, GS navigation (Splat-Nav),
language Gaussian maps, splat physics coupling, robot sim2real with splats (RoboGSim/GSWorld/RE3SIM),
NuRec in Isaac Sim, humanoid RL (perceptive locomotion, WBC, motion imitation), unitree R1 papers, VLA
landscape, privileged/DAgger distillation, multi-teacher on-policy distillation, dynamic/4D splats,
long-term GS maps, GS-SLAM benchmarks/datasets, and master-thesis project lists.

Round 2 (this document, ~20 queries) covered:
robot foundation models (GR00T, π0, VLA scaling), world models and generative simulation (Genie, Cosmos,
WAM taxonomies), humanoid hardware/commercial state, dexterous manipulation and hands (113-method survey),
loco-manipulation (Whole-Body UMI, PAMoR, AMBIT), simulators (Newton VBD, MuJoCo Warp, Isaac Lab 3.0),
benchmarks (ObjectNav-HM3D, GOAT, LIBERO/RoboCasa), tactile/event sensing, social navigation, safety
verification (barrier certificates), edge deployment (Jetson Thor), warehouse multi-robot coordination
(RMFS), and contribution-type norms for robotics theses.

---

## 2. The 2026 landscape in nine areas

### 2.1 Robot foundation models / VLA

NVIDIA's GTC 2026 (March) made the full-stack bet explicit: **Isaac GR00T N** open VLA models that are
post-trainable per embodiment, a **Physical AI Data Factory** (Cosmos world models + OSMO orchestration to
turn one real scenario into thousands of synthetic ones), **Isaac Teleop** for XR demonstration capture,
**Isaac Lab 3.0**, and **Jetson Thor/Orin** for edge inference with **cuVSLAM** on embedded
([GTC 2026 report](https://blockchain.news/news/nvidia-isaac-gr00t-robotics-platform-gtc-2026)). The
GR00T X-Embodiment dataset reportedly passed 10M downloads — data, not architecture, is the bottleneck
narrative.

Humanoid-specific VLAs are opening up: **Psi0** (RSS'26, "humanoid VLA towards universal humanoid
intelligence") open-sources nine real-world tasks with data collection, fine-tuning and deployment
([repo](https://github.com/physical-superintelligence-lab/Psi0)). Fine-tuning a humanoid VLA on one A5000
remains marginal, but *evaluation* and *small-scale post-training* studies are becoming feasible.

**Thesis angle:** small-compute studies of post-training/adaptation (LoRA, data mixtures, sim-to-real for
VLA) are now publishable as workshops; training a VLA from scratch is not.

### 2.2 World models and generative simulation

The world-model literature is consolidating into taxonomies of **world models vs world-action models**
(200+ papers in one 2026 survey; Genie-style generative environments, Cosmos post-trained into driving
simulators and robot policies). For a robotics thesis the useful framing is *simulation as a learned
function*: generate observations/rollouts instead of rendering them.

**Thesis angle:** at master scale, the tractable version is **evaluation of a world model as a
data-generator/simulator for one task** (e.g., does Cosmos-style synthetic video improve a navigation
policy?), not training a world model.

### 2.3 Humanoid hardware and commercial state

Industry context (mid-2026 reporting): Boston Dynamics began commercial Atlas shipments in 2026; Unitree's
smaller **R1 Air starts around $4.9k**; Tesla's Optimus is not purchasable as of mid-2026
([industry report](http://www.china.org.cn/2026-08/11/content_118642980.shtml)). This matters for thesis
risk: the R1 is one of the few *affordable* humanoids with an active research community, so experiments on
it are representative and reproducible for other groups.

### 2.4 Dexterous manipulation and hands

A 2026 survey catalogues **113 learning-based dexterous-manipulation methods** with their hands, simulators
and reporting practices — and notes weak reward/code reporting
([survey](https://scixplorer.org/abs/2026arXiv261033486A/abstract)). Tactile and event-based sensing are
entering manipulation pipelines (e.g., multi-resolution tactile imitation with GelSight/Evetac,
[2026](https://browse-export.arxiv.org/abs/2606.06281)). The R1's default hands are simple; dexterous work
is hardware-limited here.

### 2.5 Loco-manipulation (whole-body)

Whole-body humanoid skills are moving from locomotion to manipulation-while-walking: Whole-Body UMI and
PAMoR (Aug 2026) transfer human motion to humanoid loco-manipulation, and AMBIT (Sep 2026) demonstrates
bimanual tracking on a **Unitree R1**. This is directly adjacent to our stack (we have walking + arms
unused), and it is compute-hungry but sim-feasible.

### 2.6 Simulators and physics

Newton (GPU) added contact-rich manipulation and locomotion capabilities, including VBD solver for cables
and two-way coupling with MuJoCo Warp
([NVIDIA blog 2026](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics)).
Our Isaac Lab 3.0 stack runs on Newton — so the local infrastructure is aligned with where the field is
going, which lowers the cost of any physics/contact-related thesis.

### 2.7 Neural rendering for sim2real

The real2sim story is now productised: **NuRec is GA**, ingests real sensor data, outputs interactive USD
scenes rendered with gsplat inside Isaac Sim/CARLA/AlpaSim, with Cosmos-based Fixer for artifacts and Asset
Harvester for object removal
([NuRec](https://developer.nvidia.cn/omniverse/nurec),
[GTC 2026](https://blockchain.news/news/nvidia-isaac-gr00t-robotics-platform-gtc-2026)). Independent
coverage frames photorealistic digital twins via NeRF/3DGS as the direction
([Lacuna](https://lacuna.tiptreesystems.com/direction/photorealistic-simulation-via-neural-rendering/txn_155ae1270c604822b1ec45012b26632b)).

### 2.8 Safety, verification, deployment

Learned policies are beginning to get verifiable safety wrappers: learning-enabled barrier certificates
with LP verification (2026), and NVIDIA's **Halos** full-stack safety system. For humanoids in shared
spaces this is an active gap: our deployed R1 runner has safety ramps and watchdogs, but no formal
guarantees.

### 2.9 Evaluation and benchmarks

Simulation benchmarks (LIBERO, RoboCasa, ObjectNav-HM3D, GOAT) dominate; real-world humanoid benchmarks
remain rare. Warehouse environments and multi-robot fleets (RMFS coordination, 2026) are studied in
operations research and robotics separately. The recurring complaint in surveys is **evaluation and
reporting quality**, which is a legitimate thesis contribution when done rigorously.

---

## 3. New gap candidates from the broader search

Beyond A1–A12 in the companion document:

- **B1. Small-compute post-training of a humanoid VLA for one warehouse skill.** Take GR00T N/Psi0-class
  open weights, post-train for pick-and-place or navigation in our warehouse, evaluate data efficiency
  (splat-generated synthetic data vs teleop). Gap: data-efficiency studies for humanoid VLA adaptation are
  thin; expensive at full scale but a *narrow skill* is feasible.
- **B2. World models as synthetic data generators for humanoid locomotion/navigation.** Evaluate whether
  video/world-model rollouts improve policy robustness vs standard DR. Risks: compute, evaluation of
  generated data quality.
- **B3. Loco-manipulation on the R1**: walking + arm reaching/pushing with whole-body control, distilled
  from scripted/teleop demonstrations. We have the walker and the Isaac asset; arms are currently unused.
- **B4. Safety wrapper for a deployed learned walker**: barrier-certificate or runtime-monitor layer on top
  of our 83-obs policy, with evidence in sim (falls prevented) and on hardware (if available).
- **B8. Edge deployment study**: distilled policy + VGGT-scale perception on Jetson Thor/Orin — which
  model sizes/rates are realistic for humanoid navigation? Hardware dependent.

---

## 4. Revised ranking (method-only, easiest first)

Ease 10 = start today with existing workspace; publication potential for a master's thesis in brackets.

| # | Gap | Ease | Effort | Hardware | Type |
|---|---|---|---|---|---|
| 1 | **A1** contact-anchored metric scale for legged VGGT-SLAM | 9 | 8?10 wk | sim | method (RA-L/IROS) |
| 2 | **A2** splat novelty/change with uncertainty + planner | 8 | 10?12 wk | optional | method (ICRA) |
| 3 | **A3** auto-collider generation from splats | 7 | 8?10 wk | sim | method/tool |
| 4 | **A5** visual-vs-collider fidelity ablation | 6 | 14?20 wk | robot ideal | empirical study |
| 5 | **B4** safety wrapper for the deployed walker | 6 | 10?12 wk | optional | method+eval |
| 6 | **B3** R1 loco-manipulation (walk + arm contacts) | 6 | 10?12 wk | sim (+robot) | system |
| 7 | **A4** photoreal real2sim locomotion sim2real | 5 | 16?20 wk | robot needed | system+study (high impact) |
| 8 | **A6** planning under online-SLAM map uncertainty | 5 | 12?14 wk | sim | method (offline-splat version closed) |
| 9 | **A7** VGGT distillation into onboard control | 4 | 14?16 wk | sim | method |
| 10 | **A11** multi-session map maintenance | 4 | 18+ wk | sim | research-grade |
| 11 | **B1** humanoid VLA post-training for one skill (navigation only, no hands) | 4 | 14?16 wk | GPU-heavy | adaptation study |
| 12 | **A8** dynamic 4D maps (predictive-navigation use only; mapping closed) | 3 | 18+ wk | sim | research-grade |

Removed as benchmark-only (direction): warehouse benchmark, fleet traffic benchmark, distillation
benchmark, social-navigation benchmark, VLA-evaluation benchmark. Removed as closed: planning under
offline-splat uncertainty, dynamic 4D mapping, VLA navigation evaluation.

## 5. What this means for our workspace

- **Cheapest high-value theses** remain perception/mapping fusion (A1, A2); both are methods we can start
  today with existing infrastructure.
- **Best infrastructure fit with the industry direction**: splat-based simulation (NuRec GA, Isaac Lab 3.0,
  Newton) — the local stack is exactly the reference stack of the field, so a real2sim thesis (A3/A4/A5)
  will be reproducible by others and aligned with NVIDIA's public roadmap.
- **Highest risk**: VLA and world-model theses (compute-limited on a 16 GB laptop GPU); hardware-limited
  ones (B4/B8/B7) depend on R1 access and on the robot's hand configuration, which we have not verified.
- **Positioning**: a thesis built on one strong method claim (A1 or A2) with a rigorous evaluation protocol is
  the safest path; the real2sim system (A3+A4) is the highest-impact path if hardware and time allow.

---

## 6. Recommended sequencing for a master's

1. **Month 1-2:** build the evaluation protocol on simulator ground truth (cheap, reusable, de-risks everything after).
2. **Month 2–5:** one method contribution on top (A1 scale anchoring or A2 novelty layer).
3. **Month 5–7:** extension toward deployment or sim2real (A4/A5 if hardware, otherwise A6/A10).
4. **Month 7-8:** writing + open-sourcing the code and evaluation scripts.

This sequencing gives two publishable artefacts even if the ambitious extension fails.

---

## 7. References (added in this round)

- NVIDIA GTC 2026 full-stack robotics report — https://blockchain.news/news/nvidia-isaac-gr00t-robotics-platform-gtc-2026
- NVIDIA NuRec (GA, Isaac Sim integration) — https://developer.nvidia.cn/omniverse/nurec
- NVIDIA Newton contact-rich manipulation/locomotion — https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics
- NVIDIA Jetson Thor / edge AI — https://blogs.nvidia.com/blog/jetson-thor-robotics-edge-ai-agent/
- Psi0 humanoid VLA (RSS'26) — https://github.com/physical-superintelligence-lab/Psi0
- World models / world-action models taxonomy (2026 survey) — https://zenodo.org/records/20130369
- World models critical survey (2026) — https://zenodo.org/records/23409100
- Learning dexterous manipulation survey (113 methods, 2026) — https://scixplorer.org/abs/2026arXiv261033486A/abstract
- Multi-resolution tactile imitation (2026) — https://browse-export.arxiv.org/abs/2606.06281
- Humanoid industry state mid-2026 — http://www.china.org.cn/2026-08/11/content_118642980.shtml
- Photorealistic simulation via neural rendering — https://lacuna.tiptreesystems.com/direction/photorealistic-simulation-via-neural-rendering/txn_155ae1270c604822b1ec45012b26632b
- Learning-enabled barrier certificates (2026) — https://link.springer.com/chapter/10.1007/978-3-032-26204-2_22
- Warehouse multi-phase coordination (2026) — https://www.nature.com/articles/s41598-026-63868-3_reference.pdf
- ObjectNav-HM3D / embodied navigation benchmark numbers — https://huggingface.co/buckets/huggingchat/papers-content/tree/2404/2404.06609.md
- Physical AI 2027 outlook (data, models, hardware) — https://vmblog.com/bylines/will-physical-ai-have-a-breakthrough-in-2027-itll-come-down-to-data-models-and-hardware/