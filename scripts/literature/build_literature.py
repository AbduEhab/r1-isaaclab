"""Build a ~300-paper literature corpus by citation chaining ("children of children").

For each curated seed paper (classic robotics/SLAM -> 2026 frontier) this walks both directions:
  * ancestors: the seed's references (the papers it is built on),
  * children: papers citing the seed,
  * grandchildren: papers citing the most cited children.
Everything is scored, deduplicated, theme-tagged and stored under C:\\per\\thesis\\literature
(metadata CSV/JSON/BibTeX + open-access PDFs where available).

Run:  C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\literature\\build_literature.py
Resumable: raw API pulls are cached in literature/_cache.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = r"C:\per\thesis\literature"
CACHE = os.path.join(ROOT, "_cache")
PDFS = os.path.join(ROOT, "pdfs")
MAILTO = "research@example.com"
BASE = "https://api.openalex.org"
SELECT = ("id,doi,display_name,publication_year,cited_by_count,primary_location,"
          "authorships,open_access,best_oa_location,type,referenced_works,language")

# ---------------------------------------------------------------- curated seeds
SEEDS = {
    "foundations_planning": [
        "Real-time obstacle avoidance for manipulators and mobile robots",
        "A formal basis for the heuristic determination of minimum cost paths",
        "Rapidly-exploring random trees: A new tool for path planning",
        "A robust layered control system for a mobile robot",
        "Motor schema based navigation for a mobile robot",
        "Monte Carlo localization for mobile robots",
        "A probabilistic approach to concurrent mapping and localization for mobile robots",
        "Directed sonar sensing for mobile robot navigation",
        "Simultaneous localization and mapping: part I",
        "iSAM: Incremental smoothing and mapping",
        "g2o: A general framework for graph optimization",
        "Real-time loop closure in 2D LIDAR SLAM",
        "LSD-SLAM: Large-scale direct monocular SLAM",
        "SVO: Fast semi-direct monocular visual odometry",
        "POMDPs for robotic tasks",
        "Coastal navigation with mobile robots",
    ],
    "occupancy_and_classic_slam": [
        "High resolution maps from wide angle sonar",
        "Estimating uncertain spatial relationships in robotics",
        "Representation and estimation of spatial uncertainty",
        "Directed sonar sensing for mobile robot navigation",
        "Learning metric-topological maps for indoor mobile robot navigation",
        "An efficient probabilistic 3D mapping algorithm for large-scale environments",
        "FastSLAM: A factored solution to the simultaneous localization and mapping problem",
        "ORB-SLAM: A Versatile and Accurate Monocular SLAM System",
        "ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras",
        "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM",
        "Direct sparse odometry",
        "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator",
    ],
    "map_maintenance_longterm": [
        "Dynamic maps for long-term operation of mobile service robots",
        "Towards lifelong visual maps",
        "Fremen: Frequency map enhancement for long-term mobile robot autonomy in changing environments",
        "Dynamic pose graph SLAM: Long-term mapping in low dynamic environments",
        "Towards lifelong feature-based mapping in semi-static environments",
        "Map building with mobile robots in dynamic environments",
        "Mobile robot simultaneous localization and mapping in dynamic environments",
        "A systematic literature review on long-term localization and mapping for mobile robots",
        "Chamelion: Reliable change detection for long-term LiDAR mapping in transient environments",
        "A long-term SLAM system with graph-structured maps and online maintenance using dual-layer octree change detection",
        "LiLoc: Lifelong localization using adaptive submap joining and egocentric factor graph",
        "GS-LTS: 3D Gaussian splatting based adaptive modeling for long-term service robots",
    ],
    "change_detection": [
        "Online probabilistic change detection in feature-based maps",
        "Change detection in 3D point clouds",
        "Argos: Adapt rich geometric priors for generalizable online scene-change-detection",
        "Beyond waypoints: Semantic-centric autonomy with unreliable maps through learned abstractions",
        "Multi-session Gaussian splatting reconstruction via LIVO trajectory alignment",
        "Dynamic map maintenance for long-term robot operation",
    ],
    "gaussian_splatting": [
        "NeRF: Representing scenes as neural radiance fields for view synthesis",
        "Instant neural graphics primitives with a multiresolution hash encoding",
        "3D Gaussian Splatting for real-time radiance field rendering",
        "2D Gaussian splatting for geometrically accurate radiance fields",
        "Mip-Splatting: Alias-free 3D Gaussian splatting",
        "3DGS: 3D Gaussian splatting for real-time rendering",
    ],
    "gs_slam_feedforward": [
        "SplaTAM: Splat, track and map 3D Gaussians for dense RGB-D SLAM",
        "Gaussian Splatting SLAM",
        "Gaussian-LIC: Photo-realistic LiDAR-inertial-camera SLAM with 3D Gaussian splatting",
        "Splat-SLAM: Globally optimized RGB-only SLAM with 3D Gaussians",
        "VGGT: Visual geometry grounded transformer",
        "DUSt3R: Geometric 3D vision made easy",
        "MASt3R: Matching and stereo from DUSt3R",
        "VGGT-SLAM 2.0: Real-time dense feed-forward scene reconstruction",
        "Uncertainty-driven 3-D Gaussian splatting for robust real-time RGB-D SLAM",
        "BayesianGS-SLAM: Uncertainty-aware neural rendering SLAM via probabilistic formulation",
        "DUAL-GS: Dynamic-uncertainty-aware Gaussian SLAM for mobile robots with LiDAR-inertial-camera fusion",
        "ProDyG: Progressive dynamic Gaussian splatting for online reconstruction",
    ],
    "splat_navigation": [
        "Splat-Nav: Safe real-time robot navigation in Gaussian splatting maps",
        "SemSafe-3DGS: Semantic risk-aware active navigation in uncertain 3D Gaussian splatting maps",
        "Active next-best-view optimization for risk-averse path planning",
        "Multi-scale Gaussian-language map for zero-shot embodied navigation and reasoning",
        "ATLAS Navigator: Active task-driven language-embedded Gaussian splatting",
        "AOV-GS: Active open-vocabulary semantic mapping with 3D Gaussian splatting",
    ],
    "calibration_conformal": [
        "On calibration of modern neural networks",
        "A tutorial on conformal prediction",
        "Safe planning in dynamic environments using conformal prediction",
        "Perceive with confidence: Statistical safety assurances for navigation with learning-based perception",
        "Learnable conformal prediction for safe and efficient robotics under perception and planning uncertainties",
        "Sample-efficient safety assurances using conformal prediction",
        "Uncertainty quantification for robotic visual perception: A focused review",
    ],
    "risk_aware_planning_safety": [
        "How should a robot assess risk? Towards an axiomatic theory of risk in robotics",
        "Chance-constrained optimal path planning with obstacles",
        "Risk-sensitive finite-horizon Markov decision processes",
        "On the risk-sensitive control of Markov decision processes",
        "Convex optimization of chance constrained Markov decision processes",
        "Risk-sensitive safety analysis using conditional value-at-risk",
        "Barrier certificates for nonlinear model validation",
        "Safe reinforcement learning via shielding",
        "Uncertainty-aware optimization-based collision avoidance via Wasserstein distributionally robust chance constraints",
        "Smooth convex optimization for chance-constrained motion planning",
        "Learning-enabled barrier certificates",
    ],
    "humanoid_navigation_embodied": [
        "What the elevation map cannot see: Semantic-aware locomotion and execution-aware navigation for humanoid robot",
        "HumanoidVLN: A physics-grounded simulator and benchmark for vision-language navigation across diverse humanoid embodiments",
        "TANGO: Humanoid navigation in cluttered environments with a whole-body vision-language-action model",
        "DPed-VLN: A benchmark for socially compliant vision-and-language navigation in dynamic pedestrian environments",
        "HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation",
        "Socially aware motion planning with deep reinforcement learning",
        "Human-aware robot navigation using social force model",
    ],
    "legged_rl_distillation": [
        "Learning agile and dynamic motor skills for legged robots",
        "Learning quadrupedal locomotion over challenging terrain",
        "Learning to walk in minutes using massively parallel deep reinforcement learning",
        "Rapid motor adaptation for legged robots",
        "Learning human-to-humanoid real-time whole-body teleoperation",
        "OmniH2O: Universal and dexterous human-to-humanoid whole-body teleoperation and learning",
        "ASAP: Aligning simulation and real-world physics for learning agile humanoid whole-body skills",
        "Expressive whole-body control for humanoid robots",
        "Unified humanoid fall-safety policy from a few demonstrations",
        "SOLO: Stable omni-terrain long-horizon perceptive humanoid locomotion",
        "Generalizing agentic robotics via multi-teacher on-policy distillation",
        "Robo-MOPD: Multi-teacher on-policy distillation for robot planners",
        "E2DRL: End-to-end differentiable reinforcement learning for 3D obstacle avoidance in quadruped robots",
    ],
    "splat_sim_real2sim": [
        "GaussGym: An open-source real-to-sim framework for learning locomotion from pixels",
        "RoboGSim: A real2sim2real robotic Gaussian splatting simulator",
        "GSWorld: Closed-loop photorealistic simulation suite for robot manipulation",
        "RE3SIM: Generating high-fidelity simulation data via 3D-photorealistic real-to-sim for robotic manipulation",
        "PhysGaussian: Physics-integrated 3D Gaussians for generative dynamics",
        "Mesh-GS digital twin physics-aware real2sim2real",
    ],
}

# how many descendants/ancestors to pull per seed
ANCESTORS_PER_SEED = 40
CHILDREN_PER_SEED = 50
GRANDCHILD_PARENTS = 12          # per theme, take this many children and pull their citers
GRANDCHILDREN_PER_PARENT = 25
TARGET_TOTAL = 320


def get_json(url: str, retries: int = 6):
    import urllib.error
    for attempt in range(retries):
        time.sleep(0.35 + 0.1 * attempt)  # polite pacing, extra delay on retries
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"lit-review ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(6.0 * (attempt + 1))
            elif attempt == retries - 1:
                print(f"  ! HTTP {exc.code} :: {url[:110]}")
                return None
            else:
                time.sleep(2.0 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  ! request failed: {exc} :: {url[:110]}")
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def cache_path(key: str) -> str:
    return os.path.join(CACHE, re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:180] + ".json")


def cached(key: str, fn):
    p = cache_path(key)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    data = fn()
    if data is not None:
        json.dump(data, open(p, "w", encoding="utf-8"))
    return data


def work_fields(w) -> dict:
    src = (w.get("primary_location") or {}).get("source") or {}
    authors = []
    for a in (w.get("authorships") or [])[:6]:
        nm = (a.get("author") or {}).get("display_name")
        if nm:
            authors.append(nm)
    oa = w.get("open_access") or {}
    best = w.get("best_oa_location") or {}
    return {
        "id": w.get("id", "").split("/")[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": w.get("display_name") or "",
        "year": w.get("publication_year") or 0,
        "venue": src.get("display_name") or "",
        "citations": w.get("cited_by_count") or 0,
        "authors": authors,
        "oa_url": best.get("pdf_url") or oa.get("oa_url") or "",
        "type": w.get("type") or "",
        "refs": len(w.get("referenced_works") or []),
    }


def search_seed(title: str):
    url = (f"{BASE}/works?search={urllib.parse.quote(title)}&per-page=1"
           f"&mailto={MAILTO}&select={SELECT}")
    data = cached("search_" + title, lambda: get_json(url))
    if not data or not data.get("results"):
        return None
    return data["results"][0]


def batch_by_ids(ids: list[str]):
    out = {}
    ids = [i for i in ids if i]
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        flt = "|".join(chunk)
        url = f"{BASE}/works?filter=openalex_id:{flt}&per-page=40&mailto={MAILTO}&select={SELECT}"
        data = cached("batch_" + chunk[0] + f"_{len(chunk)}", lambda u=url: get_json(u))
        if data:
            for w in data.get("results", []):
                out[w["id"].split("/")[-1]] = work_fields(w)
        time.sleep(0.12)
    return out


def citers(work_id: str, limit: int):
    url = (f"{BASE}/works?filter=cites:{work_id}&sort=cited_by_count:desc&per-page={limit}"
           f"&mailto={MAILTO}&select={SELECT}")
    data = cached("citers_" + work_id + f"_{limit}", lambda: get_json(url))
    if not data:
        return []
    return [work_fields(w) for w in data.get("results", [])]


def main() -> int:
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(PDFS, exist_ok=True)

    provenance: dict[str, list[str]] = {}
    themes: dict[str, set] = {}
    meta: dict[str, dict] = {}

    def add(fields, theme, tag):
        wid = fields["id"]
        if not wid:
            return
        if wid not in meta:
            meta[wid] = fields
        provenance.setdefault(wid, [])
        if tag not in provenance[wid]:
            provenance[wid].append(tag)
        themes.setdefault(wid, set()).add(theme)

    print(f"[lit] resolving {sum(len(v) for v in SEEDS.values())} seeds ...")
    seed_ids = {}
    for theme, titles in SEEDS.items():
        for title in titles:
            w = search_seed(title)
            if not w:
                print(f"  ! seed not found: {title[:70]}")
                continue
            f = work_fields(w)
            add(f, theme, "seed")
            seed_ids.setdefault(theme, []).append(f["id"])
            print(f"  seed [{theme}] {f['year']} {f['title'][:60]} (cites {f['citations']})")

    # ancestors (references of the seeds) and children (citers)
    print("[lit] walking ancestors and children ...")
    ref_ids = []
    children_by_theme: dict[str, list[dict]] = {}
    for theme, wids in seed_ids.items():
        for wid in wids:
            wjson = cached("work_" + wid, lambda u=f"{BASE}/works/{wid}?mailto={MAILTO}&select={SELECT}": get_json(u))
            if wjson:
                refs = (wjson.get("referenced_works") or [])[:ANCESTORS_PER_SEED]
                ref_ids.extend(r.split("/")[-1] for r in refs)
            kids = citers(wid, CHILDREN_PER_SEED)
            children_by_theme.setdefault(theme, []).extend(kids)
            for k in kids:
                add(k, theme, "child")
    print(f"[lit] {len(ref_ids)} ancestor ids, {sum(len(v) for v in children_by_theme.values())} children")

    ancestors = batch_by_ids(ref_ids)
    for wid, f in ancestors.items():
        add(f, "ancestor", "ancestor")

    # grandchildren: citers of the most cited children of each theme
    print("[lit] walking children-of-children ...")
    for theme, kids in children_by_theme.items():
        kids_sorted = sorted({k["id"]: k for k in kids}.values(), key=lambda x: -x["citations"])
        for k in kids_sorted[:GRANDCHILD_PARENTS]:
            for g in citers(k["id"], GRANDCHILDREN_PER_PARENT):
                add(g, theme, "grandchild")

    # ---------------------------------------------------------------- selection
    def theme_of(wid):
        ts = [t for t in themes.get(wid, set()) if t != "ancestor"]
        return ts[0] if ts else "ancestor"

    ROBOTICS = re.compile(r"robot|slam|locali[sz]ation|mapping|navigation|humanoid|legged|manipulat|embodied|"
                          r"gaussian splat|splat|radiance|neural rendering|lidar|stereo|visual[- ]inertial|"
                          r"visual odometry|odometry|pose estimation|scene|perception|autonomous|relocali", re.I)
    NOT_ROBOTICS = re.compile(r"portfolio|taxonomic|seismic|quantum|finance|economic|medical|climat|astrophys|"
                              r"genetic algorithm|protein|market", re.I)

    def is_robotics(wid):
        m = meta[wid]
        text = (m["title"] + " " + (m["venue"] or ""))[:300]
        return bool(ROBOTICS.search(text)) and not bool(NOT_ROBOTICS.search(text))

    THEME_QUOTAS = {
        "gaussian_splatting": 26, "gs_slam_feedforward": 28, "splat_navigation": 20,
        "change_detection": 18, "map_maintenance_longterm": 20, "calibration_conformal": 24,
        "risk_aware_planning_safety": 20, "humanoid_navigation_embodied": 14,
        "legged_rl_distillation": 20, "occupancy_and_classic_slam": 20, "splat_sim_real2sim": 12,
        "foundations_planning": 18, "ancestor": 50,
    }

    selected = []
    seen = set()

    def take(wid):
        if wid in seen or wid not in meta:
            return False
        seen.add(wid)
        selected.append(wid)
        return True

    # seeds first
    for wid in meta:
        if "seed" in provenance.get(wid, []):
            take(wid)

    # foundational ancestors, but only robotics-relevant ones
    anc = [w for w in meta if "ancestor" in provenance.get(w, []) and w not in seen and is_robotics(w)]
    for w in sorted(anc, key=lambda x: -meta[x]["citations"])[:THEME_QUOTAS["ancestor"]]:
        take(w)

    # per-theme quotas: recent frontier first, then most cited
    pool = [w for w in meta if w not in seen]
    for theme, quota in THEME_QUOTAS.items():
        if theme == "ancestor":
            continue
        have = sum(1 for w in selected if theme in themes.get(w, set()))
        if have >= quota:
            continue
        cands = [w for w in pool if theme in themes.get(w, set()) and w not in seen]
        recent = sorted([w for w in cands if meta[w]["year"] >= 2023], key=lambda w: -meta[w]["citations"])
        older = sorted([w for w in cands if meta[w]["year"] < 2023], key=lambda w: -meta[w]["citations"])
        for w in recent + older:
            if have >= quota or len(selected) >= TARGET_TOTAL:
                break
            if take(w):
                have += 1

    # fill the remainder, but never inflate one theme beyond 1.6x its quota
    def theme_count(t):
        return sum(1 for w in selected if t in themes.get(w, set()))

    rest = [w for w in meta if w not in seen and is_robotics(w)]
    rest.sort(key=lambda w: (-(meta[w]["year"] >= 2023), -meta[w]["citations"]))
    for w in rest:
        if len(selected) >= TARGET_TOTAL:
            break
        t = theme_of(w)
        if t in THEME_QUOTAS and theme_count(t) >= int(THEME_QUOTAS[t] * 1.6):
            continue
        take(w)

    selected = [(wid, provenance[wid][-1] if provenance.get(wid) else "child") for wid in selected]
    print(f"[lit] selected {len(selected)} papers")

    # ---------------------------------------------------------------- write metadata
    rows = []
    for idx, (wid, why) in enumerate(selected, 1):
        m = meta[wid]
        rows.append({"nr": idx, "theme": theme_of(wid), "provenance": "+".join(provenance[wid]),
                     "why": why, **m, "pdf": ""})

    with open(os.path.join(ROOT, "papers.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(os.path.join(ROOT, "papers.json"), "w", encoding="utf-8"), indent=1)

    # bibtex
    def slug(s, n=40):
        return re.sub(r"[^A-Za-z0-9]+", "_", s)[:n].strip("_")

    with open(os.path.join(ROOT, "bibtex.bib"), "w", encoding="utf-8") as f:
        for r in rows:
            key = f"{slug(r['authors'][0].split()[-1]) if r['authors'] else 'anon'}{r['year']}_{slug(r['title'], 25)}"
            authors = " and ".join(r["authors"]) if r["authors"] else "Unknown"
            f.write(f"@article{{{key},\n  title = {{{r['title']}}},\n  author = {{{authors}}},\n"
                    f"  year = {{{r['year']}}},\n  journal = {{{r['venue']}}},\n"
                    f"  doi = {{{r['doi']}}},\n  note = {{{r['theme']}; {r['provenance']}}}\n}}\n\n")

    # themed markdown indexes
    by_theme = os.path.join(ROOT, "by_theme")
    os.makedirs(by_theme, exist_ok=True)
    for theme in sorted(set(r["theme"] for r in rows)):
        items = [r for r in rows if r["theme"] == theme]
        with open(os.path.join(by_theme, theme + ".md"), "w", encoding="utf-8") as f:
            f.write(f"# {theme} ({len(items)} papers)\n\n")
            for r in sorted(items, key=lambda x: (-x["citations"], x["year"])):
                f.write(f"- {r['year']} — **{r['title']}** ({r['venue'] or 'preprint'}), "
                        f"{r['citations']} citations, {r['provenance']}\n")

    with open(os.path.join(ROOT, "_selected.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("[lit] metadata written (papers.csv / papers.json / bibtex.bib / by_theme/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
