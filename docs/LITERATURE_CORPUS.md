# Literature corpus (citation-chained, Nov 2026)

**Location:** `C:\per\thesis\literature` (metadata + PDFs) — built for the thesis background chapter and
the gap analysis.

## What it is

**300 papers** spanning 1968 → 2026, assembled by citation chaining rather than keyword lists:

- **~105 curated seeds** across 13 themes: classical robotics/planning/SLAM (potential fields, A*, RRT,
  subsumption, motor schemas, MCL, occupancy mapping, FastSLAM, ORB-SLAM family), long-term mapping /
  map maintenance, change detection, Gaussian splatting, GS-SLAM & feed-forward reconstruction, splat-map
  navigation, calibration/conformal prediction, risk-aware planning & safety, humanoid navigation/embodied
  AI, legged RL & distillation, splat-based real2sim.
- **Ancestors**: the papers each seed is built on (3 209 candidate references, robotics-filtered).
- **Children**: papers citing each seed (4 010 candidates).
- **Children of children**: papers citing the most-cited children (second citation hop), which is what the
  professor asked for.
- Ranking within each theme by recency + citation count; per-theme quotas keep it balanced instead of
  letting one hot area dominate.

## Files

| file | content |
|---|---|
| `papers.csv` / `papers.json` | 300 rows: theme, provenance (`seed`/`ancestor`/`child`/`grandchild`), year, venue, citations, DOI, OA URL, local PDF |
| `bibtex.bib` | BibTeX for all 300 |
| `by_theme/*.md` | 13 reading lists sorted by citations |
| `pdfs/` | **166 open-access PDFs** |
| `missing_pdfs.csv` | 134 papers without an OA PDF (DOI included) |
| `_cache/` | raw OpenAlex responses (reproducible; lets the builder re-run offline) |

## Composition

| theme | papers |
|---|---|
| gs_slam_feedforward | 35 |
| calibration_conformal | 33 |
| change_detection | 27 |
| ancestor (foundational spine) | 27 |
| foundations_planning | 26 |
| gaussian_splatting | 25 |
| risk_aware_planning_safety | 25 |
| splat_navigation | 25 |
| map_maintenance_longterm | 23 |
| legged_rl_distillation | 23 |
| humanoid_navigation_embodied | 20 |
| occupancy_and_classic_slam | 8 |
| splat_sim_real2sim | 3 |

Years: 17 papers before 2000, 26 in the 2000s, 40 in the 2010s, 217 from 2020–2026 (34 from 2026).

## How it was built

```powershell
.venv\Scripts\python.exe scripts\literature\build_literature.py      # OpenAlex citation chaining
.venv\Scripts\python.exe scripts\literature\download_pdfs.py         # OA PDFs + index
.venv\Scripts\python.exe scripts\literature\fetch_missing_pdfs.py    # Semantic Scholar / arXiv fallback
```

APIs: OpenAlex (metadata, references, citing works), Semantic Scholar batch (OA PDF fallback), arXiv
(title → PDF fallback). Rate-limited politely with caching.

## Caveats

- 134 papers are paywalled/not OA: metadata + DOI are present, PDFs are not. Add them manually if your
  institution has access.
- A few 2026 preprints have no citation counts yet (they are included as frontier seeds/descendants).
- Citation graphs reflect OpenAlex coverage; very recent work is under-cited by construction.
- To go beyond 300, re-run the builder with a larger `TARGET_TOTAL` and more seeds; the cache makes it cheap.
