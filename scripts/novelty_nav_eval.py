"""Planning demo: does the novelty layer change the path and prevent collisions?

Compares two planners on the same warehouse task:
  * stale planner   - A* on the prior map only (new boxes unknown -> drives through them)
  * novelty planner - A* on the prior map + calibrated novelty clusters (detours around them)

Ground truth: the boxes recorded in live_meta.json. A path "collides" when it enters a GT box footprint
inflated by the robot radius.

    C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\novelty_nav_eval.py
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

ROBOT_RADIUS = 0.35
CELL = 0.10


def load_depth(d):
    return np.load(d.replace(".png", "_depth.npy")).astype(np.float32)


def backproject(depth, c2w_cv, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = np.isfinite(z) & (z > 0.2) & (z < 60)
    z = np.where(valid, z, 0)
    x_cam = -(us - cx) / fx * z
    y_cam = (vs - cy) / fy * z
    cam = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)
    cam = cam[valid.ravel()]
    hom = np.concatenate([cam, np.ones((len(cam), 1))], axis=1)
    world = (c2w_cv @ hom.T).T[:, :3]
    return world


def astar(occ, start, goal):
    from heapq import heappop, heappush

    nx, ny = occ.shape
    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])
    start = tuple(int(v) for v in start)
    goal = tuple(int(v) for v in goal)
    if occ[start] or occ[goal]:
        return None
    open_heap = [(h(start, goal), 0.0, start)]
    came, cost = {}, {start: 0.0}
    closed = set()
    while open_heap:
        _, c, cur = heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dx, dy, w in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or occ[nb]:
                continue
            nc = c + w
            if nc < cost.get(nb, 1e18):
                cost[nb] = nc
                came[nb] = cur
                heappush(open_heap, (nc + h(nb, goal), nc, nb))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\per\thesis\runs\novelty_proof")
    ap.add_argument("--out", default=r"C:\per\thesis\runs\novelty_proof\analysis")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.root, "live_meta.json")))
    det = json.load(open(os.path.join(args.out, "depth_prior_detections.json")))
    K = np.array(meta["frames"][0]["K"], dtype=np.float64)

    # region around the live path
    xs = [f["c2w"][0][3] for f in meta["frames"]]
    ys = [f["c2w"][1][3] for f in meta["frames"]]
    x0, x1 = min(xs) - 2.0, max(xs) + 2.0
    y0, y1 = min(ys) - 3.0, max(ys) + 3.0
    nx, ny = int(np.ceil((x1 - x0) / CELL)), int(np.ceil((y1 - y0) / CELL))

    def to_cell(xy):
        return int((xy[0] - x0) / CELL), int((xy[1] - y0) / CELL)

    def to_xy(cell):
        return np.array([x0 + (cell[0] + 0.5) * CELL, y0 + (cell[1] + 0.5) * CELL])

    # prior occupancy from the empty-warehouse depth observations
    prior_occ = np.zeros((nx, ny), dtype=bool)
    D_CV = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    for f in meta["frames"][::2]:
        depth = load_depth(os.path.join(args.root, "prior", "live", f["name"]))
        c2w_cv = np.array(f["c2w"], dtype=np.float64) @ D_CV
        w = backproject(depth, c2w_cv, K)
        h = w[:, 2]
        keep = h > 0.35
        for p in w[keep][::3]:
            c = to_cell(p[:2])
            if 0 <= c[0] < nx and 0 <= c[1] < ny:
                prior_occ[c] = True
    # inflate prior obstacles by the robot radius
    r = int(np.ceil(ROBOT_RADIUS / CELL))
    k = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    cv2.circle(k, (r, r), r, 1, -1)
    prior_infl = cv2.dilate(prior_occ.astype(np.uint8), k).astype(bool)

    # novelty obstacles from detected clusters
    nov_occ = np.zeros((nx, ny), dtype=bool)
    for c in det["clusters"]:
        cc = to_cell(c["centre"][:2])
        ex, ey = c["extent"][0], c["extent"][1]
        rx = int(np.ceil((ex / 2 + ROBOT_RADIUS) / CELL))
        ry = int(np.ceil((ey / 2 + ROBOT_RADIUS) / CELL))
        nov_occ[max(0, cc[0] - rx):cc[0] + rx + 1, max(0, cc[1] - ry):cc[1] + ry + 1] = True

    # task: drive along the row of boxes, from just before the first to just after the last
    bx = np.array([b["centre"][0] for b in meta["boxes"]])
    by = np.mean([b["centre"][1] for b in meta["boxes"]])
    row_y = float(np.median([meta["frames"][i]["c2w"][1][3] for i in range(10)]))
    start_xy = np.array([bx.min() - 2.0, row_y])
    goal_xy = np.array([bx.max() + 2.0, row_y])
    start, goal = to_cell(start_xy), to_cell(goal_xy)

    def snap(cell, occ):
        if not occ[cell]:
            return cell
        for rad in range(1, 20):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    c = (cell[0] + dx, cell[1] + dy)
                    if 0 <= c[0] < nx and 0 <= c[1] < ny and not occ[c]:
                        return c
        return cell

    # planning: stale map (prior only) vs novelty-aware map
    stale_occ = prior_infl.copy()
    nov_occ_full = prior_infl | nov_occ
    for _ in range(12):
        if not stale_occ[start] and not stale_occ[goal]:
            break
        start_xy[0] += 0.15
        goal_xy[0] -= 0.15
        start, goal = to_cell(start_xy), to_cell(goal_xy)
    start_stale, goal_stale = snap(start, stale_occ), snap(goal, stale_occ)
    start_nov, goal_nov = snap(start, nov_occ_full), snap(goal, nov_occ_full)
    print(f"[plan] grid {nx}x{ny}, bounds x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}], occupied {prior_occ.mean():.2f}")
    print(f"[plan] start {start} snapped {start_stale} occ {stale_occ[start_stale]}, goal {goal} snapped {goal_stale} occ {stale_occ[goal_stale]}")
    # keep a small free disc around start/goal (the robot is there, not an obstacle)
    def clear_disc(occ, c, rad=3):
        occ[max(0, c[0]-rad):c[0]+rad+1, max(0, c[1]-rad):c[1]+rad+1] = False
    clear_disc(stale_occ, start_stale); clear_disc(stale_occ, goal_stale)
    clear_disc(nov_occ_full, start_nov); clear_disc(nov_occ_full, goal_nov)
    path_stale = astar(stale_occ, start_stale, goal_stale)
    path_nov = astar(nov_occ_full, start_nov, goal_nov)

    # collision check against GT boxes (inflated by robot radius)
    gt_occ = np.zeros((nx, ny), dtype=bool)
    for b in meta["boxes"]:
        bc, size = b["centre"], b["size"]
        rx = int(np.ceil((size[0] / 2 + ROBOT_RADIUS) / CELL))
        ry = int(np.ceil((size[1] / 2 + ROBOT_RADIUS) / CELL))
        cc = to_cell(bc[:2])
        gt_occ[max(0, cc[0] - rx):cc[0] + rx + 1, max(0, cc[1] - ry):cc[1] + ry + 1] = True

    def stats(path):
        if path is None:
            return {"found": False}
        pts = np.array([to_xy(c) for c in path])
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        coll = sum(1 for c in path if gt_occ[c])
        return {"found": True, "start": to_xy(path[0]).tolist(), "goal": to_xy(path[-1]).tolist(),
                "length_m": length, "collision_cells": coll, "n_cells": len(path)}

    def poly(path):
        return [[float(x), float(y)] for x, y in (to_xy(c) for c in path)] if path else []
    res = {"stale_map": stats(path_stale), "novelty_map": stats(path_nov),
           "stale_path": poly(path_stale), "novelty_path": poly(path_nov),
           "gt_boxes": meta["boxes"], "clusters": det["clusters"],
           "detection": {"precision": det["precision"], "recall": det["recall"],
                         "gt_coverage": det["gt_coverage"], "calibration_sweep": det["calibration_sweep"]}}
    with open(os.path.join(args.out, "planning_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("stale   :", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["stale_map"].items()})
    print("novelty :", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["novelty_map"].items()})

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(prior_occ.T, origin="lower", cmap="gray", alpha=0.5,
              extent=[x0, x1, y0, y1])
    for b in meta["boxes"]:
        bc, size = b["centre"], b["size"]
        ax.add_patch(plt.Rectangle((bc[0] - size[0] / 2, bc[1] - size[1] / 2), size[0], size[1],
                                   edgecolor="red", facecolor="red", alpha=0.35, label="GT new object"))
    for c in det["clusters"]:
        ax.plot(c["centre"][0], c["centre"][1], "g^", markersize=10, label="detected novelty" if c is det["clusters"][0] else "")
    for path, col, lab in ((path_stale, "tab:blue", "stale-map A*"), (path_nov, "tab:orange", "novelty-aware A*")):
        if path:
            p = np.array([to_xy(c) for c in path])
            ax.plot(p[:, 0], p[:, 1], "--" if col == "tab:blue" else "-", color=col, linewidth=2, label=lab)
    ax.plot(*start_xy, "ko", label="start", markersize=8)
    ax.plot(*goal_xy, "k*", label="goal", markersize=12)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_title("Stale map vs novelty-aware planning (top view)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "planning_topview.png"), dpi=130)
    print("wrote planning_topview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
