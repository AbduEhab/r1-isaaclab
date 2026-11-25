"""Splat map -> planning, and new-obstacle detection by depth differencing.

Uses the R1 3DGS pipeline's artefacts:
  runs/r1/r1_splat_data/scene.npz   camera poses (world->camera), K, sparse points
  runs/r1/r1_splat_out/model.npz    43 321 Gaussians (pos, quat, scale, opac, color)

Two things this demonstrates:
1. A prior map from the splat: gaussian centres -> 2D occupancy grid (floor vs obstacle) -> A*.
2. New-obstacle detection: render the splat's depth from the current camera pose, compare with the
   live depth image (here: the same view plus an injected obstacle), and take the residual pixels
   where the live depth is *nearer* than the splat - that is the new object. Then re-plan with the
   novelty cells added so A* routes around it.

    C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\splat_novelty.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS = os.path.dirname(REPO)
SCENE = os.path.join(THESIS, "runs", "r1", "r1_splat_data", "scene.npz")
MODEL = os.path.join(THESIS, "runs", "r1", "r1_splat_out", "model.npz")
OUT = os.path.join(THESIS, "runs", "r1", "splat_novelty")

CELL = 0.10
INFLATE = 0.20
FLOOR_BAND = 0.2


# ------------------------------------------------------------------ map
def occupancy_from_points(points: np.ndarray, cell: float = CELL):
    xy = points[:, :2]
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    nx = int(np.ceil((hi[0] - lo[0]) / cell))
    ny = int(np.ceil((hi[1] - lo[1]) / cell))
    occ = np.zeros((nx, ny), dtype=bool)
    free = np.zeros((nx, ny), dtype=bool)
    hits = np.zeros((nx, ny), dtype=np.int32)

    def idx(p):
        return (int((p[0] - lo[0]) / cell), int((p[1] - lo[1]) / cell))

    for p in points:
        i, j = idx(p[:2])
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        if p[2] < FLOOR_BAND:  # floor / low points -> free
            free[i, j] = True
        else:
            hits[i, j] += 1
    occ = hits >= 2  # single stray gaussians are floaters, not walls
    # inflate obstacles by the robot radius
    r = max(1, int(round(INFLATE / cell)))
    pad = np.pad(occ, r, mode="constant", constant_values=True)
    inf = np.zeros_like(occ)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                inf |= pad[r + dx: r + dx + nx, r + dy: r + dy + ny]
    return inf, free, lo, hi


def astar(occ, lo, cell, start_xy, goal_xy):
    nx, ny = occ.shape
    s = (int((start_xy[0] - lo[0]) / cell), int((start_xy[1] - lo[1]) / cell))
    g = (int((goal_xy[0] - lo[0]) / cell), int((goal_xy[1] - lo[1]) / cell))
    free = ~occ
    def clear(c):
        r = int(0.6 / cell)
        i0, i1 = max(0, c[0] - r), min(nx, c[0] + r + 1)
        j0, j1 = max(0, c[1] - r), min(ny, c[1] + r + 1)
        free[i0:i1, j0:j1] = True

    clear(s)
    clear(g)
    if not (0 <= s[0] < nx and 0 <= s[1] < ny) or not (0 <= g[0] < nx and 0 <= g[1] < ny):
        print(f"[astar] start {s} or goal {g} outside grid {free.shape}")
        return None
    import heapq

    nbrs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    heap = [(0.0, s)]
    came = {}
    cost = {s: 0.0}
    expanded = 0
    while heap:
        _, cur = heapq.heappop(heap)
        expanded += 1
        if cur == g:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            xy = np.array([[lo[0] + (c[0] + 0.5) * cell, lo[1] + (c[1] + 0.5) * cell] for c in reversed(path)])
            return xy
        for dx, dy in nbrs:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or not free[nxt]:
                continue
            ng = cost[cur] + np.hypot(dx, dy) * cell
            if ng < cost.get(nxt, np.inf):
                cost[nxt] = ng
                came[nxt] = cur
                heapq.heappush(heap, (ng + np.hypot(nxt[0] - g[0], nxt[1] - g[1]) * cell, nxt))
    print(f"[astar] exhausted after {expanded} expansions, reached {len(cost)} cells")
    return None


# ------------------------------------------------------------------ depth render
def render_depth(points: np.ndarray, pose_w2c: np.ndarray, K: np.ndarray, W: int, H: int):
    R, t = pose_w2c[:3, :3], pose_w2c[:3, 3]
    p_cam = points @ R.T + t
    z = p_cam[:, 2]
    valid = z > 0.2
    u = K[0, 0] * p_cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inside = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi[inside], ui[inside]), z[inside].astype(np.float32))
    return depth


def depth_to_colour(depth: np.ndarray, max_depth: float = 8.0):
    d = np.clip(depth, 0, max_depth)
    vis = (1.0 - d / max_depth) * np.isfinite(depth)
    return (np.clip(vis, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sc = np.load(SCENE, allow_pickle=True)
    poses, K, names, points = sc["poses"], sc["K"], sc["names"], sc["points"]
    model = np.load(MODEL)
    gpos = model["pos"]
    keep = np.asarray(model["opac"]) > 0.3
    gpos = gpos[keep]
    print(f"[splat] {len(points)} sparse points, {len(gpos)} solid gaussians, {len(poses)} poses, K=\n{K}")

    # validate the pose convention: positions should sit inside the point cloud bounds
    c2w_pos = poses[:, :3, 3]
    w2c_pos = -np.einsum("nij,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    print(f"[splat] point bbox {points.min(axis=0).round(2)} .. {points.max(axis=0).round(2)}")
    print(f"[splat] pose translation column mean {c2w_pos.mean(axis=0).round(2)}")
    print(f"[splat] w2c camera centre mean     {w2c_pos.mean(axis=0).round(2)}")

    # ---- 1. prior map from the splat, and an A* route across it
    # camera centres come from the world->camera pose: C = -R^T t
    centers = -np.einsum("nij,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    near = np.linalg.norm(centers - np.median(centers, axis=0), axis=1) < 12.0
    centers = centers[near]
    mid = centers.mean(axis=0)
    keep2 = np.linalg.norm(gpos - mid, axis=1) < 20.0
    gpos = gpos[keep2]
    occ, free, lo, hi = occupancy_from_points(gpos)
    print(f"[map] grid {occ.shape}, occupied {occ.mean() * 100:.1f} %")
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    i0, i1 = np.unravel_index(int(np.argmax(dist)), dist.shape)
    start_xy, goal_xy = centers[i0][:2], centers[i1][:2]

    def snap(p):
        ci, cj = int((p[0] - lo[0]) / CELL), int((p[1] - lo[1]) / CELL)
        for r in range(0, 40):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    i, j = ci + dx, cj + dy
                    if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and not occ[i, j]:
                        return np.array([lo[0] + (i + 0.5) * CELL, lo[1] + (j + 0.5) * CELL])
        return p

    start_xy, goal_xy = snap(start_xy), snap(goal_xy)
    path = astar(occ, lo, CELL, start_xy, goal_xy)
    prior_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if path is not None else None
    print(f"[map] A* path start {start_xy.round(2)} goal {goal_xy.round(2)} length {prior_len}")

    # ---- 2. render the splat's depth from a pose and inject a new obstacle
    Wd, Hd = 376, 240
    # the COLMAP intrinsics are for the 1280x1080 source frames
    ORIG_W, ORIG_H = 1280.0, 1080.0
    sx, sy = Wd / ORIG_W, Hd / ORIG_H
    Ks = K.copy()
    Ks[0, :] *= sx
    Ks[1, :] *= sy
    # pick the pose that sees farthest in the centre of the image (fewest occlusions)
    candidates = [poses[i] for i in np.linspace(0, len(poses) - 1, 25).astype(int)]
    best, best_score = candidates[0], -1.0
    for cand in candidates:
        d = render_depth(gpos, cand, Ks, Wd, Hd)
        centre = d[Hd // 4: 3 * Hd // 4, Wd // 4: 3 * Wd // 4]
        fin = np.isfinite(centre)
        score = float(fin.mean()) * (float(np.median(centre[fin])) if fin.any() else 0.0)
        if score > best_score:
            best, best_score = cand, score
    pose = best
    prior_depth = render_depth(gpos, pose, Ks, Wd, Hd)
    known_raw = np.isfinite(prior_depth)
    # close the holes left by the sparse splat so the difference happens where geometry exists
    filled = np.where(known_raw, prior_depth, 0.0).astype(np.float32)
    filled = cv2.dilate(filled, np.ones((9, 9), np.uint8))
    prior_filled = np.where(filled > 0.0, filled, np.inf).astype(np.float32)
    known = np.isfinite(prior_filled)

    # the new obstacle: a 0.9 m wide, 0.9 m tall box at 2.0 m in front of the camera
    fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
    box_z = 2.0
    u0, u1 = int(cx + fx * (-0.45) / box_z), int(cx + fx * (0.45) / box_z)
    v0, v1 = int(cy + fy * (-0.45) / box_z), int(cy + fy * (0.45) / box_z)
    live_depth = prior_filled.copy()
    live_depth[v0:v1, u0:u1] = np.minimum(live_depth[v0:v1, u0:u1], box_z)

    # ---- 3. novelty = pixels whose live depth is nearer than the splat depth
    residual = np.zeros_like(prior_depth, dtype=bool)
    both = known
    residual[both] = (prior_filled[both] - live_depth[both]) > 0.25
    box_mask = np.zeros_like(residual)
    box_mask[v0:v1, u0:u1] = True
    box_residual = residual & box_mask
    tp = int(box_residual.sum())
    recall = tp / max(1, int(box_mask.sum()))
    precision = tp / max(1, int(residual.sum()))
    print(f"[novelty] box pixels {int(box_mask.sum())}, detected {tp}, "
          f"recall {100 * recall:.1f} %, precision {100 * precision:.1f} %")

    # ---- 4. re-plan with novelty cells added (a new obstacle on the route)
    if path is not None and len(path) > 2:
        # the new obstacle appears right on the planned route (e.g. a cart just parked there)
        nov_xy = path[len(path) // 2].copy()
        occ2 = occ.copy()
        ci = int((nov_xy[0] - lo[0]) / CELL)
        cj = int((nov_xy[1] - lo[1]) / CELL)
        r = int(INFLATE / CELL)
        occ2[max(0, ci - r): ci + r, max(0, cj - r): cj + r] = True
        path2 = astar(occ2, lo, CELL, start_xy, goal_xy)
        len2 = float(np.sum(np.linalg.norm(np.diff(path2, axis=0), axis=1))) if path2 is not None else None
        print(f"[map] replanned length {len2} (was {prior_len})")
    else:
        nov_xy = np.zeros(2)
        path2, len2 = None, None

    # ---- figures
    def save_map(p, name):
        vis = np.zeros((*occ.shape, 3), dtype=np.uint8)
        vis[~occ] = (40, 40, 40)
        vis[occ] = (200, 60, 60)
        if p is not None:
            for q in p:
                ci, cj = int((q[0] - lo[0]) / CELL), int((q[1] - lo[1]) / CELL)
                if 0 <= ci < occ.shape[0] and 0 <= cj < occ.shape[1]:
                    cv2.circle(vis, (cj, ci), 1, (60, 255, 60), -1)
        cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(np.flipud(vis.transpose(1, 0, 2)), cv2.COLOR_RGB2BGR))

    save_map(path, "prior_map_astar.png")
    save_map(path2, "novelty_replan.png")
    cv2.imwrite(os.path.join(OUT, "prior_depth.png"), depth_to_colour(prior_filled))
    cv2.imwrite(os.path.join(OUT, "live_depth.png"), depth_to_colour(live_depth))
    cv2.imwrite(os.path.join(OUT, "novelty_mask.png"), (residual * 255).astype(np.uint8))

    summary = {
        "poses": len(poses),
        "gaussians": int(len(gpos)),
        "map_shape": list(occ.shape),
        "occupied_fraction": float(occ.mean()),
        "astar_length_m": prior_len,
        "astar_length_after_novelty_m": len2,
        "novelty_recall": recall,
        "novelty_precision": precision,
        "new_obstacle_world_xy": nov_xy.round(3).tolist(),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[splat] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
