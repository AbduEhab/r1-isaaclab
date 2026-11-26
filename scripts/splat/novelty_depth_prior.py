"""Novelty detection using a prior map that stores captured depth observations, plus calibration metrics.

The prior map is a capture of the EMPTY warehouse taken from exactly the live viewpoints, so this is a
fair map-vs-reality test: the prior observation says what should be there, the live depth says what is
there. Changed geometry (the 4 new boxes) shows up as live depth being nearer than the prior observation.

    C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\splat\\novelty_depth_prior.py
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def backproject(depth, mask, c2w_cv, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    # the captured "world" convention has image-x increasing to the robot's left; flip it into the
    # OpenCV right-handed frame used for back-projection (verified against the GT boxes)
    x_cam = -(xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    p = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    return (c2w_cv @ p.T).T[:, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=r"C:\per\thesis\runs\novelty_proof\prior\live")
    ap.add_argument("--live", default=r"C:\per\thesis\runs\novelty_proof\live")
    ap.add_argument("--meta", default=r"C:\per\thesis\runs\novelty_proof\live_meta.json")
    ap.add_argument("--out", default=r"C:\per\thesis\runs\novelty_proof\analysis")
    ap.add_argument("--tau-floor", type=float, default=0.15)
    ap.add_argument("--tau-rel", type=float, default=0.03)
    ap.add_argument("--min-points", type=int, default=25)
    ap.add_argument("--cell", type=float, default=0.20)
    ap.add_argument("--max-frames", type=int, default=72)
    ap.add_argument("--noise-rel", type=float, default=0.0, help="live depth noise: sigma = noise_rel * z^2 [m]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(args.meta))
    boxes = meta["boxes"]
    frames = meta["frames"][: args.max_frames]
    K = np.array(frames[0]["K"], dtype=np.float64)
    D_CV = np.array([[0.0, 0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

    def load_depth(frames_dir, name):
        return np.load(os.path.join(frames_dir, name.replace(".png", "_depth.npy"))).astype(np.float32)

    def project_box_poly(b, c2w_cv):
        """Project a GT box to an image polygon (convex hull of its projected corners)."""
        cx, cy, cz = b["centre"]
        hx, hy, hz = (s / 2 for s in b["size"])
        corners = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz, 1.0]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        w2c = np.linalg.inv(c2w_cv)
        cam = (w2c @ corners.T).T[:, :3]
        if (cam[:, 2] < 0.2).any():
            return None
        # image-x increases to the robot's left in the captured convention (same flip as back-projection)
        u = -K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
        v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
        pts = np.stack([u, v], axis=1)
        hull = cv2.convexHull(pts.astype(np.float32))
        if cv2.contourArea(hull) < 30:
            return None
        return hull

    def run(tau_floor, tau_rel, collect=False):
        pts_all, score_all = [], []
        frame_stats = []
        overlays = {}
        for f in frames:
            name = f["name"]
            prior = load_depth(args.prior, name)
            live = load_depth(args.live, name)
            if args.noise_rel > 0.0:
                rs = np.random.default_rng(abs(hash(name)) % (2**32) + args.seed)
                live = live + rs.normal(0.0, 1.0, size=live.shape).astype(np.float32) * (args.noise_rel * live ** 2)
            c2w_cv = np.array(f["c2w"], dtype=np.float64) @ D_CV
            valid = np.isfinite(prior) & np.isfinite(live) & (prior > 0.2) & (prior < 60) & (live > 0.2) & (live < 60)
            tau = np.maximum(tau_floor, tau_rel * prior)
            delta = prior - live
            nov = valid & (delta > tau)
            # frame-level GT evaluation
            det = 0
            box_masks = []
            for b in boxes:
                poly = project_box_poly(b, c2w_cv)
                if poly is None:
                    continue
                pmask = np.zeros_like(nov)
                cv2.fillConvexPoly(pmask, poly.astype(np.int32), 1)
                pmask = pmask.astype(bool)
                box_masks.append(pmask)
                visible = pmask & valid
                if visible.sum() < 25:
                    continue
                if (nov & pmask).sum() > 0.25 * visible.sum():
                    det += 1
            outside = np.ones_like(nov, dtype=bool)
            for pmask in box_masks:
                outside &= ~pmask
            fp_pixels = int((nov & outside).sum())
            frame_stats.append({"name": name, "gt_visible": len(box_masks), "detected": det,
                                "fp_pixels": fp_pixels})
            if collect and len(overlays) < 8:
                vis = cv2.cvtColor((live.clip(0, 20) / 20 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                vis[nov] = (0, 0, 255)
                for pmask in box_masks:
                    ys, xs = np.nonzero(pmask)
                    if len(xs):
                        cv2.rectangle(vis, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 1)
                overlays[name] = vis
            if collect:
                world = backproject(live, nov, c2w_cv, K)
                keep = (world[:, 2] > 0.05) & (world[:, 2] < 3.0)
                if keep.any():
                    pts_all.append(world[keep])
                    score_all.append(np.clip(delta[nov][keep] / tau[nov][keep], 0, 3))
        if collect:
            pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
            scores = np.concatenate(score_all) if score_all else np.zeros((0,))
            return pts, scores, frame_stats, overlays
        return None, None, frame_stats, {}

    def cluster(pts, scores):
        """2D occupancy grid + connected components (one cluster per changed object)."""
        from scipy import ndimage
        if len(pts) == 0:
            return []
        keys = np.floor(pts[:, :2] / args.cell).astype(np.int64)
        lo = keys.min(axis=0) - 1
        grid = np.zeros((int(keys[:, 0].max() - lo[0]) + 2, int(keys[:, 1].max() - lo[1]) + 2), dtype=bool)
        gi, gj = keys[:, 0] - lo[0], keys[:, 1] - lo[1]
        grid[gi, gj] = True
        labels, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
        point_labels = labels[gi, gj]
        out = []
        for lab in np.unique(point_labels):
            if lab == 0:
                continue
            mask = point_labels == lab
            if mask.sum() < args.min_points:
                continue
            p, s = pts[mask], scores[mask]
            out.append({"centre": p.mean(axis=0).tolist(),
                        "extent": (p.max(axis=0) - p.min(axis=0)).tolist(),
                        "n_points": int(mask.sum()), "mean_score": float(s.mean())})
        out.sort(key=lambda c: -c["n_points"])
        return out

    def match(clusters):
        used = set()
        for c in clusters:
            cc = np.array(c["centre"])
            c["gt"] = None
            for gi, b in enumerate(boxes):
                bc = np.array(b["centre"][:2])
                half = np.array(b["size"][:2]) / 2 + 0.45
                if abs(cc[0] - bc[0]) <= half[0] and abs(cc[1] - bc[1]) <= half[1]:
                    c["gt"] = gi
                    used.add(gi)
                    break
        return used

    pts, scores, fstats, overlays = run(args.tau_floor, args.tau_rel, collect=True)
    clusters = cluster(pts, scores)
    used = match(clusters)
    matched = sum(1 for c in clusters if c["gt"] is not None)
    precision = matched / max(1, len(clusters))
    recall = len(used) / max(1, len(boxes))
    frame_rec = np.mean([min(1.0, fs["detected"] / max(1, fs["gt_visible"])) for fs in fstats if fs["gt_visible"] > 0])
    fp_rate = np.mean([fs["fp_pixels"] for fs in fstats])
    print(f"[depth-prior] {len(pts)} novelty points -> {len(clusters)} clusters | P={precision:.2f} R={recall:.2f} "
          f"| frame recall={frame_rec:.2f} | mean FP pixels/frame={fp_rate:.0f}")

    sweep = []
    for mult in (0.5, 1.0, 2.0, 3.0):
        p2, s2, _, _ = run(args.tau_floor * mult, args.tau_rel * mult, collect=True)
        c2 = cluster(p2, s2)
        u2 = match(c2)
        m2 = sum(1 for c in c2 if c["gt"] is not None)
        sweep.append({"tau_floor_m": args.tau_floor * mult, "tau_rel": args.tau_rel * mult,
                      "n_points": int(len(p2)), "clusters": len(c2),
                      "precision": m2 / max(1, len(c2)), "recall": len(u2) / max(1, len(boxes))})
        print(f"[depth-prior] tau={args.tau_floor * mult:.2f} m -> clusters={len(c2)} P={sweep[-1]['precision']:.2f} "
              f"R={sweep[-1]['recall']:.2f}")

    # GT footprint coverage by matched clusters (false-safe area)
    coverage = []
    for gi, b in enumerate(boxes):
        bc = np.array(b["centre"][:2])
        hx, hy = b["size"][0] / 2, b["size"][1] / 2
        cells, hit = 0, 0
        for dx in np.arange(-hx, hx + 1e-6, args.cell / 2):
            for dy in np.arange(-hy, hy + 1e-6, args.cell / 2):
                p = np.array([bc[0] + dx, bc[1] + dy])
                cells += 1
                if any(c["gt"] == gi and np.linalg.norm(np.array(c["centre"][:2]) - p) < 0.7 for c in clusters):
                    hit += 1
        coverage.append({"box": gi, "covered_fraction": hit / max(1, cells)})

    np.save(os.path.join(args.out, "novelty_points.npy"), np.hstack([pts, scores[:, None]]) if len(pts) else np.zeros((0, 4)))
    for name, img in overlays.items():
        cv2.imwrite(os.path.join(args.out, "overlay_" + name), img)
    out = {"n_frames": len(frames), "n_novelty_points": int(len(pts)), "clusters": clusters,
           "precision": precision, "recall": recall, "frame_recall": float(frame_rec),
           "mean_fp_pixels_per_frame": float(fp_rate), "calibration_sweep": sweep, "gt_coverage": coverage,
           "frame_stats": fstats,
           "thresholds": {"tau_floor": args.tau_floor, "tau_rel": args.tau_rel, "min_points": args.min_points, "cell": args.cell},
           "prior_map": "empty-warehouse depth observations captured from the same 72 viewpoints"}
    with open(os.path.join(args.out, "depth_prior_detections.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[depth-prior] wrote {args.out}/depth_prior_detections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
