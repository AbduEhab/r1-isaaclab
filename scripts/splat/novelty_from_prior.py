"""Detect new objects by rendering the prior splat and differencing against the live depth.

Prior map: runs/splat_flycam/final/gsplat_model.npz (empty warehouse).
Live data: runs/novelty_proof/live (warehouse + 4 new boxes, with GT positions in live_meta.json).

    docker run --rm --gpus all -v C:\\per\\thesis\\runs:/data \\
        -v C:\\per\\thesis\\r1-isaaclab\\scripts\\splat:/code gsplat:25.10 \\
        python /code/novelty_from_prior.py --model /data/splat_flycam/final/gsplat_model.npz \\
        --meta /data/novelty_proof/live_meta.json --frames /data/novelty_proof/live \\
        --out /data/novelty_proof/analysis
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch


# captured camera poses are in Isaac's optical convention (forward +X, up +Z); gsplat needs OpenCV
D_CV = np.array([[0.0, 0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])


def activated(model):
    return dict(
        means=torch.tensor(model["means"], dtype=torch.float32, device="cuda"),
        quats=torch.nn.functional.normalize(torch.tensor(model["quats"], dtype=torch.float32, device="cuda"), dim=-1),
        scales=torch.exp(torch.tensor(model["scales"], dtype=torch.float32, device="cuda")),
        opacities=torch.sigmoid(torch.tensor(model["opacities"], dtype=torch.float32, device="cuda")),
        colors=torch.clamp(torch.tensor(model["colors"], dtype=torch.float32, device="cuda"), 0.0, 1.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-floor", type=float, default=0.20, help="minimum depth margin [m] for novelty")
    ap.add_argument("--tau-rel", type=float, default=0.04, help="relative depth margin (fraction of prior depth)")
    ap.add_argument("--min-points", type=int, default=40, help="min novelty points for a cluster")
    ap.add_argument("--cell", type=float, default=0.20, help="cluster grid resolution [m]")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from gsplat import rasterization

    import train_gsplat_warehouse as T  # noqa: F401  (camera convention helpers)

    model_npz = np.load(args.model)
    params = activated(model_npz)
    K = np.array(model_npz["K"], dtype=np.float32)
    W, H = int(model_npz["W"]), int(model_npz["H"])
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    meta = json.load(open(args.meta))
    boxes = meta["boxes"]
    frames = meta["frames"]
    print(f"[novelty] prior model {len(params['means'])} Gaussians, {len(frames)} live frames, {len(boxes)} GT boxes")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def detect(frames_subset, tau_floor, tau_rel):
        """Return world novelty points for a depth margin (tau_floor, tau_rel)."""
        pts_all, score_all = [], []
        overlays = {}
        for f in frames_subset:
            img_path = os.path.join(args.frames, f["name"])
            depth = np.load(img_path.replace(".png", "_depth.npy")).astype(np.float32)
            c2w = np.array(f["c2w"], dtype=np.float64)
            c2w_cv = c2w @ D_CV
            w2c = np.linalg.inv(c2w_cv)
            viewmats = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
            with torch.no_grad():
                renders, alphas, _ = rasterization(
                    params["means"], params["quats"], params["scales"], params["opacities"], params["colors"],
                    viewmats, Ks, W, H, sh_degree=None, render_mode="RGB+ED")
            prior_depth = renders[0, :, :, 3].cpu().numpy()
            prior_rgb = renders[0, :, :, :3].clamp(0, 1).cpu().numpy()
            alpha = alphas[0].squeeze(-1).cpu().numpy()
            valid = (alpha > 0.5) & np.isfinite(prior_depth) & (prior_depth > 0.2) & (prior_depth < 60.0)
            live_valid = np.isfinite(depth) & (depth > 0.2) & (depth < 60.0)
            tau = np.maximum(tau_floor, tau_rel * prior_depth)
            delta = prior_depth - depth
            nov = valid & live_valid & (delta > tau)
            ys, xs = np.nonzero(nov)
            if len(xs) == 0:
                continue
            z = depth[ys, xs]
            x_cam = (xs - cx) / fx * z
            y_cam = (ys - cy) / fy * z
            p_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
            world = (c2w_cv @ p_cam.T).T[:, :3]
            keep = (world[:, 2] > 0.05) & (world[:, 2] < 3.0)
            if not keep.any():
                continue
            pts_all.append(world[keep])
            score_all.append(np.clip(delta[ys, xs][keep] / tau[ys, xs][keep], 0, 3))
            if len(overlays) < 6:
                vis = np.zeros((H, W, 3), dtype=np.uint8)
                vis[nov] = (0, 0, 255)
                pri = (prior_depth.clip(0, 20) / 20 * 255).astype(np.uint8)
                liv = (depth.clip(0, 20) / 20 * 255).astype(np.uint8)
                overlays[f["name"]] = np.hstack([
                    cv2.cvtColor(pri, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(liv, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor((prior_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    vis,
                ])
        if not pts_all:
            return np.zeros((0, 3)), np.zeros((0,)), overlays
        return np.concatenate(pts_all), np.concatenate(score_all), overlays

    def cluster(pts, scores):
        if len(pts) == 0:
            return []
        xy = pts[:, :2]
        keys = np.floor(xy / args.cell).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inv)
        clusters = []
        for k in np.argsort(counts)[::-1]:
            if counts[k] < args.min_points:
                continue
            sel = inv == k
            p = pts[sel]
            clusters.append({
                "centre": p.mean(axis=0).tolist(),
                "extent": (p.max(axis=0) - p.min(axis=0)).tolist(),
                "n_points": int(sel.sum()),
                "mean_score": float(scores[sel].mean()),
            })
        return clusters

    def match(clusters):
        used_gt = set()
        matched = 0
        for c in clusters:
            cc = np.array(c["centre"])
            best, best_d = None, 1e9
            for gi, b in enumerate(boxes):
                bc = np.array(b["centre"])
                d = float(np.linalg.norm(cc[:2] - bc[:2]))
                if d < best_d:
                    best, best_d = gi, d
            if best is not None and best_d < 0.8 and best not in used_gt:
                c["gt"] = best
                used_gt.add(best)
                matched += 1
            else:
                c["gt"] = None
        return matched, used_gt

    # main run
    pts, scores, overlays = detect(frames, args.tau_floor, args.tau_rel)
    clusters = cluster(pts, scores)
    matched, used_gt = match(clusters)
    precision = matched / max(1, len(clusters))
    recall = len(used_gt) / max(1, len(boxes))
    print(f"[novelty] {len(pts)} novelty points -> {len(clusters)} clusters | precision {precision:.2f} recall {recall:.2f}")

    # calibration sweep on the depth margin
    sweep = []
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        p2, s2, _ = detect(frames, args.tau_floor * mult, args.tau_rel * mult)
        c2 = cluster(p2, s2)
        m2, u2 = match(c2)
        sweep.append({
            "tau_floor_m": args.tau_floor * mult,
            "tau_rel": args.tau_rel * mult,
            "clusters": len(c2),
            "precision": m2 / max(1, len(c2)),
            "recall": len(u2) / max(1, len(boxes)),
        })
        print(f"[novelty] tau_floor={args.tau_floor * mult:.2f} -> clusters={len(c2)} "
              f"P={sweep[-1]['precision']:.2f} R={sweep[-1]['recall']:.2f}")

    # how much GT footprint area is covered by detections (completeness / false-safe area)
    coverage = []
    for gi, b in enumerate(boxes):
        bc = np.array(b["centre"][:2])
        hx, hy = b["size"][0] / 2, b["size"][1] / 2
        n_cells, hit = 0, 0
        for dx in np.arange(-hx, hx + 1e-6, args.cell / 2):
            for dy in np.arange(-hy, hy + 1e-6, args.cell / 2):
                p = np.array([bc[0] + dx, bc[1] + dy])
                n_cells += 1
                if any(np.linalg.norm(np.array(c["centre"][:2]) - p) < 0.6 for c in clusters if c["gt"] == gi):
                    hit += 1
        coverage.append({"box": gi, "covered_fraction": hit / max(1, n_cells)})

    # save
    np.save(os.path.join(args.out, "novelty_points.npy"), np.hstack([pts, scores[:, None]]) if len(pts) else np.zeros((0, 4)))
    for name, img in overlays.items():
        cv2.imwrite(os.path.join(args.out, "overlay_" + name), img)
    result = {
        "n_frames": len(frames),
        "n_novelty_points": int(len(pts)),
        "clusters": clusters,
        "precision": precision,
        "recall": recall,
        "calibration_sweep": sweep,
        "gt_coverage": coverage,
        "thresholds": {"tau_floor": args.tau_floor, "tau_rel": args.tau_rel, "min_points": args.min_points, "cell": args.cell},
    }
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[novelty] wrote {args.out}/detections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
