"""Evaluate a VGGT-SLAM run against the warehouse ground truth.

VGGT-SLAM writes `poses.txt` (frame_id x y z qx qy qz qw) and `poses_points.pcd`. The capture wrote
`meta.json` with the ground-truth camera-to-world pose of every frame and the intrinsics.

    python eval_vggt_warehouse.py --poses /out/poses.txt --meta /data/vggt_warehouse/meta.json \
        --points /out/poses_points.pcd --out /out
"""

import argparse
import json
import os

import numpy as np


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity that maps src onto dst (returns s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(args.meta))
    frames = meta["frames"]

    rows = []
    for line in open(args.poses):
        vals = [float(v) for v in line.split()]
        if len(vals) >= 8:
            rows.append(vals)
    rows = np.array(rows)
    frame_ids = rows[:, 0].astype(int)
    est_pos = rows[:, 1:4]
    print(f"[eval] {len(rows)} estimated poses, {len(frames)} ground-truth frames")

    gt_pos = np.array([np.array(frames[i]["c2w"])[:3, 3] for i in frame_ids if i < len(frames)])
    keep = frame_ids < len(frames)
    est_pos = est_pos[keep]
    if len(est_pos) < 3:
        print("[eval] too few poses to align")
        return 1

    s, R, t = umeyama(est_pos, gt_pos)
    aligned = (s * (R @ est_pos.T)).T + t
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    ate = float(np.sqrt((err ** 2).mean()))
    print(f"[eval] Sim(3) alignment: scale {s:.4f}, ATE RMSE {ate:.4f} m, max {err.max():.4f} m")

    per_frame = []
    for fid, e in zip(frame_ids[keep], err):
        per_frame.append({"frame_id": int(fid), "error_m": float(e)})

    # trajectory plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], "k.-", ms=4, label="ground truth")
        ax.plot(aligned[:, 0], aligned[:, 1], "r.-", ms=4, label="VGGT-SLAM (aligned)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"VGGT-SLAM on Isaac warehouse: ATE {ate:.3f} m (scale {s:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "vggt_trajectory.png"), dpi=130)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] trajectory plot failed: {exc}")

    summary = {"poses": int(len(rows)), "aligned": int(len(est_pos)), "scale": s,
               "ate_rmse_m": ate, "ate_max_m": float(err.max()), "per_frame": per_frame}

    if args.points and os.path.exists(args.points):
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(args.points)
            pts = np.asarray(pcd.points)
            print(f"[eval] point cloud {len(pts)} points, "
                  f"extent {pts.min(0).round(2)} .. {pts.max(0).round(2)}")
            gt_stack = np.array([np.array(f["c2w"])[:3, 3] for f in frames])
            gmin, gmax = gt_stack.min(0), gt_stack.max(0)
            inside = np.all((pts[:, :2] > gmin[:2] - 5) & (pts[:, :2] < gmax[:2] + 5), axis=1)
            summary["points"] = int(len(pts))
            summary["points_near_trajectory"] = int(inside.sum())
            summary["points_min"] = pts.min(0).round(3).tolist()
            summary["points_max"] = pts.max(0).round(3).tolist()

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 8))
                sub = pts[np.random.choice(len(pts), min(60000, len(pts)), replace=False)]
                ax.scatter(sub[:, 0], sub[:, 1], c=np.clip((sub[:, 2] - 0.1) / 2.5, 0, 1),
                           s=0.4, cmap="viridis")
                ax.plot(gt_pos[:, 0], gt_pos[:, 1], "r-", lw=1.5, label="trajectory")
                ax.set_aspect("equal")
                ax.set_title("VGGT-SLAM warehouse point cloud (top view, colour = height)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(args.out, "vggt_map_topdown.png"), dpi=130)
                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"[eval] map plot failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] point cloud read failed: {exc}")

    with open(os.path.join(args.out, "vggt_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
