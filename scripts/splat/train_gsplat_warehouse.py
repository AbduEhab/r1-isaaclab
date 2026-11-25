"""Train a Gaussian splat of the Isaac warehouse with NVIDIA's gsplat (no COLMAP).

Input: the walkthrough capture - RGB frames, metric depth, and exact camera poses/intrinsics from
Isaac Sim. The poses come from the simulator, so no SfM is needed; depth back-projection initialises
the Gaussians. Training is plain 3DGS optimisation (L1 + SSIM) through the gsplat CUDA rasterizer.

    docker run --rm --gpus all -v C:\\per\\thesis\\runs\\warehouse_splat:/data \
        -v C:\\per\\thesis\\r1-isaaclab\\scripts\\splat:/code gsplat:25.10 \
        python /code/train_gsplat_warehouse.py --frames /data/frames --meta /data/walk_meta.json \
        --out /data/out --iters 7000
"""

import argparse
import json
import os

import numpy as np
import torch


# The captured camera pose is in Isaac's "world" convention: forward = +X, up = +Z. gsplat/COLMAP
# needs the OpenCV frame: x right (= up x forward), y down (=-up), z forward. Columns of the pose
# rotation are [right, up, forward] in the captured convention, so:
#   x_cv = R[:,1], y_cv = -R[:,2], z_cv = R[:,0]
D_CV = np.array([[0.0, 0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])


def prim_to_cv(c2w_prim: np.ndarray) -> np.ndarray:
    return c2w_prim @ D_CV


def to_gsplat_viewmat(c2w_prim: np.ndarray) -> np.ndarray:
    return np.linalg.inv(prim_to_cv(c2w_prim))


def load_dataset(frames_dir, meta_path):
    meta = json.load(open(meta_path))
    frames = meta["frames"]
    K = np.array(frames[0]["K"], dtype=np.float32)
    H, W = meta["height"], meta["width"]
    images, depths, viewmats, Ks, c2ws = [], [], [], [], []
    for f in frames:
        rgb_path = os.path.join(frames_dir, f["name"])
        depth_path = rgb_path.replace(".png", "_depth.npy")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue
        img = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(depth_path).astype(np.float32)
        c2w = np.array(f["c2w"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        images.append(torch.from_numpy(img).permute(2, 0, 1))
        depths.append(torch.from_numpy(depth))
        viewmats.append(torch.from_numpy(to_gsplat_viewmat(c2w)).float())  # OpenCV-convention view matrix
        Ks.append(torch.from_numpy(np.array(f["K"], dtype=np.float32)))
        c2ws.append(c2w)
    return {
        "images": torch.stack(images),
        "depths": torch.stack(depths),
        "viewmats": torch.stack(viewmats),
        "Ks": torch.stack(Ks),
        "c2ws": np.stack(c2ws),
        "K": K,
        "H": H,
        "W": W,
    }


def init_from_depth(data, device, stride=5, voxel=0.05, max_points=180_000):
    """Back-project depth into world points, then voxel-downsample."""
    K = data["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = data["H"], data["W"]
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us, vs = us.ravel(), vs.ravel()
    xs = (us - cx) / fx
    ys = (vs - cy) / fy
    pts_all, col_all = [], []
    for i in range(len(data["images"])):
        if i % 3 != 0:  # every third frame is plenty for an initialisation
            continue
        d = data["depths"][i].numpy()[vs, us]
        valid = np.isfinite(d) & (d > 0.25) & (d < 40.0)
        if not valid.any():
            continue
        z = d[valid]
        p_cam = np.stack([xs[valid] * z, ys[valid] * z, z, np.ones_like(z)], axis=1)
        world = (prim_to_cv(data["c2ws"][i]) @ p_cam.T).T[:, :3]
        cols = data["images"][i].permute(1, 2, 0).numpy()[vs[valid], us[valid]]
        pts_all.append(world)
        col_all.append(cols)
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    print(f"[gsplat] back-projected {len(pts)} points")

    # voxel downsample: keep one point per voxel (the first), colours averaged-per-voxel
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pts_ds = np.zeros((len(counts), 3), dtype=np.float64)
    col_ds = np.zeros((len(counts), 3), dtype=np.float64)
    for j, s in enumerate(starts):
        idx = order[s: s + counts[j]]
        pts_ds[j] = pts[idx].mean(axis=0)
        col_ds[j] = cols[idx].mean(axis=0)
    if len(pts_ds) > max_points:
        sel = np.random.default_rng(0).choice(len(pts_ds), max_points, replace=False)
        pts_ds, col_ds = pts_ds[sel], col_ds[sel]
    print(f"[gsplat] {len(pts_ds)} gaussians after {voxel} m voxelisation")

    n = len(pts_ds)
    means = torch.tensor(pts_ds, dtype=torch.float32, device=device)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), np.log(voxel * 0.8), dtype=torch.float32, device=device)
    opacities = torch.full((n,), -2.197, dtype=torch.float32, device=device)  # sigmoid -> 0.1
    colors = torch.tensor(col_ds, dtype=torch.float32, device=device)
    return means, quats, scales, opacities, colors


def activated(params):
    """gsplat rasterization expects activated values, not the raw 3DGS parameters."""
    return dict(
        means=params["means"],
        quats=torch.nn.functional.normalize(params["quats"], dim=-1),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.clamp(params["colors"], 0.0, 1.0),
    )


def save_ply(path, means, quats, scales, opacities, colors):
    """Standard 3DGS PLY (SH DC colour, log scales, quaternion rotation)."""
    means = means.detach().cpu().numpy()
    quats = quats.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    opacities = opacities.detach().cpu().numpy()
    colors = np.clip(colors.detach().cpu().numpy(), 0.0, 1.0)
    n = len(means)
    sh = (colors - 0.5) / 0.28209479177387814
    rot = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    fields = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
              + [f"f_rest_{i}" for i in range(45)]
              + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    with open(path, "wb") as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for name in fields:
            header += f"property float {name}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        data = np.zeros((n, len(fields)), dtype=np.float32)
        data[:, 0:3] = means
        data[:, 6:9] = sh
        data[:, 54] = opacities
        data[:, 55:58] = scales
        data[:, 58:62] = rot
        f.write(data.tobytes())
    print(f"[gsplat] wrote {path} ({n} gaussians)")


def render_flythrough(path, means, quats, scales, opacities, colors, c2ws, K, W, H):
    """Smooth camera path through the capture poses + a turntable, written to an MP4."""
    import cv2
    from gsplat import rasterization

    Ks = torch.tensor(K, dtype=torch.float32, device=means.device)[None]
    fps = 20

    def render_pose(c2w_prim):
        viewmats = torch.tensor(to_gsplat_viewmat(c2w_prim), dtype=torch.float32, device=means.device)[None]
        with torch.no_grad():
            a = activated({"means": means, "quats": quats, "scales": scales,
                           "opacities": opacities, "colors": colors})
            renders, _, _ = rasterization(a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
                                          viewmats, Ks, W, H, sh_degree=None, render_mode="RGB")
        img = renders[0].clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)

    # render the recorded walkthrough: interpolate between consecutive GT poses (slerp), which are
    # exactly the views the Gaussians were optimised against, so the video shows the reconstruction
    # quality rather than extrapolation artefacts
    from scipy.spatial.transform import Rotation, Slerp

    idx = np.arange(0, len(c2ws), 2)
    c2ws_s = c2ws[idx]
    times = np.linspace(0.0, 1.0, len(c2ws_s))
    rots = Rotation.from_matrix(c2ws_s[:, :3, :3])
    slerp = Slerp(times, rots)
    pos_t = np.array([c2ws_s[:, :3, 3]])
    frames = []
    for t in np.linspace(0.0, 1.0, 150):
        Rm = slerp([t]).as_matrix()[0]
        pos = np.array([np.interp(t, times, c2ws_s[:, :3, 3][:, k]) for k in range(3)])
        c2w_source = np.eye(4)
        c2w_source[:3, :3], c2w_source[:3, 3] = Rm, pos
        frames.append(c2w_source)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i, c2w in enumerate(frames):
        img = render_pose(c2w)
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if (i + 1) % 20 == 0:
            print(f"[gsplat] rendered {i + 1}/{len(frames)} video frames", flush=True)
    out.release()
    print(f"[gsplat] wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=180_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    data = load_dataset(args.frames, args.meta)
    print(f"[gsplat] {len(data['images'])} frames, {data['W']}x{data['H']}")
    images = data["images"].to(device)
    viewmats = data["viewmats"].to(device)
    Ks = data["Ks"].to(device)

    means, quats, scales, opacities, colors = init_from_depth(data, device, voxel=args.voxel, max_points=args.max_points)
    params = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}
    learnable = {k: torch.nn.Parameter(v.requires_grad_(True)) for k, v in params.items()}
    optimizers = {
        "means": torch.optim.Adam([learnable["means"]], lr=1.6e-4),
        "quats": torch.optim.Adam([learnable["quats"]], lr=1e-3),
        "scales": torch.optim.Adam([learnable["scales"]], lr=5e-3),
        "opacities": torch.optim.Adam([learnable["opacities"]], lr=5e-2),
        "colors": torch.optim.Adam([learnable["colors"]], lr=2.5e-3),
    }

    from gsplat import rasterization

    n_train = len(images)
    rng = np.random.default_rng(0)
    losses = []
    if not args.no_train:
        for it in range(1, args.iters + 1):
            i = int(rng.integers(n_train))
            a = activated(learnable)
            renders, alphas, _ = rasterization(
                a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
                viewmats[i][None], Ks[i][None], data["W"], data["H"],
                sh_degree=None, render_mode="RGB")
            gt = images[i][None]
            out_rgb = renders.permute(0, 3, 1, 2)  # gsplat returns HWC
            l1 = torch.abs(out_rgb - gt).mean()
            loss = l1
            try:
                from gsplat import ssim as _ssim  # noqa: F401
                loss = 0.8 * l1 + 0.2 * (1.0 - _ssim(out_rgb, gt))
            except Exception:
                pass
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers.values():
                opt.step()
            losses.append(float(loss.item()))
            if it % 500 == 0:
                print(f"[gsplat] iter {it}/{args.iters} loss {np.mean(losses[-200:]):.5f}", flush=True)

    model = {
        "means": learnable["means"].detach(),
        "quats": learnable["quats"].detach(),
        "scales": learnable["scales"].detach(),
        "opacities": learnable["opacities"].detach(),
        "colors": learnable["colors"].detach(),
    }
    np.savez(os.path.join(args.out, "gsplat_model.npz"),
             **{k: v.cpu().numpy() for k, v in model.items()},
             K=data["K"], W=data["W"], H=data["H"])
    save_ply(os.path.join(args.out, "warehouse_splat.ply"), **model)

    render_flythrough(os.path.join(args.out, "warehouse_splat_flythrough.mp4"),
                      **model, c2ws=data["c2ws"], K=data["K"], W=data["W"], H=data["H"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
