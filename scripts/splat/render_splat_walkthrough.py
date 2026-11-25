"""Render a walkthrough video of the trained warehouse splat at the recorded camera poses.

The Gaussians were optimised against these exact views, so rendering at them shows the reconstruction
quality instead of interpolation fog.

    docker run --rm --gpus all -v C:\\per\\thesis\\runs\\warehouse_splat:/data \
        -v C:\\per\\thesis\\r1-isaaclab\\scripts\\splat:/code gsplat:25.10 \
        python /code/render_splat_walkthrough.py --model /data/final/gsplat_model.npz \
        --meta /data/walk_meta.json --out /data/final
"""

import argparse
import os

import cv2
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    import json

    import train_gsplat_warehouse as T
    from gsplat import rasterization

    model = np.load(args.model)
    params = {k: torch.tensor(model[k], dtype=torch.float32, device="cuda")
              for k in ("means", "quats", "scales", "opacities", "colors")}
    act = T.activated(params)
    K = model["K"]
    W, H = int(model["W"]), int(model["H"])
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]

    meta = json.load(open(args.meta))
    frames = meta["frames"][:: args.stride]
    print(f"[render] {len(frames)} poses, {W}x{H}")

    os.makedirs(args.out, exist_ok=True)
    video_path = os.path.join(args.out, "warehouse_splat_walkthrough.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for i, f in enumerate(frames):
        c2w = np.array(f["c2w"], dtype=np.float64)
        # the capture pose is in Isaac's forward=+X/up=+Z convention (see train_gsplat_warehouse)
        w2c = np.linalg.inv(T.prim_to_cv(c2w))
        viewmats = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
        with torch.no_grad():
            renders, _, _ = rasterization(act["means"], act["quats"], act["scales"], act["opacities"],
                                          act["colors"], viewmats, Ks, W, H,
                                          sh_degree=None, render_mode="RGB")
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % 30 == 0:
            cv2.imwrite(os.path.join(args.out, f"pose_{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"[render] {i}/{len(frames)}", flush=True)
    writer.release()
    print(f"[render] wrote {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
